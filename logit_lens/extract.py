"""Per-checkpoint logit-lens extraction (GPU).

For every SemCor occurrence's target word, and at every (checkpoint, layer), this
teacher-forces the sentence through Pythia and asks: given everything up to (but
not including) the target's subtoken, what logprob and rank did the model give to
the actual subtoken -- i.e. was the target word the model's most likely
continuation? Multi-subtoken words get one row per subtoken (aggregate at
analysis time; see ``aggregate.py``).

One pickle per checkpoint at ``<storage>/results/logprobs_step<N>.pkl``, written
after that checkpoint completes so the job is resumable: re-running skips any
checkpoint whose file already exists. Each checkpoint's HF snapshot is purged
after use (``purge_cache``) to keep the small home quota bounded.
"""

from __future__ import annotations

import gc
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import hf_cache_dir, load_config, store
from .model import (
    load_model_and_tokenizer, load_tokenizer, logit_lens_logprobs, pick_device,
    purge_model_cache, self_check,
)
from .semcor import align_tokenizer, load_or_build_occurrences

log = logging.getLogger(__name__)


def _needed_positions(occ_df: pd.DataFrame) -> "dict[int, list[int]]":
    """sent_idx -> sorted unique query positions (subtok_pos - 1) needed across all
    occurrences in that sentence. The token AT position p is predicted by the
    hidden state produced FROM position p-1 (causal LM), so to score subtoken i we
    read the model's output at i-1. The word's first subtoken has no left context
    if it opens the sentence (query position -1); that subtoken is unscoreable and
    dropped (logged in ``extract_checkpoint``)."""
    by_sent: dict[int, set[int]] = defaultdict(set)
    for row in occ_df.itertuples():
        if row.subtok_indices is None:
            continue
        for i in row.subtok_indices:
            if i - 1 >= 0:
                by_sent[row.sent_idx].add(i - 1)
    return {s: sorted(p) for s, p in by_sent.items()}


def _sentence_batches(sent_input_ids: dict, sent_idxs: "list[int]", batch_size: int, pad_id: int):
    """Yield (sids, input_ids[b,t], attention_mask[b,t]) batches, sentences sorted
    by length so padding waste stays low (mirrors pythia_wsd_analysis_clean.ipynb's approach)."""
    sorted_sids = sorted(sent_idxs, key=lambda s: len(sent_input_ids[s]))
    for b0 in range(0, len(sorted_sids), batch_size):
        sids = sorted_sids[b0:b0 + batch_size]
        lens = [len(sent_input_ids[s]) for s in sids]
        max_len = max(lens)
        input_ids = torch.full((len(sids), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((len(sids), max_len), dtype=torch.long)
        for bi, s in enumerate(sids):
            ids = sent_input_ids[s]
            input_ids[bi, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn_mask[bi, :len(ids)] = 1
        yield sids, input_ids, attn_mask


@torch.inference_mode()
def extract_checkpoint(
    cfg: dict, step: int, occ_df: pd.DataFrame, sent_input_ids: dict,
    layers: "list[int]", out_path: Path, device: torch.device, cache_dir: Path,
) -> Path:
    model_name = cfg["model"]
    revision = cfg["revision_format"].format(step=step)
    batch_size, max_length = cfg["batch_size"], cfg["max_length"]

    log.info("Loading %s @ %s on %s (cache %s)", model_name, revision, device, cache_dir)
    model, tokenizer = load_model_and_tokenizer(model_name, revision, cache_dir, device)
    assert hasattr(model, "gpt_neox") and hasattr(model, "embed_out"), \
        f"{model_name} isn't a GPTNeoX/Pythia model; logit_lens_logprobs assumes gpt_neox.final_layer_norm + embed_out"
    pad_id = tokenizer.pad_token_id
    num_layers = model.config.num_hidden_layers
    for l in layers:
        if not 1 <= l <= num_layers:
            raise ValueError(f"layer {l} out of range 1..{num_layers} for {model_name}")

    needed = _needed_positions(occ_df)
    # sent_idx -> occurrence rows relevant to it, with (occ_row_id, subtok_indices)
    occs_by_sent: dict[int, list[tuple]] = defaultdict(list)
    for occ_row_id, row in enumerate(occ_df.itertuples()):
        if row.subtok_indices is not None and row.sent_idx in needed:
            occs_by_sent[row.sent_idx].append((occ_row_id, row.word_id, row.synset, row.subtok_indices))

    sent_idxs = [s for s in occs_by_sent if len(sent_input_ids[s]) <= max_length]
    n_truncated = len(occs_by_sent) - len(sent_idxs)
    if n_truncated:
        log.warning("%d sentence(s) exceed max_length=%d, dropped", n_truncated, max_length)

    records: list[dict] = []
    n_no_context = 0
    t0 = time.time()
    checked = False
    n_batches = (len(sent_idxs) + batch_size - 1) // batch_size
    for sids, input_ids, attn_mask in tqdm(
        _sentence_batches(sent_input_ids, sent_idxs, batch_size, pad_id),
        total=n_batches, desc=f"step {step}", unit="batch", leave=False, dynamic_ncols=True,
    ):
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
        # out.logits (top-layer, full vocab) is only needed once, to sanity-check the
        # logit lens against the model's real output; skip computing it otherwise
        # (logits_to_keep=1) since our own logit_lens_logprobs covers every requested
        # layer, including the last, from hidden_states directly.
        logits_to_keep = 0 if not checked else 1
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_hidden_states=True, logits_to_keep=logits_to_keep)

        if not checked:
            self_check(model, out, num_layers)
            checked = True

        # (batch_idx, query_pos) pairs needed in this batch, in a fixed order
        query_pairs: list[tuple[int, int]] = []
        pair_index: dict[tuple[int, int], int] = {}
        for bi, sid in enumerate(sids):
            for qp in needed[sid]:
                key = (bi, qp)
                pair_index[key] = len(query_pairs)
                query_pairs.append(key)
        if not query_pairs:
            continue
        bidx = torch.tensor([p[0] for p in query_pairs], device=device)
        qpos = torch.tensor([p[1] for p in query_pairs], device=device)

        logprobs_by_layer = {}
        for layer in layers:
            hidden_rows = out.hidden_states[layer][bidx, qpos, :]        # (n_needed, hidden)
            logprobs_by_layer[layer] = logit_lens_logprobs(model, hidden_rows)  # (n_needed, vocab)

        for bi, sid in enumerate(sids):
            ids = sent_input_ids[sid]
            for occ_row_id, word_id, synset, subtok_indices in occs_by_sent[sid]:
                for sub_pos, tok_i in enumerate(subtok_indices):
                    qp = tok_i - 1
                    if qp < 0:
                        n_no_context += 1
                        continue
                    row_idx = pair_index[(bi, qp)]
                    target_id = ids[tok_i]
                    for layer in layers:
                        lp = logprobs_by_layer[layer][row_idx]
                        target_lp = lp[target_id].item()
                        rank = int((lp > lp[target_id]).sum().item()) + 1
                        records.append({
                            "step": step, "layer": layer, "word_id": word_id, "synset": synset,
                            "occ_row_id": occ_row_id, "subtok_pos": sub_pos,
                            "n_subtokens": len(subtok_indices), "target_token_id": target_id,
                            "logprob": target_lp, "rank": rank, "is_top1": rank == 1,
                        })
        del out, logprobs_by_layer

    if n_no_context:
        log.info("%d subtoken(s) opened their sentence (no left context), skipped", n_no_context)

    df = pd.DataFrame.from_records(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)
    log.info("step %d: %d rows -> %s (%.1fmin)", step, len(df), out_path, (time.time() - t0) / 60)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return out_path


def extract(cfg: dict, force: bool = False, steps: "list[int] | None" = None,
            layers: "list[int] | None" = None) -> Path:
    occ_dir = store(cfg, "occurrences")
    results_dir = store(cfg, "results")
    occ_df, _targets_df = load_or_build_occurrences(cfg, occ_dir)

    device = pick_device(cfg["device"])
    cache_dir = hf_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)

    steps = steps if steps is not None else cfg["checkpoints"]
    layers = layers if layers is not None else cfg["layers"]

    # tokenizer/alignment is checkpoint-independent (Pythia's vocab never changes
    # across steps) -- built once and reused for every checkpoint.
    revision = cfg["revision_format"].format(step=steps[0])
    tokenizer = load_tokenizer(cfg["model"], revision, cache_dir)
    occ_df, sent_input_ids = align_tokenizer(occ_df, tokenizer)
    del tokenizer

    for step in tqdm(steps, desc="checkpoints", unit="ckpt", dynamic_ncols=True):
        out_path = results_dir / f"logprobs_step{step}.pkl"
        if out_path.exists() and not force:
            log.info("%s already extracted, skipping", out_path)
            continue
        extract_checkpoint(cfg, step, occ_df, sent_input_ids, layers, out_path, device, cache_dir)
        if cfg["purge_cache"]:
            purge_model_cache(cfg["model"], cache_dir)
    return results_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract(load_config())
