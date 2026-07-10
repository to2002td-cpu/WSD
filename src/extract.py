"""
For every instance (sentence + char span of the target word) and every
checkpoint, we run the model under inference mode with hidden states on and
mean-pool the sub-tokens overlapping the target span, at each configured layer.

Output: one compressed .npz per checkpoint at
``<emb_dir>/<model_name>/step<N>.npz`` containing

    vectors: float16 [n_instances, n_layers, hidden_size]
    layers:  int     [n_layers]      (which hidden layer each column is)
    step:    int

Row i corresponds to line i of dataset.jsonl (order is the alignment key).
Weights are never modified; everything runs under torch.inference_mode().
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _target_token_indices(offsets, start: int, end: int) -> "list[int]":
    """Indices of sub-tokens overlapping the char span [start, end)."""
    return [i for i, (s, e) in enumerate(offsets) if s < end and e > start and e > s]


@torch.inference_mode()
def extract_checkpoint(
    model_name: str,
    step: int,
    records: "list[dict]",
    emb_dir: Path,
    layers: "list[int]",
    batch_size: int,
    max_length: int,
    device: torch.device,
    cache_dir: str | None = None,
) -> Path:
    """Run one checkpoint over all instances and save pooled vectors."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = f"step{step}"
    out_path = emb_dir / f"{revision}.npz"
    if out_path.exists():
        log.info("%s already extracted, skipping", out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s @ %s on %s", model_name, revision, device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=_model_dtype(device), cache_dir=cache_dir
    )
    model.to(device)
    model.eval()

    max_layer = model.config.num_hidden_layers
    for layer in layers:
        if not (1 <= layer <= max_layer):
            raise ValueError(f"Layer {layer} out of range 1..{max_layer}")

    hidden = model.config.hidden_size
    vectors = np.zeros((len(records), len(layers), hidden), dtype=np.float16)

    n_truncated = 0
    progress = tqdm(
        range(0, len(records), batch_size),
        desc=f"  {revision} embeddings", unit="batch", leave=False, dynamic_ncols=True,
    )
    for b0 in progress:
        batch = records[b0 : b0 + batch_size]
        enc = tokenizer(
            [r["sentence"] for r in batch],
            return_tensors="pt", padding=True, truncation=True,
            max_length=max_length, return_offsets_mapping=True,
        )
        offset_mapping = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = torch.stack([out.hidden_states[layer] for layer in layers]).float()
        for j, rec in enumerate(batch):
            idx = _target_token_indices(
                offset_mapping[j].tolist(), rec["target_start"], rec["target_end"]
            )
            if not idx:  # target fell beyond truncation
                n_truncated += 1
                continue
            pooled = hs[:, j, idx, :].mean(dim=1)  # [n_layers, H]
            vectors[b0 + j] = pooled.cpu().numpy().astype(np.float16)

    if n_truncated:
        log.warning("%d/%d targets lost to truncation (rows left as zeros)",
                    n_truncated, len(records))
    np.savez_compressed(out_path, vectors=vectors, step=step, layers=np.array(layers))
    log.info("Saved %s (%s)", out_path, vectors.shape)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return out_path


def _purge_model_cache(model_name: str, cache_dir: str | None) -> None:
    """Delete the HF cache entry (all downloaded revisions) of the model.

    Each Pythia-6.9b revision is a full ~14 GB snapshot; on a cluster we drop
    the weights as soon as a checkpoint's .npz is written.
    """
    import shutil

    from huggingface_hub.constants import HF_HUB_CACHE

    target = Path(cache_dir or HF_HUB_CACHE) / ("models--" + model_name.replace("/", "--"))
    if target.exists():
        shutil.rmtree(target)
        log.info("Purged HF cache %s", target)


def extract(cfg: dict, records: "list[dict]", only: "list[str] | None" = None) -> Path:
    """Extract every configured checkpoint over the dataset."""
    from .config import run_paths

    ex = cfg["extract"]
    model_name = ex["model"]
    _, emb_dir = run_paths(cfg, only)
    device = pick_device(ex["device"])

    for step in tqdm(ex["steps"], desc="checkpoints", unit="ckpt", dynamic_ncols=True):
        extract_checkpoint(
            model_name, step, records, emb_dir, ex["layers"],
            batch_size=ex["batch_size"], max_length=ex["max_length"],
            device=device, cache_dir=ex["cache_dir"],
        )
        if ex["purge_cache"]:
            _purge_model_cache(model_name, ex["cache_dir"])
    return emb_dir
