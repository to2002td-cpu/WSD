"""Stage 2 (GPU): mean-pool the target token's hidden states at each checkpoint.

One compressed .npz per checkpoint at ``<emb_dir>/<model>/step<N>.npz`` with
``vectors`` float16 [n_instances, n_layers, hidden], ``layers`` and ``step``.
Row i aligns with line i of dataset.jsonl. Weights are never modified.
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
    return [i for i, (s, e) in enumerate(offsets) if s < end and e > start and e > s]


@torch.inference_mode()
def extract_checkpoint(
    model_name: str,
    step: int,
    revision: str,
    records: "list[dict]",
    emb_dir: Path,
    layers: "list[int]",
    batch_size: int,
    max_length: int,
    device: torch.device,
    cache_dir: str | None = None,
) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = emb_dir / f"step{step}.npz"
    if out_path.exists():
        log.info("%s already extracted, skipping", out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s @ %s on %s", model_name, revision, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=_model_dtype(device), cache_dir=cache_dir
    )
    model.to(device)
    model.eval()

    max_layer = model.config.num_hidden_layers
    for layer in layers:
        if not 1 <= layer <= max_layer:
            raise ValueError(f"Layer {layer} out of range 1..{max_layer} for {model_name}")

    vectors = np.zeros((len(records), len(layers), model.config.hidden_size), dtype=np.float16)
    n_truncated = 0
    progress = tqdm(range(0, len(records), batch_size),
                    desc=f"  {revision}", unit="batch", leave=False, dynamic_ncols=True)
    for b0 in progress:
        batch = records[b0:b0 + batch_size]
        enc = tokenizer([r["sentence"] for r in batch], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = torch.stack([out.hidden_states[layer] for layer in layers]).float()
        for j, rec in enumerate(batch):
            idx = _target_token_indices(offsets[j].tolist(), rec["target_start"], rec["target_end"])
            if not idx:                                  # target beyond truncation -> row left zero
                n_truncated += 1
                continue
            vectors[b0 + j] = hs[:, j, idx, :].mean(dim=1).cpu().numpy().astype(np.float16)

    if n_truncated:
        log.warning("%d/%d targets lost to truncation", n_truncated, len(records))
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
    """Drop the model's HF cache (each revision is a full snapshot) so disk stays
    bounded to one checkpoint on the cluster."""
    import shutil

    from huggingface_hub.constants import HF_HUB_CACHE

    target = Path(cache_dir or HF_HUB_CACHE) / ("models--" + model_name.replace("/", "--"))
    if target.exists():
        shutil.rmtree(target)
        log.info("Purged HF cache %s", target)


def _revision(ex: dict, step: int, i: int) -> str:
    """HF revision for a checkpoint. Pythia uses ``step{N}`` (the default format);
    models whose branch names embed extra fields (e.g. OLMo token counts) list
    them explicitly in ``extract.revisions`` parallel to ``steps``."""
    revisions = ex.get("revisions")
    if revisions:
        return revisions[i]
    return ex.get("revision_format", "step{step}").format(step=step)


def extract(cfg: dict, records: "list[dict]", only: "list[str] | None" = None) -> Path:
    from .config import run_paths

    ex = cfg["extract"]
    _, emb_dir = run_paths(cfg, only)
    device = pick_device(ex["device"])
    for i, step in enumerate(tqdm(ex["steps"], desc="checkpoints", unit="ckpt", dynamic_ncols=True)):
        extract_checkpoint(
            ex["model"], step, _revision(ex, step, i), records, emb_dir, ex["layers"],
            batch_size=ex["batch_size"], max_length=ex["max_length"],
            device=device, cache_dir=ex["cache_dir"],
        )
        if ex["purge_cache"]:
            _purge_model_cache(ex["model"], ex["cache_dir"])
    return emb_dir
