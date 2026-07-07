"""Extract contextual representations of target words from Pythia checkpoints.

For every instance (sentence + char span of the target word) and every
checkpoint, we run the model in inference mode with `output_hidden_states=True`
and mean-pool the hidden states of the sub-tokens that overlap the target
span, at every layer (embedding layer 0 included, so layer separation can be
contrasted against the static-embedding baseline).

Output: one compressed .npz per checkpoint at
`<out_dir>/<model_name>/step<N>.npz` containing

    vectors: float16 array [n_instances, n_layers + 1, hidden_size]

where row i corresponds to line i of the dataset JSONL (order is the
alignment key between embeddings and metadata).

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

# Pythia released checkpoints: step0, powers of two up to 512, then every
# 1000 steps up to 143000. Log-spaced default so early training is covered.
DEFAULT_STEPS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000,
]


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
        return (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    return torch.float32


def _target_token_indices(
    offsets: "list[tuple[int, int]]", start: int, end: int
) -> "list[int]":
    """Indices of sub-tokens overlapping the char span [start, end)."""
    return [
        i for i, (s, e) in enumerate(offsets) if s < end and e > start and e > s
    ]


@torch.inference_mode()
def extract_checkpoint(
    model_name: str,
    step: int,
    records: "list[dict]",
    out_dir: Path,
    batch_size: int = 16,
    max_length: int = 128,
    device: torch.device | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Run one checkpoint over all instances and save pooled vectors."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    revision = f"step{step}"
    out_path = out_dir / model_name.split("/")[-1] / f"{revision}.npz"
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
        model_name,
        revision=revision,
        dtype=_model_dtype(device),
        cache_dir=cache_dir,
    )
    model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers + 1  # + embedding layer
    hidden = model.config.hidden_size
    vectors = np.zeros((len(records), n_layers, hidden), dtype=np.float16)

    n_truncated = 0
    for b0 in tqdm(
        range(0, len(records), batch_size), desc=revision, unit="batch"
    ):
        batch = records[b0 : b0 + batch_size]
        enc = tokenizer(
            [r["sentence"] for r in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offset_mapping = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        # [n_layers, B, T, H] -> pooled per instance
        hs = torch.stack(out.hidden_states).float()

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
        log.warning(
            "%d/%d targets lost to truncation (rows left as zeros)",
            n_truncated,
            len(records),
        )
    np.savez_compressed(out_path, vectors=vectors, step=step)
    log.info("Saved %s (%s)", out_path, vectors.shape)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return out_path


def extract_all(
    model_name: str,
    steps: "list[int]",
    records: "list[dict]",
    out_dir: Path,
    **kwargs,
) -> "list[Path]":
    paths = []
    for step in steps:
        paths.append(
            extract_checkpoint(model_name, step, records, out_dir, **kwargs)
        )
    return paths
