"""Shared loaders for the extracted per-checkpoint embeddings, used by the purity
and KDE analyses. Pairs each dataset row with its pooled hidden-state vectors."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Distinct, print-legible categorical palette (Tableau-10 order, faint hues last).
SENSE_PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
                 "#76b7b2", "#9c755f", "#ff9da7", "#edc948", "#bab0ac"]


def gloss(sense: str, limit: int = 55) -> str:
    try:
        from nltk.corpus import wordnet as wn

        d = wn.synset(sense).definition()
        return d if len(d) <= limit else d[: limit - 3] + "..."
    except Exception:
        return ""


def load_vectors(npz_path: Path, cache_dir: Path | None):
    """Checkpoint vectors [N, L, H]. With cache_dir, keep an uncompressed .npy twin
    and memory-map it (the .npz is fully decompressed on every load)."""
    if cache_dir is None:
        return np.load(npz_path)["vectors"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy = cache_dir / (npz_path.stem + ".npy")
    if not npy.exists():
        log.info("Caching %s -> %s", npz_path.name, npy.name)
        tmp = npy.with_name(npy.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, np.load(npz_path)["vectors"])
        tmp.rename(npy)
    return np.load(npy, mmap_mode="r")


def layers_of(npz_path: Path) -> "list[int]":
    return [int(x) for x in np.load(npz_path)["layers"]]


def valid_mask(npz_files: list[Path], cache_dir: Path | None) -> np.ndarray:
    """Rows that survived tokenizer truncation (same across checkpoints)."""
    if cache_dir is not None and (cache_dir / "valid.npy").exists():
        return np.load(cache_dir / "valid.npy")
    last = np.asarray(load_vectors(npz_files[0], cache_dir)[:, -1, :], np.float32)
    valid = np.linalg.norm(last, axis=1) > 0
    if cache_dir is not None:
        np.save(cache_dir / "valid.npy", valid)
    return valid


def displayed_senses(meta: pd.DataFrame, min_per_sense: int, k: int) -> "list[str]":
    """Senses to keep: >= min_per_sense occurrences, k best-attested."""
    counts = meta["sense"].value_counts()
    return [s for s, c in counts.items() if c >= min_per_sense][:k]


def population(records, word, pos, valid) -> "tuple[pd.DataFrame, str]":
    """Valid rows of (word, pos), in dataset order."""
    meta = pd.DataFrame(records)
    meta["row"] = np.arange(len(meta))
    g = meta[meta["word"] == word]
    if g.empty:
        raise SystemExit(f"'{word}' not found in the dataset.")
    if pos is None:
        pos = g["pos"].value_counts().idxmax()
    g = g[g["pos"] == pos]
    g = g[valid[g["row"].to_numpy()]]
    if g.empty:
        raise SystemExit(f"'{word}' ({pos}) has no valid rows.")
    return g.reset_index(drop=True), pos


def sorted_checkpoints(emb_dir: Path) -> "list[Path]":
    npz = sorted(emb_dir.glob("step*.npz"), key=lambda p: int(p.stem.removeprefix("step")))
    if not npz:
        raise SystemExit(f"No step*.npz in {emb_dir}")
    return npz
