"""Shared embedding loaders and a per-(word, layer, step) 2-D UMAP disk cache
(``<emb_dir>/_umap_cache/...``), computed once and reused by the KDE figures."""

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


def _load_vectors(npz_path: Path, cache_dir: Path | None):
    """Checkpoint vectors [N, L, H]. With cache_dir, keep an uncompressed .npy
    twin and memory-map it (the .npz must be fully decompressed on every load)."""
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


def _layers(npz_path: Path) -> "list[int]":
    """Which hidden layer each vector column corresponds to (stored at extract)."""
    return [int(x) for x in np.load(npz_path)["layers"]]


def _valid_mask(npz_files: list[Path], cache_dir: Path | None) -> np.ndarray:
    """Rows that survived tokenizer truncation (same across checkpoints)."""
    if cache_dir is not None and (cache_dir / "valid.npy").exists():
        return np.load(cache_dir / "valid.npy")
    last = np.asarray(_load_vectors(npz_files[0], cache_dir)[:, -1, :], np.float32)
    valid = np.linalg.norm(last, axis=1) > 0
    if cache_dir is not None:
        np.save(cache_dir / "valid.npy", valid)
    return valid


def _umap2(X: np.ndarray) -> np.ndarray:
    """2-D UMAP of per-dimension-standardized X (standardizing first stops the
    few huge-variance "rogue" embedding dimensions from dominating distances)."""
    import umap

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="euclidean", random_state=0)
    return reducer.fit_transform(Xs).astype(np.float32)


def displayed_senses(meta: pd.DataFrame, min_per_sense: int, k: int) -> "list[str]":
    """Senses to draw: >= min_per_sense occurrences, k best-attested."""
    counts = meta["sense"].value_counts()
    return [s for s, c in counts.items() if c >= min_per_sense][:k]


def _population(records, word, pos, valid) -> "tuple[pd.DataFrame, str]":
    """Valid rows of (word, pos), in dataset order. This is the UMAP population."""
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


def load_umap(emb_dir: Path, records, word, pos, cache_dir):
    """(meta, pos, steps, layers, coords) with coords[(step, layer)] = [N, 2].

    Coordinates are read from the per-(step, layer) cache, computed on first use.
    """
    npz_files = sorted(emb_dir.glob("step*.npz"),
                       key=lambda p: int(p.stem.removeprefix("step")))
    if not npz_files:
        raise SystemExit(f"No step*.npz in {emb_dir}")
    valid = _valid_mask(npz_files, cache_dir)
    meta, pos = _population(records, word, pos, valid)
    rows = meta["row"].to_numpy()

    steps = [int(p.stem.removeprefix("step")) for p in npz_files]
    layers = _layers(npz_files[0])

    ucache = emb_dir / "_umap_cache" / f"{word}.{pos}"
    ucache.mkdir(parents=True, exist_ok=True)

    coords: dict[tuple[int, int], np.ndarray] = {}
    for npz, step in zip(npz_files, steps):
        mm = None
        for j, layer in enumerate(layers):
            f = ucache / f"step{step}.layer{layer}.npy"
            if f.exists():
                P = np.load(f)
                if P.shape[0] == len(rows):
                    coords[(step, layer)] = P
                    continue
            if mm is None:
                mm = _load_vectors(npz, cache_dir)
            P = _umap2(np.asarray(mm[rows, j, :], np.float32))
            np.save(f, P)
            coords[(step, layer)] = P
            log.info("UMAP cached step %d layer %d (%d pts)", step, layer, len(rows))
    return meta, pos, steps, layers, coords
