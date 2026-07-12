"""2-D UMAP of the embeddings, cached per (word, layer, step) under
``<emb_dir>/_umap_cache/`` and reused by the KDE figures."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .embeddings import layers_of, load_vectors, population, sorted_checkpoints, valid_mask

log = logging.getLogger(__name__)


def _umap2(X: np.ndarray) -> np.ndarray:
    """2-D UMAP of per-dimension-standardized X (standardizing first stops a few
    huge-variance dimensions from dominating distances)."""
    import umap

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="euclidean", random_state=0)
    return reducer.fit_transform(Xs).astype(np.float32)


def load_umap(emb_dir: Path, records, word, pos, cache_dir):
    """(meta, pos, steps, layers, coords) with coords[(step, layer)] = [N, 2],
    read from the per-(step, layer) cache and computed on first use."""
    npz_files = sorted_checkpoints(emb_dir)
    valid = valid_mask(npz_files, cache_dir)
    meta, pos = population(records, word, pos, valid)
    rows = meta["row"].to_numpy()
    steps = [int(p.stem.removeprefix("step")) for p in npz_files]
    layers = layers_of(npz_files[0])

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
                mm = load_vectors(npz, cache_dir)
            P = _umap2(np.asarray(mm[rows, j, :], np.float32))
            np.save(f, P)
            coords[(step, layer)] = P
            log.info("UMAP cached step %d layer %d (%d pts)", step, layer, len(rows))
    return meta, pos, steps, layers, coords
