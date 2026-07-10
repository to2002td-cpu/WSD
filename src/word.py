"""Visualize a word's senses in UMAP space, over a layer × checkpoint grid.

For every (layer, checkpoint) we UMAP the target word's occurrences
(per-dimension standardized first, to stop the transformer's rogue dimensions
from dominating the distances) and scatter them coloured by WordNet sense — so
you can see, across depth and training, where the senses sit in one blob or pull
apart.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .plots import INK, _style

log = logging.getLogger(__name__)

MIN_PER_SENSE = 6    # a sense needs this many occurrences to be shown
# Distinct, print-legible categorical palette (Tableau-10 order, faint hues last).
SENSE_PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
                 "#76b7b2", "#9c755f", "#ff9da7", "#edc948", "#bab0ac"]


def _gloss(sense: str) -> str:
    try:
        from nltk.corpus import wordnet as wn

        d = wn.synset(sense).definition()
        return d if len(d) <= 55 else d[:52] + "..."
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


def _select(records, emb_dir, word, pos, min_per_sense, cache_dir):
    """Locate the target word's two best-attested senses. Returns
    (rows, labels, senses, pos, valid, npz_files)."""
    meta = pd.DataFrame(records)
    meta["row"] = np.arange(len(meta))
    g = meta[meta["word"] == word]
    if g.empty:
        raise SystemExit(f"'{word}' not found in the dataset.")
    if pos is None:
        pos = g["pos"].value_counts().idxmax()
    g = g[g["pos"] == pos]

    npz_files = sorted(emb_dir.glob("step*.npz"),
                       key=lambda p: int(p.stem.removeprefix("step")))
    if not npz_files:
        raise SystemExit(f"No step*.npz in {emb_dir}")
    valid = _valid_mask(npz_files, cache_dir)
    g = g[valid[g["row"].to_numpy()]]

    counts = g["sense"].value_counts()   # descending
    # Keep only synsets whose own lemma IS the target word — e.g. drop
    # clock_time.n.01 / fourth_dimension.n.01, where "time" is just a synonym.
    senses = [s for s, c in counts.items()
              if c >= min_per_sense
              and s.rsplit(".", 2)[0] == word][:len(SENSE_PALETTE)]
    if len(senses) < 2:
        raise SystemExit(
            f"'{word}' ({pos}) has <2 senses with >= {min_per_sense} "
            f"occurrences; counts: {dict(counts)}")
    g = g[g["sense"].isin(senses)]
    return g["row"].to_numpy(), g["sense"].to_numpy(), senses, pos, valid, npz_files


def _umap2(X: np.ndarray) -> np.ndarray:
    """2-D UMAP of per-dimension-standardized X (standardizing first stops the
    few huge-variance "rogue" embedding dimensions from dominating distances)."""
    import umap

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="euclidean", random_state=0)
    return reducer.fit_transform(Xs)


def visualize_word(records, emb_dir, word, pos, fig_dir,
                   cache_dir: Path | None = None,
                   min_per_sense: int = MIN_PER_SENSE) -> None:
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    rows, labels, senses, pos, valid, npz_files = _select(
        records, emb_dir, word, pos, min_per_sense, cache_dir)
    color = {s: SENSE_PALETTE[i] for i, s in enumerate(senses)}

    steps = [int(p.stem.removeprefix("step")) for p in npz_files]
    layers = _layers(npz_files[0])           # data column j -> layers[j]

    nrow, ncol = len(layers), len(steps)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.8 * nrow),
                             squeeze=False)

    for c, (npz, step) in enumerate(zip(npz_files, steps)):
        mm = _load_vectors(npz, cache_dir)
        for r, layer in enumerate(layers):
            ax = axes[r][c]
            W = np.asarray(mm[rows, r, :], np.float32)   # word rows at this layer
            P = _umap2(W)
            for s in senses:
                m = labels == s
                ax.scatter(P[m, 0], P[m, 1], s=12, c=color[s], alpha=0.85,
                           edgecolor="white", linewidth=0.2)
            if r == 0:
                ax.set_title(f"step {step:,}", color=INK, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"layer {layer}", color=INK, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        log.info("step %d: UMAP done (%d layers)", step, len(layers))

    handles = [plt.Line2D([], [], marker="o", ls="", color=color[s],
                          label=f"{s} — {_gloss(s)}") for s in senses]
    fig.legend(handles=handles, loc="lower center", ncols=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"“{word}” ({pos}) — {len(senses)} senses in UMAP space "
                 f"(rows = layers, columns = checkpoints)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = fig_dir / f"{word}_umap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)


def plot(cfg: dict, word: str, pos: str | None) -> None:
    """Config-driven entry point: UMAP grid for one word."""
    from .config import store
    from .dataset import build_dataset

    ex, p = cfg["extract"], cfg["plot"]
    emb_dir = store(cfg, ex["emb_dir"]) / ex["model"].split("/")[-1]
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg)
    visualize_word(
        records, emb_dir, word, pos, store(cfg, p["fig_dir"]),
        cache_dir=cache_dir, min_per_sense=p["min_per_sense"],
    )
