"""Per-word UMAP density grid (layer × checkpoint), thermal instead of scatter.

Same layout, selection and UMAP as `plot`, but each panel is a 2-D histogram of
the word's point concentration (hexbin, log-scaled) — so the overplotting that
hides 10k markers in the scatter becomes a legible thermal map. Senses are
blended together here; use `plot` for the per-sense colouring.
"""

from __future__ import annotations

import logging

import numpy as np

from .plots import INK, _style
from .word import _layers, _load_vectors, _select, _umap2

log = logging.getLogger(__name__)


def visualize_density(records, emb_dir, word, pos, fig_dir,
                      cache_dir=None, min_per_sense: int = 6,
                      gridsize: int = 60, cmap: str = "inferno") -> None:
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    rows, _labels, senses, pos, _valid, npz_files = _select(
        records, emb_dir, word, pos, min_per_sense, cache_dir)

    steps = [int(p.stem.removeprefix("step")) for p in npz_files]
    layers = _layers(npz_files[0])

    nrow, ncol = len(layers), len(steps)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.8 * nrow),
                             squeeze=False)

    for c, (npz, step) in enumerate(zip(npz_files, steps)):
        mm = _load_vectors(npz, cache_dir)
        for r, layer in enumerate(layers):
            ax = axes[r][c]
            W = np.asarray(mm[rows, r, :], np.float32)   # word rows at this layer
            P = _umap2(W)
            ax.hexbin(P[:, 0], P[:, 1], gridsize=gridsize, cmap=cmap,
                      bins="log", linewidths=0.0, rasterized=True)
            ax.set_facecolor("black")   # empty cells read as cold
            ax.grid(False)
            if r == 0:
                ax.set_title(f"step {step:,}", color=INK, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"layer {layer}", color=INK, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        log.info("step %d: density done (%d layers)", step, len(layers))

    fig.suptitle(f"“{word}” ({pos}) — UMAP point density over {len(senses)} "
                 f"senses (rows = layers, columns = checkpoints; hot = dense)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    out = fig_dir / f"{word}_density.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)


def heatmap(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None) -> None:
    """Config-driven entry point: UMAP density grid for one word."""
    from .config import run_paths, store
    from .dataset import build_dataset

    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg, only)
    visualize_density(
        records, emb_dir, word, pos, store(cfg, p["fig_dir"]),
        cache_dir=cache_dir, min_per_sense=p["min_per_sense"],
    )
