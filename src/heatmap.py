"""Per-word UMAP density grid (layer × checkpoint), thermal instead of scatter.

Same layout and cached coordinates as `plot`, but each panel is a 2-D histogram
of point concentration (hexbin, log-scaled) — so the overplotting that hides 10k
markers in the scatter becomes a legible thermal map. Senses are blended here;
use `plot` for per-sense colour or `density` for per-sense KDE.
"""

from __future__ import annotations

import logging

from .plots import INK, _style
from .umapcache import load_umap

log = logging.getLogger(__name__)


def visualize_density(records, emb_dir, word, pos, fig_dir,
                      cache_dir=None, gridsize: int = 60, cmap: str = "inferno") -> None:
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    meta, pos, steps, layers, coords = load_umap(emb_dir, records, word, pos, cache_dir)
    n_senses = meta["sense"].nunique()

    nrow, ncol = len(layers), len(steps)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.8 * nrow),
                             squeeze=False)
    for c, step in enumerate(steps):
        for r, layer in enumerate(layers):
            ax = axes[r][c]
            P = coords[(step, layer)]
            ax.hexbin(P[:, 0], P[:, 1], gridsize=gridsize, cmap=cmap,
                      bins="log", linewidths=0.0, rasterized=True)
            ax.set_facecolor("black")   # empty cells read as cold
            ax.grid(False)
            if r == 0:
                ax.set_title(f"step {step:,}", color=INK, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"layer {layer}", color=INK, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"“{word}” ({pos}) — UMAP point density over {n_senses} senses "
                 f"(rows = layers, columns = checkpoints; hot = dense)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    out = fig_dir / f"{word}_thermal.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)


def heatmap(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None) -> None:
    """Config-driven entry point: UMAP thermal density grid for one word."""
    from .config import run_paths, store
    from .dataset import build_dataset

    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg, only)
    visualize_density(
        records, emb_dir, word, pos, store(cfg, p["fig_dir"]), cache_dir=cache_dir,
    )
