"""Visualize a word's senses in UMAP space, over a layer × checkpoint grid.

For every (layer, checkpoint) the target word's occurrences are scattered in
their cached 2-D UMAP coordinates, coloured by WordNet sense — so you can see,
across depth and training, where the senses sit in one blob or pull apart.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .plots import INK, _style
from .umapcache import SENSE_PALETTE, displayed_senses, gloss, load_umap

log = logging.getLogger(__name__)


def visualize_word(records, emb_dir, word, pos, fig_dir,
                   cache_dir: Path | None = None, min_per_sense: int = 0) -> None:
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    meta, pos, steps, layers, coords = load_umap(emb_dir, records, word, pos, cache_dir)
    senses = displayed_senses(meta, min_per_sense, len(SENSE_PALETTE))
    if len(senses) < 2:
        raise SystemExit(f"'{word}' ({pos}) has <2 senses with >= {min_per_sense} occurrences.")
    color = {s: SENSE_PALETTE[i] for i, s in enumerate(senses)}
    sense_arr = meta["sense"].to_numpy()

    nrow, ncol = len(layers), len(steps)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.8 * nrow),
                             squeeze=False)
    for c, step in enumerate(steps):
        for r, layer in enumerate(layers):
            ax = axes[r][c]
            P = coords[(step, layer)]
            for s in senses:
                m = sense_arr == s
                ax.scatter(P[m, 0], P[m, 1], s=3, c=color[s], alpha=0.35,
                           edgecolor="none", rasterized=True)
            if r == 0:
                ax.set_title(f"step {step:,}", color=INK, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"layer {layer}", color=INK, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

    handles = [plt.Line2D([], [], marker="o", ls="", color=color[s],
                          label=f"{s} — {gloss(s)}") for s in senses]
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


def plot(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None) -> None:
    """Config-driven entry point: UMAP scatter grid for one word."""
    from .config import run_paths, store
    from .dataset import build_dataset

    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg, only)
    visualize_word(
        records, emb_dir, word, pos, store(cfg, p["fig_dir"]),
        cache_dir=cache_dir, min_per_sense=p["min_per_sense"],
    )
