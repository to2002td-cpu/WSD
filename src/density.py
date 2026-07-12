"""Per-word UMAP scatter + per-sense Gaussian-KDE overlay (layer × checkpoint grid)."""

from __future__ import annotations

import logging

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from .plots import INK, _style
from .umapcache import SENSE_PALETTE, displayed_senses, gloss, load_umap

log = logging.getLogger(__name__)

KDE_MAX = 800      # subsample per sense before fitting the KDE (speed)
GRID = 70          # KDE evaluation grid resolution


def _alpha_cmap(color: str, amax: float = 0.5):
    r, g, b, _ = to_rgba(color)
    return LinearSegmentedColormap.from_list("", [(r, g, b, 0.0), (r, g, b, amax)])


def _panel(ax, P, sense_arr, senses, color, rng):
    from scipy.stats import gaussian_kde

    x0, x1 = P[:, 0].min(), P[:, 0].max()
    y0, y1 = P[:, 1].min(), P[:, 1].max()
    mx, my = 0.05 * (x1 - x0 + 1e-9), 0.05 * (y1 - y0 + 1e-9)
    xx, yy = np.mgrid[x0 - mx:x1 + mx:GRID * 1j, y0 - my:y1 + my:GRID * 1j]
    grid = np.vstack([xx.ravel(), yy.ravel()])

    for s in senses:
        pts = P[sense_arr == s]
        ax.scatter(pts[:, 0], pts[:, 1], s=2, c=color[s], alpha=0.25,
                   edgecolor="none", rasterized=True)
        if len(pts) < 15:
            continue
        sample = pts if len(pts) <= KDE_MAX else pts[rng.choice(len(pts), KDE_MAX, replace=False)]
        try:
            Z = gaussian_kde(sample.T)(grid).reshape(xx.shape)
        except Exception:
            continue
        if Z.max() <= 0:
            continue
        levels = np.linspace(Z.max() * 0.2, Z.max(), 5)
        ax.contourf(xx, yy, Z, levels=levels, cmap=_alpha_cmap(color[s]), antialiased=True)


def visualize_kde(records, emb_dir, word, pos, fig_dir,
                  cache_dir=None, min_per_sense: int = 0) -> None:
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    meta, pos, steps, layers, coords = load_umap(emb_dir, records, word, pos, cache_dir)
    senses = displayed_senses(meta, min_per_sense, len(SENSE_PALETTE))
    if len(senses) < 2:
        raise SystemExit(f"'{word}' ({pos}) has <2 senses with >= {min_per_sense} occurrences.")
    color = {s: SENSE_PALETTE[i] for i, s in enumerate(senses)}
    sense_arr = meta["sense"].to_numpy()
    rng = np.random.default_rng(0)

    nrow, ncol = len(layers), len(steps)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.8 * nrow),
                             squeeze=False)
    for c, step in enumerate(steps):
        for r, layer in enumerate(layers):
            ax = axes[r][c]
            _panel(ax, coords[(step, layer)], sense_arr, senses, color, rng)
            if r == 0:
                ax.set_title(f"step {step:,}", color=INK, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"layer {layer}", color=INK, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        log.info("KDE step %d done (%d layers)", step, len(layers))

    handles = [plt.Line2D([], [], marker="o", ls="", color=color[s],
                          label=f"{s} — {gloss(s)}") for s in senses]
    fig.legend(handles=handles, loc="lower center", ncols=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"“{word}” ({pos}) — {len(senses)} senses, per-sense KDE "
                 f"(rows = layers, columns = checkpoints)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = fig_dir / f"{word}_density.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)


def density(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None) -> None:
    """Config-driven entry point: per-sense KDE scatter grid for one word."""
    from .config import run_paths, store
    from .dataset import build_dataset

    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg, only)
    visualize_kde(
        records, emb_dir, word, pos, store(cfg, p["fig_dir"]),
        cache_dir=cache_dir, min_per_sense=p["min_per_sense"],
    )
