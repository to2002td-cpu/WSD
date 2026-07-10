"""Single large publication panel for one (layer, checkpoint).

Colour encodes the WordNet sense, marker shape encodes the writing style (the
most frequent styles get distinct markers; the rest share an "other" dot), and a
translucent per-sense Gaussian-KDE cloud sits behind the points. Saved as a
300-dpi PNG and a vector PDF (points rasterized, text kept as editable fonts).
"""

from __future__ import annotations

import logging

import numpy as np

from .density import GRID, KDE_MAX, _alpha_cmap
from .plots import BASELINE, INK, MUTED, _style
from .umapcache import SENSE_PALETTE, displayed_senses, gloss, load_umap

log = logging.getLogger(__name__)

# Distinct filled markers, in decreasing legibility order.
STYLE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "8"]
POINTS_PER_SENSE = 350     # subsample the marker layer so shapes stay legible


def _pick(value, options, name):
    if value is None:
        return options[len(options) // 2] if name == "layer" else options[-1]
    if value not in options:
        raise SystemExit(f"{name} {value} not extracted; available: {options}")
    return value


def _kde_cloud(ax, P, mask, cmap, xx, yy, grid):
    from scipy.stats import gaussian_kde

    pts = P[mask]
    if len(pts) < 15:
        return
    if len(pts) > KDE_MAX:
        pts = pts[np.random.default_rng(0).choice(len(pts), KDE_MAX, replace=False)]
    try:
        Z = gaussian_kde(pts.T)(grid).reshape(xx.shape)
    except Exception:
        return
    if Z.max() > 0:
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.max() * 0.2, Z.max(), 6),
                    cmap=cmap, antialiased=True, zorder=1)


def visualize_panel(records, emb_dir, word, pos, fig_dir, layer, step,
                    cache_dir=None, min_per_sense: int = 0,
                    n_styles: int = len(STYLE_MARKERS)) -> None:
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    meta, pos, steps, layers, coords = load_umap(emb_dir, records, word, pos, cache_dir)
    layer = _pick(layer, layers, "layer")
    step = _pick(step, steps, "step")

    senses = displayed_senses(meta, min_per_sense, len(SENSE_PALETTE))
    if len(senses) < 2:
        raise SystemExit(f"'{word}' ({pos}) has <2 senses with >= {min_per_sense} occurrences.")
    color = {s: SENSE_PALETTE[i] for i, s in enumerate(senses)}

    top_styles = list(meta["style"].value_counts().index[:n_styles])
    marker = {st: STYLE_MARKERS[i] for i, st in enumerate(top_styles)}

    P = coords[(step, layer)]
    sense_arr = meta["sense"].to_numpy()
    style_arr = meta["style"].fillna("?").to_numpy()
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_facecolor("white")
    ax.grid(False)

    x0, x1 = P[:, 0].min(), P[:, 0].max()
    y0, y1 = P[:, 1].min(), P[:, 1].max()
    mx, my = 0.05 * (x1 - x0 + 1e-9), 0.05 * (y1 - y0 + 1e-9)
    xx, yy = np.mgrid[x0 - mx:x1 + mx:GRID * 1j, y0 - my:y1 + my:GRID * 1j]
    grid = np.vstack([xx.ravel(), yy.ravel()])

    for s in senses:                                   # KDE clouds behind points
        _kde_cloud(ax, P, sense_arr == s, _alpha_cmap(color[s], 0.45), xx, yy, grid)

    for s in senses:                                   # subsampled marker layer
        idx = np.where(sense_arr == s)[0]
        if len(idx) > POINTS_PER_SENSE:
            idx = rng.choice(idx, POINTS_PER_SENSE, replace=False)
        for st in top_styles:
            m = idx[style_arr[idx] == st]
            if len(m):
                ax.scatter(P[m, 0], P[m, 1], s=26, c=color[s], marker=marker[st],
                           alpha=0.75, edgecolor="white", linewidths=0.3,
                           zorder=3, rasterized=True)
        other = idx[~np.isin(style_arr[idx], top_styles)]
        if len(other):
            ax.scatter(P[other, 0], P[other, 1], s=7, c=color[s], marker=".",
                       alpha=0.4, edgecolor="none", zorder=2, rasterized=True)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(BASELINE)
    ax.set_title(f"“{word}” ({pos}) — UMAP at layer {layer}, step {step:,}\n"
                 f"colour = sense · marker = style", color=INK)

    sense_handles = [plt.Line2D([], [], marker="o", ls="", color=color[s], markersize=8,
                                label=f"{s} — {gloss(s, 28)}") for s in senses]
    style_handles = [plt.Line2D([], [], marker=marker[st], ls="", color="0.25", markersize=8,
                                label=st) for st in top_styles]
    style_handles.append(plt.Line2D([], [], marker=".", ls="", color="0.25", markersize=8,
                                    label="other styles"))
    leg1 = ax.legend(handles=sense_handles, title="Sense", loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, labelcolor=INK)
    leg1.get_title().set_color(INK)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=style_handles, title="Style (marker)", loc="upper left",
                     bbox_to_anchor=(1.02, 0.44), borderaxespad=0.0, labelcolor=MUTED)
    leg2.get_title().set_color(INK)

    stem = fig_dir / f"{word}_panel_L{layer}_s{step}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
    log.info("Wrote %s.{png,pdf}", stem)


def panel(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None,
          layer: int | None = None, step: int | None = None) -> None:
    """Config-driven entry point: single publication panel for one word."""
    from .config import run_paths, store
    from .dataset import build_dataset

    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg, only)
    visualize_panel(
        records, emb_dir, word, pos, store(cfg, p["fig_dir"]), layer, step,
        cache_dir=cache_dir, min_per_sense=p["min_per_sense"],
    )
