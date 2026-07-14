"""Per-condition PCA of contextual embeddings (layer x checkpoint grid).

Each panel is an independent 2-D PCA of one (layer, checkpoint): hidden states are
L2-normalized, then projected onto that condition's top two principal components.
Signs are aligned to the layer's final checkpoint so a row stops mirror-flipping,
and each panel is annotated with its cumulative explained variance. Because every
panel has its own basis, the axes are not comparable across panels — the figure
shows how sense structure sharpens within each condition, not motion across them.

Coords are cached per word so re-styling only re-plots.
"""

from __future__ import annotations

import logging

import numpy as np

from .embeddings import (displayed_senses, gloss, layers_of, load_vectors,
                        population, sorted_checkpoints, valid_mask)
from .plots import EDGE, INK_SOFT, MUTED, PAPER, SENSE_COLORS, _style, fmt_step, save

log = logging.getLogger(__name__)

BATCH = 10000      # rows per IncrementalPCA / transform batch


def _l2(chunk: np.ndarray) -> np.ndarray:
    x = chunk.astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _pca2(rows16: np.ndarray):
    """PCA -> 2 of L2-normalized rows. Returns coords [n, 2], the cumulative
    explained-variance ratio, and the two loading vectors [2, D]."""
    from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=2)
    for i in range(0, len(rows16), BATCH):
        ipca.partial_fit(_l2(rows16[i:i + BATCH]))
    out = np.empty((len(rows16), 2), np.float32)
    for i in range(0, len(rows16), BATCH):
        out[i:i + BATCH] = ipca.transform(_l2(rows16[i:i + BATCH]))
    return out, float(ipca.explained_variance_ratio_[:2].sum()), ipca.components_[:2]


def _compute(emb_dir, rows, steps, layers, cache_dir):
    """Per-condition PCA over every (step, layer), sign-aligned per layer to the
    final checkpoint. Returns coords [nS, nL, N, 2] and cum-EVR [nS, nL]."""
    npz_files = sorted_checkpoints(emb_dir)
    nS, nL, N = len(steps), len(layers), len(rows)
    D = np.load(npz_files[0])["vectors"].shape[-1]

    X = np.empty((nS, nL, N, D), np.float16)               # [step, layer, token, dim]
    for si, npz in enumerate(npz_files):
        mm = load_vectors(npz, cache_dir)
        X[si] = np.transpose(np.asarray(mm[rows], np.float16), (1, 0, 2))
        del mm

    coords = np.empty((nS, nL, N, 2), np.float32)
    evr = np.empty((nS, nL), np.float32)
    for li in range(nL):
        fits = [_pca2(X[si, li]) for si in range(nS)]
        ref = fits[-1][2]
        for si, (out, ev, comp) in enumerate(fits):
            for pc in (0, 1):
                if float(np.dot(comp[pc], ref[pc])) < 0:       # arbitrary PCA sign
                    out[:, pc] *= -1
            coords[si, li] = out
            evr[si, li] = ev
        log.info("PCA layer %d done", layers[li])
    return coords, evr


def _load_or_compute(emb_dir, word, pos, rows, steps, layers, cache_dir):
    cache = emb_dir / "_pca_cache" / f"{word}.{pos}.npz"
    if cache.exists():
        d = np.load(cache)
        if d["coords"].shape == (len(steps), len(layers), len(rows), 2):
            return d["coords"], d["evr"]
    coords, evr = _compute(emb_dir, rows, steps, layers, cache_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, coords=coords, evr=evr)
    return coords, evr


# Figure

def _extent(P, pad=0.06):
    """Square limits covering every point (equal aspect, nothing cropped)."""
    (x0, y0), (x1, y1) = P.min(0), P.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * (1 + pad) + 1e-9
    return cx - half, cx + half, cy - half, cy + half


def _short(text, n=46):
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _panel(ax, P, sense_arr, senses, color, evr):
    for s in senses:
        pts = P[sense_arr == s]
        ax.scatter(pts[:, 0], pts[:, 1], s=2.5, c=color[s], alpha=0.6,
                   edgecolor="none", rasterized=True)
    x0, x1, y0, y1 = _extent(P)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(EDGE); sp.set_linewidth(0.6)
    import matplotlib.patheffects as pe
    ax.text(0.04, 0.95, f"{evr:.0%}", transform=ax.transAxes, va="top", ha="left",
            fontsize=5.5, color=MUTED,
            path_effects=[pe.withStroke(linewidth=1.3, foreground=PAPER)])


def visualize(records, emb_dir, word, pos, fig_dir, cache_dir=None, min_per_sense=0):
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    valid = valid_mask(sorted_checkpoints(emb_dir), cache_dir)
    meta, pos = population(records, word, pos, valid)
    senses = displayed_senses(meta, min_per_sense, len(SENSE_COLORS))
    if len(senses) < 2:
        raise SystemExit(f"'{word}' ({pos}) has <2 senses with >= {min_per_sense} occurrences.")
    rows = meta["row"].to_numpy()
    steps = [int(p.stem.removeprefix("step")) for p in sorted_checkpoints(emb_dir)]
    layers = layers_of(sorted_checkpoints(emb_dir)[0])
    color = {s: SENSE_COLORS[i] for i, s in enumerate(senses)}
    sense_arr = meta["sense"].to_numpy()

    coords, evr = _load_or_compute(emb_dir, word, pos, rows, steps, layers, cache_dir)

    nrow, ncol = len(layers), len(steps)
    fig, axes = plt.subplots(nrow, ncol, figsize=(0.9 * ncol + 0.5, 0.9 * nrow + 0.7),
                             squeeze=False)
    for c, step in enumerate(steps):
        for r, layer in enumerate(layers):
            ax = axes[r][c]
            _panel(ax, coords[c, r], sense_arr, senses, color, float(evr[c, r]))
            if r == 0:
                ax.set_title(fmt_step(step), color=MUTED, pad=3, fontsize=7)
            if c == 0:
                ax.set_ylabel(str(layer), color=MUTED, rotation=0,
                              ha="right", va="center", labelpad=5, fontsize=7)

    left, right, top, bot = 0.052, 0.995, 0.925, 0.105
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bot, wspace=0.07, hspace=0.07)

    # Meta-axes: the grid's two dimensions, named once (columns = training,
    # rows = depth) instead of repeating "step"/"layer" on every panel.
    fig.text((left + right) / 2, 0.975, "training step  $\\rightarrow$",
             ha="center", va="center", fontsize=8.5, color=INK_SOFT)
    fig.text(0.011, (top + bot) / 2, "layer", rotation=90,
             ha="center", va="center", fontsize=8.5, color=INK_SOFT)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=5.5, color=color[s],
                          label=f"{s} — {_short(gloss(s))}") for s in senses]
    fig.text(right, 0.975, "% = variance in PC 1–2", ha="right", va="center",
             fontsize=6, color=MUTED)
    fig.legend(handles=handles, loc="lower center", ncols=min(len(senses), 3),
               frameon=False, fontsize=6.5, handletextpad=0.4, columnspacing=1.4,
               bbox_to_anchor=((left + right) / 2, -0.005))
    save(fig, fig_dir / f"{word}_pca.png", dpi=400)
    log.info("Wrote %s", fig_dir / f"{word}_pca.png")


def pca(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None) -> None:
    from .config import run_paths, store
    from .dataset import build_dataset

    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg, only)
    visualize(records, emb_dir, word, pos, store(cfg, p["fig_dir"]),
              cache_dir=cache_dir, min_per_sense=p["min_per_sense"])
