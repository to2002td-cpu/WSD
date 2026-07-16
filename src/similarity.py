"""Sense x sense similarity matrix, one panel per (layer, checkpoint).

For each sense, a small sample of its occurrences (a contiguous block) is drawn;
blocks are concatenated in sense order and their pairwise cosine similarity is
plotted as one image, so a sense's block is visibly clean (bright on-block, dark
off-block) when it's separated, and bleeds into a neighbouring block when it
overlaps -- no reduction to a single per-sense number, and no assumption that a
sense forms one compact cluster.

Contextual embeddings are anisotropic: at a given (layer, checkpoint), even
unrelated tokens from unrelated words share a large positive cosine similarity
(measured on this corpus: 0.4-0.9 between random pairs, not ~0). Left uncorrected
this washes out the sense-block signal in a uniformly bright baseline. Each
vector is therefore mean-centered against that (layer, checkpoint)'s corpus-wide
mean -- over every valid occurrence of every word, not just this word's -- before
normalizing, which is what the cosine similarity should isolate.
"""

from __future__ import annotations

import logging

import numpy as np

from .embeddings import layers_of, load_vectors, sorted_checkpoints, valid_mask
from .purity import prep_word

log = logging.getLogger(__name__)


def _blocks(y: np.ndarray, n_senses: int, points_per_sense: int, rng) -> "tuple[np.ndarray, list[int]]":
    """Up to ``points_per_sense`` indices per sense, concatenated in sense order
    (so the similarity matrix built from them has contiguous per-sense blocks).
    Returns (indices, block sizes in the same order)."""
    idx, sizes = [], []
    for c in range(n_senses):
        ci = np.where(y == c)[0]
        if len(ci) == 0:
            continue
        take = ci if len(ci) <= points_per_sense else rng.choice(ci, points_per_sense, replace=False)
        idx.append(take)
        sizes.append(len(take))
    return (np.concatenate(idx) if idx else np.array([], dtype=int)), sizes


def _similarity_at(vectors, rows, idx, layer_idx: int, center: np.ndarray) -> np.ndarray:
    """Full pairwise cosine-similarity matrix of the sampled points at one layer,
    after removing the corpus-wide anisotropic direction ``center``."""
    X = np.asarray(vectors[rows[idx], layer_idx, :], np.float64) - center
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X @ X.T


def _compute(npz_files, rows, idx, cache_dir, valid: np.ndarray):
    """The similarity matrix at every (layer, checkpoint), one pass over the
    checkpoints. ``valid`` scopes the anisotropy correction's mean to every
    valid occurrence of every word in the corpus, not just this word's."""
    steps = [int(p.stem.removeprefix("step")) for p in npz_files]
    layers = layers_of(npz_files[0])
    n = len(idx)
    M = np.empty((len(steps), len(layers), n, n), np.float32)
    for si, npz in enumerate(npz_files):
        vectors = load_vectors(npz, cache_dir)
        for li in range(len(layers)):
            center = np.asarray(vectors[valid, li, :], np.float64).mean(axis=0)
            M[si, li] = _similarity_at(vectors, rows, idx, li, center)
        del vectors
        log.info("similarity checkpoint %d done", steps[si])
    return steps, layers, M


def _grid(steps, layers, sizes, M, word, out_path):
    """One small heatmap per (layer, checkpoint): rows = layer (top = shallowest,
    matching the PCA grid), columns = checkpoint. Grey lines mark sense-block
    boundaries; the diagonal (always 1, self-similarity) is masked so the color
    scale reflects the meaningful pairwise values instead. Titled explicitly as
    mean-centered, since the anisotropy correction shifts values well outside the
    [0, 1] range a reader would expect from plain cosine similarity."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import TwoSlopeNorm

    from .plots import EDGE, GRID, INK_SOFT, MUTED, PURITY_CMAP, _style, fmt_step, save

    _style()
    import matplotlib.pyplot as plt

    nS, nL = len(steps), len(layers)
    Mv = M.copy()
    for m in Mv.reshape(-1, *Mv.shape[-2:]):
        np.fill_diagonal(m, np.nan)
    vmin, vmax = float(np.nanmin(Mv)), float(np.nanmax(Mv))
    cmap = PURITY_CMAP.copy(); cmap.set_bad(GRID)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=min(vmin, -1e-6), vmax=max(vmax, 1e-6))
    bounds = np.cumsum(sizes)[:-1] - 0.5                        # block boundary positions

    fig, axes = plt.subplots(nL, nS, figsize=(0.62 * nS + 0.5, 0.62 * nL + 0.6), squeeze=False)
    for si, step in enumerate(steps):
        for li, layer in enumerate(layers):
            ax = axes[li][si]
            ax.imshow(np.ma.masked_invalid(Mv[si, li]), cmap=cmap, norm=norm, interpolation="nearest")
            for b in bounds:
                ax.axvline(b, color=EDGE, lw=0.4); ax.axhline(b, color=EDGE, lw=0.4)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(EDGE); sp.set_linewidth(0.5)
            if li == 0:
                ax.set_title(fmt_step(step), color=MUTED, fontsize=6.5, pad=2)
            if si == 0:
                ax.set_ylabel(str(layer), color=MUTED, rotation=0, ha="right", va="center", fontsize=6.5)

    fig.subplots_adjust(left=0.065, right=0.9, top=0.85, bottom=0.09, wspace=0.06, hspace=0.06)
    fig.suptitle(f"{word}: sense $\\times$ sense similarity", y=0.99, fontsize=9, color=INK_SOFT)
    fig.text(0.48, 0.9, "training step  $\\rightarrow$", ha="center", fontsize=8.5, color=INK_SOFT)
    fig.text(0.015, 0.5, "layer", rotation=90, ha="center", va="center", fontsize=8.5, color=INK_SOFT)
    fig.text(0.48, 0.015, "(mean-centered per layer/checkpoint, removes the corpus-wide "
            "anisotropic direction -- not raw cosine similarity)",
            ha="center", fontsize=6.5, color=MUTED)

    sm = ScalarMappable(norm=norm, cmap=PURITY_CMAP); sm.set_array([])
    cax = fig.add_axes((0.915, 0.18, 0.015, 0.65))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("mean-centered cosine similarity", color=INK_SOFT, fontsize=7)
    cbar.outline.set_edgecolor(EDGE); cbar.ax.tick_params(labelsize=6, colors=MUTED)
    save(fig, out_path, dpi=400)
    log.info("Wrote %s", out_path)


def visualize(records, emb_dir, word, pos, out_dir, *, cache_dir=None, min_per_sense=0,
             points_per_sense=40, seed=0) -> None:
    """The sense x sense similarity grid for one word, given an already-loaded
    corpus. Shared by the single-word CLI and the corpus-wide driver so the
    dataset and per-checkpoint embeddings aren't reloaded per word."""
    npz_files = sorted_checkpoints(emb_dir)
    valid = valid_mask(npz_files, cache_dir)
    plan = prep_word(records, word, pos, valid, min_per_sense)
    if plan is None:
        raise SystemExit(f"'{word}' has <2 senses with >= {min_per_sense} occurrences.")

    rng = np.random.default_rng(seed)
    idx, sizes = _blocks(plan["y"], len(plan["senses"]), points_per_sense, rng)
    steps, layers, M = _compute(npz_files, plan["rows"], idx, cache_dir, valid)

    out_dir.mkdir(parents=True, exist_ok=True)
    _grid(steps, layers, sizes, M, word, out_dir / f"{word}_similarity.png")


def similarity_corpus(records, emb_dir, words, pos, out_dir, **kwargs) -> None:
    """``visualize`` for many words, skipping (with a warning) any that don't
    have enough senses."""
    for word in words:
        try:
            visualize(records, emb_dir, word, pos, out_dir, **kwargs)
        except SystemExit as e:
            log.warning("skip %s: %s", word, e)


def similarity(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None,
              points_per_sense: int = 40, seed: int = 0) -> None:
    from .config import run_paths, store
    from .dataset import build_dataset

    m = cfg.get("purity", {})
    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg, only)

    out_dir = store(cfg, m.get("out_dir", "purity")) / "similarity"
    visualize(records, emb_dir, word, pos, out_dir, cache_dir=cache_dir,
             min_per_sense=p["min_per_sense"], points_per_sense=points_per_sense, seed=seed)
