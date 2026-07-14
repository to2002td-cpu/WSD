"""Stage 3 (CPU): sense-cluster purity of contextual embeddings.

For each occurrence, ``purity`` at neighbourhood size ``k`` is the fraction of its
``k`` cosine-nearest neighbours (raw hidden states, self excluded) that share its
WordNet sense, averaged over a balanced sample. ``chance = 1/K`` (K balanced
senses) is the fully-mixed floor. Being local, purity is robust to a sense
splitting into several clusters; sweeping ``k`` shows how far it reaches.

Compute and plot are separate: purity is written to ``{word}_purity.csv`` and the
heatmap grid is rendered from it, so re-running only re-plots (pass ``force`` to
recompute the O(n^2) purity).
"""

from __future__ import annotations

import csv
import logging

import numpy as np

from .embeddings import (
    displayed_senses,
    layers_of,
    load_vectors,
    population,
    sorted_checkpoints,
    valid_mask,
)

log = logging.getLogger(__name__)

MIN_SAMPLES = 20     # a sense needs at least this many occurrences to be scored


# Purity computation

def _balanced_idx(y: np.ndarray, k: int, max_per_sense: int, rng) -> np.ndarray:
    """Up to ``max_per_sense`` indices per sense, so chance is a clean 1/K."""
    out = []
    for c in range(k):
        ci = np.where(y == c)[0]
        if len(ci) < MIN_SAMPLES:
            continue
        out.append(ci if len(ci) <= max_per_sense else rng.choice(ci, max_per_sense, replace=False))
    return np.concatenate(out) if out else np.array([], dtype=int)


MARGIN_M = 10        # neighbours per side for the robust margin (mean over m nearest)


def _purity(Zn: np.ndarray, yl: np.ndarray, ks: "list[int]"):
    """From L2-normalized rows: purity at each k (fraction of the k cosine-nearest
    neighbours sharing the label; k clamps to n-1) and a per-point robust margin
    (mean cosine to the MARGIN_M nearest same-class minus to the nearest
    other-class — the metric gap that purity, being rank-based, discards)."""
    n = len(yl)
    ks_eff = {k: min(k, n - 1) for k in ks}
    kmax = max(ks_eff.values())
    S = Zn @ Zn.T
    np.fill_diagonal(S, -np.inf)
    part = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
    rows = np.arange(n)[:, None]
    nn = part[rows, np.argsort(-S[rows, part], axis=1)]        # neighbours, sorted desc
    same = (yl[nn] == yl[:, None])
    pur = {k: float(same[:, :ke].mean()) for k, ke in ks_eff.items()}

    sims = np.take_along_axis(S, nn, axis=1)                   # their cosine sims, desc

    def _topm_mean(mask):
        pick = mask & (np.cumsum(mask, axis=1) <= MARGIN_M)
        cnt = pick.sum(1)
        return np.where(cnt > 0, np.where(pick, sims, 0.0).sum(1) / np.maximum(cnt, 1), np.nan)

    margin = _topm_mean(same) - _topm_mean(~same)
    return pur, margin


def _purity_at(X, y, n_senses, max_per_sense, ks, rng):
    idx = _balanced_idx(y, n_senses, max_per_sense, rng)
    if len(idx) == 0:
        return None
    ys = y[idx]
    present = np.unique(ys)
    if len(present) < 2:
        return None
    remap = {c: i for i, c in enumerate(present)}
    yl = np.fromiter((remap[c] for c in ys), dtype=int, count=len(ys))
    Z = X[idx].astype(np.float64)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    return _purity(Z, yl, ks)


def _prep_word(records, word, pos, valid, min_per_sense, max_senses):
    meta, pos = population(records, word, pos, valid)
    senses = displayed_senses(meta, min_per_sense, max_senses)
    if len(senses) < 2:
        return None
    sense_arr = meta["sense"].to_numpy()
    y = np.array([senses.index(s) for s in sense_arr])
    return dict(pos=pos, rows=meta["row"].to_numpy(), senses=senses, y=y)


def _compute(npz_files, plans, ks, max_per_sense, cache_dir, seed):
    """Fill each plan's ``sweep`` in a single pass over the checkpoints."""
    steps = [int(p.stem.removeprefix("step")) for p in npz_files]
    layers = layers_of(npz_files[0])
    for plan in plans.values():
        plan["sweep"] = {k: np.full((len(layers), len(steps)), np.nan) for k in ks}
        plan["margins"] = {}                                   # (li, si) -> per-point margins
    rng = np.random.default_rng(seed)
    for si, (npz, step) in enumerate(zip(npz_files, steps)):
        vectors = load_vectors(npz, cache_dir)
        for li in range(len(layers)):
            for plan in plans.values():
                X = np.asarray(vectors[plan["rows"], li, :], np.float32)
                res = _purity_at(X, plan["y"], len(plan["senses"]), max_per_sense, ks, rng)
                if res is not None:
                    pur, margin = res
                    for k, pu in pur.items():
                        plan["sweep"][k][li, si] = pu
                    plan["margins"][(li, si)] = margin.astype(np.float32)
        del vectors
        log.info("checkpoint %d done (%d words)", step, len(plans))
    return steps, layers


# CSV I/O

def _write_csv(path, steps, layers, ks, sweep, chance, n_senses):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "layer", "k", "n_senses", "chance", "purity", "purity_minus_chance"])
        for k in ks:
            for li, layer in enumerate(layers):
                for si, step in enumerate(steps):
                    pu = sweep[k][li, si]
                    w.writerow([step, layer, k, n_senses, chance, pu, pu - chance])
    log.info("Wrote %s", path.name)


MARGIN_SUB = 800     # per-cell margin subsample kept for the ridgeline


def _save_margins(path, steps, layers, margins, rng):
    """A subsample of per-point margins per (step, layer) -> npz for the ridgeline."""
    A = np.full((len(steps), len(layers), MARGIN_SUB), np.nan, np.float32)
    for (li, si), arr in margins.items():
        arr = arr[np.isfinite(arr)]
        if len(arr):
            take = arr if len(arr) <= MARGIN_SUB else rng.choice(arr, MARGIN_SUB, replace=False)
            A[si, li, :len(take)] = take
    np.savez(path, margins=A, steps=np.array(steps), layers=np.array(layers))
    log.info("Wrote %s", path.name)


def _read_csv(path):
    """(steps, layers, ks, sweep, chance) from a purity CSV."""
    from collections import defaultdict

    vals: dict = defaultdict(lambda: np.nan)
    ks_set, steps_set, layers_set = set(), set(), set()
    chance = np.nan
    for r in csv.DictReader(path.open()):
        k, s, l = int(r["k"]), int(r["step"]), int(r["layer"])
        ks_set.add(k); steps_set.add(s); layers_set.add(l)
        chance = float(r["chance"])
        if r["purity"] not in ("", "nan"):
            vals[(k, l, s)] = float(r["purity"])
    ks, steps, layers = sorted(ks_set), sorted(steps_set), sorted(layers_set)
    sweep = {k: np.array([[vals[(k, l, s)] for s in steps] for l in layers]) for k in ks}
    return steps, layers, ks, sweep, chance


# Drivers

def _subdirs(out_dir):
    """Self-describing output folders: data (CSVs + margin npz) and one per figure."""
    sd = {name: out_dir / name for name in ("data", "heatmap", "auc", "pk", "margin")}
    for p in sd.values():
        p.mkdir(parents=True, exist_ok=True)
    return sd


def purity_corpus(records, emb_dir, words, pos, out_dir, *, max_per_sense: int,
                  max_senses: int, knn_ks: "list[int]", min_per_sense: int,
                  cache_dir=None, seed: int = 0, force: bool = False):
    """Purity for many words. Words whose ``data/{word}_purity.csv`` exists are
    reused (unless ``force``); the rest are computed in one checkpoint pass. Every
    word is then rendered from its CSV, and the corpus aggregate is refreshed.
    Outputs land in ``out_dir/{data,heatmap,auc,pk,margin}/``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = _subdirs(out_dir)
    ks = sorted(set(knn_ks))
    todo = [w for w in words if force or not (sd["data"] / f"{w}_purity.csv").exists()]

    if todo:
        npz_files = sorted_checkpoints(emb_dir)
        valid = valid_mask(npz_files, cache_dir)
        plans = {}
        for word in todo:
            try:
                plan = _prep_word(records, word, pos, valid, min_per_sense, max_senses)
            except SystemExit as e:
                log.warning("skip %s: %s", word, e)
                continue
            if plan is None:
                log.warning("skip %s: <2 senses with >= %d occurrences", word, min_per_sense)
                continue
            plans[word] = plan
        log.info("Purity: computing %d word(s) (k sweep %s)", len(plans), ks)
        if plans:
            steps, layers = _compute(npz_files, plans, ks, max_per_sense, cache_dir, seed)
            rng_sub = np.random.default_rng(seed)
            for word, plan in plans.items():
                _write_csv(sd["data"] / f"{word}_purity.csv", steps, layers, ks,
                           plan["sweep"], 1.0 / len(plan["senses"]), len(plan["senses"]))
                _save_margins(sd["data"] / f"{word}_margin.npz", steps, layers, plan["margins"], rng_sub)

    rendered = 0
    for word in words:
        csv_path = sd["data"] / f"{word}_purity.csv"
        if not csv_path.exists():
            continue
        steps, layers, kk, sweep, chance = _read_csv(csv_path)
        _heatmap_grid(kk, steps, layers, sweep, sd["heatmap"] / f"{word}_purity.png")
        _auc_heatmap(kk, steps, layers, sweep, sd["auc"] / f"{word}_purity_auc.png")
        _pk_curves(kk, steps, layers, sweep, chance, sd["pk"] / f"{word}_purity_pk.png")
        margin_path = sd["data"] / f"{word}_margin.npz"
        if margin_path.exists():
            d = np.load(margin_path)
            _margin_ridgeline(d["margins"], list(d["steps"]), list(d["layers"]),
                              sd["margin"] / f"{word}_margin.png")
        rendered += 1
    _aggregate(sd, seed)
    log.info("Purity done: %d word(s) in %s", rendered, out_dir)


def purity(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None,
           force: bool = False) -> None:
    from .config import run_paths, store

    m = cfg.get("purity", {})
    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    from .dataset import build_dataset
    records = build_dataset(cfg, only)
    purity_corpus(
        records, emb_dir, [word], pos, store(cfg, m.get("out_dir", "purity")),
        max_per_sense=m.get("max_per_sense", 1000), max_senses=m.get("max_senses", 12),
        knn_ks=m.get("knn_ks", [5, 10, 20, 50, 100, 200, 500, 1000, 2000]),
        min_per_sense=p["min_per_sense"], cache_dir=cache_dir, seed=m.get("seed", 0), force=force,
    )


# Purity heatmap grid (one panel per checkpoint)

def _surface_z(ks, layers, sweep, si):
    """purity[layer, k] grid for one checkpoint (rows = layer, cols = k)."""
    return np.array([[sweep[k][li, si] for k in ks] for li in range(len(layers))])


def _heatmap_grid(ks, steps, layers, sweep, out_path):
    """Purity heatmaps — one panel per checkpoint, cells = purity(layer, k) on a
    shared diverging 0..1 scale. No title: the caption carries it."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from .plots import EDGE, GRID, INK_SOFT, MUTED, PURITY_CMAP, WIDE_WIDTH, _style, fmt_step, save

    _style()
    import matplotlib.pyplot as plt

    cmap = PURITY_CMAP.copy()
    cmap.set_bad(GRID)                                          # missing cells recede to grid grey
    norm = Normalize(0, 1)

    nk, nL, nS = len(ks), len(layers), len(steps)
    ncol = 5
    nrow = int(np.ceil(nS / ncol))
    xt = list(range(0, nk, 2))                                  # every other k, avoids clutter

    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDE_WIDTH, 1.45 * nrow + 0.35),
                             squeeze=False)
    for idx in range(nrow * ncol):
        ax = axes[idx // ncol][idx % ncol]
        if idx >= nS:
            ax.axis("off")
            continue
        Z = np.ma.masked_invalid(_surface_z(ks, layers, sweep, idx))   # (nL, nk)
        ax.imshow(Z, cmap=cmap, norm=norm, origin="lower", aspect="auto",
                  interpolation="nearest")

        ax.set_xticks(np.arange(-0.5, nk, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nL, 1), minor=True)
        ax.grid(which="minor", color=GRID, linewidth=0.6)       # hairline grid delineates every cell
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(EDGE); sp.set_linewidth(0.6)

        is_bottom = idx + ncol >= nS
        is_left = idx % ncol == 0
        ax.set_xticks(xt)
        ax.set_xticklabels([fmt_step(ks[i]) for i in xt], rotation=45, ha="right") if is_bottom \
            else ax.set_xticklabels([])
        ax.set_yticks(range(nL))
        ax.set_yticklabels([str(l) for l in layers]) if is_left else ax.set_yticklabels([])
        ax.set_title(f"step {fmt_step(steps[idx])}", pad=3, fontsize=7.5, color=MUTED)

    # Meta-axes named once (as in the PCA figure), instead of a per-panel k/layer label.
    left, right, top, bot = 0.085, 0.9, 0.93, 0.15
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bot, hspace=0.28, wspace=0.1)
    fig.text((left + right) / 2, 0.04, "$k$", ha="center", va="center",
             fontsize=8.5, color=INK_SOFT)
    fig.text(0.022, (top + bot) / 2, "layer", rotation=90, ha="center", va="center",
             fontsize=8.5, color=INK_SOFT)

    sm = ScalarMappable(norm=norm, cmap=PURITY_CMAP); sm.set_array([])
    cax = fig.add_axes((0.915, 0.2, 0.013, 0.6))
    cbar = fig.colorbar(sm, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("$k$-NN sense purity", color=INK_SOFT)
    cbar.outline.set_edgecolor(EDGE); cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    save(fig, out_path, dpi=400)
    log.info("Wrote %s", out_path)


def _auc_heatmap(ks, steps, layers, sweep, out_path):
    """The k-sweep collapsed to one number per (layer, step): purity integrated
    over log k (P(k) -> 1/K as k grows, so this measures how slowly separation
    decays with neighbourhood scale). A single layer x step heatmap."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from .plots import EDGE, GRID, INK_SOFT, MUTED, PURITY_CMAP, _style, fmt_step, save

    _style()
    import matplotlib.pyplot as plt

    nL, nS = len(layers), len(steps)
    lk = np.log10(ks)
    span = lk[-1] - lk[0]
    A = np.full((nL, nS), np.nan)
    for li in range(nL):
        for si in range(nS):
            col = np.array([sweep[k][li, si] for k in ks], float)
            if np.all(np.isfinite(col)):
                A[li, si] = np.sum(np.diff(lk) * (col[:-1] + col[1:]) / 2) / span  # log-k mean purity

    cmap = PURITY_CMAP.copy(); cmap.set_bad(GRID)
    norm = Normalize(0, 1)
    fig, ax = plt.subplots(figsize=(0.45 * nS + 1.2, 0.34 * nL + 1.2))
    ax.imshow(np.ma.masked_invalid(A), cmap=cmap, norm=norm, origin="lower",
              aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, nS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nL, 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.6)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(EDGE); sp.set_linewidth(0.6)
    ax.set_xticks(range(nS)); ax.set_xticklabels([fmt_step(s) for s in steps], rotation=45, ha="right")
    ax.set_yticks(range(nL)); ax.set_yticklabels([str(l) for l in layers])
    ax.set_xlabel("training step  $\\rightarrow$", color=INK_SOFT, labelpad=4)
    ax.set_ylabel("layer", color=INK_SOFT, labelpad=4)

    sm = ScalarMappable(norm=norm, cmap=PURITY_CMAP); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.03, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("mean $k$-NN purity (log-$k$ AUC)", color=INK_SOFT)
    cbar.outline.set_edgecolor(EDGE); cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    fig.tight_layout()
    save(fig, out_path, dpi=400)
    log.info("Wrote %s", out_path)


def _pk_curves(ks, steps, layers, sweep, chance, out_path):
    """P(k) = purity vs neighbourhood size k (log x), one curve per checkpoint
    (light -> dark = training), one panel per layer. The slope reads the scale of
    separation (steep = only local); the AUC heatmap is the area under each curve."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    from .plots import EDGE, INK_SOFT, MUTED, WIDE_WIDTH, _style, fmt_step, save

    _style()
    import matplotlib.pyplot as plt

    nS, nL = len(steps), len(layers)
    ramp = LinearSegmentedColormap.from_list("tr", ["#CBD8EE", "#8FB0DB", "#1F4E8C"])

    fig, axes = plt.subplots(1, nL, figsize=(WIDE_WIDTH, 1.9), sharey=True, squeeze=False)
    for li, ax in enumerate(axes[0]):
        for si in range(nS):
            y = np.array([sweep[k][li, si] for k in ks])
            ax.plot(ks, y, color=ramp(si / (nS - 1)), lw=1.1, solid_capstyle="round")
        if np.isfinite(chance):
            ax.axhline(chance, color=MUTED, lw=0.7, ls=(0, (3, 2)))
            if li == 0:
                ax.text(ks[0] * 1.1, chance + 0.03, "chance", fontsize=6, color=MUTED)
        ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.set_xticks([ks[0], 100, ks[-1]]); ax.set_xticklabels([fmt_step(ks[0]), "100", fmt_step(ks[-1])])
        ax.set_title(f"layer {layers[li]}", color=MUTED, fontsize=7.5)
        ax.set_yticks([0, 0.5, 1.0]); ax.tick_params(colors=MUTED)
        if li:
            ax.tick_params(labelleft=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_edgecolor(EDGE); ax.spines[s].set_linewidth(0.6)

    fig.text(0.5, 0.02, "$k$  (log)", ha="center", fontsize=8.5, color=INK_SOFT)
    fig.text(0.007, 0.55, "sense purity  $P(k)$", rotation=90, va="center", ha="center",
             fontsize=8.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.065, right=0.9, top=0.86, bottom=0.19, wspace=0.12)
    sm = ScalarMappable(norm=Normalize(0, nS - 1), cmap=ramp); sm.set_array([])
    cax = fig.add_axes((0.915, 0.2, 0.013, 0.62))
    cb = fig.colorbar(sm, cax=cax, ticks=range(0, nS, 2))
    cb.ax.set_yticklabels([fmt_step(steps[i]) for i in range(0, nS, 2)], fontsize=6)
    cb.set_label("training step", color=INK_SOFT, fontsize=8)
    cb.outline.set_edgecolor(EDGE); cb.ax.tick_params(length=0, colors=MUTED)
    save(fig, out_path, dpi=400)
    log.info("Wrote %s", out_path)


def _margin_ridgeline(A, steps, layers, out_path):
    """Ridgeline of the per-point margin distribution across checkpoints (bottom =
    init, top = final), one panel per layer. Mass moving from ~0 to positive = the
    class boundary widening over training. Dashed line at margin 0."""
    from scipy.stats import gaussian_kde

    from .plots import EDGE, INK_SOFT, MUTED, SENSE_COLORS, WIDE_WIDTH, _style, fmt_step, save

    _style()
    import matplotlib.pyplot as plt

    nS, nL, _ = A.shape
    finite = A[np.isfinite(A)]
    lo, hi = np.percentile(finite, [0.5, 99.5])
    pad = 0.06 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 200)
    color = SENSE_COLORS[0]
    overlap = 1.7

    fig, axes = plt.subplots(1, nL, figsize=(WIDE_WIDTH, 0.26 * nS + 1.0), sharex=True, squeeze=False)
    for li, ax in enumerate(axes[0]):
        for si in range(nS):
            m = A[si, li]; m = m[np.isfinite(m)]
            if len(m) < 5:
                continue
            try:
                d = gaussian_kde(m)(xs)
            except Exception:
                continue
            d = d / d.max() * overlap
            ax.fill_between(xs, si, si + d, color=color, alpha=0.5, lw=0, zorder=si)
            ax.plot(xs, si + d, color=INK_SOFT, lw=0.5, zorder=si)
        ax.axvline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=nS + 1)
        ax.set_title(f"layer {layers[li]}", color=MUTED, fontsize=7.5)
        ax.set_ylim(-0.2, nS - 1 + overlap + 0.2)
        ax.set_yticks(range(nS))
        ax.set_yticklabels([fmt_step(s) for s in steps] if li == 0 else [], fontsize=6)
        ax.tick_params(which="both", length=0, colors=MUTED)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_edgecolor(EDGE); ax.spines["bottom"].set_linewidth(0.6)

    fig.text(0.5, 0.02, "margin   (cos to same $-$ cos to other)", ha="center",
             fontsize=8.5, color=INK_SOFT)
    fig.text(0.007, 0.55, "training step", rotation=90, va="center", ha="center",
             fontsize=8.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.14, wspace=0.08)
    save(fig, out_path, dpi=400)
    log.info("Wrote %s", out_path)


def _aggregate(sd, seed):
    """Pool every word's ``data/*_purity.csv`` (and ``*_margin.npz``) into a
    corpus-level CSV + figures."""
    from collections import defaultdict

    files = sorted(p for p in sd["data"].glob("*_purity.csv") if not p.name.startswith("corpus"))
    if not files:
        return
    pur: dict = defaultdict(list)
    cha: dict = defaultdict(list)
    ks_set, steps_set, layers_set = set(), set(), set()
    for f in files:
        for r in csv.DictReader(f.open()):
            k, s, l = int(r["k"]), int(r["step"]), int(r["layer"])
            ks_set.add(k); steps_set.add(s); layers_set.add(l)
            if r["purity"] not in ("", "nan"):
                pur[(k, l, s)].append(float(r["purity"]))
                cha[(k, l, s)].append(float(r["chance"]))

    ks, steps, layers = sorted(ks_set), sorted(steps_set), sorted(layers_set)
    sweep = {k: np.full((len(layers), len(steps)), np.nan) for k in ks}
    chance = np.full((len(layers), len(steps)), np.nan)
    for k in ks:
        for li, l in enumerate(layers):
            for si, s in enumerate(steps):
                if pur[(k, l, s)]:
                    sweep[k][li, si] = float(np.mean(pur[(k, l, s)]))
                    chance[li, si] = float(np.mean(cha[(k, l, s)]))

    with (sd["data"] / "corpus_purity.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "layer", "k", "n_words", "mean_chance", "mean_purity", "std_purity"])
        for k in ks:
            for li, l in enumerate(layers):
                for si, s in enumerate(steps):
                    v = pur[(k, l, s)]
                    w.writerow([s, l, k, len(v), chance[li, si], sweep[k][li, si],
                                float(np.std(v)) if v else np.nan])

    _heatmap_grid(ks, steps, layers, sweep, sd["heatmap"] / "corpus_purity.png")
    _auc_heatmap(ks, steps, layers, sweep, sd["auc"] / "corpus_purity_auc.png")
    _pk_curves(ks, steps, layers, sweep, float(np.nanmean(chance)), sd["pk"] / "corpus_purity_pk.png")
    _aggregate_margins(sd, seed)
    log.info("Aggregated %d words -> corpus_purity.png", len(files))


def _aggregate_margins(sd, seed):
    """Pool every word's margin distribution per (step, layer) into a corpus
    ridgeline (all words' margins concatenated, then subsampled)."""
    files = sorted(p for p in sd["data"].glob("*_margin.npz") if not p.name.startswith("corpus"))
    if not files:
        return
    npz = [np.load(f) for f in files]
    steps = list(npz[0]["steps"]); layers = list(npz[0]["layers"])
    per = [d["margins"] for d in npz]                          # each [nS, nL, M]
    nS, nL = len(steps), len(layers)
    rng = np.random.default_rng(seed)
    A = np.full((nS, nL, MARGIN_SUB), np.nan, np.float32)
    for si in range(nS):
        for li in range(nL):
            vals = np.concatenate([p[si, li][np.isfinite(p[si, li])] for p in per])
            if len(vals):
                take = vals if len(vals) <= MARGIN_SUB else rng.choice(vals, MARGIN_SUB, replace=False)
                A[si, li, :len(take)] = take
    np.savez(sd["data"] / "corpus_margin.npz", margins=A, steps=np.array(steps), layers=np.array(layers))
    _margin_ridgeline(A, steps, layers, sd["margin"] / "corpus_margin.png")
