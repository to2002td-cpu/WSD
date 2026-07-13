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


# --------------------------------------------------------------------------- #
# Purity computation                                                          #
# --------------------------------------------------------------------------- #

def _balanced_idx(y: np.ndarray, k: int, max_per_sense: int, rng) -> np.ndarray:
    """Up to ``max_per_sense`` indices per sense, so chance is a clean 1/K."""
    out = []
    for c in range(k):
        ci = np.where(y == c)[0]
        if len(ci) < MIN_SAMPLES:
            continue
        out.append(ci if len(ci) <= max_per_sense else rng.choice(ci, max_per_sense, replace=False))
    return np.concatenate(out) if out else np.array([], dtype=int)


def _purity(Zn: np.ndarray, yl: np.ndarray, ks: "list[int]") -> "dict[int, float]":
    """Purity at each k from L2-normalized rows. k beyond the sample clamps to
    n-1, where purity necessarily equals chance."""
    n = len(yl)
    ks_eff = {k: min(k, n - 1) for k in ks}
    kmax = max(ks_eff.values())
    S = Zn @ Zn.T
    np.fill_diagonal(S, -np.inf)
    part = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
    rows = np.arange(n)[:, None]
    nn = part[rows, np.argsort(-S[rows, part], axis=1)]
    same = (yl[nn] == yl[:, None])
    return {k: float(same[:, :ke].mean()) for k, ke in ks_eff.items()}


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
    rng = np.random.default_rng(seed)
    for si, (npz, step) in enumerate(zip(npz_files, steps)):
        vectors = load_vectors(npz, cache_dir)
        for li in range(len(layers)):
            for plan in plans.values():
                X = np.asarray(vectors[plan["rows"], li, :], np.float32)
                res = _purity_at(X, plan["y"], len(plan["senses"]), max_per_sense, ks, rng)
                if res is not None:
                    for k, pu in res.items():
                        plan["sweep"][k][li, si] = pu
        del vectors
        log.info("checkpoint %d done (%d words)", step, len(plans))
    return steps, layers


# --------------------------------------------------------------------------- #
# CSV I/O                                                                      #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Drivers                                                                      #
# --------------------------------------------------------------------------- #

def purity_corpus(records, emb_dir, words, pos, out_dir, *, max_per_sense: int,
                  max_senses: int, knn_ks: "list[int]", min_per_sense: int,
                  cache_dir=None, seed: int = 0, force: bool = False):
    """Purity for many words. Words whose ``{word}_purity.csv`` exists are reused
    (unless ``force``); the rest are computed in one checkpoint pass. Every word is
    then rendered from its CSV, and the corpus aggregate is refreshed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(set(knn_ks))
    todo = [w for w in words if force or not (out_dir / f"{w}_purity.csv").exists()]

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
            for word, plan in plans.items():
                _write_csv(out_dir / f"{word}_purity.csv", steps, layers, ks,
                           plan["sweep"], 1.0 / len(plan["senses"]), len(plan["senses"]))

    rendered = 0
    for word in words:
        csv_path = out_dir / f"{word}_purity.csv"
        if not csv_path.exists():
            continue
        steps, layers, kk, sweep, _ = _read_csv(csv_path)
        _heatmap_grid(kk, steps, layers, sweep, out_dir / f"{word}_purity.png")
        rendered += 1
    _aggregate(out_dir)
    log.info("Purity done: %d figure(s) in %s", rendered, out_dir)


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


# --------------------------------------------------------------------------- #
# Purity heatmap grid (one panel per checkpoint)                               #
# --------------------------------------------------------------------------- #

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


def _aggregate(out_dir):
    """Average every ``*_purity.csv`` into corpus-level CSV + heatmap grid."""
    from collections import defaultdict

    files = sorted(p for p in out_dir.glob("*_purity.csv") if not p.name.startswith("corpus"))
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

    with (out_dir / "corpus_purity.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "layer", "k", "n_words", "mean_chance", "mean_purity", "std_purity"])
        for k in ks:
            for li, l in enumerate(layers):
                for si, s in enumerate(steps):
                    v = pur[(k, l, s)]
                    w.writerow([s, l, k, len(v), chance[li, si], sweep[k][li, si],
                                float(np.std(v)) if v else np.nan])

    _heatmap_grid(ks, steps, layers, sweep, out_dir / "corpus_purity.png")
    log.info("Aggregated %d words -> corpus_purity.png", len(files))
