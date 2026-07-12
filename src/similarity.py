"""Stage 3 (CPU): sense-cluster purity of contextual embeddings.

For each occurrence, ``purity`` at neighbourhood size ``k`` is the fraction of its
``k`` cosine-nearest neighbours (raw hidden states, self excluded) that share its
WordNet sense, averaged over a balanced sample. ``chance = 1/K`` (K balanced
senses) is the fully-mixed floor. Being local, purity is robust to a sense
splitting into several clusters. Sweeping ``k`` shows how far the purity reaches;
the result is one 3-D surface over (k, layer) per checkpoint.
"""

from __future__ import annotations

import csv
import logging

import numpy as np

from .umapcache import _layers, _load_vectors, _population, _valid_mask, displayed_senses

log = logging.getLogger(__name__)

MIN_SAMPLES = 20     # a sense needs at least this many occurrences to be scored


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
    meta, pos = _population(records, word, pos, valid)
    senses = displayed_senses(meta, min_per_sense, max_senses)
    if len(senses) < 2:
        return None
    sense_arr = meta["sense"].to_numpy()
    y = np.array([senses.index(s) for s in sense_arr])
    return dict(pos=pos, rows=meta["row"].to_numpy(), senses=senses, y=y)


# --------------------------------------------------------------------------- #
# Drivers                                                                      #
# --------------------------------------------------------------------------- #

def similarity_corpus(records, emb_dir, words, pos, out_dir, *, max_per_sense: int,
                      max_senses: int, knn_ks: "list[int]", min_per_sense: int,
                      cache_dir=None, seed: int = 0):
    """Purity for many words, loading each checkpoint once. Writes a per-word CSV
    and 3-D surface, then the corpus aggregate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(emb_dir.glob("step*.npz"), key=lambda p: int(p.stem.removeprefix("step")))
    if not npz_files:
        raise SystemExit(f"No step*.npz in {emb_dir}")
    valid = _valid_mask(npz_files, cache_dir)
    steps = [int(p.stem.removeprefix("step")) for p in npz_files]
    layers = _layers(npz_files[0])
    ks = sorted(set(knn_ks))

    plans: dict[str, dict] = {}
    for word in words:
        try:
            plan = _prep_word(records, word, pos, valid, min_per_sense, max_senses)
        except SystemExit as e:
            log.warning("skip %s: %s", word, e)
            continue
        if plan is None:
            log.warning("skip %s: <2 senses with >= %d occurrences", word, min_per_sense)
            continue
        plan.update(sweep={k: np.full((len(layers), len(steps)), np.nan) for k in ks},
                    chance=1.0 / len(plan["senses"]))
        plans[word] = plan
    log.info("Purity: %d/%d words usable (k sweep %s)", len(plans), len(words), ks)

    rng = np.random.default_rng(seed)
    for si, (npz, step) in enumerate(zip(npz_files, steps)):
        vectors = _load_vectors(npz, cache_dir)
        for li in range(len(layers)):
            for plan in plans.values():
                X = np.asarray(vectors[plan["rows"], li, :], np.float32)
                res = _purity_at(X, plan["y"], len(plan["senses"]), max_per_sense, ks, rng)
                if res is not None:
                    for k, pu in res.items():
                        plan["sweep"][k][li, si] = pu
        del vectors
        log.info("checkpoint %d done (%d words)", step, len(plans))

    for word, plan in plans.items():
        _write_csv(out_dir / f"{word}_purity.csv", steps, layers, ks,
                   plan["sweep"], plan["chance"], len(plan["senses"]))
        _surface(f"“{word}” ({plan['pos']}) — sense-cluster purity (k × layer per checkpoint)",
                 ks, steps, layers, plan["sweep"], plan["chance"], out_dir / f"{word}_purity.png")
    _aggregate(out_dir)
    return plans


def similarity_word(records, emb_dir, word, pos, out_dir, **kw):
    plans = similarity_corpus(records, emb_dir, [word], pos, out_dir, **kw)
    if not plans:
        raise SystemExit(f"'{word}' ({pos}) has <2 senses with >= {kw['min_per_sense']} occurrences.")
    return plans


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


# --------------------------------------------------------------------------- #
# 3-D purity surface (one panel per checkpoint)                                #
# --------------------------------------------------------------------------- #

def _surface_z(ks, layers, sweep, si):
    return np.array([[sweep[k][li, si] for k in ks] for li in range(len(layers))])


def _surface(title, ks, steps, layers, sweep, chance, out_path):
    """One 3-D purity surface over (k, layer) per checkpoint, shared z ∈ [0,1]
    with a chance floor. Colour = purity (= height); read across the grid for the
    training trajectory."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from .plots import INK, _style

    _style()
    import matplotlib.pyplot as plt

    nk, nL, nS = len(ks), len(layers), len(steps)
    Xk, Yl = np.meshgrid(np.arange(nk), np.arange(nL))
    cmap = plt.cm.viridis
    ncol = 5
    nrow = int(np.ceil(nS / ncol))

    fig = plt.figure(figsize=(3.5 * ncol, 3.7 * nrow))
    for si, step in enumerate(steps):
        ax = fig.add_subplot(nrow, ncol, si + 1, projection="3d")
        Z = _surface_z(ks, layers, sweep, si)
        ax.plot_surface(Xk, Yl, np.full_like(Z, chance), color="0.6", alpha=0.12,
                        linewidth=0, shade=False)
        ax.plot_surface(Xk, Yl, Z, cmap=cmap, vmin=0, vmax=1, rstride=1, cstride=1,
                        edgecolor=(0, 0, 0, 0.15), linewidth=0.2, antialiased=True)
        ax.set_xticks(range(nk)); ax.set_xticklabels([f"{k:,}" for k in ks],
                                                     rotation=45, fontsize=5.5, ha="right")
        ax.set_yticks(range(nL)); ax.set_yticklabels([f"L{l}" for l in layers], fontsize=6.5)
        ax.set_zlim(0, 1); ax.set_zticks([0, 0.5, 1.0]); ax.tick_params(labelsize=7)
        ax.set_xlabel("k", fontsize=8, labelpad=4)
        ax.set_ylabel("layer", fontsize=8, labelpad=4)
        ax.set_title(f"step {step:,}", color=INK, fontsize=11, pad=-2)
        ax.view_init(elev=24, azim=-58)
        ax.set_box_aspect((1.3, 1.0, 0.8))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("white"); axis.pane.set_alpha(0.4)

    for j in range(nS, nrow * ncol):
        fig.add_subplot(nrow, ncol, j + 1).axis("off")

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, ticks=np.linspace(0, 1, 6),
                        fraction=0.015, pad=0.06, shrink=0.5)
    cbar.set_label("k-NN sense purity  (surface height)", fontsize=10)
    fig.suptitle(title, color=INK, fontsize=15, y=1.0)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def _aggregate(out_dir):
    """Average every ``*_purity.csv`` into corpus-level CSV + 3-D surface."""
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

    _surface(f"Top-{len(files)} nouns — mean sense-cluster purity (k × layer per checkpoint)",
             ks, steps, layers, sweep, float(np.nanmean(chance)), out_dir / "corpus_purity.png")
    log.info("Aggregated %d words -> corpus_purity.png", len(files))


def similarity(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None) -> None:
    from .config import run_paths, store
    from .dataset import build_dataset

    m = cfg.get("similarity", {})
    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg, only)
    similarity_word(
        records, emb_dir, word, pos, store(cfg, m.get("out_dir", "similarity")),
        max_per_sense=m.get("max_per_sense", 1000), max_senses=m.get("max_senses", 12),
        knn_ks=m.get("knn_ks", [5, 10, 20, 50, 100, 200, 500, 1000, 2000]),
        min_per_sense=p["min_per_sense"], cache_dir=cache_dir, seed=m.get("seed", 0),
    )
