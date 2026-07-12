"""Sense-cluster purity of contextual embeddings (k-NN neighbourhood purity).

The KDE panels show a word's occurrences forming clusters, with a single WordNet
sense sometimes split across several clusters. The question that matters is not
how far one sense-blob sits from another (senses are multimodal, so blob-distance
misleads) but whether the clusters are **pure**: pick any occurrence and look at
its nearest neighbours in the model's embedding space — do they share its sense?

We answer with the most basic similarity primitive — cosine nearest neighbours on
the raw pooled hidden states (no UMAP, no standardisation). For a balanced sample
of a word's occurrences, ``purity`` at neighbourhood size ``k`` is the mean, over
occurrences, of the fraction of the ``k`` nearest neighbours carrying the same
sense. Being local, purity is unaffected by a sense fragmenting into several
clusters: as long as each neighbourhood is single-sense, purity stays high.
``chance = 1/K`` (K balanced senses) is the fully-mixed floor.

Sweeping ``k`` traces how far that purity reaches: a curve that stays high as k
grows means senses are pure at scale, one that decays to chance means only tight
local pockets are pure. The result is rendered as one 3-D purity surface over
(k, layer) per checkpoint, so the training trajectory reads across the grid.
"""

from __future__ import annotations

import csv
import logging

import numpy as np

from .umapcache import (
    _layers,
    _load_vectors,
    _population,
    _valid_mask,
    displayed_senses,
)

log = logging.getLogger(__name__)

MIN_SAMPLES = 20     # a sense needs at least this many occurrences to be scored


# --------------------------------------------------------------------------- #
# Purity measure                                                              #
# --------------------------------------------------------------------------- #

def _balanced_idx(y: np.ndarray, k: int, max_per_sense: int, rng) -> np.ndarray:
    """Up to ``max_per_sense`` row indices per sense (balanced, so purity's chance
    baseline is a clean 1/K and no frequent sense dominates the neighbourhoods)."""
    out = []
    for c in range(k):
        ci = np.where(y == c)[0]
        if len(ci) < MIN_SAMPLES:
            continue
        out.append(ci if len(ci) <= max_per_sense
                   else rng.choice(ci, max_per_sense, replace=False))
    return np.concatenate(out) if out else np.array([], dtype=int)


def _purity(Zn: np.ndarray, yl: np.ndarray, ks: "list[int]") -> "dict[int, float]":
    """Purity at each k from L2-normalized rows: rank cosine nearest neighbours
    once (self excluded), then average the same-sense fraction over the k nearest.
    k beyond the sample all clamp to n-1, where purity necessarily equals chance."""
    n = len(yl)
    ks_eff = {k: min(k, n - 1) for k in ks}            # requested k -> usable k (clamped to data)
    kmax = max(ks_eff.values())
    S = Zn @ Zn.T
    np.fill_diagonal(S, -np.inf)                       # never a neighbour of itself
    part = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]     # kmax nearest (unordered)
    rows = np.arange(n)[:, None]
    nn = part[rows, np.argsort(-S[rows, part], axis=1)]        # ... ordered nearest-first
    same = (yl[nn] == yl[:, None])                     # neighbour shares sense? n×kmax
    return {k: float(same[:, :ke].mean()) for k, ke in ks_eff.items()}


def _purity_at(X, y, n_senses, max_per_sense, ks, rng):
    """(purity_by_k, K) for one (step, layer) over a balanced, L2-normalized
    subsample, or ``None`` if fewer than two senses survive."""
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
    return _purity(Z, yl, ks), len(present)


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
    """Neighbourhood-purity analysis for many words, loading each checkpoint once.
    Writes a per-word purity summary CSV and the 3-D purity surface (PNG + HTML),
    then the corpus aggregate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(emb_dir.glob("step*.npz"),
                       key=lambda p: int(p.stem.removeprefix("step")))
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
        vectors = _load_vectors(npz, cache_dir)                  # loaded once for all words
        for li, layer in enumerate(layers):
            for plan in plans.values():
                X = np.asarray(vectors[plan["rows"], li, :], np.float32)
                res = _purity_at(X, plan["y"], len(plan["senses"]), max_per_sense, ks, rng)
                if res is None:
                    continue
                for k, pu in res[0].items():
                    plan["sweep"][k][li, si] = pu
        del vectors
        log.info("checkpoint %d done (%d words)", step, len(plans))

    for word, plan in plans.items():
        _write_summary_csv(out_dir / f"{word}_purity.csv", steps, layers, ks,
                           plan["sweep"], plan["chance"], len(plan["senses"]))
        _surface_png(f"“{word}” ({plan['pos']}) — sense-cluster purity surface (k × layer), one panel per checkpoint",
                     ks, steps, layers, plan["sweep"], plan["chance"],
                     out_dir / f"{word}_purity_surface.png")
        _surface_html(f"“{word}” ({plan['pos']}) — purity surface (k × layer); drag to scrub checkpoints",
                      ks, steps, layers, plan["sweep"], plan["chance"],
                      out_dir / f"{word}_purity_surface.html")
    similarity_aggregate(out_dir)
    return plans


def similarity_word(records, emb_dir, word, pos, out_dir, **kw):
    """Single-word convenience wrapper over :func:`similarity_corpus`."""
    plans = similarity_corpus(records, emb_dir, [word], pos, out_dir, **kw)
    if not plans:
        raise SystemExit(f"'{word}' ({pos}) has <2 senses with >= {kw['min_per_sense']} occurrences.")
    return plans


# --------------------------------------------------------------------------- #
# Output: CSV                                                                  #
# --------------------------------------------------------------------------- #

def _write_summary_csv(path, steps, layers, ks, sweep, chance, n_senses):
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
# Output: 3-D purity surface (one panel per checkpoint)                        #
# --------------------------------------------------------------------------- #

def _surface_z(ks, layers, sweep, si):
    """purity[layer, k] grid for one checkpoint (rows = layers, cols = k)."""
    return np.array([[sweep[k][li, si] for k in ks] for li in range(len(layers))])


def _surface_png(title, ks, steps, layers, sweep, chance, out_path):
    """One 3-D purity surface per checkpoint, each over (neighbours k, hidden
    layer) on a shared z ∈ [0,1] with a chance floor. Reading across the grid
    (early → late) the surface lifts off chance and domes over the middle layers
    as senses organize; within a panel the fall along k shows how far the purity
    reaches. Colour encodes purity (= surface height)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from .plots import INK, _style

    _style()
    import matplotlib.pyplot as plt

    nk, nL, nS = len(ks), len(layers), len(steps)
    Xk, Yl = np.meshgrid(np.arange(nk), np.arange(nL))           # X = k (col), Y = layer (row)
    norm = Normalize(0, 1)
    cmap = plt.cm.viridis
    ncol = 5
    nrow = int(np.ceil(nS / ncol))

    fig = plt.figure(figsize=(3.5 * ncol, 3.7 * nrow))
    for si, step in enumerate(steps):
        ax = fig.add_subplot(nrow, ncol, si + 1, projection="3d")
        Z = _surface_z(ks, layers, sweep, si)                    # (nL, nk)
        ax.plot_surface(Xk, Yl, np.full_like(Z, chance), color="0.6",
                        alpha=0.12, linewidth=0, shade=False)     # chance floor
        ax.plot_surface(Xk, Yl, Z, cmap=cmap, vmin=0, vmax=1, rstride=1, cstride=1,
                        edgecolor=(0, 0, 0, 0.15), linewidth=0.2, antialiased=True)
        ax.set_xticks(range(nk)); ax.set_xticklabels([f"{k:,}" for k in ks],
                                                     rotation=45, fontsize=5.5, ha="right")
        ax.set_yticks(range(nL)); ax.set_yticklabels([f"L{l}" for l in layers], fontsize=6.5)
        ax.set_zlim(0, 1); ax.set_zticks([0, 0.5, 1.0]); ax.tick_params(labelsize=7)
        ax.set_xlabel("k", fontsize=8, labelpad=4)
        ax.set_ylabel("layer", fontsize=8, labelpad=4)          # z = purity (shared colorbar)
        ax.set_title(f"step {step:,}", color=INK, fontsize=11, pad=-2)
        ax.view_init(elev=24, azim=-58)
        ax.set_box_aspect((1.3, 1.0, 0.8))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("white"); axis.pane.set_alpha(0.4)

    for j in range(nS, nrow * ncol):                            # hide unused cells
        fig.add_subplot(nrow, ncol, j + 1).axis("off")

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, ticks=np.linspace(0, 1, 6),
                        fraction=0.015, pad=0.06, shrink=0.5)
    cbar.set_label("k-NN sense purity  (surface height)", fontsize=10)
    fig.suptitle(title, color=INK, fontsize=15, y=1.0)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def _surface_html(title, ks, steps, layers, sweep, chance, out_path):
    """Interactive 3-D purity surface over (neighbours k, layer); a checkpoint
    slider morphs the surface over training. Fixed z ∈ [0,1] and a chance plane so
    every step is directly comparable."""
    import plotly.graph_objects as go

    nk, nL = len(ks), len(layers)
    xi, yi = list(range(nk)), list(range(nL))
    fig = go.Figure()
    fig.add_trace(go.Surface(                                    # always-visible chance plane
        z=[[chance] * nk for _ in range(nL)], x=xi, y=yi, showscale=False,
        opacity=0.15, colorscale=[[0, "gray"], [1, "gray"]], hoverinfo="skip", name="chance"))
    for si, step in enumerate(steps):
        Z = [[float(v) for v in row] for row in _surface_z(ks, layers, sweep, si)]
        fig.add_trace(go.Surface(
            z=Z, x=xi, y=yi, cmin=0, cmax=1, colorscale="Viridis", showscale=True,
            colorbar=dict(title="purity"), visible=(step == steps[-1]),
            contours={"z": {"show": True, "usecolormap": True, "width": 1}}))

    slider = [dict(method="update", label=f"{step:,}",
                   args=[{"visible": [True] + [t == step for t in steps]}])
              for step in steps]
    fig.update_layout(
        title=title, width=880, height=720, template="plotly_white",
        scene=dict(
            xaxis=dict(tickvals=xi, ticktext=[str(k) for k in ks], title="neighbours k"),
            yaxis=dict(tickvals=yi, ticktext=[f"L{l}" for l in layers], title="layer"),
            zaxis=dict(range=[0, 1], title="k-NN purity")),
        sliders=[dict(active=len(steps) - 1, steps=slider,
                      currentvalue=dict(prefix="checkpoint step: "), pad=dict(t=30))])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    log.info("Wrote %s", out_path)


def similarity_aggregate(out_dir):
    """Average every ``*_purity.csv`` into a corpus-level purity summary CSV and
    3-D surface (mean purity over all analysed words)."""
    from collections import defaultdict

    files = sorted(p for p in out_dir.glob("*_purity.csv") if not p.name.startswith("corpus"))
    if not files:
        log.warning("No *_purity.csv in %s to aggregate", out_dir)
        return
    acc: dict = defaultdict(list)                      # (k, layer, step) -> [purity per word]
    chance_acc: dict = defaultdict(list)
    ks_set, steps_set, layers_set = set(), set(), set()
    for f in files:
        for r in csv.DictReader(f.open()):
            k, s, l = int(r["k"]), int(r["step"]), int(r["layer"])
            ks_set.add(k); steps_set.add(s); layers_set.add(l)
            if r["purity"] not in ("", "nan"):
                acc[(k, l, s)].append(float(r["purity"]))
                chance_acc[(k, l, s)].append(float(r["chance"]))

    ks, steps, layers = sorted(ks_set), sorted(steps_set), sorted(layers_set)
    sweep = {k: np.full((len(layers), len(steps)), np.nan) for k in ks}
    chance = np.full((len(layers), len(steps)), np.nan)
    for k in ks:
        for li, l in enumerate(layers):
            for si, s in enumerate(steps):
                if acc[(k, l, s)]:
                    sweep[k][li, si] = float(np.mean(acc[(k, l, s)]))
                    chance[li, si] = float(np.mean(chance_acc[(k, l, s)]))

    with (out_dir / "corpus_purity.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "layer", "k", "n_words", "mean_chance", "mean_purity", "std_purity"])
        for k in ks:
            for li, l in enumerate(layers):
                for si, s in enumerate(steps):
                    v = acc[(k, l, s)]
                    w.writerow([s, l, k, len(v), chance[li, si], sweep[k][li, si],
                                float(np.std(v)) if v else np.nan])

    mean_chance = float(np.nanmean(chance))
    _surface_png(f"Top-{len(files)} nouns — mean sense-cluster purity surface (k × layer), one panel per checkpoint",
                 ks, steps, layers, sweep, mean_chance, out_dir / "corpus_purity_surface.png")
    _surface_html(f"Top-{len(files)} nouns — mean purity surface (k × layer); drag to scrub checkpoints",
                  ks, steps, layers, sweep, mean_chance, out_dir / "corpus_purity_surface.html")
    log.info("Aggregated %d words -> %s", len(files), out_dir / "corpus_purity_surface.png")


def similarity(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None) -> None:
    """Config-driven entry point: neighbourhood-purity 3-D surface for one word."""
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
