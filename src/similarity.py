"""Local neighbourhood purity of contextual sense embeddings.

The KDE panels show that a word's occurrences form clusters, and that a single
WordNet sense may split into several clusters. The question that matters is not
"is one sense-blob far from another" (senses are multimodal, so blob-distance /
silhouette is misleading) but: **are the clusters pure?** Pick any occurrence and
look at its nearest neighbours in the model's embedding space — do they share its
sense (a pure, same-meaning neighbourhood) or is everything mixed?

This is the most basic similarity primitive — cosine nearest neighbours on the
raw pooled hidden states (no UMAP, no standardisation) — turned into two
transparent, bounded measures, per checkpoint and layer:

  * **purity** — for each occurrence, the fraction of its k nearest neighbours
    that carry the same sense, averaged over occurrences. 1 = every neighbourhood
    is single-sense; ``chance = 1/K`` (K balanced senses) = fully mixed.
  * **neighbour-membership matrix** N — N_ab = fraction of sense a's neighbours
    that belong to sense b (rows sum to 1). The diagonal is per-sense purity; the
    off-diagonal shows which senses get confused with which.

Being local, purity is unaffected by a sense fragmenting into several clusters:
as long as each neighbourhood is single-sense, purity is high. Reported against
its chance baseline (``purity - chance``, on a fixed [-1, 1] scale) so panels are
directly comparable and 0 always means "mixed".
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


def _balanced_idx(y: np.ndarray, k: int, max_per_sense: int, rng) -> np.ndarray:
    """Up to ``max_per_sense`` row indices per sense (balanced, so purity's chance
    baseline is a clean 1/K and no sense dominates the neighbourhoods)."""
    out = []
    for c in range(k):
        ci = np.where(y == c)[0]
        if len(ci) < MIN_SAMPLES:
            continue
        out.append(ci if len(ci) <= max_per_sense
                   else rng.choice(ci, max_per_sense, replace=False))
    return np.concatenate(out) if out else np.array([], dtype=int)


def _purity(Zn: np.ndarray, yl: np.ndarray, K: int, ks: "list[int]", ref_k: int):
    """(purity per k, per-sense purity at ref_k, membership matrix N at ref_k).

    Cosine nearest neighbours (self excluded) are ranked once; purity at each k
    is the mean, over points, of the same-sense fraction among the k nearest
    neighbours. N_ab (at ``ref_k``) averages, over points of sense a, the fraction
    of their neighbours in sense b (row-stochastic)."""
    n = len(yl)
    ks_eff = {k: min(k, n - 1) for k in ks}            # requested k -> usable k (clamped to data)
    kmax = max(ks_eff.values())
    S = Zn @ Zn.T
    np.fill_diagonal(S, -np.inf)                       # never a neighbour of itself
    part = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]     # kmax nearest (unordered)
    rows = np.arange(n)[:, None]
    nn = part[rows, np.argsort(-S[rows, part], axis=1)]        # ... ordered nearest-first
    neigh = yl[nn]                                     # neighbour senses, n×kmax
    same = (neigh == yl[:, None])                      # n×kmax boolean

    # Key by the requested k; k beyond the data all clamp to n-1 (purity -> chance).
    purity_by_k = {k: float(same[:, :ke].mean()) for k, ke in ks_eff.items()}

    ref = min(ref_k, kmax)
    memb = np.stack([(neigh[:, :ref] == c).mean(1) for c in range(K)], axis=1)  # n×K
    per_sense = np.array([same[yl == c, :ref].mean() if np.any(yl == c) else np.nan
                          for c in range(K)])
    N = np.stack([memb[yl == c].mean(0) if np.any(yl == c) else np.full(K, np.nan)
                  for c in range(K)])
    return purity_by_k, per_sense, N


def _purity_at(X, y, k, max_per_sense, ks, ref_k, rng):
    """(purity_by_k, per_sense, N, present) for one (step, layer), or ``None`` if
    fewer than two senses survive the balanced sample."""
    idx = _balanced_idx(y, k, max_per_sense, rng)
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
    purity_by_k, per_sense, N = _purity(Z, yl, len(present), ks, ref_k)
    return purity_by_k, per_sense, N, present


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
                      max_senses: int, knn_k: int, knn_ks: "list[int]", min_per_sense: int,
                      cache_dir=None, seed: int = 0):
    """Neighbourhood-purity analysis for many words, loading each checkpoint once.
    Writes per-word CSVs + figures (purity panels + the layer×step membership grid
    + the purity-vs-k sweep) and the corpus aggregate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(emb_dir.glob("step*.npz"),
                       key=lambda p: int(p.stem.removeprefix("step")))
    if not npz_files:
        raise SystemExit(f"No step*.npz in {emb_dir}")
    valid = _valid_mask(npz_files, cache_dir)
    steps = [int(p.stem.removeprefix("step")) for p in npz_files]
    layers = _layers(npz_files[0])
    ks = sorted(set(knn_ks) | {knn_k})                 # sweep, with the reference k included

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
                    chance=1.0 / len(plan["senses"]), detail=[], Ns={})
        plans[word] = plan
    log.info("Purity: %d/%d words usable (k sweep %s, ref k=%d)", len(plans), len(words), ks, knn_k)

    rng = np.random.default_rng(seed)
    for si, (npz, step) in enumerate(zip(npz_files, steps)):
        vectors = _load_vectors(npz, cache_dir)                  # loaded once for all words
        for li, layer in enumerate(layers):
            for word, plan in plans.items():
                X = np.asarray(vectors[plan["rows"], li, :], np.float32)
                res = _purity_at(X, plan["y"], len(plan["senses"]), max_per_sense, ks, knn_k, rng)
                if res is None:
                    continue
                purity_by_k, per_sense, N, present = res
                for k, pu in purity_by_k.items():
                    plan["sweep"][k][li, si] = pu
                plan["Ns"][(li, si)] = (N, present)
                for local, s_idx in enumerate(present):
                    plan["detail"].append(dict(step=step, layer=layer,
                                               sense=plan["senses"][s_idx],
                                               purity=float(per_sense[local])))
        del vectors
        log.info("checkpoint %d done (%d words)", step, len(plans))

    for word, plan in plans.items():
        plan["purity"] = plan["sweep"][knn_k]           # reference-k array for the heatmaps
        _write_word_csvs(out_dir, word, steps, layers, ks, plan)
        _save_matrices(out_dir, word, steps, layers, plan)
        _purity_panels(f"“{word}” ({plan['pos']}) — sense-cluster purity in embedding space (k={knn_k})",
                       steps, layers, plan, out_dir / f"{word}_purity.png")
        _matrix_grid(word, plan, steps, layers, knn_k, out_dir / f"{word}_membership.png")
        _sweep_surface_png(f"“{word}” ({plan['pos']}) — sense-cluster purity surface (k × layer), one panel per checkpoint",
                           ks, steps, layers, plan["sweep"], plan["chance"],
                           out_dir / f"{word}_purity_vs_k.png")
        _sweep_surface_html(f"“{word}” ({plan['pos']}) — purity surface (k × layer); drag to scrub checkpoints",
                            ks, steps, layers, plan["sweep"], plan["chance"],
                            out_dir / f"{word}_purity_surface.html")
    similarity_aggregate(out_dir, knn_k)
    return plans


def similarity_word(records, emb_dir, word, pos, out_dir, **kw):
    """Single-word convenience wrapper over :func:`similarity_corpus`."""
    plans = similarity_corpus(records, emb_dir, [word], pos, out_dir, **kw)
    if not plans:
        raise SystemExit(f"'{word}' ({pos}) has <2 senses with >= {kw['min_per_sense']} occurrences.")
    return plans


# --------------------------------------------------------------------------- #
# Output: CSVs + saved matrices                                                #
# --------------------------------------------------------------------------- #

def _write_word_csvs(out_dir, word, steps, layers, ks, plan):
    with (out_dir / f"{word}_purity_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "layer", "k", "n_senses", "chance", "purity", "purity_minus_chance"])
        nse, chance = len(plan["senses"]), plan["chance"]
        for k in ks:
            for li, layer in enumerate(layers):
                for si, step in enumerate(steps):
                    pu = plan["sweep"][k][li, si]
                    w.writerow([step, layer, k, nse, chance, pu, pu - chance])
    with (out_dir / f"{word}_purity_per_sense.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "layer", "sense", "purity"])
        w.writeheader()
        w.writerows(plan["detail"])
    log.info("Wrote %s_purity_summary.csv (+per-sense detail)", word)


def _save_matrices(out_dir, word, steps, layers, plan):
    """Persist every layer×step neighbour-membership matrix as data."""
    k = len(plan["senses"])
    cube = np.full((len(layers), len(steps), k, k), np.nan)
    for (li, si), (N, present) in plan["Ns"].items():
        cube[li, si][np.ix_(present, present)] = N
    np.savez_compressed(out_dir / f"{word}_membership.npz", matrices=cube,
                        steps=np.array(steps), layers=np.array(layers),
                        senses=np.array(plan["senses"]))


# --------------------------------------------------------------------------- #
# Output: figures                                                              #
# --------------------------------------------------------------------------- #

def _annot_panel(ax, M, steps, layers, cmap, vmin, vmax, fmt, title):
    from .plots import INK

    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xticks(range(len(steps)), [f"{s:,}" for s in steps], rotation=45, ha="right")
    ax.set_yticks(range(len(layers)), [f"L{l}" for l in layers])
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("hidden layer")
    ax.set_title(title, color=INK, fontsize=10.5)
    mid = 0.5 * (vmin + vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if v < mid else "black", fontsize=7)
    return im


def _purity_panels(title, steps, layers, plan, out_path):
    """Two heatmaps on fixed scales: raw purity [0,1], and purity − chance on a
    fixed [-1, 1] diverging scale (0 = mixed / chance-level)."""
    from .plots import INK, _style

    purity, chance = plan["purity"], plan["chance"]
    _style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(2.1 * len(steps) + 3.0, 1.5 * len(layers) + 2.4))
    im = _annot_panel(axes[0], purity, steps, layers, "viridis", 0.0, 1.0, ".2f",
                      f"k-NN purity (fraction of same-sense neighbours; chance ≈ {chance:.2f})")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02).set_label("purity")

    im = _annot_panel(axes[1], purity - chance, steps, layers, "RdBu_r", -1.0, 1.0, ".2f",
                      "purity − chance (normalized; 0 = fully mixed, 1 = perfectly pure)")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02).set_label("purity − chance")

    fig.suptitle(title, color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def _matrix_grid(word, plan, steps, layers, ref_k, out_path):
    """Every layer×step neighbour-membership matrix on one fixed [-1, 1] diverging
    scale (N − chance). White = chance-level mixing, red diagonal = pure same-sense
    neighbourhoods, blue = under-represented: flat at init, diagonal reddens as the
    senses organize."""
    from .plots import INK, _style

    chance = plan["chance"]
    cells = {(li, si): (N - chance) for (li, si), (N, _) in plan["Ns"].items()}
    if not cells:
        return

    _style()
    import matplotlib.pyplot as plt

    nL, nS = len(layers), len(steps)
    fig, axes = plt.subplots(nL, nS, figsize=(1.15 * nS + 1.5, 1.15 * nL + 1.5),
                             squeeze=False)
    im = None
    for li in range(nL):
        for si in range(nS):
            ax = axes[li][si]
            ax.set_xticks([]); ax.set_yticks([])
            C = cells.get((li, si))
            if C is not None:
                im = ax.imshow(C, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            else:
                ax.set_facecolor("#eeeeee")
            if si == 0:
                ax.set_ylabel(f"L{layers[li]}", fontsize=9, rotation=0, ha="right", va="center")
            if li == nL - 1:
                ax.set_xlabel(f"{steps[si]:,}", fontsize=8, rotation=45, ha="right")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("neighbour fraction − chance  (red diagonal = pure)")
    fig.suptitle(f"“{word}” ({plan['pos']}) — neighbour-membership matrix (k={ref_k}), every layer × step",
                 color=INK, fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def _surface_z(ks, layers, sweep, si):
    """purity[layer, k] grid for one checkpoint (rows = layers, cols = k)."""
    return np.array([[sweep[k][li, si] for k in ks] for li in range(len(layers))])


def _sweep_surface_png(title, ks, steps, layers, sweep, chance, out_path):
    """Hero figure: one 3-D purity surface per checkpoint, each over (neighbours k,
    hidden layer), on a shared z ∈ [0,1] with a chance floor. Read across the grid
    (early → late) to watch the whole surface lift off chance and swell over the
    middle layers as the senses organize; within a panel the fall along k shows how
    far that purity reaches."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from .plots import INK, _style

    _style()
    import matplotlib.pyplot as plt

    nk, nL, nS = len(ks), len(layers), len(steps)
    Xk, Yl = np.meshgrid(np.arange(nk), np.arange(nL))            # X = k (col), Y = layer (row)
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
        ax.set_ylabel("layer", fontsize=8, labelpad=4)      # z = purity (see shared colorbar)
        ax.set_title(f"step {step:,}", color=INK, fontsize=11, pad=-2)
        ax.view_init(elev=24, azim=-58)
        ax.set_box_aspect((1.3, 1.0, 0.8))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("white"); axis.pane.set_alpha(0.4)

    for j in range(nS, nrow * ncol):                             # hide unused cells
        fig.add_subplot(nrow, ncol, j + 1).axis("off")

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, ticks=np.linspace(0, 1, 6),
                        fraction=0.015, pad=0.06, shrink=0.5)
    cbar.set_label("k-NN sense purity  (surface height)", fontsize=10)
    fig.suptitle(title, color=INK, fontsize=15, y=1.0)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def _sweep_surface_html(title, ks, steps, layers, sweep, chance, out_path):
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


def similarity_aggregate(out_dir, ref_k):
    """Average every ``*_purity_summary.csv`` into corpus-level purity panels (at
    the reference k) and a corpus purity-vs-k sweep, with per-cell std in the CSV."""
    from collections import defaultdict

    files = sorted(p for p in out_dir.glob("*_purity_summary.csv")
                   if not p.name.startswith("corpus"))
    if not files:
        log.warning("No *_purity_summary.csv in %s to aggregate", out_dir)
        return
    pur_acc: dict = defaultdict(list)                   # (k, l, s) -> [purity per word]
    chance_acc: dict = defaultdict(list)
    ks_set, steps_set, layers_set = set(), set(), set()
    for f in files:
        for r in csv.DictReader(f.open()):
            k, s, l = int(r["k"]), int(r["step"]), int(r["layer"])
            ks_set.add(k); steps_set.add(s); layers_set.add(l)
            if r["purity"] not in ("", "nan"):
                pur_acc[(k, l, s)].append(float(r["purity"]))
                chance_acc[(k, l, s)].append(float(r["chance"]))

    ks, steps, layers = sorted(ks_set), sorted(steps_set), sorted(layers_set)
    sweep = {k: np.full((len(layers), len(steps)), np.nan) for k in ks}
    chance = np.full((len(layers), len(steps)), np.nan)
    for k in ks:
        for li, l in enumerate(layers):
            for si, s in enumerate(steps):
                if pur_acc[(k, l, s)]:
                    sweep[k][li, si] = float(np.mean(pur_acc[(k, l, s)]))
                    chance[li, si] = float(np.mean(chance_acc[(k, l, s)]))

    with (out_dir / "corpus_purity_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "layer", "k", "n_words", "mean_chance", "mean_purity",
                    "std_purity", "mean_purity_minus_chance"])
        for k in ks:
            for li, l in enumerate(layers):
                for si, s in enumerate(steps):
                    v = pur_acc[(k, l, s)]
                    w.writerow([s, l, k, len(v), chance[li, si], sweep[k][li, si],
                                float(np.std(v)) if v else np.nan,
                                sweep[k][li, si] - chance[li, si]])

    # Corpus chance varies by word; use the mean chance for the labelled baseline.
    mean_chance = float(np.nanmean(chance))
    plan = dict(purity=sweep[ref_k], chance=mean_chance)
    _purity_panels(f"Top-{len(files)} nouns — mean sense-cluster purity in embedding space (k={ref_k})",
                   steps, layers, plan, out_dir / "corpus_purity.png")
    _sweep_surface_png(f"Top-{len(files)} nouns — mean sense-cluster purity surface (k × layer), one panel per checkpoint",
                       ks, steps, layers, sweep, mean_chance, out_dir / "corpus_purity_vs_k.png")
    _sweep_surface_html(f"Top-{len(files)} nouns — mean purity surface (k × layer); drag to scrub checkpoints",
                        ks, steps, layers, sweep, mean_chance, out_dir / "corpus_purity_surface.html")
    log.info("Aggregated %d words -> %s", len(files), out_dir / "corpus_purity.png")


def similarity(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None) -> None:
    """Config-driven entry point: neighbourhood-purity analysis for one word."""
    from .config import run_paths, store
    from .dataset import build_dataset

    m = cfg.get("similarity", {})
    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg, only)
    similarity_word(
        records, emb_dir, word, pos, store(cfg, m.get("out_dir", "similarity")),
        max_per_sense=m.get("max_per_sense", 300), max_senses=m.get("max_senses", 12),
        knn_k=m.get("knn_k", 10), knn_ks=m.get("knn_ks", [5, 10, 20, 50, 100]),
        min_per_sense=p["min_per_sense"], cache_dir=cache_dir, seed=m.get("seed", 0),
    )
