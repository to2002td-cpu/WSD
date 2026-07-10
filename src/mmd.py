"""Kernel two-sample test (Maximum Mean Discrepancy) between senses.

An MMD-based two-sample test (Gretton et al., *A Kernel Two-Sample Test*, JMLR
2012) applied pairwise between a word's WordNet senses in the model's raw
per-layer embedding space. For every (checkpoint, layer) and every pair of
senses (a, b) we test

    H0 : P_a = P_b     (the two sense-conditional embedding distributions agree)

with the unbiased quadratic-time MMD^2 estimator, an RBF kernel whose bandwidth
is set by the median heuristic, and a label-permutation null. The ground truth
is that every pair is a genuinely distinct sense, so the fraction of pairs the
test can tell apart — per step and layer — is a statistical measure of how far
the model has pulled the senses apart, alongside the visual KDE clouds.

Everything runs on the raw pooled hidden states (not the 2-D UMAP, whose
non-linear projection would distort the distances the kernel relies on).
"""

from __future__ import annotations

import csv
import logging

import numpy as np

from .umapcache import (
    SENSE_PALETTE,
    _layers,
    _load_vectors,
    _population,
    _valid_mask,
    displayed_senses,
    gloss,
)

log = logging.getLogger(__name__)

MIN_SAMPLES = 20     # skip a sense (and its pairs) below this many occurrences


# --------------------------------------------------------------------------- #
# Core estimator + permutation test                                           #
# --------------------------------------------------------------------------- #

def _rbf_gram(Z: np.ndarray) -> np.ndarray:
    """RBF Gram matrix with the median-heuristic bandwidth.

    Uses exp(-||x-y||^2 / median(||.-.||^2)); i.e. sigma^2 = median(d2)/2, the
    common median heuristic, computed on the pooled sample's pairwise squared
    distances."""
    sq = np.einsum("ij,ij->i", Z, Z)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(d2, 0.0, out=d2)
    iu = np.triu_indices(len(Z), k=1)
    med = float(np.median(d2[iu])) if iu[0].size else 1.0
    if med <= 0:
        med = 1.0
    return np.exp(-d2 / med)


def _mmd2_unbiased(K: np.ndarray, m: int) -> float:
    """Unbiased MMD^2 estimate (Gretton 2012, Lemma 6) from a pooled Gram matrix
    whose first ``m`` rows/cols are sample X and the rest sample Y."""
    n = K.shape[0] - m
    Kxx, Kyy, Kxy = K[:m, :m], K[m:, m:], K[:m, m:]
    sxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    return float(sxx + syy - 2.0 * Kxy.mean())


def kernel_two_sample_test(X: np.ndarray, Y: np.ndarray, n_perm: int, rng) -> "tuple[float, float]":
    """(MMD^2, p-value) for H0: X and Y are drawn from the same distribution.

    The p-value comes from a label-permutation null. The permutation statistic
    is the *biased* quadratic form w^T K w (w = +1/m on X, -1/n on Y); its
    diagonal contribution is constant across permutations, so the permutation
    distribution matches the unbiased one up to that shift and the test is
    identical — while being a single matmul over all permutations at once.
    """
    Z = np.vstack([X, Y]).astype(np.float64)
    m, n = len(X), len(Y)
    N = m + n
    K = _rbf_gram(Z)
    mmd2 = _mmd2_unbiased(K, m)

    w = np.concatenate([np.full(m, 1.0 / m), np.full(n, -1.0 / n)])
    obs = float(w @ K @ w)
    V = np.empty((N, n_perm))
    for b in range(n_perm):
        V[:, b] = w[rng.permutation(N)]
    stats = np.einsum("ij,ij->j", V, K @ V)          # w^T K w per permutation
    p = (1 + int(np.count_nonzero(stats >= obs))) / (n_perm + 1)
    return mmd2, p


# --------------------------------------------------------------------------- #
# Driver over (step, layer, sense-pair)                                       #
# --------------------------------------------------------------------------- #

def _prep_word(records, word, pos, valid, min_per_sense, max_senses):
    """Per-word test plan: (rows, senses, local_idx, pairs). Raises SystemExit
    (via ``_population``) or returns None if the word has <2 usable senses."""
    meta, pos = _population(records, word, pos, valid)
    senses = displayed_senses(meta, min_per_sense, max_senses)
    if len(senses) < 2:
        return None
    sense_arr = meta["sense"].to_numpy()
    return dict(pos=pos, rows=meta["row"].to_numpy(), senses=senses,
                local_idx={s: np.where(sense_arr == s)[0] for s in senses},
                pairs=[(a, b) for i, a in enumerate(senses) for b in senses[i + 1:]])


def _subsample(X_all, local_idx, senses, max_per_sense, rng):
    """One bounded, standardized sample per sense (drawn once per step×layer)."""
    samp = {}
    for s in senses:
        Xs = X_all[local_idx[s]]
        if len(Xs) > max_per_sense:
            Xs = Xs[rng.choice(len(Xs), max_per_sense, replace=False)]
        samp[s] = Xs
    return samp


def _run_pairs(samp, pairs, n_perm, alpha, step, layer, rng, csv_rows):
    """Test every sense pair at one (step, layer); append per-pair CSV rows.
    Returns (n_tested, n_significant, sum_mmd2)."""
    n_tested = n_diff = 0
    mmd2_sum = 0.0
    for a, b in pairs:
        Xa, Xb = samp[a], samp[b]
        if len(Xa) < MIN_SAMPLES or len(Xb) < MIN_SAMPLES:
            continue
        mmd2, p = kernel_two_sample_test(Xa, Xb, n_perm, rng)
        sig = p < alpha
        n_tested += 1
        n_diff += int(sig)
        mmd2_sum += mmd2
        csv_rows.append(dict(step=step, layer=layer, sense_a=a, sense_b=b,
                             n_a=len(Xa), n_b=len(Xb), mmd2=mmd2, p_value=p,
                             significant=int(sig)))
    return n_tested, n_diff, mmd2_sum


def _write_word_csvs(out_dir, word, csv_rows, steps, layers, frac, eff, n_pairs):
    with (out_dir / f"{word}_mmd.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "layer", "sense_a", "sense_b",
                                                "n_a", "n_b", "mmd2", "p_value", "significant"])
        writer.writeheader()
        writer.writerows(csv_rows)
    with (out_dir / f"{word}_mmd_summary.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "layer", "n_pairs", "frac_significant", "mean_mmd2"])
        for li, layer in enumerate(layers):
            for si, step in enumerate(steps):
                writer.writerow([step, layer, n_pairs, frac[li, si], eff[li, si]])
    log.info("Wrote %s_mmd.csv (+summary, %d pairwise tests)", word, len(csv_rows))


def mmd_word(records, emb_dir, word, pos, out_dir, *, n_perm: int, max_per_sense: int,
             alpha: float, max_senses: int, standardize: bool, min_per_sense: int,
             cache_dir=None, seed: int = 0):
    """Run the pairwise sense MMD test for one word across every checkpoint and
    layer; write the per-pair + summary CSVs. Returns (steps, layers, frac, eff,
    n_pairs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(emb_dir.glob("step*.npz"),
                       key=lambda p: int(p.stem.removeprefix("step")))
    if not npz_files:
        raise SystemExit(f"No step*.npz in {emb_dir}")

    valid = _valid_mask(npz_files, cache_dir)
    plan = _prep_word(records, word, pos, valid, min_per_sense, max_senses)
    if plan is None:
        raise SystemExit(f"'{word}' ({pos}) has <2 senses with >= {min_per_sense} occurrences.")

    steps = [int(p.stem.removeprefix("step")) for p in npz_files]
    layers = _layers(npz_files[0])
    rows, senses, local_idx, pairs = plan["rows"], plan["senses"], plan["local_idx"], plan["pairs"]
    rng = np.random.default_rng(seed)

    frac = np.full((len(layers), len(steps)), np.nan)   # fraction of pairs significant
    eff = np.full((len(layers), len(steps)), np.nan)     # mean MMD^2 effect size
    csv_rows: list[dict] = []
    for si, (npz, step) in enumerate(zip(npz_files, steps)):
        mm = _load_vectors(npz, cache_dir)                       # [N_all, L, H]
        for li, layer in enumerate(layers):
            X_all = np.asarray(mm[rows, li, :], np.float32)      # word population, meta order
            if standardize:
                X_all = (X_all - X_all.mean(0)) / (X_all.std(0) + 1e-6)
            samp = _subsample(X_all, local_idx, senses, max_per_sense, rng)
            nt, nd, ms = _run_pairs(samp, pairs, n_perm, alpha, step, layer, rng, csv_rows)
            if nt:
                frac[li, si], eff[li, si] = nd / nt, ms / nt
            log.info("%s step %d layer %d: %d/%d significant, mean MMD² = %.3f",
                     word, step, layer, nd, nt, eff[li, si])
        del mm

    _write_word_csvs(out_dir, word, csv_rows, steps, layers, frac, eff, len(pairs))
    return steps, layers, frac, eff, len(pairs)


def mmd_corpus(records, emb_dir, words, pos, out_dir, *, n_perm: int, max_per_sense: int,
               alpha: float, max_senses: int, standardize: bool, min_per_sense: int,
               cache_dir=None, seed: int = 0):
    """Batch analysis over many words that loads each checkpoint's vectors only
    once (all words share the pass), so no unzipped ``.npy`` cache is needed.
    Writes per-word CSVs + heatmaps and the corpus aggregate. Words with <2
    usable senses are skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(emb_dir.glob("step*.npz"),
                       key=lambda p: int(p.stem.removeprefix("step")))
    if not npz_files:
        raise SystemExit(f"No step*.npz in {emb_dir}")
    valid = _valid_mask(npz_files, cache_dir)
    steps = [int(p.stem.removeprefix("step")) for p in npz_files]
    layers = _layers(npz_files[0])

    plans: dict[str, dict] = {}
    for word in words:
        try:
            plan = _prep_word(records, word, pos, valid, min_per_sense, max_senses)
        except SystemExit as e:                                  # word absent / no valid rows
            log.warning("skip %s: %s", word, e)
            continue
        if plan is None:
            log.warning("skip %s: <2 senses with >= %d occurrences", word, min_per_sense)
            continue
        plan.update(frac=np.full((len(layers), len(steps)), np.nan),
                    eff=np.full((len(layers), len(steps)), np.nan), csv_rows=[])
        plans[word] = plan
    log.info("MMD corpus: %d/%d words usable", len(plans), len(words))

    rng = np.random.default_rng(seed)
    for si, (npz, step) in enumerate(zip(npz_files, steps)):
        vectors = _load_vectors(npz, cache_dir)                  # [N_all, L, H], loaded once
        for li, layer in enumerate(layers):
            for word, plan in plans.items():
                X = np.asarray(vectors[plan["rows"], li, :], np.float32)
                if standardize:
                    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
                samp = _subsample(X, plan["local_idx"], plan["senses"], max_per_sense, rng)
                nt, nd, ms = _run_pairs(samp, plan["pairs"], n_perm, alpha, step, layer,
                                        rng, plan["csv_rows"])
                if nt:
                    plan["frac"][li, si], plan["eff"][li, si] = nd / nt, ms / nt
        del vectors
        log.info("checkpoint %d done (%d words)", step, len(plans))

    for word, plan in plans.items():
        _write_word_csvs(out_dir, word, plan["csv_rows"], steps, layers,
                         plan["frac"], plan["eff"], len(plan["pairs"]))
        _heatmap(f"“{word}” ({plan['pos']}) — sense separation by MMD kernel two-sample test",
                 steps, layers, plan["frac"], plan["eff"], out_dir / f"{word}_mmd.png", alpha)
    mmd_aggregate(out_dir, alpha)
    return plans


def _panel(ax, M, steps, layers, cmap, vmax, fmt):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax, origin="upper")
    ax.set_xticks(range(len(steps)), [f"{s:,}" for s in steps], rotation=45, ha="right")
    ax.set_yticks(range(len(layers)), [f"L{l}" for l in layers])
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("hidden layer")
    thresh = vmax * 0.6
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if v < thresh else "black", fontsize=7.5)
    return im


def _heatmap(title, steps, layers, frac, eff, out_path, alpha):
    from .plots import INK, _style

    _style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(1.05 * len(steps) + 2.5, 1.4 * len(layers) + 2.5))
    im0 = _panel(axes[0], eff, steps, layers, "viridis", float(np.nanmax(eff)) or 1.0, ".3f")
    axes[0].set_title("mean MMD² effect size (how strongly senses are separated)",
                      color=INK, fontsize=11)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02).set_label("mean MMD²")

    im1 = _panel(axes[1], frac, steps, layers, "magma", 1.0, ".2f")
    axes[1].set_title(f"fraction of sense-pairs significant (p < {alpha})",
                      color=INK, fontsize=11)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02).set_label("fraction significant")

    fig.suptitle(title, color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def mmd_visualize(records, emb_dir, word, pos, out_dir, **kw):
    """Run the test and render the layer×step separation heatmaps."""
    alpha = kw["alpha"]
    steps, layers, frac, eff, _ = mmd_word(records, emb_dir, word, pos, out_dir, **kw)
    _heatmap(f"“{word}” ({pos}) — sense separation by MMD kernel two-sample test",
             steps, layers, frac, eff, out_dir / f"{word}_mmd.png", alpha)
    return steps, layers, frac, eff


def mmd_aggregate(out_dir, alpha: float):
    """Average every ``*_mmd_summary.csv`` in ``out_dir`` into one corpus-level
    layer×step heatmap (mean effect size and mean significant-fraction over all
    analysed words) — the headline result across the corpus."""
    import csv as _csv
    from collections import defaultdict

    files = sorted(p for p in out_dir.glob("*_mmd_summary.csv")
                   if not p.name.startswith("corpus"))
    if not files:
        log.warning("No *_mmd_summary.csv in %s to aggregate", out_dir)
        return
    eff_acc: dict = defaultdict(list)
    frac_acc: dict = defaultdict(list)
    steps_set, layers_set = set(), set()
    for f in files:
        for r in _csv.DictReader(f.open()):
            s, l = int(r["step"]), int(r["layer"])
            steps_set.add(s); layers_set.add(l)
            for key, acc in (("mean_mmd2", eff_acc), ("frac_significant", frac_acc)):
                v = r[key]
                if v not in ("", "nan"):
                    acc[(l, s)].append(float(v))

    steps, layers = sorted(steps_set), sorted(layers_set)
    eff = np.full((len(layers), len(steps)), np.nan)
    frac = np.full((len(layers), len(steps)), np.nan)
    for li, l in enumerate(layers):
        for si, s in enumerate(steps):
            if eff_acc[(l, s)]:
                eff[li, si] = float(np.mean(eff_acc[(l, s)]))
            if frac_acc[(l, s)]:
                frac[li, si] = float(np.mean(frac_acc[(l, s)]))

    with (out_dir / "corpus_mmd_summary.csv").open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["step", "layer", "n_words", "mean_frac_significant", "mean_mmd2"])
        for li, l in enumerate(layers):
            for si, s in enumerate(steps):
                w.writerow([s, l, len(eff_acc[(l, s)]), frac[li, si], eff[li, si]])
    _heatmap(f"Top-{len(files)} nouns — mean sense separation by MMD kernel two-sample test",
             steps, layers, frac, eff, out_dir / "corpus_mmd.png", alpha)
    log.info("Aggregated %d words -> %s", len(files), out_dir / "corpus_mmd.png")


def mmd(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None) -> None:
    """Config-driven entry point: pairwise sense MMD test + heatmap for one word."""
    from .config import run_paths, store
    from .dataset import build_dataset

    m = cfg["mmd"]
    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg, only)
    mmd_visualize(
        records, emb_dir, word, pos, store(cfg, m["out_dir"]),
        n_perm=m["n_perm"], max_per_sense=m["max_per_sense"], alpha=m["alpha"],
        max_senses=m["max_senses"], standardize=m["standardize"],
        min_per_sense=p["min_per_sense"], cache_dir=cache_dir, seed=m["seed"],
    )
