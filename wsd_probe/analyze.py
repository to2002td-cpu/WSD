"""Measure sense separation in the extracted representations.

For every (word, layer, checkpoint) we compare cosine distances between
representations of the target word grouped by annotated sense:

  * intra: mean pairwise cosine distance between instances of the SAME sense
  * inter: mean pairwise cosine distance between instances of DIFFERENT senses
  * ratio = inter / intra  (> 1 means senses are farther apart than chance
    variation within a sense; internally normalized, hence comparable across
    checkpoints despite drifting representation geometry/anisotropy)
  * centroid_dist: mean cosine distance between sense centroids
  * silhouette: silhouette score of the sense labeling (cosine metric)
  * p_perm: permutation test p-value for the statistic (inter - intra); sense
    labels are shuffled `n_permutations` times, so significance is evaluated
    without any distributional assumption.

Results are written as one tidy CSV, one row per (word, layer, step).
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _pairwise_cosine_dist(x: np.ndarray) -> np.ndarray:
    """Full [n, n] cosine distance matrix (float32)."""
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-8)
    dist = 1.0 - x @ x.T
    # Clean up float error so sklearn accepts it as a precomputed distance
    # matrix: non-negative, exactly zero diagonal.
    np.clip(dist, 0.0, None, out=dist)
    np.fill_diagonal(dist, 0.0)
    return dist


def _mean_intra_inter(
    dist: np.ndarray, labels: np.ndarray
) -> "tuple[float, float]":
    same = labels[:, None] == labels[None, :]
    triu = np.triu(np.ones_like(same, dtype=bool), k=1)
    intra = dist[same & triu]
    inter = dist[~same & triu]
    return float(intra.mean()), float(inter.mean())


def _permutation_pvalue(
    dist: np.ndarray,
    labels: np.ndarray,
    observed: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """P(inter - intra >= observed) under random relabeling of instances."""
    count = 0
    perm = labels.copy()
    for _ in range(n_permutations):
        rng.shuffle(perm)
        intra, inter = _mean_intra_inter(dist, perm)
        if inter - intra >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


def analyze_word(
    vectors: np.ndarray,  # [n_instances, n_layers, hidden]
    senses: np.ndarray,   # [n_instances] sense labels
    n_permutations: int,
    rng: np.random.Generator,
) -> "list[dict]":
    """Per-layer separation metrics for one word at one checkpoint."""
    from sklearn.metrics import silhouette_score

    _, sense_ids = np.unique(senses, return_inverse=True)
    n_layers = vectors.shape[1]
    rows = []
    for layer in range(n_layers):
        x = vectors[:, layer, :]
        dist = _pairwise_cosine_dist(x)
        intra, inter = _mean_intra_inter(dist, sense_ids)
        observed = inter - intra

        centroids = np.stack(
            [x[sense_ids == s].mean(axis=0) for s in np.unique(sense_ids)]
        )
        cdist = _pairwise_cosine_dist(centroids)
        centroid_dist = float(
            np.mean([cdist[i, j] for i, j in
                     itertools.combinations(range(len(centroids)), 2)])
        )

        try:
            sil = float(silhouette_score(dist, sense_ids, metric="precomputed"))
        except ValueError:
            sil = float("nan")

        p = _permutation_pvalue(dist, sense_ids, observed, n_permutations, rng)
        rows.append(
            dict(
                layer=layer,
                intra_dist=intra,
                inter_dist=inter,
                ratio=inter / intra if intra > 0 else float("nan"),
                centroid_dist=centroid_dist,
                silhouette=sil,
                p_perm=p,
            )
        )
    return rows


def analyze_all(
    records: "list[dict]",
    embeddings_dir: Path,
    out_csv: Path,
    n_permutations: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the analysis for every word x layer x checkpoint found on disk."""
    meta = pd.DataFrame(records)
    meta["row"] = np.arange(len(meta))

    npz_files = sorted(
        embeddings_dir.glob("step*.npz"),
        key=lambda p: int(p.stem.removeprefix("step")),
    )
    if not npz_files:
        raise FileNotFoundError(f"No step*.npz found in {embeddings_dir}")
    log.info("Analyzing %d checkpoints from %s", len(npz_files), embeddings_dir)

    all_rows = []
    for npz_path in npz_files:
        step = int(npz_path.stem.removeprefix("step"))
        vectors = np.load(npz_path)["vectors"]
        # Rows lost to truncation were left as zeros -> drop them.
        valid = np.linalg.norm(
            vectors[:, -1, :].astype(np.float32), axis=1
        ) > 0
        rng = np.random.default_rng(seed)

        for (word, pos), group in meta.groupby(["word", "pos"]):
            rows_idx = group["row"].to_numpy()
            ok = valid[rows_idx]
            g = group[ok]
            # A sense may lose instances to truncation; keep senses that
            # still have >= 2 examples, and words with >= 2 such senses.
            counts = g["sense"].value_counts()
            keep_senses = counts[counts >= 2].index
            g = g[g["sense"].isin(keep_senses)]
            if g["sense"].nunique() < 2:
                continue

            word_vecs = vectors[g["row"].to_numpy()]
            senses = g["sense"].to_numpy()
            for row in analyze_word(word_vecs, senses, n_permutations, rng):
                row.update(
                    word=word,
                    pos=pos,
                    step=step,
                    n_senses=int(g["sense"].nunique()),
                    n_instances=len(g),
                )
                all_rows.append(row)
        log.info("step %d done", step)

    df = pd.DataFrame(all_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Wrote %d rows to %s", len(df), out_csv)
    return df


def summarize(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Aggregate over words: mean separation and share of significant words
    per (step, layer)."""
    agg = (
        df.groupby(["step", "layer"])
        .agg(
            ratio_mean=("ratio", "mean"),
            silhouette_mean=("silhouette", "mean"),
            centroid_dist_mean=("centroid_dist", "mean"),
            frac_significant=("p_perm", lambda p: float((p < alpha).mean())),
            n_words=("word", "nunique"),
        )
        .reset_index()
    )
    return agg
