"""Measure sense separation in the extracted representations.

Centroid-based analysis (no pairwise n x n matrices). For every
(word, layer, checkpoint) we summarize each annotated sense by its centroid
(the mean representation of its instances) and measure separation relative to
within-sense spread:

  * intra: mean cosine distance from each instance to its OWN sense centroid
    (within-sense scatter / compactness)
  * inter: mean cosine distance from each instance to the OTHER senses'
    centroids
  * ratio = inter / intra  (> 1 means senses sit farther apart than the spread
    within a sense; internally normalized, hence comparable across checkpoints
    despite drifting representation geometry/anisotropy)
  * centroid_dist: mean cosine distance between sense centroids
  * silhouette: nearest-centroid silhouette, mean over instances of
    (b - a) / max(a, b) where a = distance to own centroid and b = distance to
    the nearest other centroid (in [-1, 1]; higher = better separated)

The own-sense (intra / a) distances use a LEAVE-ONE-OUT centroid: each
instance is compared to the mean of the *other* instances of its sense, never
to a centroid it helped define. Without this, an instance is trivially close
to its own centroid and even random data would look separated (ratio > 1,
silhouette > 0) when senses have few instances. Distances to other senses'
centroids need no correction (the instance never contributed to them).

All statistics are O(n_instances x n_senses), i.e. linear in the number of
instances, so no full pairwise distance matrix is ever built.

Results are written as one tidy CSV, one row per (word, layer, step).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _unit(x: np.ndarray) -> np.ndarray:
    """Row-normalize to unit L2 norm (safe against zero rows)."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def analyze_word(
    vectors: np.ndarray,  # [n_instances, n_layers, hidden]
    senses: np.ndarray,   # [n_instances] sense labels
) -> "list[dict]":
    """Per-layer, centroid-based separation metrics for one word/checkpoint."""
    _, sense_ids = np.unique(senses, return_inverse=True)
    n_instances = len(sense_ids)
    n_senses = int(sense_ids.max()) + 1
    inst_idx = np.arange(n_instances)
    counts = np.bincount(sense_ids, minlength=n_senses).astype(np.float32)
    n_layers = vectors.shape[1]

    rows = []
    for layer in range(n_layers):
        u = _unit(vectors[:, layer, :].astype(np.float32))  # [n, d] unit vecs

        # Per-sense sum and spherical centroid (mean of unit vectors,
        # re-normalized to a direction so cosine distance is well defined).
        sums = np.zeros((n_senses, u.shape[1]), dtype=np.float32)
        np.add.at(sums, sense_ids, u)
        cu = _unit(sums)                                     # [g, d] unit

        # Distance of every instance to every sense centroid: [n, g]. These
        # centroids include the instance for its own sense, so the own-sense
        # column is replaced below by a leave-one-out distance.
        dist_to = 1.0 - u @ cu.T

        # Leave-one-out own-sense centroid: (sum_of_sense - u_i)/(count-1).
        own_counts = counts[sense_ids][:, None]
        loo = (sums[sense_ids] - u) / np.maximum(own_counts - 1.0, 1.0)
        loo = _unit(loo)
        own = 1.0 - np.einsum("nd,nd->n", u, loo)            # [n] LOO distance

        intra = float(own.mean())
        # Mean distance to the (g-1) other (un-corrected) centroids.
        others_sum = dist_to.sum(axis=1) - dist_to[inst_idx, sense_ids]
        inter = float((others_sum / (n_senses - 1)).mean())

        # Nearest-centroid silhouette: a = LOO own distance, b = nearest other.
        other = dist_to.copy()
        other[inst_idx, sense_ids] = np.inf
        nearest_other = other.min(axis=1)                    # [n] = b
        sil = float(
            ((nearest_other - own)
             / np.maximum(np.maximum(own, nearest_other), 1e-12)).mean()
        )

        # Mean pairwise cosine distance between the g sense centroids.
        cc = 1.0 - cu @ cu.T
        iu, ju = np.triu_indices(n_senses, k=1)
        centroid_dist = float(cc[iu, ju].mean())

        rows.append(
            dict(
                layer=layer,
                intra_dist=intra,
                inter_dist=inter,
                ratio=inter / intra if intra > 0 else float("nan"),
                centroid_dist=centroid_dist,
                silhouette=sil,
            )
        )
    return rows


def analyze_all(
    records: "list[dict]",
    embeddings_dir: Path,
    out_csv: Path,
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

    from tqdm import tqdm

    all_rows = []
    for npz_path in tqdm(
        npz_files, desc="checkpoints", unit="ckpt", dynamic_ncols=True
    ):
        step = int(npz_path.stem.removeprefix("step"))
        vectors = np.load(npz_path)["vectors"]
        # Rows lost to truncation were left as zeros -> drop them.
        valid = np.linalg.norm(
            vectors[:, -1, :].astype(np.float32), axis=1
        ) > 0

        word_groups = meta.groupby(["word", "pos"])
        words_bar = tqdm(
            word_groups, desc=f"  step{step} words", unit="word",
            leave=False, dynamic_ncols=True,
        )
        for (word, pos), group in words_bar:
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
            for row in analyze_word(word_vecs, senses):
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


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over words: mean separation per (step, layer)."""
    agg = (
        df.groupby(["step", "layer"])
        .agg(
            ratio_mean=("ratio", "mean"),
            silhouette_mean=("silhouette", "mean"),
            centroid_dist_mean=("centroid_dist", "mean"),
            n_words=("word", "nunique"),
        )
        .reset_index()
    )
    return agg
