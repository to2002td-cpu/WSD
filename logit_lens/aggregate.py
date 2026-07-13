"""Merge the per-checkpoint pickles from ``extract.py`` into tidy dataframes.

Rolls the subtoken-level rows up to:
- occurrence level: one row per (step, layer, occurrence) -- summed logprob
  across the word's subtokens (teacher-forced joint logprob), whether every
  subtoken was individually top-1, mean rank.
- word level: one row per (step, layer, word_id) -- mean occurrence logprob,
  fraction of occurrences that were all-top1, mean rank, occurrence count.

The word-level table's keys (word_id, layer, step) match the ``sweep_df`` /
``prop`` tables built in ``twosampletest.ipynb``, so the two can be joined
directly to correlate separability with predictability.
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import load_config, store

log = logging.getLogger(__name__)


def load_all_checkpoints(results_dir) -> pd.DataFrame:
    paths = sorted(results_dir.glob("logprobs_step*.pkl"))
    if not paths:
        raise SystemExit(f"No logprobs_step*.pkl found in {results_dir}; run extract first")
    dfs = [pd.read_pickle(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    log.info("Loaded %d checkpoint file(s), %d subtoken rows", len(paths), len(df))
    return df


def occurrence_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (step, layer, occ_row_id): joint logprob + all-subtokens-top1."""
    g = df.groupby(["step", "layer", "occ_row_id", "word_id", "synset"], as_index=False).agg(
        logprob=("logprob", "sum"),
        mean_rank=("rank", "mean"),
        max_rank=("rank", "max"),
        all_top1=("is_top1", "all"),
        n_subtokens_scored=("subtok_pos", "count"),
    )
    return g


def word_level(occ_df: pd.DataFrame, targets_df: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """One row per (step, layer, word_id): mean logprob, top1 rate, mean rank."""
    g = occ_df.groupby(["step", "layer", "word_id"], as_index=False).agg(
        mean_logprob=("logprob", "mean"),
        top1_rate=("all_top1", "mean"),
        mean_rank=("mean_rank", "mean"),
        n_occ=("occ_row_id", "count"),
    )
    if targets_df is not None:
        g = g.merge(targets_df[["word_id", "freq_bin"]], on="word_id", how="left")
    return g.sort_values(["word_id", "step", "layer"]).reset_index(drop=True)


def aggregate(cfg: dict) -> "tuple[pd.DataFrame, pd.DataFrame]":
    results_dir = store(cfg, "results")
    occ_dir = store(cfg, "occurrences")

    subtok_df = load_all_checkpoints(results_dir)
    occ_lvl = occurrence_level(subtok_df)
    word_lvl = word_level(occ_lvl)

    targets_path = occ_dir / "targets_df.csv"
    if targets_path.exists():
        import ast
        targets_df = pd.read_csv(targets_path, converters={
            "usable_senses": ast.literal_eval, "usable_counts": ast.literal_eval})
        word_lvl = word_level(occ_lvl, targets_df)

    occ_out = results_dir / "occurrence_logprobs.pkl"
    word_out = results_dir / "word_summary.csv"
    occ_lvl.to_pickle(occ_out)
    word_lvl.to_csv(word_out, index=False)
    log.info("Wrote %s (%d rows), %s (%d rows)", occ_out, len(occ_lvl), word_out, len(word_lvl))
    return occ_lvl, word_lvl


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    aggregate(load_config())
