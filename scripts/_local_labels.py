"""Reconstruct per-row sense labels for a locally-extracted run.

The extracted embeddings (``data/<model>/<lemma>.<pos>/step*.npz``) carry no
sense labels; those live in the dataset JSONL under the pipeline's
``storage_root``, which is not present locally. They are, however, fully
determined by the pipeline: ``generate`` collects exactly ``per_synset``
sentences for each WordNet sense of a lemma, and ``build_dataset`` lays the
extracted rows out as equal, contiguous per-sense blocks in sorted-by-sense
order. Rebuilding them here is identical to what re-running the pipeline would
produce, and needs no API calls.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def sorted_senses(word: str, pos: str) -> "list[str]":
    """The lemma's WordNet senses for ``pos``, in ``build_dataset`` grouping
    order (sorted by sense id, lemma held constant)."""
    from nltk.corpus import wordnet as wn

    return sorted(s.name() for s in wn.synsets(word, pos=pos))


def reconstruct_records(word: str, pos: str, n_rows: int) -> "list[dict]":
    """Synthetic per-row records carrying just the reconstructed sense label.

    Row i's sense is ``senses[i // block]`` with ``block = n_rows / n_senses``.
    Only ``word``/``pos``/``sense`` are read by the KDE and MMD builders."""
    senses = sorted_senses(word, pos)
    if n_rows % len(senses):
        raise SystemExit(
            f"{n_rows} rows is not divisible by {len(senses)} senses; the equal-block "
            "reconstruction does not hold — supply the real dataset JSONL instead."
        )
    block = n_rows // len(senses)
    records = [{"word": word, "pos": pos, "sense": s} for s in senses for _ in range(block)]
    log.info("Reconstructed %d rows: %d senses x %d each", n_rows, len(senses), block)
    return records


def local_row_count(cache_dir) -> int:
    """Number of valid extracted rows (the UMAP / MMD population size)."""
    return int(np.load(cache_dir / "valid.npy").sum())
