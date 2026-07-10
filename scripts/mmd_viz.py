"""Run the sense MMD kernel two-sample test on the local ``bank.n`` run.

Validation / proof-of-concept for the statistical sense-separation analysis
before the top-100 run. Uses the deterministically reconstructed sense labels
(see ``scripts/_local_labels.py``) and the raw per-layer embeddings under
``data/Pythia-6.9B/bank.n``, writing:

  * mmd/bank_mmd.csv   -- one row per (step, layer, sense-pair): MMD^2, p, significant
  * mmd/bank_mmd.png   -- layer x step heatmap of the fraction of sense-pairs split

Run:  uv run --extra plot python scripts/mmd_viz.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _local_labels import local_row_count, reconstruct_records   # noqa: E402
from src.config import load_config                               # noqa: E402
from src.mmd import mmd_visualize                                # noqa: E402

EMB_DIR = ROOT / "data" / "Pythia-6.9B" / "bank.n"
OUT_DIR = ROOT / "mmd"
WORD, POS = "bank", "n"
MIN_PER_SENSE = 6


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    m = load_config()["mmd"]
    cache_dir = EMB_DIR / "_npy_cache"
    records = reconstruct_records(WORD, POS, local_row_count(cache_dir))

    mmd_visualize(
        records, EMB_DIR, WORD, POS, OUT_DIR,
        n_perm=m["n_perm"], max_per_sense=m["max_per_sense"], alpha=m["alpha"],
        max_senses=m["max_senses"], standardize=m["standardize"],
        min_per_sense=MIN_PER_SENSE, cache_dir=cache_dir, seed=m["seed"],
    )


if __name__ == "__main__":
    main()
