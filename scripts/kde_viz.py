"""Build both KDE visualizations for the local Pythia-6.9B ``bank.n`` run.

The extracted embeddings live at ``data/Pythia-6.9B/bank.n`` with their shared
UMAP cache, but the dataset JSONL that carries the per-row sense labels is not
present locally. Those labels are reconstructed deterministically (see
``scripts/_local_labels.py``) and fed to the shared KDE builders, writing:

  * figures/bank_density.png      -- the classic per-sense KDE grid (layer x checkpoint)
  * figures/bank_kde_slider.html  -- a per-layer KDE column with a checkpoint slider

Run:  uv run --extra plot python scripts/kde_viz.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _local_labels import local_row_count, reconstruct_records   # noqa: E402
from src.density import visualize_kde                            # noqa: E402
from src.kdehtml import build_kde_html                           # noqa: E402

EMB_DIR = ROOT / "data" / "Pythia-6.9B" / "bank.n"
FIG_DIR = ROOT / "figures"
WORD, POS = "bank", "n"
MIN_PER_SENSE = 6                              # matches config.yaml plot.min_per_sense


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cache_dir = EMB_DIR / "_npy_cache"
    records = reconstruct_records(WORD, POS, local_row_count(cache_dir))

    visualize_kde(records, EMB_DIR, WORD, POS, FIG_DIR,
                  cache_dir=cache_dir, min_per_sense=MIN_PER_SENSE)
    build_kde_html(records, EMB_DIR, WORD, POS, FIG_DIR / f"{WORD}_kde_slider.html",
                   cache_dir=cache_dir, min_per_sense=MIN_PER_SENSE)


if __name__ == "__main__":
    main()
