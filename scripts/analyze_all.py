"""Run the sense-cluster purity analysis over the whole corpus.

For every lemma in the configured ``lemmas_file``, computes k-NN purity (once,
cached to ``{word}_purity.csv``) and renders the 3-D surface, plus a corpus
aggregate. Re-running only re-plots from the CSVs; ``--force`` recomputes.

    uv run --extra plot python scripts/analyze_all.py --model pythia-6.9b
    uv run --extra plot python scripts/analyze_all.py --force   # recompute
    uv run --extra plot python scripts/analyze_all.py --kde     # also KDE grids
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve, run_paths, store   # noqa: E402
from src.dataset import build_dataset                           # noqa: E402
from src.purity import purity_corpus                            # noqa: E402
from src.synsets import read_lemmas                             # noqa: E402

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="base config (default configs/default.yaml)")
    ap.add_argument("--model", default=None, help="model config under configs/models/")
    ap.add_argument("--lemmas", default=None, help="lemma list under lemmas/")
    ap.add_argument("--pos", default="n", help="WordNet POS to analyse (default n)")
    ap.add_argument("-f", "--force", action="store_true", help="recompute purity (else re-plot from CSV)")
    ap.add_argument("--kde", action="store_true", help="also emit the per-word KDE density grid (PNG)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.model, args.lemmas)
    m, p = cfg["purity"], cfg["plot"]
    lemmas = read_lemmas(resolve(cfg["lemmas_file"]))
    log.info("Analysing %d lemmas (pos=%s)", len(lemmas), args.pos)

    records = build_dataset(cfg, None)                          # full corpus, loaded once
    _, emb_dir = run_paths(cfg, None)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    out = store(cfg, m["out_dir"])

    purity_corpus(
        records, emb_dir, lemmas, args.pos, out,
        max_per_sense=m["max_per_sense"], max_senses=m["max_senses"],
        knn_ks=m["knn_ks"], min_per_sense=p["min_per_sense"],
        cache_dir=cache_dir, seed=m["seed"], force=args.force,
    )

    if args.kde:                                                # optional per-word KDE grid (needs UMAP)
        from src.density import visualize_kde
        fig_dir = store(cfg, p["fig_dir"])
        for word in lemmas:
            try:
                visualize_kde(records, emb_dir, word, args.pos, fig_dir,
                              cache_dir=cache_dir, min_per_sense=p["min_per_sense"])
            except SystemExit as e:
                log.warning("skip KDE %s: %s", word, e)

    log.info("Analysis done -> %s", out)


if __name__ == "__main__":
    main()
