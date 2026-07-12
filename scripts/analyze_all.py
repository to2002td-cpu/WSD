"""Run the cosine-similarity sense-organization analysis over the whole corpus.

Loads the (full-corpus) dataset once, then for every lemma in the configured
``lemmas_file`` measures, per (checkpoint, layer), how much more similar
same-sense occurrences are than different-sense ones (intra vs inter cosine
similarity, Δ = intra − inter, and the ratio Q). Writes each word's CSVs +
figures and a corpus-level aggregate heatmap. Monosemous words are skipped.

    uv run --extra plot python scripts/analyze_all.py            # similarity for all words
    uv run --extra plot python scripts/analyze_all.py --kde      # also KDE PNG + slider HTML

Reads the same paths the pipeline wrote (``run_paths`` with no ``--only``), so
run it after ``extract`` on the shared embeddings.
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
from src.similarity import similarity_corpus                    # noqa: E402
from src.synsets import read_lemmas                             # noqa: E402

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pos", default="n", help="WordNet POS to analyse (default n)")
    ap.add_argument("--kde", action="store_true", help="also emit KDE PNG + slider HTML per word")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    m, p = cfg["similarity"], cfg["plot"]
    lemmas = read_lemmas(resolve(cfg["lemmas_file"]))
    log.info("Analysing %d lemmas (pos=%s)", len(lemmas), args.pos)

    records = build_dataset(cfg, None)                          # full corpus, loaded once
    _, emb_dir = run_paths(cfg, None)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    sim_out = store(cfg, m["out_dir"])

    # One pass over the checkpoints scores every word (each .npz loaded once).
    similarity_corpus(
        records, emb_dir, lemmas, args.pos, sim_out,
        max_per_sense=m["max_per_sense"], max_senses=m["max_senses"],
        knn_k=m["knn_k"], knn_ks=m["knn_ks"], min_per_sense=p["min_per_sense"],
        cache_dir=cache_dir, seed=m["seed"],
    )

    if args.kde:                                                # optional, per-word (needs UMAP)
        from src.density import visualize_kde
        from src.kdehtml import build_kde_html
        fig_dir = store(cfg, p["fig_dir"])
        for word in lemmas:
            try:
                visualize_kde(records, emb_dir, word, args.pos, fig_dir,
                              cache_dir=cache_dir, min_per_sense=p["min_per_sense"])
                build_kde_html(records, emb_dir, word, args.pos,
                               fig_dir / f"{word}_kde_slider.html",
                               cache_dir=cache_dir, min_per_sense=p["min_per_sense"])
            except SystemExit as e:
                log.warning("skip KDE %s: %s", word, e)

    log.info("Analysis done -> %s", sim_out)


if __name__ == "__main__":
    main()
