"""CLI for the WSD probing pipeline. Everything is driven by config.yaml.

    python -m src synsets             # lemmas.txt -> synsets.json
    python -m src generate            # stage 1 (CPU): synthesize sentences
    python -m src extract             # stage 2 (GPU): pool hidden states
    python -m src density LEMMA [--pos n]  # per-word per-sense KDE grid (PNG)
    python -m src kde     LEMMA [--pos n]  # interactive per-layer KDE slider (HTML)

Use --config to point at a different YAML. Every stage is resumable: existing
outputs are reused (generation tops up short senses, extraction skips
checkpoints already on disk, the dataset is built once).
"""

from __future__ import annotations

import argparse
import logging

from .config import load_config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(prog="src", description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("synsets", help="build synsets.json from lemmas.txt")
    sub.add_parser("generate", help="stage 1 (CPU): synthesize sentences via API")

    e = sub.add_parser("extract", help="stage 2 (GPU): pool hidden states per checkpoint")
    e.add_argument("--only", nargs="+", metavar="LEMMA.POS",
                   help="restrict to these generated files (e.g. bank.n); default = all")

    for name, helptext in [
        ("density", "per-word UMAP scatter + per-sense KDE grid (PNG)"),
        ("kde", "per-word interactive per-layer KDE slider over checkpoints (HTML)"),
        ("mmd", "per-word pairwise sense MMD kernel two-sample test + heatmap"),
    ]:
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("word", help="target lemma, e.g. 'bank'")
        sp.add_argument("--pos", default=None, help="WordNet POS (n/v/a/r); default = most frequent")
        sp.add_argument("--only", nargs="+", metavar="LEMMA.POS",
                        help="read the run extracted with the same --only (e.g. bank.n)")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "synsets":
        from .config import resolve, store
        from .synsets import load_or_build_synsets
        load_or_build_synsets(store(cfg, cfg["synsets_cache"]), resolve(cfg["lemmas_file"]),
                              cfg.get("synsets_pos"))
    elif args.cmd == "generate":
        from .generate import generate
        generate(cfg)
    elif args.cmd == "extract":
        from .dataset import build_dataset
        from .extract import extract
        extract(cfg, build_dataset(cfg, args.only), args.only)
    elif args.cmd == "density":
        from .density import density
        density(cfg, args.word, args.pos, args.only)
    elif args.cmd == "kde":
        from .kdehtml import kde_slider
        kde_slider(cfg, args.word, args.pos, args.only)
    elif args.cmd == "mmd":
        from .mmd import mmd
        mmd(cfg, args.word, args.pos, args.only)


if __name__ == "__main__":
    main()
