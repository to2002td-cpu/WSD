"""CLI for the WSD probing pipeline. Everything is driven by configs/.

    python -m src synsets  [--model M] [--lemmas L]
    python -m src generate [--model M] [--lemmas L]       # stage 1 (CPU): synthesize + push to HF
    python -m src extract   --model M  [--only LEMMA.POS]  # stage 2 (GPU): pool hidden states
    python -m src purity   WORD --model M [--pos n] [-f]   # stage 3 (CPU): k-NN purity heatmap grid
    python -m src pca      WORD --model M [--pos n]        # per-condition PCA grid (PDF + PNG)
    python -m src push     [--repo-id ID] [--public]       # push sentences to a HF dataset repo

--model picks configs/models/<M>.yaml, --lemmas picks lemmas/<L>.txt, --config
overrides the base (configs/default.yaml). Stages are resumable.
"""

from __future__ import annotations

import argparse
import logging

from .config import load_config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="base config (default configs/default.yaml)")
    common.add_argument("--model", default=None, help="model config under configs/models/ (e.g. pythia-6.9b)")
    common.add_argument("--lemmas", default=None, help="lemma list under lemmas/ (e.g. nouns_top50)")

    parser = argparse.ArgumentParser(prog="src", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synsets", parents=[common], help="build synsets.json from the lemma list")
    g = sub.add_parser("generate", parents=[common], help="stage 1 (CPU): synthesize sentences via API")
    g.add_argument("--no-push", action="store_true", help="skip the HF push after generation")

    e = sub.add_parser("extract", parents=[common], help="stage 2 (GPU): pool hidden states per checkpoint")
    e.add_argument("--only", nargs="+", metavar="LEMMA.POS", help="restrict to these generated files")

    for name, helptext in [
        ("pca", "per-word per-condition PCA grid (layer × checkpoint)"),
        ("purity", "per-word k-NN sense-cluster purity heatmap grid (k × layer per checkpoint)"),
    ]:
        sp = sub.add_parser(name, parents=[common], help=helptext)
        sp.add_argument("word", help="target lemma, e.g. 'bank'")
        sp.add_argument("--pos", default=None, help="WordNet POS (n/v/a/r); default = most frequent")
        sp.add_argument("--only", nargs="+", metavar="LEMMA.POS", help="read the run extracted with the same --only")
        if name == "purity":
            sp.add_argument("-f", "--force", action="store_true", help="recompute purity (else re-plot from CSV)")

    ps = sub.add_parser("push", parents=[common], help="push generated sentences to a HF dataset repo")
    ps.add_argument("--repo-id", default=None, help="target dataset repo (else config hub.repo_id)")
    ps.add_argument("--public", action="store_true", help="create a public repo (default private)")

    args = parser.parse_args()
    cfg = load_config(args.config, args.model, args.lemmas)

    if args.cmd == "synsets":
        from .config import resolve, store
        from .synsets import load_or_build_synsets
        load_or_build_synsets(store(cfg, cfg["synsets_cache"]), resolve(cfg["lemmas_file"]),
                              cfg.get("synsets_pos"))
    elif args.cmd == "generate":
        from .generate import generate
        generate(cfg)
        if not args.no_push:
            from .hub import push_dataset
            push_dataset(cfg)
    elif args.cmd == "extract":
        from .dataset import build_dataset
        from .extract import extract
        extract(cfg, build_dataset(cfg, args.only), args.only)
    elif args.cmd == "pca":
        from .pca import pca
        pca(cfg, args.word, args.pos, args.only)
    elif args.cmd == "purity":
        from .purity import purity
        purity(cfg, args.word, args.pos, args.only, args.force)
    elif args.cmd == "push":
        from .hub import push_dataset
        push_dataset(cfg, args.repo_id, private=not args.public)


if __name__ == "__main__":
    main()
