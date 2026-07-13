"""CLI for the SemCor logit-lens probe (separability vs. LM predictability).

    python -m logit_lens occurrences               # stage 0 (CPU): build occ_df from SemCor
    python -m logit_lens extract [--force]          # stage 1 (GPU): logprobs per checkpoint x layer
                                 [--steps N [N ...]] [--layers L [L ...]]
    python -m logit_lens aggregate                  # stage 2 (CPU): tidy word x layer x step summary

All three stages are resumable/idempotent: occurrences and per-checkpoint
extractions are skipped if their output already exists (``--force`` to redo).
Config lives in ``logit_lens/config.yaml``; override with ``--config``.
"""

from __future__ import annotations

import argparse
import logging

from .config import load_config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(prog="logit_lens", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config YAML (default logit_lens/config.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("occurrences", help="stage 0 (CPU): build occ_df / targets_df from SemCor")

    e = sub.add_parser("extract", help="stage 1 (GPU): logit-lens logprobs per checkpoint x layer")
    e.add_argument("-f", "--force", action="store_true", help="recompute even if a checkpoint's file exists")
    e.add_argument("--steps", type=int, nargs="+", default=None, help="override config checkpoints")
    e.add_argument("--layers", type=int, nargs="+", default=None, help="override config layers")

    sub.add_parser("aggregate", help="stage 2 (CPU): merge checkpoints into word x layer x step summary")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "occurrences":
        from .config import store
        from .semcor import load_or_build_occurrences
        load_or_build_occurrences(cfg, store(cfg, "occurrences"))
    elif args.cmd == "extract":
        from .extract import extract
        extract(cfg, force=args.force, steps=args.steps, layers=args.layers)
    elif args.cmd == "aggregate":
        from .aggregate import aggregate
        aggregate(cfg)


if __name__ == "__main__":
    main()
