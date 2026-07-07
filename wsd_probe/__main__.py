"""CLI for the sense-separation probing experiment.

    python -m wsd_probe data     [--min-examples-per-sense N] [--max-words K] ...
    python -m wsd_probe extract  [--model M] [--steps s1,s2,...] ...
    python -m wsd_probe analyze  [--n-permutations N]
    python -m wsd_probe plot
    python -m wsd_probe all      (runs the four stages in order)

Every stage is idempotent: existing outputs are reused (extraction skips
checkpoints already on disk), so the pipeline can be resumed after an
interruption.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import analyze, data, extract, plots

log = logging.getLogger("wsd_probe")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "results" / "dataset.jsonl"
DEFAULT_EMB_DIR = ROOT / "results" / "embeddings"
DEFAULT_ANALYSIS = ROOT / "results" / "analysis.csv"
DEFAULT_SUMMARY = ROOT / "results" / "summary.csv"
DEFAULT_FIG_DIR = ROOT / "results" / "figures"
DEFAULT_MODEL = "EleutherAI/pythia-6.9b"


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--min-examples-per-sense", type=int, default=0)
    p.add_argument("--max-examples-per-sense", type=int, default=9999)
    p.add_argument("--max-words", type=int, default=9999)
    p.add_argument("--min-sent-tokens", type=int, default=0)
    p.add_argument("--max-sent-tokens", type=int, default=9999)
    p.add_argument("--seed", type=int, default=42)


def _add_extract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--steps",
        default=",".join(str(s) for s in extract.DEFAULT_STEPS),
        help="Comma-separated pre-training steps (Pythia revisions)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default=None, help="cuda | mps | cpu (auto)")
    p.add_argument("--emb-dir", type=Path, default=DEFAULT_EMB_DIR)
    p.add_argument("--cache-dir", default=None, help="HF cache directory")


def _add_analyze_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--analysis-csv", type=Path, default=DEFAULT_ANALYSIS)
    p.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)


def _add_plot_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)


def run_data(args) -> "list[dict]":
    if args.dataset.exists():
        log.info("Dataset %s exists, reusing", args.dataset)
        return data.load_dataset(args.dataset)
    return data.build_dataset(
        args.dataset,
        min_examples_per_sense=args.min_examples_per_sense,
        max_examples_per_sense=args.max_examples_per_sense,
        max_words=args.max_words,
        min_sent_tokens=args.min_sent_tokens,
        max_sent_tokens=args.max_sent_tokens,
        seed=args.seed,
    )


def run_extract(args, records) -> Path:
    steps = [int(s) for s in str(args.steps).split(",")]
    extract.extract_all(
        args.model,
        steps,
        records,
        args.emb_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=extract.pick_device(args.device),
        cache_dir=args.cache_dir,
    )
    return args.emb_dir / args.model.split("/")[-1]


def run_analyze(args, records, emb_dir: Path):
    df = analyze.analyze_all(
        records, emb_dir, args.analysis_csv,
        n_permutations=args.n_permutations, seed=args.seed,
    )
    summary = analyze.summarize(df)
    summary.to_csv(args.summary_csv, index=False)
    log.info("Wrote %s", args.summary_csv)
    return df, summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(prog="wsd_probe", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("data", help="mine ambiguous words from SemCor")
    _add_data_args(p)

    p = sub.add_parser("extract", help="extract hidden states per checkpoint")
    _add_data_args(p)
    _add_extract_args(p)

    p = sub.add_parser("analyze", help="compute separation metrics + stats")
    _add_data_args(p)
    _add_extract_args(p)
    _add_analyze_args(p)

    p = sub.add_parser("plot", help="produce figures")
    _add_data_args(p)
    _add_extract_args(p)
    _add_analyze_args(p)
    _add_plot_args(p)

    p = sub.add_parser("all", help="data -> extract -> analyze -> plot")
    _add_data_args(p)
    _add_extract_args(p)
    _add_analyze_args(p)
    _add_plot_args(p)

    args = parser.parse_args()

    records = run_data(args)
    if args.cmd == "data":
        return

    if args.cmd in ("extract", "all"):
        emb_dir = run_extract(args, records)
    else:
        emb_dir = args.emb_dir / args.model.split("/")[-1]

    if args.cmd == "extract":
        return

    if args.cmd in ("analyze", "all") or not args.analysis_csv.exists():
        df, summary = run_analyze(args, records, emb_dir)
    else:
        import pandas as pd

        df = pd.read_csv(args.analysis_csv)
        summary = pd.read_csv(args.summary_csv)

    if args.cmd in ("plot", "all"):
        plots.make_all_plots(df, summary, args.fig_dir)


if __name__ == "__main__":
    main()
