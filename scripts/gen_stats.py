"""Live statistics on the generated corpus: progress toward per_synset targets,
per-lemma / per-(sense, style) balance, and a live generation rate.

    uv run python scripts/gen_stats.py --lemmas top100Lemmas --model pythia-6.9b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve, store   # noqa: E402
from src.synsets import load_or_build_synsets, monosemous, read_lemmas   # noqa: E402

STATE_FILE = ".gen_stats_state.json"


def _senses_by_lemma_pos(cfg: dict) -> "dict[tuple[str, str], int]":
    """(lemma, pos) -> n_senses in scope, from the same synsets cache generate() builds."""
    synsets = load_or_build_synsets(store(cfg, cfg["synsets_cache"]),
                                    resolve(cfg["lemmas_file"]), cfg.get("synsets_pos"))
    return {(g["lemma"], g["pos"]): len(g["senses"]) for g in synsets}


def _read_pair_counts(path: Path) -> "tuple[int, Counter]":
    """(rows, valid sentences per (sense_id, style)) for one lemma's generated file."""
    per_pair: Counter = Counter()
    rows = 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows += 1
            per_pair[(r.get("sense_id"), r.get("style"))] += 1
    return rows, per_pair


def _rate(state_path: Path, rows_total: int) -> "float | None":
    """rows/sec since the last invocation. A small state file (not per-record
    timestamps) so this stays cheap even at hundreds of thousands of rows."""
    now = time.time()
    prev = None
    if state_path.exists():
        try:
            prev = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            prev = None
    state_path.write_text(json.dumps({"t": now, "rows": rows_total}))
    if not prev or now <= prev["t"]:
        return None
    dt = now - prev["t"]
    return (rows_total - prev["rows"]) / dt if dt > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--lemmas", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.model, args.lemmas)
    gen = cfg["generate"]
    out_dir = store(cfg, gen["out_dir"])
    n_styles = len(json.loads(resolve(gen["styles_file"]).read_text()))
    per_synset = gen["per_synset"]
    per_style = per_synset // n_styles

    lemmas = read_lemmas(resolve(cfg["lemmas_file"]))
    senses = _senses_by_lemma_pos(cfg)
    mono = set(monosemous(lemmas))

    rows_total = target_total = 0
    table = []
    for lemma in sorted(lemmas):
        poses = [pos for (l, pos) in senses if l == lemma]
        n_senses = sum(senses[(lemma, pos)] for pos in poses)
        target = n_senses * per_synset
        rows = 0
        lo = hi = None
        for pos in poses:
            path = out_dir / f"{lemma}.{pos}.jsonl"
            if not path.exists():
                continue
            r, per_pair = _read_pair_counts(path)
            rows += r
            if per_pair:
                counts = list(per_pair.values())
                lo = min(counts) if lo is None else min(lo, min(counts))
                hi = max(counts) if hi is None else max(hi, max(counts))
        rows_total += rows
        target_total += target
        pct = (rows / target * 100) if target else 0.0
        table.append((lemma, rows, target, pct, lo, hi, lemma in mono))

    rate = _rate(out_dir / STATE_FILE, rows_total)

    table.sort(key=lambda r: r[3])   # least-complete lemmas first

    print(f"Generated corpus: {out_dir}")
    n_mono = len(mono)
    print(f"Lemmas: {len(lemmas)} in scope ({n_mono} monosemous -- no sense "
          f"contrast possible, still generated for){'!' if n_mono else ''}")
    pct_total = (rows_total / target_total * 100) if target_total else 0.0
    print(f"Rows:  {rows_total:,} / {target_total:,}  ({pct_total:.1f}%)")
    if rate is None:
        print("Rate:  n/a (first check this run -- rerun to see a rate)")
    else:
        line = f"Rate:  {rate:.2f} rows/s (since last check)"
        if rate > 0 and rows_total < target_total:
            eta_h = (target_total - rows_total) / rate / 3600
            line += f", ETA {eta_h:.1f}h"
        print(line)

    print(f"\n{'lemma':<18}{'rows':>10}{'target':>10}{'%':>7}   pair min-max / {per_style}")
    for lemma, rows, target, pct, lo, hi, is_mono in table:
        pair = f"{lo}-{hi}" if lo is not None else "-"
        flag = "  (monosemous)" if is_mono else ""
        print(f"{lemma:<18}{rows:>10,}{target:>10,}{pct:>6.1f}%   {pair}{flag}")


if __name__ == "__main__":
    main()
