from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)


def _pos_from_sense(sense: str) -> str:
    """'virus.n.01' -> 'n' (satellite adjectives 's' merged into 'a')."""
    pos = sense.rsplit(".", 2)[1]
    return "a" if pos == "s" else pos


def _locate_target(sentence: str, word: str, pos: str | None = None):
    """Char span (start, end, surface) of the target word: exact whole-word, then
    a stem-sharing inflection ("virus"->"viruses"), then WordNet morphology for
    irregulars ("foot"->"feet"). None if absent."""
    esc = re.escape(word)
    for pattern in (rf"\b{esc}\b", rf"\b{esc}[a-zA-Z]*\b"):
        m = re.search(pattern, sentence, re.IGNORECASE)
        if m:
            return m.start(), m.end(), m.group(0)
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return None
    for m in re.finditer(r"[A-Za-z]+", sentence):
        base = wn.morphy(m.group(0).lower(), pos) if pos else wn.morphy(m.group(0).lower())
        if base == word:
            return m.start(), m.end(), m.group(0)
    return None


def _read_generated(gen_dir: Path, only: "list[str] | None" = None) -> "list[dict]":
    if only:
        paths = [gen_dir / f"{s.removesuffix('.jsonl')}.jsonl" for s in only]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"No such generated file(s) in {gen_dir}: {missing}")
    else:
        paths = sorted(gen_dir.glob("*.jsonl"))
    rows: list[dict] = []
    for path in sorted(paths):
        with path.open() as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def build_dataset(cfg: dict, only: "list[str] | None" = None) -> "list[dict]":
    """Build (or reuse) the extraction dataset from the generated sentences,
    locating each target span. ``only`` restricts to given stems (e.g. ['bank.n'])."""
    ds = cfg["dataset"]
    from .config import run_paths, store

    out_path, _ = run_paths(cfg, only)
    if out_path.exists():
        log.info("Dataset %s exists, reusing", out_path)
        return load_dataset(out_path)

    gen_dir = store(cfg, cfg["generate"]["out_dir"])
    raw = _read_generated(gen_dir, only)
    log.info("Loaded %d generated sentences from %s", len(raw), gen_dir)

    rng = random.Random(ds["seed"])
    by_sense: dict[tuple, list[dict]] = defaultdict(list)
    n_invalid = n_nolocate = 0
    for i, gen in enumerate(raw):
        if ds["require_valid"] and gen.get("valid") is False:
            n_invalid += 1
            continue
        sentence = gen.get("sentence")
        if not sentence:
            continue
        sense = gen["sense_id"]
        word = (gen.get("word") or sense.rsplit(".", 2)[0]).lower()
        pos = _pos_from_sense(sense)
        loc = _locate_target(sentence, word, pos)
        if loc is None:
            n_nolocate += 1
            continue
        start, end, surface = loc
        by_sense[(word, pos, sense)].append({
            "word": word,
            "pos": pos,
            "sense": sense,
            "sentence": sentence,
            "target_start": start,
            "target_end": end,
            "surface": surface,
            "sent_id": i,
            "style": gen.get("style"),
            "model": gen.get("model"),
            "seed": gen.get("seed"),
        })

    records: list[dict] = []
    for _key, recs in sorted(by_sense.items()):
        if len(recs) < ds["min_examples_per_sense"]:
            continue
        if len(recs) > ds["max_examples_per_sense"]:
            recs = rng.sample(recs, ds["max_examples_per_sense"])
        records.extend(recs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    log.info(
        "Kept %d instances across %d senses "
        "(dropped %d invalid, %d word-not-found); wrote %s",
        len(records), len({(r["word"], r["pos"], r["sense"]) for r in records}),
        n_invalid, n_nolocate, out_path,
    )
    return records


def load_dataset(path: Path) -> "list[dict]":
    with path.open() as f:
        return [json.loads(line) for line in f]
