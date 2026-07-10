"""Stage 1 (CPU): generate synthetic sense-annotated sentences via the chat API.

For every synset sense we ask the model, one request at a time, for a single
sentence using the target word in exactly that sense (the prompt makes it
reason, draft, then self-check). We keep going until `per_synset` valid
sentences are collected, giving up on a sense after `max_attempts` requests.

Output is one resumable JSONL per lemma at `<out_dir>/<lemma>.<pos>.jsonl`;
re-running tops up whichever senses are short of their target.
"""

from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

from .config import resolve, store
from .synsets import load_or_build_synsets

log = logging.getLogger(__name__)


def build_prompt(template: str, word: str, sense: dict, style: str) -> str:
    return (
        template
        .replace("{word}", word)
        .replace("{definition}", sense["definition"])
        .replace("{style}", style)
    )


def _iter_json_objects(text: str):
    """Yield (start, end, obj) for every top-level {...} block that parses as JSON.

    A brace-aware, string/escape-aware scan, so JSON embedded in the model's
    reasoning is recovered even when braces appear in the surrounding prose.
    """
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    yield start, i + 1, json.loads(text[start:i + 1])
                except ValueError:
                    pass
                start = None


def parse_result(text: str) -> "dict | None":
    """Take the LAST JSON object carrying a non-empty "sentence" as the answer;
    everything before it is the reasoning trace."""
    for start, _end, obj in reversed(list(_iter_json_objects(text))):
        sentence = obj.get("sentence")
        if isinstance(sentence, str) and sentence.strip():
            valid = obj.get("valid")
            return {
                "sentence": sentence.strip(),
                "valid": bool(valid) if valid is not None else None,
                "reasoning": text[:start].strip(),
            }
    return None


# --------------------------------------------------------------------------- #
# API client                                                                  #
# --------------------------------------------------------------------------- #

class Client:
    """Thin wrapper over the chat-completions endpoint."""

    def __init__(self, cfg: dict):
        self.url = cfg["api_url"]
        self.model = cfg["model"]
        self.max_tokens = cfg["max_tokens"]
        key = resolve(cfg["api_key_file"]).read_text().strip()
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

    def complete(self, prompt: str, seed: int) -> "str | None":
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": self.max_tokens,
            "seed": seed,  # per-request seed -> diversity + reproducibility
        }
        try:
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        except requests.RequestException as exc:
            log.debug("request error: %s", exc)
            return None
        if not resp.ok:
            log.debug("HTTP %s: %s", resp.status_code, resp.text[:200])
            return None
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError, TypeError):
            log.debug("unexpected response: %s", resp.text[:200])
            return None


def _make_record(client, template, lemma, sense, styles) -> "dict | None":
    """One API call -> a valid record, or None (error / invalid / unparseable)."""
    style = random.choice(styles)
    seed = random.randint(0, 2**31 - 1)
    text = client.complete(build_prompt(template, lemma, sense, style), seed)
    if not text:
        return None
    parsed = parse_result(text)
    if not parsed or not parsed["valid"]:
        return None
    return {
        "word": lemma,
        "sense_id": sense["id"],
        "sense_definition": sense["definition"],
        "style": style,
        "model": client.model,
        "seed": seed,
        "valid": True,
        "sentence": parsed["sentence"],
        "reasoning": parsed["reasoning"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Per-lemma generation with resumable top-up                                  #
# --------------------------------------------------------------------------- #

def _valid_counts(path: Path) -> "dict[str, int]":
    """Valid sentences already present per sense id (empty if no file yet)."""
    counts: dict[str, int] = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                if line.strip():
                    sid = json.loads(line)["sense_id"]
                    counts[sid] = counts.get(sid, 0) + 1
    return counts


def _fill_lemma(client, template, lemma, senses, need, cap, styles, pool, out) -> "dict[str, int]":
    """Fill every under-target sense of one lemma, requesting in rounds.

    All of a lemma's senses are topped up together: each round submits, across
    every sense still short of its target, one task per outstanding sentence and
    runs them through the shared worker pool at once -- so the pool stays
    saturated instead of draining between senses. A round retries whatever
    failed; a sense stops once it hits `need` valid records or `cap` attempts.
    Returns valid records written per sense id.
    """
    by_id = {s["id"]: s for s in senses}
    got = {sid: 0 for sid in need}
    attempts = {sid: 0 for sid in need}
    while True:
        tasks = []  # sense dicts to request this round, mixed across senses
        for sid in need:
            room = min(need[sid] - got[sid], cap - attempts[sid])
            if room > 0:
                tasks.extend([by_id[sid]] * room)
                attempts[sid] += room
        if not tasks:
            break
        random.shuffle(tasks)  # keep every worker busy on a mix of senses
        for rec in pool.map(lambda s: _make_record(client, template, lemma, s, styles), tasks):
            if rec is not None:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                got[rec["sense_id"]] += 1
        out.flush()
    return got


def generate(cfg: dict) -> None:
    gen = cfg["generate"]
    synsets = load_or_build_synsets(
        store(cfg, cfg["synsets_cache"]), resolve(cfg["lemmas_file"]), cfg.get("synsets_pos")
    )

    styles = json.loads(resolve(gen["styles_file"]).read_text())
    template = resolve(gen["prompt_file"]).read_text()
    client = Client(gen)
    per_synset = gen["per_synset"]
    max_attempts = gen["max_attempts"]
    out_dir = store(cfg, gen["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=gen["workers"]) as pool:
        for group in tqdm(synsets, desc="Lemmas", unit="lemma"):
            path = out_dir / f"{group['lemma']}.{group['pos']}.jsonl"
            have = _valid_counts(path)
            need = {s["id"]: per_synset - have.get(s["id"], 0)
                    for s in group["senses"] if per_synset - have.get(s["id"], 0) > 0}
            if not need:
                continue
            with path.open("a") as out:
                got = _fill_lemma(
                    client, template, group["lemma"], group["senses"], need,
                    max_attempts, styles, pool, out,
                )
            for sid, n in need.items():
                if got[sid] < n:
                    log.warning("%s %s: %d/%d after %d-attempt cap",
                                group["lemma"], sid, have.get(sid, 0) + got[sid],
                                per_synset, max_attempts)
    log.info("Generation complete -> %s", out_dir)
