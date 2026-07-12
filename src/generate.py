"""Stage 1 (CPU): synthesize sense-annotated sentences via the chat API.

For every WordNet sense we request one sentence at a time, keeping only valid
self-checked ones, until each (sense, style) pair reaches ``per_synset //
n_styles`` — so every style is sampled equally. Output is one resumable JSONL per
lemma at ``<out_dir>/<lemma>.<pos>.jsonl``; re-running tops up short pairs.
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
    return (template.replace("{word}", word)
            .replace("{definition}", sense["definition"]).replace("{style}", style))


def _iter_json_objects(text: str):
    """Yield (start, end, obj) for every top-level {...} that parses as JSON
    (brace/string-aware, so JSON embedded in the model's prose is recovered)."""
    depth = 0
    start = None
    in_str = esc = False
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
    """Take the last JSON object carrying a non-empty "sentence" as the answer."""
    for start, _end, obj in reversed(list(_iter_json_objects(text))):
        sentence = obj.get("sentence")
        if isinstance(sentence, str) and sentence.strip():
            valid = obj.get("valid")
            return {"sentence": sentence.strip(),
                    "valid": bool(valid) if valid is not None else None,
                    "reasoning": text[:start].strip()}
    return None


class Client:
    def __init__(self, cfg: dict):
        self.url = cfg["api_url"]
        self.model = cfg["model"]
        self.max_tokens = cfg["max_tokens"]
        key = resolve(cfg["api_key_file"]).read_text().strip()
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    def complete(self, prompt: str, seed: int) -> "str | None":
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "max_tokens": self.max_tokens, "seed": seed}
        try:
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        except requests.RequestException as exc:
            log.debug("request error: %s", exc)
            return None
        if not resp.ok:
            log.debug("HTTP %s: %s", resp.status_code, resp.text[:200])
            return None
        try:
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError, TypeError):
            log.debug("unexpected response: %s", resp.text[:200])
            return None


def _make_record(client, template, lemma, sense, style, seed) -> "dict | None":
    text = client.complete(build_prompt(template, lemma, sense, style), seed)
    if not text:
        return None
    parsed = parse_result(text)
    if not parsed or not parsed["valid"]:
        return None
    return {"word": lemma, "sense_id": sense["id"], "sense_definition": sense["definition"],
            "style": style, "model": client.model, "seed": seed, "valid": True,
            "sentence": parsed["sentence"], "reasoning": parsed["reasoning"],
            "created_at": datetime.now(timezone.utc).isoformat()}


def _counts(path: Path) -> "dict[tuple[str, str], int]":
    """Valid sentences already present per (sense_id, style)."""
    counts: dict[tuple[str, str], int] = {}
    if path.exists():
        for line in path.open():
            if line.strip():
                r = json.loads(line)
                key = (r["sense_id"], r.get("style"))
                counts[key] = counts.get(key, 0) + 1
    return counts


def _fill_lemma(client, template, lemma, senses, styles, per_style, cap, pool, out, rng, have):
    """Fill every (sense, style) pair of one lemma to ``per_style`` valid
    sentences, requesting in rounds until the target or the per-pair attempt cap
    ``cap`` is reached. Styles are filled equally by construction."""
    by_id = {s["id"]: s for s in senses}
    pairs = [(s["id"], style) for s in senses for style in styles]
    got = {p: have.get(p, 0) for p in pairs}
    attempts = {p: 0 for p in pairs}
    while True:
        tasks = []
        for p in pairs:
            room = min(per_style - got[p], cap - attempts[p])
            if room > 0:
                tasks.extend([p] * room)
                attempts[p] += room
        if not tasks:
            break
        rng.shuffle(tasks)
        jobs = [(by_id[sid], style, rng.randrange(2**31)) for sid, style in tasks]
        for (sid, style), rec in zip(tasks, pool.map(
                lambda j: _make_record(client, template, lemma, j[0], j[1], j[2]), jobs)):
            if rec is not None:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                got[(sid, style)] += 1
        out.flush()
    return got


def generate(cfg: dict) -> None:
    gen = cfg["generate"]
    synsets = load_or_build_synsets(store(cfg, cfg["synsets_cache"]),
                                    resolve(cfg["lemmas_file"]), cfg.get("synsets_pos"))
    styles = json.loads(resolve(gen["styles_file"]).read_text())
    template = resolve(gen["prompt_file"]).read_text()
    client = Client(gen)
    per_style, rem = divmod(gen["per_synset"], len(styles))
    if rem:
        log.warning("per_synset %d not divisible by %d styles; using %d per style (%d total)",
                    gen["per_synset"], len(styles), per_style, per_style * len(styles))
    out_dir = store(cfg, gen["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=gen["workers"]) as pool:
        for group in tqdm(synsets, desc="Lemmas", unit="lemma"):
            path = out_dir / f"{group['lemma']}.{group['pos']}.jsonl"
            have = _counts(path)
            rng = random.Random(f"{gen.get('seed', 0)}-{group['lemma']}")   # deterministic per lemma
            with path.open("a") as out:
                got = _fill_lemma(client, template, group["lemma"], group["senses"], styles,
                                  per_style, gen["max_attempts"], pool, out, rng, have)
            for (sid, style), n in got.items():
                if n < per_style:
                    log.warning("%s %s/%s: %d/%d after cap", group["lemma"], sid, style, n, per_style)
    log.info("Generation complete -> %s", out_dir)
