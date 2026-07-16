"""Stage 1 (CPU): synthesize sense-annotated sentences via the chat API.

Up to ``batch_n`` sentences per request: the backend (vLLM behind a LiteLLM
proxy) serves the OpenAI ``n`` parameter as one batch of independent stochastic
completions, which is far cheaper than one request per sentence (measured:
~16 rows/s at concurrency=4, batch_n=50 vs. ~3.2 rows/s at 32 sequential
one-sentence requests -- past concurrency=4 the backend itself is the ceiling,
more concurrent requests just fail with 503/RetryError instead of adding
throughput). Every WordNet (sense, style) pair is filled to ``per_synset //
n_styles`` valid, self-checked sentences, so styles are balanced; any sentence a
batch comes back missing (invalid, malformed, or a failed request) leaves the
pair short, topped up by a later smaller batch until ``max_attempts`` is spent.
All pairs across all lemmas share a single work queue: workers stay busy on
whatever still needs sentences instead of draining one lemma at a time. Output is
one resumable JSONL per lemma at ``<out_dir>/<lemma>.<pos>.jsonl``; re-running
tops up short pairs.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .config import resolve, store
from .synsets import load_or_build_synsets, monosemous, read_lemmas

log = logging.getLogger(__name__)

# Sustained rows/sec measured 2026-07-16 at concurrency=4, batch_n=50
# (ilaas/gemma-4-31b) -- a rough throughput estimate for that operating point,
# not a guarantee, and NOT linear in concurrency (see module docstring: this
# backend's ceiling is around concurrency=4, not something a bigger --workers
# buys you more of).
EMPIRICAL_ROWS_PER_SEC = 16.27


def _format_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, _s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m"
    return "<1m"


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
    """Chat-completions client over a connection-pooled session, so concurrent
    workers reuse TCP/TLS connections and retry the proxy's transient 5xx."""

    def __init__(self, cfg: dict):
        self.url = cfg["api_url"]
        self.model = cfg["model"]
        self.max_tokens = cfg["max_tokens"]
        key = resolve(cfg["api_key_file"]).read_text().strip()

        pool = cfg["workers"]
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=(502, 503, 504),
                      allowed_methods=frozenset({"POST"}))
        adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=retry)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json",
                                     "Authorization": f"Bearer {key}"})
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def complete(self, prompt: str, seed: int, n: int = 1) -> "list[str]":
        """Up to ``n`` independent completions of ``prompt`` from a single
        request. Empty on any failure (network error, non-2xx, malformed
        body) -- callers retry the whole batch, since the backend rejects
        ``n`` outright rather than returning a partial batch."""
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "max_tokens": self.max_tokens, "seed": seed, "n": n}
        try:
            resp = self.session.post(self.url, json=payload, timeout=180)
        except requests.RequestException as exc:
            log.debug("request error: %s", exc)
            return []
        if not resp.ok:
            log.debug("HTTP %s: %s", resp.status_code, resp.text[:200])
            return []
        try:
            return [c["message"]["content"].strip() for c in resp.json()["choices"]]
        except (ValueError, KeyError, IndexError, TypeError):
            log.debug("unexpected response: %s", resp.text[:200])
            return []


def _make_records(client, template, lemma, sense, style, seed, n) -> "list[dict]":
    """The valid, self-checked records from one batch of up to ``n`` completions
    (fewer if some choices were invalid, empty, or the whole request failed)."""
    texts = client.complete(build_prompt(template, lemma, sense, style), seed, n)
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for i, text in enumerate(texts):
        if not text:
            continue
        parsed = parse_result(text)
        if not parsed or not parsed["valid"]:
            continue
        records.append({"word": lemma, "sense_id": sense["id"], "sense_definition": sense["definition"],
                        "style": style, "model": client.model, "seed": seed, "choice_index": i,
                        "valid": True, "sentence": parsed["sentence"], "reasoning": parsed["reasoning"],
                        "created_at": now})
    return records


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


class _LemmaFile:
    """Thread-safe appender for one lemma's JSONL output."""

    def __init__(self, path: Path):
        self._fh = path.open("a")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


@dataclass
class _Pair:
    """One (sense, style) pair being filled to ``target`` valid sentences."""
    lemma: str
    sense: dict
    style: str
    target: int
    writer: _LemmaFile
    max_attempts: int
    rng: random.Random = field(repr=False)
    got: int = 0
    attempts: int = 0
    pending: int = 0

    def next_seed(self) -> int:
        return self.rng.randrange(2**31)

    def needs_work(self) -> bool:
        return self.got < self.target and self.attempts < self.max_attempts


def _build_pairs(synsets, styles, per_style, out_dir, seed, max_attempts):
    """One ``_Pair`` per (lemma, sense, style) still short of ``per_style``, plus
    the per-lemma writers to close afterwards."""
    pairs, writers = [], []
    for group in synsets:
        path = out_dir / f"{group['lemma']}.{group['pos']}.jsonl"
        have = _counts(path)
        writer = _LemmaFile(path)
        writers.append(writer)
        for sense in group["senses"]:
            for style in styles:
                target = per_style - have.get((sense["id"], style), 0)
                if target <= 0:
                    continue
                rng = random.Random(f"{seed}-{group['lemma']}-{sense['id']}-{style}")
                pairs.append(_Pair(group["lemma"], sense, style, target, writer,
                                   max_attempts=max_attempts, rng=rng))
    return pairs, writers


def _run_queue(pairs, client, template, workers, batch_n, progress) -> None:
    """Drive all pairs concurrently through a shy sliding window of at most
    ``workers`` in-flight batch requests, round-robin so every pair makes
    progress. Each in-flight request asks for exactly as many completions as
    the pair still needs (capped at ``batch_n``), so at most one batch is ever
    in flight per pair; any request that comes back short (invalid choices, or
    the whole request failing) leaves the pair short, picked up again by
    ``fill()`` with a smaller batch until ``target`` is met or ``max_attempts``
    is spent."""
    ready = deque(pairs)
    inflight: dict = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        def fill() -> None:
            rounds = 0
            while len(inflight) < workers and ready and rounds < len(ready):
                pair = ready[0]
                if not pair.needs_work():
                    ready.popleft(); rounds = 0; continue
                remaining = pair.target - pair.got - pair.pending
                if remaining <= 0:                        # enough already in flight
                    ready.rotate(-1); rounds += 1; continue
                n = min(batch_n, remaining, pair.max_attempts - pair.attempts)
                pair.attempts += n
                pair.pending += n
                fut = pool.submit(_make_records, client, template, pair.lemma,
                                  pair.sense, pair.style, pair.next_seed(), n)
                inflight[fut] = (pair, n)
                ready.rotate(-1); rounds = 0

        fill()
        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                pair, n = inflight.pop(fut)
                pair.pending -= n
                for rec in fut.result():
                    if pair.got >= pair.target:
                        break
                    pair.writer.write(rec)
                    pair.got += 1
                    progress.update(1)
            fill()


def generate(cfg: dict) -> None:
    gen = cfg["generate"]
    lemmas = read_lemmas(resolve(cfg["lemmas_file"]))
    synsets = load_or_build_synsets(store(cfg, cfg["synsets_cache"]),
                                    resolve(cfg["lemmas_file"]), cfg.get("synsets_pos"))
    n_senses = sum(len(g["senses"]) for g in synsets)
    log.info("%d lemmas -> %d (lemma, pos) group(s), %d sense(s) in scope",
             len(lemmas), len(synsets), n_senses)

    mono = monosemous(lemmas)
    if mono:
        log.warning("%d lemma(s) have <2 WordNet senses across ALL POS (no sense "
                    "contrast possible, will be skipped downstream): %s",
                    len(mono), ", ".join(mono))

    styles = json.loads(resolve(gen["styles_file"]).read_text())
    template = resolve(gen["prompt_file"]).read_text()
    client = Client(gen)
    per_style, rem = divmod(gen["per_synset"], len(styles))
    if rem:
        log.warning("per_synset %d not divisible by %d styles; using %d per style (%d total)",
                    gen["per_synset"], len(styles), per_style, per_style * len(styles))
    out_dir = store(cfg, gen["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, writers = _build_pairs(synsets, styles, per_style, out_dir,
                                  gen.get("seed", 0), gen["max_attempts"])
    total = sum(p.target for p in pairs)
    batch_n = gen.get("batch_n", 50)
    eta = total / EMPIRICAL_ROWS_PER_SEC
    log.info("Generating %d sentence(s) over %d (sense, style) pair(s), %d workers x "
             "batch_n=%d -- est. %s", total, len(pairs), gen["workers"], batch_n, _format_duration(eta))
    try:
        with tqdm(total=total, desc="Sentences", unit="sent") as progress:
            _run_queue(pairs, client, template, gen["workers"], batch_n, progress)
    finally:
        for writer in writers:
            writer.close()

    for pair in pairs:
        if pair.got < pair.target:
            log.warning("%s %s/%s: %d/%d after %d attempts", pair.lemma, pair.sense["id"],
                        pair.style, pair.got, pair.target, pair.attempts)
    log.info("Generation complete -> %s", out_dir)
