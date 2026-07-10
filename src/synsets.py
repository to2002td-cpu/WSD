"""Enumerate the WordNet synsets to study, from a plain list of lemmas.

Reads one lemma per line from the lemmas file and, for each lemma, collects
every WordNet synset it belongs to (all content parts of speech) with its
definition, grouped by (lemma, pos):

    {"lemma": "bank", "pos": "n",
     "senses": [{"id": "bank.n.01", "definition": "..."}, ...]}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# WordNet POS we keep (satellite adjectives 's' are merged into 'a').
CONTENT_POS = {"n", "v", "a", "s", "r"}


def _ensure_nltk_data() -> None:
    import nltk

    for resource, path in [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            log.info("Downloading nltk resource: %s", resource)
            nltk.download(resource, quiet=True)


def read_lemmas(path: Path) -> "list[str]":
    """One lemma per line; blank lines and '#' comments ignored."""
    lemmas = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lemmas.append(line.lower())
    return lemmas


def _is_proper(syn, lemma: str) -> bool:
    """A proper-noun sense: the lemma is capitalized in WordNet (e.g. May,
    Turkey)"""
    match = next((l for l in syn.lemmas() if l.name().lower() == lemma), None)
    return match is not None and match.name()[:1].isupper()


def build_synsets(lemmas: "list[str]") -> "list[dict]":
    """Every WordNet synset of each lemma, grouped by (lemma, pos).

    Proper-noun senses are dropped."""
    _ensure_nltk_data()
    from nltk.corpus import wordnet as wn

    groups: dict[tuple[str, str], dict[str, str]] = {}
    for lemma in lemmas:
        for syn in wn.synsets(lemma):
            pos = syn.pos()
            if pos not in CONTENT_POS:
                continue
            if _is_proper(syn, lemma):
                continue
            pos = "a" if pos == "s" else pos
            groups.setdefault((lemma, pos), {})[syn.name()] = syn.definition()

    out = [
        {
            "lemma": lemma,
            "pos": pos,
            "senses": [{"id": i, "definition": d} for i, d in sorted(senses.items())],
        }
        for (lemma, pos), senses in sorted(groups.items())
    ]
    log.info("%d (lemma, pos) groups over %d lemmas", len(out), len(set(lemmas)))
    return out


def load_or_build_synsets(cache: Path, lemmas_file: Path) -> "list[dict]":
    """Synsets from the cache file, built from the lemmas file on first use."""
    if cache.exists():
        with cache.open() as f:
            return json.load(f)
    synsets = build_synsets(read_lemmas(lemmas_file))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        json.dump(synsets, f, indent=2)
    log.info("Wrote %s", cache)
    return synsets
