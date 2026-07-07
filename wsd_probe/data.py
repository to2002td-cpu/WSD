"""Mine ambiguous words and their sense-annotated contexts from SemCor.

SemCor (via nltk) provides sentences whose content words are annotated with
WordNet senses. We look for lemmas that occur with at least two distinct
senses, each sense having enough occurrences to support a statistical
comparison, and export every qualifying instance as one record:

    {word, pos, sense, sentence, target_start, target_end, ...}

`target_start`/`target_end` are character offsets of the target word inside
`sentence`, which lets the extraction step locate the word after subword
tokenization via the tokenizer's offset mapping.

No assumption is made about which words are interesting: candidates are mined
exhaustively and ranked by how well-supported their sense contrast is
(the instance count of their 2nd best-attested sense, i.e. how balanced the
evidence is), then the top `max_words` are kept — all thresholds are CLI
arguments.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

log = logging.getLogger(__name__)

# Word-level POS tags we keep (WordNet POS of the annotated synset).
CONTENT_POS = {"n", "v", "a", "s", "r"}  # noun, verb, adj, satellite adj, adverb


@dataclass
class Instance:
    """One occurrence of a target word with a known WordNet sense."""

    word: str          # lowercased lemma name, e.g. "bank"
    pos: str           # WordNet POS of the sense, e.g. "n"
    sense: str         # synset name, e.g. "bank.n.01"
    sentence: str      # detokenized sentence text
    target_start: int  # char offset of the target surface form in `sentence`
    target_end: int
    surface: str       # surface form as it appears in the sentence
    sent_id: int       # index of the sentence in SemCor


def _ensure_nltk_data() -> None:
    import nltk

    for resource, path in [
        ("semcor", "corpora/semcor"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            log.info("Downloading nltk resource: %s", resource)
            nltk.download(resource, quiet=True)


def _iter_semcor_instances(
    min_sent_tokens: int, max_sent_tokens: int
) -> "list[Instance]":
    """Walk SemCor and yield every single-word, sense-annotated occurrence."""
    from nltk.corpus import semcor
    from nltk.corpus.reader.wordnet import Lemma
    from nltk.tree import Tree

    from tqdm import tqdm

    instances: list[Instance] = []
    sents = semcor.tagged_sents(tag="sem")
    progress = tqdm(sents, desc="Scanning SemCor", unit="sent", dynamic_ncols=True)
    for sent_id, sent in enumerate(progress):
        # Each chunk is either a plain list of tokens (unannotated) or a Tree
        # whose label is a wordnet Lemma (annotated) or a string (named
        # entities etc., which we ignore as targets).
        tokens: list[str] = []
        targets: list[tuple[int, str, Lemma]] = []  # (token_idx, surface, lemma)
        for chunk in sent:
            if isinstance(chunk, Tree):
                leaves = chunk.leaves()
                label = chunk.label()
                # Only single-token targets with a proper Lemma annotation:
                # multi-word expressions ("take_up") would complicate span
                # matching and sense comparison.
                if (
                    isinstance(label, Lemma)
                    and len(leaves) == 1
                    and isinstance(leaves[0], str)
                    and leaves[0].isalpha()
                ):
                    targets.append((len(tokens), leaves[0], label))
                tokens.extend(l for l in leaves if isinstance(l, str))
            else:  # list of raw tokens
                tokens.extend(t for t in chunk if isinstance(t, str))

        if not (min_sent_tokens <= len(tokens) <= max_sent_tokens):
            continue

        sentence = " ".join(tokens)
        # Char offset of token i in the space-joined sentence.
        offsets = []
        pos_cursor = 0
        for t in tokens:
            offsets.append(pos_cursor)
            pos_cursor += len(t) + 1

        for tok_idx, surface, lemma in targets:
            synset = lemma.synset()
            pos = synset.pos()
            if pos not in CONTENT_POS:
                continue
            start = offsets[tok_idx]
            instances.append(
                Instance(
                    word=lemma.name().lower(),
                    pos="a" if pos == "s" else pos,  # merge satellite adjectives
                    sense=synset.name(),
                    sentence=sentence,
                    target_start=start,
                    target_end=start + len(surface),
                    surface=surface,
                    sent_id=sent_id,
                )
            )
    return instances


def build_dataset(
    out_path: Path,
    min_examples_per_sense: int = 10,
    max_examples_per_sense: int = 60,
    max_words: int = 50,
    min_sent_tokens: int = 5,
    max_sent_tokens: int = 80,
    seed: int = 0,
) -> "list[dict]":
    """Mine SemCor and write a JSONL of instances for ambiguous words.

    A word (lemma, pos) qualifies if >= 2 of its senses each have at least
    `min_examples_per_sense` occurrences. All qualifying senses are kept (not
    just two). Words are ranked by the support of their 2nd best-attested
    sense, so the top of the list is where the sense contrast is best
    evidenced, and truncated to `max_words`.
    """
    _ensure_nltk_data()
    log.info("Scanning SemCor (this takes a couple of minutes on first run)...")
    all_instances = _iter_semcor_instances(min_sent_tokens, max_sent_tokens)
    log.info("Collected %d annotated single-word occurrences", len(all_instances))

    by_word: dict[tuple[str, str], dict[str, list[Instance]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for inst in all_instances:
        by_word[(inst.word, inst.pos)][inst.sense].append(inst)

    rng = random.Random(seed)
    candidates = []
    for (word, pos), senses in by_word.items():
        strong = {
            s: insts
            for s, insts in senses.items()
            if len(insts) >= min_examples_per_sense
        }
        if len(strong) < 2:
            continue
        support = sorted((len(v) for v in strong.values()), reverse=True)
        candidates.append((support[1], support[0], word, pos, strong))

    candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))
    kept = candidates[:max_words]
    log.info(
        "%d ambiguous words qualify; keeping top %d", len(candidates), len(kept)
    )

    records: list[dict] = []
    for _, _, word, pos, strong in kept:
        for _, insts in sorted(strong.items()):
            if len(insts) > max_examples_per_sense:
                insts = rng.sample(insts, max_examples_per_sense)
            records.extend(asdict(i) for i in insts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    log.info("Wrote %d instances to %s", len(records), out_path)
    return records


def load_dataset(path: Path) -> "list[dict]":
    with path.open() as f:
        return [json.loads(line) for line in f]
