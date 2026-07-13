"""SemCor word/sense selection and occurrence-table construction.

Ported from ``pythia_wsd_analysis_clean.ipynb``: parse SemCor's sense-tagged
sentences, pick a deterministic set of polysemous target words (seed 1234), then
re-walk SemCor to collect every occurrence of their usable senses with a
character span. This is the exact procedure that produced the ``occ_df.pkl``
consumed by ``twosampletest.ipynb`` and ``perword_clustering.ipynb`` (25,976
occurrences over 60 words), so re-running it here reconstructs the same target
words without needing that (inaccessible) pickle.
"""

from __future__ import annotations

import ast
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def ensure_nltk_data() -> None:
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


def _load_semcor_sents():
    ensure_nltk_data()
    from nltk.corpus import semcor

    sents = semcor.tagged_sents(tag="sem")
    log.info("SemCor: %d sentences", len(sents))
    return sents


def _lemma_sense_records(sents) -> pd.DataFrame:
    """One row per sense-tagged chunk (word_id = 'lemma::pos', synset, sent_idx)."""
    from nltk.corpus.reader.wordnet import Lemma

    records = []
    for sent_idx, sent in enumerate(sents):
        for chunk in sent:
            if not hasattr(chunk, "label"):
                continue
            label = chunk.label()
            if not isinstance(label, Lemma):
                continue
            synset = label.synset()
            records.append({
                "word_id": f"{label.name()}::{synset.pos()}",
                "synset": synset.name(),
                "pos": synset.pos(),
                "sent_idx": sent_idx,
            })
    df = pd.DataFrame(records)
    log.info("%d sense-tagged occurrences over %d (lemma, pos) ids",
              len(df), df["word_id"].nunique())
    return df


def select_words(cfg: dict, sents=None) -> pd.DataFrame:
    """Deterministically pick target words from SemCor (pythia_wsd_analysis_clean.ipynb's
    algorithm), or load an explicit list from ``word_selection.word_list_file`` if given.

    Returns ``targets_df``: word_id, freq_bin, pos, total_occ, usable_senses
    (list[str]), usable_counts (list[int]).
    """
    ws = cfg["word_selection"]
    sents = sents if sents is not None else _load_semcor_sents()
    df = _lemma_sense_records(sents)

    # total_occ = ALL of a word's sense-tagged occurrences (every sense, not just
    # the ones meeting min_occ below) -- this is what freq-bin membership is based
    # on, matching pythia_wsd_analysis_clean.ipynb exactly.
    lemma_stats = df.groupby("word_id").agg(total_occ=("synset", "size"), pos=("pos", "first"))
    sense_level = (
        df.groupby(["word_id", "synset", "pos"]).size().reset_index(name="occ")
        .sort_values(["word_id", "occ"], ascending=[True, False])
    )
    pool = sense_level[sense_level["occ"] >= ws["min_occ"]]

    word_list_file = ws.get("word_list_file")
    if word_list_file:
        wanted = {
            line.strip() for line in Path(word_list_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        pool = pool[pool["word_id"].isin(wanted)]
        missing = wanted - set(pool["word_id"])
        if missing:
            log.warning("%d word(s) from %s have no usable sense in SemCor: %s",
                        len(missing), word_list_file, sorted(missing))
        target_words = {"custom": sorted(wanted & set(pool["word_id"]))}
    else:
        max_senses = ws.get("max_senses")

        def n_senses_ok(g):
            return ws["min_senses"] <= len(g) and (max_senses is None or len(g) <= max_senses)

        usable_word_ids = pool.groupby("word_id").filter(n_senses_ok)["word_id"].unique()

        exclude_lemmas = set(ws.get("exclude_lemmas") or ())
        exclude_pos = set(ws.get("exclude_pos") or ())
        if exclude_lemmas or exclude_pos:
            # opt-in only: the source notebook applies no such filtering, so this
            # is empty by default and only prunes the candidate pool if configured.
            usable_word_ids = [
                w for w in usable_word_ids
                if w.split("::")[0] not in exclude_lemmas and w.split("::")[1] not in exclude_pos
            ]

        usable_totals = lemma_stats.loc[lemma_stats.index.isin(usable_word_ids), "total_occ"]

        def freq_bin_of(occ: int) -> str:
            for name, lo in ws["freq_bins"]:
                if occ >= lo:
                    return name
            return ws["freq_bins"][-1][0]   # last bin is the catch-all (lo=0)

        bins: dict[str, list[str]] = defaultdict(list)
        for word_id, occ in usable_totals.items():
            bins[freq_bin_of(occ)].append(word_id)

        # One seeded RNG shared across bins, like the source notebook's single
        # `rng = random.Random(SEED)` reused across a `for b in [...]` loop -- the
        # bins MUST be sampled in the config's declared order (not e.g. dict
        # insertion order, which would depend on data and desync the RNG stream
        # from the source notebook's fixed frequent -> mid -> rare sequence).
        rng = random.Random(ws["seed"])
        target_words = {}
        for name, _lo in ws["freq_bins"]:
            words = bins.get(name, [])
            target_words[name] = rng.sample(words, min(ws["words_per_bin"], len(words)))
            log.info("freq bin %-9s: %d candidates -> %d picked", name, len(words), len(target_words[name]))

    records = []
    for freq_bin, words in target_words.items():
        for w in words:
            s = pool[pool["word_id"] == w].sort_values("occ", ascending=False)
            records.append({
                "word_id": w, "freq_bin": freq_bin, "pos": w.split("::")[1],
                "total_occ": int(lemma_stats.loc[w, "total_occ"]),
                "usable_senses": list(s["synset"]),
                "usable_counts": [int(c) for c in s["occ"]],
            })
    targets_df = pd.DataFrame(records)
    log.info("Target words: %d (%s)", len(targets_df),
              dict(Counter(targets_df["freq_bin"])) if len(targets_df) else {})
    return targets_df


def build_occurrences(targets_df: pd.DataFrame, sents=None) -> pd.DataFrame:
    """Re-walk SemCor collecting every occurrence of each target word's usable
    senses, with a character span into the reconstructed sentence text.

    Returns ``occ_df``: word_id, synset, surface, sent_idx, word_idx, n_chunk_words,
    relative_pos, char_start, char_end, sentence.
    """
    from nltk.corpus.reader.wordnet import Lemma

    sents = sents if sents is not None else _load_semcor_sents()
    target_word_ids = set(targets_df["word_id"])
    usable_set = {
        (row.word_id, s) for row in targets_df.itertuples() for s in row.usable_senses
    }

    occurrences = []
    for sent_idx, sent in enumerate(sents):
        tokens: list[str] = []
        target_positions = []
        for chunk in sent:
            if hasattr(chunk, "label"):
                label = chunk.label()
                chunk_leaves = chunk.leaves()
                n_chunk = len(chunk_leaves)
                if isinstance(label, Lemma):
                    sn = label.synset().name()
                    word_id = f"{label.name()}::{label.synset().pos()}"
                    if word_id in target_word_ids and (word_id, sn) in usable_set:
                        target_positions.append({
                            "word_idx": len(tokens), "n_chunk_words": n_chunk,
                            "word_id": word_id, "synset": sn,
                            "surface": " ".join(chunk_leaves),
                        })
                tokens.extend(chunk_leaves)
            else:
                tokens.extend(chunk if isinstance(chunk, list) else [chunk])

        if not target_positions:
            continue
        sentence_text = " ".join(tokens)
        char_offsets, pos = [], 0
        for tok in tokens:
            char_offsets.append(pos)
            pos += len(tok) + 1
        for tp in target_positions:
            wi, nw = tp["word_idx"], tp["n_chunk_words"]
            lti = wi + nw - 1
            occurrences.append({
                "word_id": tp["word_id"], "synset": tp["synset"], "surface": tp["surface"],
                "sent_idx": sent_idx, "word_idx": wi, "n_chunk_words": nw,
                "relative_pos": wi / len(tokens),
                "char_start": char_offsets[wi], "char_end": char_offsets[lti] + len(tokens[lti]),
                "sentence": sentence_text,
            })

    occ_df = pd.DataFrame(occurrences)
    log.info("occ_df: %d occurrences, %d unique sentences, %d word ids",
              len(occ_df), occ_df["sent_idx"].nunique(), occ_df["word_id"].nunique())
    return occ_df


def align_tokenizer(occ_df: pd.DataFrame, tokenizer) -> "tuple[pd.DataFrame, dict]":
    """Tokenize each unique sentence once and add a ``subtok_indices`` column to
    ``occ_df`` (the target's subtoken span, char-offset based; None if lost to
    tokenization quirks). Returns ``(occ_df, sent_input_ids)`` where
    ``sent_input_ids[sent_idx]`` is that sentence's token id list -- kept out of
    occ_df itself so it isn't duplicated once per occurrence."""
    unique_sents = occ_df[["sent_idx", "sentence"]].drop_duplicates("sent_idx")
    offsets_by_sent, sent_input_ids = {}, {}
    for row in unique_sents.itertuples():
        enc = tokenizer(row.sentence, return_offsets_mapping=True, add_special_tokens=False)
        offsets_by_sent[row.sent_idx] = enc["offset_mapping"]
        sent_input_ids[row.sent_idx] = enc["input_ids"]

    subtok_indices, n_ok, n_failed = [], 0, 0
    for row in occ_df.itertuples():
        offsets = offsets_by_sent[row.sent_idx]
        idx = [i for i, (s, e) in enumerate(offsets) if s < row.char_end and e > row.char_start]
        subtok_indices.append(idx or None)
        n_ok += bool(idx)
        n_failed += not idx
    occ_df = occ_df.copy()
    occ_df["subtok_indices"] = subtok_indices
    log.info("Alignment: %d OK, %d failed", n_ok, n_failed)
    return occ_df, sent_input_ids


def load_or_build_occurrences(cfg: dict, out_dir: Path) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """(occ_df, targets_df), cached under ``out_dir`` as targets_df.csv / occ_df.pkl.
    ``occ_df`` has no tokenizer alignment yet -- that's added by the extract stage,
    which knows which tokenizer to use."""
    targets_path, occ_path = out_dir / "targets_df.csv", out_dir / "occ_df.pkl"
    if targets_path.exists() and occ_path.exists():
        log.info("Reusing cached occurrences: %s", occ_path)
        targets_df = pd.read_csv(targets_path, converters={
            "usable_senses": ast.literal_eval, "usable_counts": ast.literal_eval})
        return pd.read_pickle(occ_path), targets_df

    sents = _load_semcor_sents()
    targets_df = select_words(cfg, sents)
    occ_df = build_occurrences(targets_df, sents)

    out_dir.mkdir(parents=True, exist_ok=True)
    targets_df.to_csv(targets_path, index=False)
    occ_df.to_pickle(occ_path)
    log.info("Wrote %s, %s", targets_path, occ_path)
    return occ_df, targets_df
