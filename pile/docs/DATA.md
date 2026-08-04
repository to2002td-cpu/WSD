# Data files: schema and interpretation

Every stage writes a plain jsonl or csv file. This document describes each one and
how to read it.

## occurrences_full.jsonl

One line per extracted occurrence of a target lemma in The Pile. Fields:

- occ_id            unique id, assigned in stream order (occ0, occ1, ...).
- lemma             the spaCy lemma, lowercased.
- pos               coarse WordNet part of speech: n, v, a, r.
- tokens            the tokens of the sentence, as a list.
- target_index      index of the target token within tokens.
- surface           the surface form of the target as it appears in the text.
- is_aux_or_modal   true if spaCy tagged the target as an auxiliary or modal; these
                    are kept here but dropped in aggregation.

The occ_id is assigned per run, so ids are not comparable across different extraction
runs. Do not reuse an annotations file from one run against occurrences from another.

## annotations_full.jsonl

One line per annotated occurrence. Occurrences whose sentence exceeds 120 tokens are
not present. Fields:

- occ_id            matches occurrences_full.jsonl.
- pred_sense_key    WordNet sense key of the chosen sense.
- pred_synset       WordNet synset name of the chosen sense (e.g. cell.n.02).
- top_prob          probability ConSeC assigned to the chosen sense; the per-occurrence
                    confidence, used for filtering.
- probs             the full probability vector over all candidate senses.

Monosemous lemma-pos pairs (a single WordNet synset) are written directly with
top_prob 1.0 and do not pass through the model.

## sense_frequency_full.csv

The aggregated per-sense frequency, produced by collect.py. One row per
(lemma, pos, synset, sense_key). Columns:

- n         number of occurrences assigned to this sense (argmax).
- n_conf    number of those occurrences whose top_prob is at least tau (default 0.8).
- share     n divided by the total occurrences of the lemma-pos; the fraction of the
            lemma that this sense accounts for.

A row appears only if n_conf is at least the floor (default 30). Read n_conf, not n,
as the reliable count: the gap between n and n_conf tells you how confidently the
sense was distinguished. share is the quantity of interest for the frequency axis,
since it is the within-lemma sense distribution.

collect.py also writes sense_frequency_full_kept_lemmas.csv, the lemma-pos pairs with
at least one synset above the floor.

## annotations_table.csv

A compact one-row-per-annotation table, produced once by build_table.py, joining the
two large files and dropping the token lists. Columns: occ_id, lemma, pos, synset,
sense_key, top_prob, is_aux_or_modal. This is the file to load for analysis: it is
small enough to hold in memory, and analysis.py re-aggregates it under any choice of
tau, floor and drop lists without touching the large jsonl files.

Typical use from a notebook:

    from analysis import load_table, sense_frequency, coverage
    df = load_table("annotations_table.csv")
    table = sense_frequency(df, tau=0.8, floor=30, drop_lemmas=["class"])
    coverage(table)

coverage returns the number of synsets and lemma-pos pairs above the floor, and how
many synsets fall in the rare tail (below 5 and 2 percent of their lemma).

## Review files

- review_sample.csv       the stratified sample to annotate: occ_id, lemma, pos,
                          predicted synset and gloss, top_prob, the sentence, and an
                          empty verdict column.
- review_sample_key.csv   maps each occ_id to its stratum (rare or frequent); kept
                          separate so the annotator judges blind.
- verdicts_<name>.jsonl   one file per annotator, occ_id and verdict (y, n, o).
                          y = the gloss matches; n = another WordNet sense fits better;
                          o = no WordNet sense fits (out of inventory).
