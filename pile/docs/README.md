# Pile-based sense frequency and geometry

This directory holds the second arm of the study: instead of generated sentences, it
works from occurrences of ambiguous lemmas in The Pile, labelled automatically with a
supervised WSD system. It produces two things: an estimate of how frequent each
WordNet sense is in the pretraining corpus, and a measurement of how the senses of a
lemma separate geometrically across Pythia checkpoints.

It is self-contained and does not import from `src/`. Output files follow the same
`step<N>.npz` convention as the generated-data pipeline, so downstream analyses can
read either source.

## Layout

    frequency/     extract occurrences from The Pile, label them with ConSeC,
                   aggregate into per-sense frequencies
    geometry/      sample a balanced dataset, extract embeddings per checkpoint,
                   measure separation and sense shape
    validation/    stratified sample and terminal reviewer for manual checking
    corpus_check/  compare the public Pile against the corpus Pythia was trained on
    jobs/          OAR job scripts for the cluster
    docs/          methodology, data schemas, annotator guide
    results/       small result tables and figures, no raw data

## What is not here

Occurrences, annotations, the compact annotation table and the embedding archives are
too large to version and are regenerable from this code. They live on team storage
under `wsd_geometry/` and `big_annotation/`. See `docs/DATA.md` for the schema of each
file and how to rebuild it.

## Two environments

The ConSeC dependencies are pinned to 2021 and conflict with modern spaCy, so the
labelling stage runs in its own conda environment. Extraction and geometry use current
libraries. `docs/METHODOLOGY.md` gives the run order.

## Main findings so far

Sense separation emerges between step 512 and step 5000 and then saturates, and it is
built in the middle layers: at layer 16 the excess k-NN purity goes 0.26, 0.40, 0.55,
0.72 across steps 0, 512, 1000, 5000, while layer 1 lags well behind.

Within a lemma, more frequent senses are not better separated; the relation is
negative and grows with training. The shape diagnostic explains why: frequent senses
occupy a broad region at the centre of the lemma's space while rare senses form tight
clusters at the periphery, so a neighbourhood measure penalises the former.

Numbers, caveats and the checks behind them are in `docs/`.
