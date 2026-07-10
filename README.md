# WSD probe

Probe sense separation in Pythia checkpoints. List lemmas in `lemmas.txt`; for
each we take every WordNet synset, generate `per_synset` sentences per sense
(chat API), then mean-pool the target token's hidden states across checkpoints.

Config: `config.yaml`. Prompt: `prompts/generate.txt`. Outputs go under
`storage_root` (override with `$WSD_STORAGE_ROOT`).

## Commands

```bash
./scripts/synsets.sh                          # lemmas.txt -> synsets.json
./scripts/generate.sh                         # stage 1 (CPU): synthesize sentences
./scripts/extract.sh   --only bank.n          # stage 2 (GPU): pool hidden states
./scripts/analyze.sh                          # stage 3 (CPU): MMD test per word + aggregate
./scripts/submit.sh                           # submit generate -> extract -> analyze as OAR jobs
```

`--only LEMMA.POS` restricts extraction (and the viz below) to given generated
files, scoping the dataset and embeddings under that tag; omit it for the whole
corpus (the top-100 run). `generate.sh`/`extract.sh`/`analyze.sh` are OAR job
scripts (`oarsub -S ...`); `submit.sh` chains them with OAR dependencies
(`./scripts/submit.sh gen-ext` skips the analysis stage). Run any stage directly
with `uv run python -m src <stage>`.

The corpus of lemmas is `config.yaml: lemmas_file` (currently
`nouns_top100.txt`, the top-100 nouns by frequency). `analyze.sh` runs the MMD
sense-separation test (below) over every lemma and writes a corpus-level
`mmd/corpus_mmd.png`; pass `--kde` to also emit the KDE PNG + slider HTML per
word.

## Visualization

Both read one shared UMAP cache (computed once per word, then reused) and draw
the same per-sense Gaussian-KDE clouds. Pass the same `--only` used for
extraction.

```bash
./scripts/density.sh bank --pos n --only bank.n   # per-sense KDE grid (layer x checkpoint), PNG
./scripts/kde.sh     bank --pos n --only bank.n   # per-layer KDE column with a checkpoint slider, HTML
```

Outputs land in `<storage>/figures/`: `bank_density.png` (static grid) and
`bank_kde_slider.html` (drag the slider to scrub checkpoints; the whole depth
column evolves together, legend toggles senses).

## Statistical test: sense separation (MMD)

A kernel two-sample test (Maximum Mean Discrepancy; Gretton et al. 2012) run
pairwise between a word's senses on the raw per-layer embeddings, giving a
statistical counterpart to the visual clouds.

```bash
./scripts/mmd.sh bank --pos n --only bank.n       # per (step, layer, sense-pair) MMD test
```

Writes to `<storage>/mmd/`: `bank_mmd.csv` (per-pair MMD², permutation p-value,
significance), `bank_mmd_summary.csv`, and `bank_mmd.png` — a layer×step heatmap
whose primary panel is the **mean MMD² effect size** (how strongly senses are
separated; rises over training, strongest in deeper layers). The binary
significance panel saturates (context alone makes senses distinguishable at every
checkpoint), so effect size is the informative axis.
