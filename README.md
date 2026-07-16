# WSD probe

How word senses separate across language-model training. For each lemma in the
study list we take every WordNet synset, generate balanced sense-annotated
sentences via a chat API, mean-pool the target token's hidden states at a set
of checkpoints × layers, and measure how those senses organize in embedding
space — via k-NN sense-cluster purity, per-condition PCA, and sense × sense
similarity matrices.

## Pipeline

```bash
./scripts/synsets.sh  --lemmas nouns_top50    # lemmas -> synsets.json
./scripts/generate.sh --lemmas nouns_top50    # stage 1 (CPU): synthesize sentences
./scripts/extract.sh  --model pythia-6.9b     # stage 2 (GPU): pool hidden states
./scripts/analyze.sh  --model pythia-6.9b     # stage 3 (CPU): purity (add --pca / --similarity)
./scripts/submit.sh all -- --model pythia-6.9b --lemmas nouns_top50   # chain as OAR jobs
```

`*.sh` are OAR job scripts; `submit.sh` chains them (`gen-ext` stops before
analysis). Run any stage directly with `uv run python -m src <stage>`. Every
stage is **idempotent**: generation tops up short `(sense, style)` pairs,
extraction skips checkpoints already on disk, the dataset is built once, and
purity is cached per word — nothing already done is recomputed on a re-run.

## Layout

```
configs/
  default.yaml        base run: includes the concern files + project keys + hub
  generation.yaml     stage-1 (API) settings
  extraction.yaml     stage-2 defaults (checkpoints, layers, cache)
  analysis.yaml       dataset, plot, purity settings
  models/*.yaml       per-model fragments (model + layers); olmo-template.yaml
lemmas/               word lists (nouns_top50.txt = default study set)
prompts/              committed generation inputs (generate.txt, styles.json)
src/                  the pipeline package (import + `python -m src <stage>`)
scripts/              thin wrappers: *.sh are OAR jobs, *.py are batch drivers
data/                 runtime data & outputs — git-ignored (see storage_root)
```

`src/` — one module per stage: `synsets` · `generate` (API + HF push) ·
`dataset` · `extract` · `purity` (k-NN purity heatmaps + margin) · `pca`
(per-condition PCA grid) · `similarity` (sense × sense similarity matrix grid)
· `embeddings` (shared loaders) · `hub` (HF push) · `plots` · `config` ·
`__main__`.

## Stage 1: generate

```bash
uv run python -m src generate --lemmas nouns_top50 --model pythia-6.9b
```

One (sense, style) pair is filled to `per_synset / n_styles` valid,
self-checked sentences (the model grades its own output; only sentences it
marks `valid: true` are kept). Requests are batched: each API call asks the
backend for up to `batch_n` independent completions of the same prompt in one
round trip (the OpenAI `n` parameter, served by vLLM's continuous batching) —
far cheaper than one request per sentence. `workers` is the number of
concurrent batch requests, not a throughput knob to crank up: past a small
number the shared backend saturates and starts failing (503s) instead of
going faster, so this pipeline's default (`workers: 4`, `batch_n: 50`) is set
from measured throughput, not guessed. Any sentence a batch comes back
missing (invalid, malformed, or a failed request) leaves its pair short,
picked up again by a later, smaller batch until `max_attempts` is spent.

Before spending any API budget, `generate` prints how many lemmas/senses are
in scope and warns about lemmas with fewer than 2 real WordNet senses across
every part of speech (no sense contrast possible, so not worth generating
for) — check that list before a big run.

## Stage 2: extract

```bash
uv run python -m src extract --model pythia-6.9b --lemmas nouns_top50
```

GPU stage. For each configured checkpoint, loads the revision, mean-pools the
target word's hidden states at each configured layer, and writes one
compressed `.npz` per checkpoint. Row `i` of a checkpoint's vectors aligns
with line `i` of the dataset — weights are never modified.

## Stage 3: analysis

Dataset assembly (`src/dataset.py`) cleans the raw generated sentences into
the extraction dataset: drops anything the model didn't mark valid, drops
sentences where the target word can't be located, and keeps everything else —
no per-sense caps or floors on the dataset itself.

```bash
uv run python -m src purity     bank --model pythia-6.9b   # k-NN sense-cluster purity
uv run python -m src pca        bank --model pythia-6.9b   # per-condition PCA grid
uv run python -m src similarity bank --model pythia-6.9b   # sense x sense similarity grid
uv run python scripts/analyze_all.py --model pythia-6.9b --pca --similarity   # whole corpus
```

**Purity**: for each occurrence, the fraction of its k cosine-nearest
neighbours (raw hidden states) that share its WordNet sense, swept over `k`
(`purity.knn_ks`) and averaged over every occurrence of every sense with
`>= plot.min_per_sense` examples. `chance = 1/K` is the fully-mixed floor.
Being local (not a single-cluster assumption), purity is robust to a sense
splitting into several clusters. Writes `<storage>/purity/`, split into
`data/` (the cached CSV + margin `.npz` — re-running only re-renders figures
from it; pass `-f/--force` to recompute) and one folder per figure
(`heatmap/`, `auc/`, `pk/`, `margin/`), plus a corpus aggregate.

**PCA**: per-condition (layer × checkpoint) 2-D scatter of the same
occurrences, colored by sense — a qualitative look behind the purity number.

**Similarity**: the full pairwise cosine-similarity matrix of a sample of
occurrences, sense blocks concatenated so a sense's block is visibly clean
when separated and bleeds into its neighbours when it overlaps. Values are
mean-centered per (layer, checkpoint) against the corpus-wide mean first, to
remove embedding anisotropy (raw cosine similarity between unrelated tokens
is otherwise far from 0) — the figure is explicitly labeled as such, not raw
cosine similarity.

k-NN purity is O(n²) in however many occurrences a word ends up with — there
is no per-word or per-sense cap, so a highly polysemous word with many
examples per sense will be slower to score than a word with few senses.

## Models

`--model NAME` merges `configs/models/NAME.yaml` (model + layers) over the
base; `--lemmas NAME` swaps the study list; `--config FILE` overrides the
base.

Pythia (160m–12b) revisions are `step{N}` (grounded against the HF refs API).
OLMo's HF-native repos carry no intermediate checkpoints, so
`olmo-template.yaml` documents the original-repo recipe (`step{N}-tokens{T}B`
branches + `trust_remote_code`) — copy and verify branch names before use.
Embeddings are namespaced by model, so runs coexist under `storage_root`.

## HuggingFace

Generation pushes automatically when it finishes; `push` runs the same step
alone:

```bash
HF_TOKEN=hf_... ./scripts/push.sh --repo-id you/wsd-sentences
```

Rebuilds the cleaned corpus from the generated sentences and uploads it as
`dataset.parquet` **and** `dataset.jsonl` (plus a dataset card). Set the
target via `hub.repo_id`, `--repo-id`, or `$HF_REPO_ID`; `--public` for a
public repo. `generate --no-push` skips the automatic upload.

To pull a pushed corpus back down, the dataset card's `configs:` metadata
makes it a standard `datasets`-library load — no custom code needed:

```python
from datasets import load_dataset
ds = load_dataset("you/wsd-sentences")   # needs HF_TOKEN (read scope) if private
```

## Conventions

- **Config is the single source of truth**, split by concern; no hard-coded
  params. `--model`/`--lemmas`/`--config` compose a run.
- **`src/` is importable, `scripts/` is thin.**
- **Outputs never touch git** — everything writes under `storage_root`
  (`$WSD_STORAGE_ROOT` > `config.storage_root` > repo); `data/` is ignored.
- **Reproducible** — generation fills every style equally with a seeded RNG;
  dataset, extraction, and analysis are seeded and deterministic.
- **Idempotent** — every stage caches its outputs; re-runs recompute nothing.

## Setup

```bash
uv sync              # pipeline deps (generate/extract/push)
uv sync --extra plot # + analysis/plotting stack (matplotlib, umap, scipy, ...)
```

`data/api.txt` holds the chat-API key (git-ignored). Point `storage_root` at
a large disk so the generated corpus and float16 embeddings do not fill the
NFS home quota.
