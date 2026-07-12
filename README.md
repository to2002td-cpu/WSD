# WSD probe

How word senses separate across language-model training. For each lemma in the
study list we take every WordNet synset, generate balanced sense-annotated
sentences (chat API), mean-pool the target token's hidden states at a set of
checkpoints × layers, and measure **sense-cluster purity** — for each occurrence,
how many of its nearest neighbours in embedding space share its sense. Purity
rises off its chance floor over training, showing senses organize.

## Layout

```
configs/
  default.yaml        base run: includes the concern files + project keys + hub
  generation.yaml     stage-1 (API) settings
  extraction.yaml     stage-2 defaults (checkpoints, layers, cache)
  analysis.yaml       dataset, plot, purity settings
  models/*.yaml        per-model fragments (model + layers); olmo-template.yaml
lemmas/               word lists (nouns_top50.txt = default study set)
prompts/              committed generation inputs (generate.txt, styles.json)
src/                  the pipeline package (import + `python -m src <stage>`)
scripts/              thin wrappers: *.sh are OAR jobs, *.py are batch drivers
data/                 runtime data & outputs — git-ignored (see storage_root)
```

`src/` — one module per stage: `synsets` · `generate` · `dataset` · `extract` ·
`purity` (k-NN purity + 3-D surface) · `density` (KDE grid) · `embeddings`
(shared loaders) · `umap_cache` (UMAP for KDE) · `hub` (HF push) · `plots` ·
`config` · `__main__`.

## Pipeline

```bash
./scripts/synsets.sh  --lemmas nouns_top50      # lemmas -> synsets.json
./scripts/generate.sh --lemmas nouns_top50      # stage 1 (CPU): synthesize sentences
./scripts/extract.sh  --model pythia-6.9b       # stage 2 (GPU): pool hidden states
./scripts/analyze.sh  --model pythia-6.9b       # stage 3 (CPU): purity 3-D surfaces
./scripts/submit.sh all -- --model pythia-6.9b --lemmas nouns_top50   # chain as OAR jobs
```

`*.sh` are OAR job scripts; `submit.sh` chains them (`... gen-ext` stops before
analysis). Run any stage directly with `uv run python -m src <stage>`. Stages are
**idempotent**: generation tops up short `(sense, style)` pairs, extraction skips
checkpoints already on disk, the dataset is built once, and purity is cached per
word — nothing already done is recomputed on a re-run.

## Models

`--model NAME` merges `configs/models/NAME.yaml` (model + layers) over the base;
`--lemmas NAME` swaps the study list; `--config FILE` overrides the base.

```bash
uv run python -m src extract --model pythia-1.4b --lemmas nouns_top50
```

Pythia (160m–12b) revisions are `step{N}` (grounded against the HF refs API).
OLMo's HF-native repos carry no intermediate checkpoints, so `olmo-template.yaml`
documents the original-repo recipe (`step{N}-tokens{T}B` branches +
`trust_remote_code`) — copy and verify branch names before use. Embeddings are
namespaced by model, so runs coexist under `storage_root`.

## Analysis: sense-cluster purity

k-NN neighbourhood purity on the raw per-layer embeddings, swept over `k`
(`purity.knn_ks`):

```bash
uv run python -m src purity bank --model pythia-6.9b   # one word
./scripts/analyze.sh --model pythia-6.9b               # whole corpus
```

Writes `<storage>/purity/`: `bank_purity.csv` (purity per step × layer × k, with
the `chance = 1/K` baseline) and `bank_purity.png` (one 3-D surface over k × layer
per checkpoint). The CSV is the cached artifact — re-running **re-renders figures
from it without recomputing**; pass `-f/--force` to recompute. `density.sh bank`
renders the per-sense KDE grid.

## Push sentences to HuggingFace

```bash
HF_TOKEN=hf_... ./scripts/push.sh --repo-id you/wsd-sentences
```

Uploads `dataset.jsonl` (the sense-annotated sentences + a dataset card) to a HF
dataset repo. Set `hub.repo_id` in the config or pass `--repo-id`; `--public` for
a public repo.

## Conventions

- **Config is the single source of truth**, split by concern; no hard-coded
  params. `--model`/`--lemmas`/`--config` compose a run.
- **`src/` is importable, `scripts/` is thin.**
- **Outputs never touch git** — everything writes under `storage_root`
  (`$WSD_STORAGE_ROOT` > `config.storage_root` > repo); `data/` is ignored.
- **Reproducible** — generation samples every style equally with a seeded
  per-lemma RNG; dataset, extraction, and purity are seeded and deterministic.
- **Idempotent** — every stage caches its outputs; re-runs recompute nothing.

## Setup

```bash
uv sync              # pipeline deps (generate/extract/push)
uv sync --extra plot # + analysis/plotting stack (matplotlib, umap, scipy, ...)
```

`data/api.txt` holds the chat-API key (git-ignored). Point `storage_root` at a
large disk so the generated corpus and float16 embeddings do not fill the NFS
home quota.
