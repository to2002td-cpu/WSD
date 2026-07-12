# WSD probe

How word senses separate across language-model training. For each lemma in the
study list we take every WordNet synset, generate balanced sense-annotated
sentences (chat API), mean-pool the target token's hidden states at a set of
checkpoints × layers, and measure **sense-cluster purity** — for each occurrence,
how many of its nearest neighbours in embedding space share its sense. Purity
rises off its chance floor over training, showing senses organize.

## Layout

```
config.yaml    base config (pythia-6.9b); single source of truth
configs/       per-model configs (Pythia/OLMo sizes) inheriting config.yaml via `base:`
lemmas/        word lists (nouns_by_frequency.txt, nouns_top50.txt = study set)
prompts/       committed generation inputs (generate.txt, styles.json)
src/           the pipeline package (import + `python -m src <stage>`)
scripts/       thin wrappers: *.sh are OAR jobs, *.py are batch drivers
data/          runtime data & outputs — git-ignored (see storage_root)
```

`src/` — one module per stage: `synsets` · `generate` · `dataset` · `extract` ·
`similarity` (k-NN purity + 3-D surface) · `density` (KDE grid) · `umapcache`
(shared loaders + UMAP cache) · `plots` (mpl style) · `config` · `__main__`.

## Pipeline

```bash
./scripts/synsets.sh                       # lemmas -> synsets.json
./scripts/generate.sh                      # stage 1 (CPU): synthesize sentences via API
./scripts/extract.sh                       # stage 2 (GPU): pool hidden states per checkpoint
./scripts/analyze.sh                       # stage 3 (CPU): k-NN purity 3-D surface, all words
./scripts/submit.sh                        # chain generate -> extract -> analyze as OAR jobs
```

`*.sh` are OAR job scripts; `submit.sh` chains them with OAR dependencies
(`./scripts/submit.sh gen-ext` stops before analysis). Run any stage directly
with `uv run python -m src <stage>`. Every stage is resumable: generation tops up
short `(sense, style)` pairs, extraction skips checkpoints already on disk, the
dataset is built once.

## Models

`config.yaml` is the base run (`EleutherAI/pythia-6.9b`). Each file in `configs/`
inherits it via `base:` and overrides only what changes — model, layers, and
(for OLMo) checkpoint revisions:

```bash
uv run python -m src extract --config configs/pythia-1.4b.yaml
./scripts/extract.sh          --config configs/olmo-7b.yaml
```

Pythia revisions are `step{N}` (default). OLMo branch names embed the token count
(e.g. `step1000-tokens5B`), so its configs list the exact names in
`extract.revisions` (parallel to `extract.steps`) — fill them from
`huggingface_hub.list_repo_refs(<model>)`. Embeddings are namespaced by model, so
runs coexist under `storage_root`.

## Analysis: sense-cluster purity

k-NN neighbourhood purity on the raw per-layer embeddings, swept over neighbour
counts `k` (`similarity.knn_ks`):

```bash
./scripts/analyze.sh                          # whole corpus (batch driver)
uv run python -m src similarity bank --pos n  # a single word
```

Writes to `<storage>/similarity/`: `bank_purity.csv` (purity per step × layer ×
k, with the `chance = 1/K` baseline) and `bank_purity.png` (one 3-D purity
surface over k × layer per checkpoint — read across the grid for the training
trajectory). `./scripts/density.sh bank --pos n` renders the per-sense KDE grid.

## Conventions

- **Config is the single source of truth** — no hard-coded params; add a key and
  read `cfg[...]`. Per-model configs stay minimal via `base:` inheritance.
- **`src/` is importable, `scripts/` is thin** — a script wires config to a `src`
  entry point and handles the OAR/GPU shell; logic lives in `src/`.
- **Outputs never touch git** — everything writes under `storage_root`
  (`$WSD_STORAGE_ROOT` > `config.storage_root` > repo); `data/` is ignored.
- **Reproducible** — generation samples every style equally (`per_synset /
  n_styles` per style) with a seeded per-lemma RNG; dataset, extraction, and
  purity are all seeded and deterministic.
- **Resumable & idempotent** stages — safe to re-submit after a walltime timeout.

## Setup

```bash
uv sync              # pipeline deps (generate/extract)
uv sync --extra plot # + analysis/plotting stack (matplotlib, umap, scipy, ...)
```

`data/api.txt` holds the chat-API key (git-ignored). Point `storage_root` at a
large disk so the generated corpus and float16 embeddings do not fill the NFS
home quota.
