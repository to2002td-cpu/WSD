# WSD probe

Probing how word senses separate across Pythia training checkpoints. For each
lemma in the study list we take every WordNet synset, generate `per_synset`
sentences per sense (chat API), mean-pool the target token's hidden states at a
set of checkpoints × layers, and measure **sense-cluster purity** — for each
occurrence, how many of its nearest neighbours in embedding space share its
sense. Purity rises off its chance floor over training, showing senses organize.

## Layout

```
config.yaml          single source of truth for every parameter
lemmas/              word lists (nouns_by_frequency.txt, nouns_top50.txt = study set)
prompts/             committed generation inputs (generate.txt, styles.json)
src/                 the pipeline package (import + `python -m src <stage>`)
scripts/             thin wrappers: *.sh are OAR jobs, *.py are batch drivers
data/                runtime data & outputs — git-ignored (see storage_root)
```

`src/` — one module per stage: `synsets` · `generate` · `dataset` · `extract` ·
`similarity` (k-NN purity + 3-D surface) · `density`/`kdehtml` (KDE viz) ·
`umapcache` (shared embedding loaders + UMAP cache) · `plots` (shared mpl style)
· `config` (paths) · `__main__` (CLI).

## Pipeline

```bash
./scripts/synsets.sh                       # lemmas -> synsets.json
./scripts/generate.sh                      # stage 1 (CPU): synthesize sentences via API
./scripts/extract.sh                       # stage 2 (GPU): pool hidden states per checkpoint
./scripts/analyze.sh                       # stage 3 (CPU): k-NN purity 3-D surface, all words
./scripts/submit.sh                        # chain generate -> extract -> analyze as OAR jobs
```

`*.sh` are OAR job scripts (`oarsub -S ...`); `submit.sh` chains them with OAR
dependencies (`./scripts/submit.sh gen-ext` stops before analysis). Run any stage
directly with `uv run python -m src <stage>`. Every stage is resumable:
generation tops up short senses, extraction skips checkpoints already on disk,
the dataset is built once.

The study set is `config.yaml: lemmas_file` (top-50 nouns); checkpoints and
layers are `extract.steps` / `extract.layers`. `analyze.sh` runs purity over
every lemma and writes `<storage>/similarity/corpus_purity_surface.png`; pass
`--kde` to also emit the KDE PNG + slider HTML per word.

## Analysis: sense-cluster purity

k-NN neighbourhood purity on the raw per-layer embeddings, swept over neighbour
counts `k` (`similarity.knn_ks`). Per word:

```bash
./scripts/analyze.sh                          # whole corpus (batch driver)
uv run python -m src similarity bank --pos n  # a single word
```

Writes to `<storage>/similarity/`: `bank_purity.csv` (purity per step × layer ×
k, with the `chance = 1/K` baseline), `bank_purity_surface.png` (one 3-D purity
surface over k × layer per checkpoint — read across the grid for the training
trajectory), and `bank_purity_surface.html` (the same surface, interactive, with
a checkpoint slider).

## Visualization: KDE clouds

Per-sense Gaussian-KDE density in the shared 2-D UMAP space (one cache per word,
reused):

```bash
./scripts/density.sh bank --pos n   # per-sense KDE grid (layer × checkpoint), PNG
./scripts/kde.sh     bank --pos n   # per-layer KDE column with a checkpoint slider, HTML
```

## Conventions

- **Config is the single source of truth** — no hard-coded params in code; add a
  key to `config.yaml` and read it via `cfg[...]`.
- **`src/` is importable, `scripts/` is thin** — a script wires config to a
  `src` entry point and handles the OAR/GPU shell; logic lives in `src/`.
- **Outputs never touch git** — everything writes under `storage_root`
  (`$WSD_STORAGE_ROOT` > `config.storage_root` > repo), and `data/` is ignored.
- **Resumable & idempotent stages** — safe to re-submit after a walltime timeout.
- **Style**: `from __future__ import annotations`, type hints, module docstrings,
  `snake_case`, `_private` helpers, `logging` over `print`; figures go through
  `plots._style()`.

## Setup

```bash
uv sync              # pipeline deps (generate/extract)
uv sync --extra plot # + plotting/analysis stack (matplotlib, umap, plotly, ...)
```

`data/api.txt` holds the chat-API key (git-ignored). Point `storage_root` at a
large disk (e.g. Grid'5000 group storage) so the generated corpus and float16
embeddings do not fill the NFS home quota.
