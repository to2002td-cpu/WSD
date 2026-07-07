# When do language models learn word senses?

Empirical probe of **sense separation** in the internal representations of
[Pythia](https://github.com/EleutherAI/pythia) across its released
pre-training checkpoints, using the sense annotations of **SemCor** (WordNet,
via nltk). No training, no weight modification — inference only.

**Question.** At which pre-training step, and in which layers, does a model
start representing the same word differently when it appears in contexts
corresponding to different senses?

## Method

1. **Data** (`wsd_probe/data.py`) — SemCor is scanned exhaustively for lemmas
   occurring with ≥ 2 WordNet senses, each attested by at least
   `--min-examples-per-sense` sentences. No a-priori word list: candidates are
   ranked by the support of their 2nd best-attested sense and truncated to
   `--max-words`. All qualifying senses of a word are kept (not just two).
   Output: `results/dataset.jsonl`, one record per occurrence with character
   offsets of the target word.

2. **Extraction** (`wsd_probe/extract.py`) — for each checkpoint
   (`revision=stepN`), sentences are run through the model with
   `output_hidden_states=True`; the representation of the target word is the
   **mean over its sub-tokens** (offset mapping handles multi-sub-token
   words), kept at **every layer** including the embedding layer (layer 0 =
   static-embedding baseline). Output: one `stepN.npz` per checkpoint with a
   `[n_instances, n_layers+1, hidden]` float16 array, row-aligned with the
   JSONL. Extraction is resumable (existing files are skipped).

3. **Analysis** (`wsd_probe/analyze.py`) — for every (word, layer,
   checkpoint): mean pairwise cosine distance within senses (*intra*) vs
   across senses (*inter*), their **ratio** (internally normalized, hence
   robust to representation anisotropy drifting over training), mean
   **centroid distance**, **silhouette score**, and a **permutation test**
   (sense labels shuffled, default 1000 permutations) giving a
   distribution-free p-value for inter > intra. Output: tidy
   `results/analysis.csv` + aggregate `results/summary.csv`.

4. **Figures** (`wsd_probe/plots.py`) — separation vs checkpoint per layer,
   layer × checkpoint heatmaps, fraction of significant words, per-word
   trajectories (word and layer selected from the data, not a priori).

## Setup

```bash
uv sync          # or: pip install torch transformers nltk numpy scipy pandas matplotlib scikit-learn tqdm
```

nltk data (semcor, wordnet) downloads automatically on first run.

## Run

```bash
# Full pipeline, default = EleutherAI/pythia-6.9b, 20 log-spaced checkpoints
uv run python -m wsd_probe all

# Or stage by stage
uv run python -m wsd_probe data
uv run python -m wsd_probe extract --model EleutherAI/pythia-6.9b --batch-size 8
uv run python -m wsd_probe analyze --n-permutations 1000
uv run python -m wsd_probe plot
```

Useful flags:

- `--steps 0,1,2,...,143000` — which Pythia revisions to load (default:
  log-spaced from 0 to 143000).
- `--device cuda|mps|cpu` — auto-detected if omitted; fp16/bf16 on CUDA.
- `--model` — any Pythia size (`pythia-70m` … `pythia-12b`); note there is no
  "7B": the closest is **6.9b**.
- `--max-words`, `--min-examples-per-sense` — dataset size/quality trade-off.
- `--cache-dir` — HuggingFace cache location (checkpoints of 6.9b are ~14 GB
  each; they can be deleted after extraction, only the .npz matter).

Reproducibility: every sampling step is seeded (`--seed`, default 0); each
stage is idempotent and resumable.

## Grid'5000 (OAR)

```bash
./scripts/submit_oar.sh                 # 1 GPU, 24h, production queue
OAR_RESOURCES="host=1/gpu=1,walltime=48:00:00" \
OAR_PROPERTY="gpu_model LIKE 'A100%'" ./scripts/submit_oar.sh --batch-size 16
```

The node payload (`scripts/oar_job.sh`) installs uv if needed, puts the HF
cache on node-local `/tmp`, and runs `wsd_probe all` with `--purge-cache` so
disk usage stays bounded to one ~14 GB checkpoint at a time. Logs (with
progress bars) land in `oar_logs/`. Extraction is resumable: if the job hits
walltime, resubmit and already-extracted checkpoints are skipped.

## Compute notes

- 6.9B in fp16 needs ~16 GB of GPU memory at batch 8, seq 128.
- Disk: embeddings are ~250 KB/instance/checkpoint for 6.9b (33 layers ×
  4096 fp16) → with the default ~2–3k instances and 20 checkpoints, plan
  ~10–15 GB for `results/embeddings/`.
- The smoke-test configuration used to validate the code:

```bash
uv run python -m wsd_probe all --model EleutherAI/pythia-70m \
  --dataset results_smoke/dataset.jsonl --emb-dir results_smoke/embeddings \
  --analysis-csv results_smoke/analysis.csv --summary-csv results_smoke/summary.csv \
  --fig-dir results_smoke/figures --max-words 6 --min-examples-per-sense 8 \
  --max-examples-per-sense 15 --steps 0,64,512,4000,32000,143000 --n-permutations 300
```

## Interpretation aids

- **ratio ≈ 1 / silhouette ≈ 0** → senses are indistinguishable at that
  layer/step; the checkpoint where the curves leave the baseline marks the
  onset of sense separation.
- **Layer 0** is the static embedding: any separation there is lexical
  leakage (e.g. capitalization, morphology), a useful control.
- `frac_significant` = share of words whose permutation p-value < 0.05.
  Caveat: pairwise distances are not independent samples; the permutation
  test is the appropriate primary test here, and p-values across
  (word × layer × step) are exploratory — apply your favourite multiplicity
  correction before making claims about individual cells.
