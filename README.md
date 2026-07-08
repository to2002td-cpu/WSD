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
   (`revision=stepN`), sentences are run through the model in
   `torch.inference_mode()` with `output_hidden_states=True`; the
   representation of the target word is the **mean over the sub-tokens that
   overlap its character span** (the tokenizer's offset mapping handles
   multi-sub-token words). Vectors are kept at a fixed set of **probe layers**
   (`extract.TARGET_LAYERS`, currently `{8, 16, 24, 32}`) — hidden-state index
   1 is the first transformer block's output, so these are mid/late layers.
   Output: one `stepN.npz` per checkpoint holding a
   `[n_instances, n_probe_layers, hidden]` float16 array, row-aligned with the
   JSONL (targets lost to sequence truncation are left as zero rows and
   dropped downstream). Extraction is resumable — existing files are skipped.

3. **Analysis** (`wsd_probe/analyze.py`) — the core measurement, **centroid
   based**. For every (word × probe-layer × checkpoint) cell it summarizes each
   sense by its centroid and computes the within-/between-sense cosine
   distances, their **ratio**, the **centroid distance**, and a
   **nearest-centroid silhouette**. All statistics are linear in the number of
   instances — no `n × n` pairwise matrix. Exact definitions are in
   [What exactly is computed](#what-exactly-is-computed) below. Output: tidy
   `results/analysis.csv` (one row per cell) + aggregate `results/summary.csv`.

4. **Figures** (`wsd_probe/plots.py`) — separation vs checkpoint per layer,
   layer × checkpoint heatmaps, per-word trajectories (word and layer selected
   from the data, not a priori).

## What exactly is computed

The unit of measurement is one **cell**: a single word `w`, at one probe layer
`ℓ`, at one pre-training checkpoint `t`. Everything below happens inside a cell;
the pipeline just repeats it over every `(w, ℓ, t)` and aggregates.

**Setup.** Inside a cell we have `n` instances of the word `w` (each an
occurrence in a different SemCor sentence), grouped into `g` annotated senses.
Instance `i` contributes:

- a vector `hᵢ ∈ ℝ^d` — the layer-`ℓ` hidden state, mean-pooled over the target
  word's sub-tokens (`d` = model hidden size, e.g. 4096 for pythia-6.9b);
- a sense label `yᵢ` — the WordNet synset SemCor annotated it with
  (e.g. `bank.n.01`).

Every vector is first L2-normalized to a unit vector `uᵢ = hᵢ / ‖hᵢ‖`, so all
geometry uses **cosine distance** `d(a, b) = 1 − a·b` (direction only, magnitude
ignored; `0` = same direction, `1` = orthogonal). The question "does the model
separate senses?" becomes "does each meaning occupy its own direction?"

**1. Sense centroids.** Each sense `s` is summarized by one prototype vector —
the mean of its instances' unit vectors, re-normalized to a direction:

```
μ_s = normalize( mean of uᵢ over instances with yᵢ = s )
```

This is the whole reason the analysis is fast: instead of comparing all `n(n−1)/2`
pairs of instances, we compare instances to `g` centroids (`g` = 2–4 senses in
practice), so the cost is `O(n · g)` per cell rather than `O(n²)`.

**2. Within-sense scatter (`intra`), with leave-one-out.** How tightly do
instances of the *same* sense cluster? For each instance we measure its cosine
distance to its own sense's centroid — but to a centroid computed **without that
instance** (leave-one-out, LOO):

```
μ_s^(−i) = normalize( mean of u over sense s, excluding instance i )
intra    = mean over i of  d( uᵢ , μ_{yᵢ}^(−i) )
```

The LOO correction is essential and easy to get wrong: if `i` helped build its
own centroid, it is trivially close to it, and **even random data would look
separated** (`ratio > 1`, `silhouette > 0`) whenever senses have few instances.
Excluding `i` removes that self-inclusion bias, so `intra` is an honest noise
floor. (Distances to *other* senses' centroids below need no correction — `i`
never contributed to them.)

**3. Between-sense distance (`inter`).** How far is an instance from the *other*
meanings?

```
inter = mean over i of  [ mean over s ≠ yᵢ of  d( uᵢ , μ_s ) ]
```

Sense separation means `inter > intra`: instances sit closer to their own sense
prototype than to competing ones.

**4. Ratio (the headline statistic).**

```
ratio = inter / intra
```

- `ratio ≈ 1` → senses are indistinguishable from within-sense noise.
- `ratio > 1` → different senses sit measurably farther apart than the same
  sense does from itself.

Taking the ratio (rather than the raw gap `inter − intra`) **internally
normalizes** the cell: if a checkpoint's whole representation space becomes more
or less anisotropic over training — inflating *all* cosine distances — both
`intra` and `inter` scale together and the ratio is unaffected. This is what
makes numbers comparable *across checkpoints*, which is the entire point of the
study.

**5. Centroid distance.** The coarsest view — are the sense *prototypes*
themselves far apart, ignoring within-sense scatter entirely:

```
centroid_dist = mean over sense pairs (s, s′) of  d( μ_s , μ_s′ )
```

**6. Nearest-centroid silhouette.** A single per-instance separation score in
`[−1, 1]`. For instance `i` let `a = d(uᵢ, μ_{yᵢ}^(−i))` (LOO distance to its own
prototype) and `b = min over s ≠ yᵢ of d(uᵢ, μ_s)` (distance to the *nearest*
competing prototype). Then

```
silhouette = mean over i of  (b − a) / max(a, b)
```

`≈ 1` = instance sits squarely with its own sense, `≈ 0` = on the boundary,
`< 0` = closer to a different sense. It is the standard clustering-quality
answer to "are the senses actually separable?", complementary to the ratio.
This is a centroid (nearest-prototype) silhouette, *not* the full pairwise
silhouette — chosen precisely to avoid the `n × n` matrix.

**Aggregation** (`summarize`). Per `(step, layer)` we average `ratio`,
`silhouette`, and `centroid_dist` over words. Reading these curves against the
checkpoint axis is how we date the *onset* of sense separation: the step at
which they lift off their `ratio ≈ 1` / `silhouette ≈ 0` baseline.

**Robustness details worth knowing when checking the method:**

- Leave-one-out on the own-sense centroid (point 2) is what keeps the null
  honest: on random vectors this method returns `ratio ≈ 1.00` and
  `silhouette ≈ 0`, as it should.
- Truncation: if the target word falls past `--max-length`, its row is left as
  zeros in the `.npz` and dropped in analysis (`valid` mask on the last-layer
  norm), so pooled-over-nothing artifacts never enter a statistic.
- After dropping truncated rows, a sense is kept only if it still has ≥ 2
  instances (LOO needs ≥ 2), and a word only if ≥ 2 such senses remain —
  otherwise `intra`/`inter` would be undefined.
- The analysis is deterministic: no sampling, no permutations, so reruns on the
  same `.npz` give identical numbers.

## Setup

```bash
uv sync          # or: pip install torch transformers nltk numpy scipy pandas matplotlib scikit-learn tqdm
```

nltk data (semcor, wordnet) downloads automatically on first run.

## Run

```bash
# Full pipeline, default = EleutherAI/pythia-6.9b, checkpoints extract.DEFAULT_STEPS
uv run python -m wsd_probe all

# Or stage by stage
uv run python -m wsd_probe data
uv run python -m wsd_probe extract --model EleutherAI/pythia-6.9b --batch-size 8
uv run python -m wsd_probe analyze
uv run python -m wsd_probe plot
```

Useful flags:

- `--steps 0,1,2,...,143000` — which Pythia revisions to load (default:
  `extract.DEFAULT_STEPS = 1000,16000,32000,64000,128000,143000`; add earlier
  steps like `0,512` to catch onset earlier in training).
- `--device cuda|mps|cpu` — auto-detected if omitted; fp16/bf16 on CUDA.
- `--model` — any Pythia size (`pythia-70m` … `pythia-12b`); note there is no
  "7B": the closest is **6.9b**.
- `--max-words`, `--min-examples-per-sense` — dataset size/quality trade-off.
- `--cache-dir` — HuggingFace cache location (checkpoints of 6.9b are ~14 GB
  each; they can be deleted after extraction, only the .npz matter).

Reproducibility: dataset sampling is seeded (`--seed`, default 42) and the
analysis is deterministic; each stage is idempotent and resumable.

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
  --max-examples-per-sense 15 --steps 0,64,512,4000,32000,143000
```

## Interpretation aids

- **ratio ≈ 1 / silhouette ≈ 0** → senses are indistinguishable at that
  layer/step; the checkpoint where the curves leave the baseline marks the
  onset of sense separation.
- **Probe layers** (`TARGET_LAYERS = {8, 16, 24, 32}`) are mid-to-late
  transformer blocks. To recover a static-embedding control (separation
  attributable to spelling/morphology alone, before any contextualization),
  add an early layer such as `1` to `TARGET_LAYERS` and re-extract.
- **centroid_dist** is the coarsest signal (prototype-to-prototype distance);
  **ratio** and **silhouette** additionally account for within-sense scatter,
  so they are the more honest "are senses separated?" measures.
- Small `n` per sense makes centroids noisy: treat single-cell numbers as
  exploratory and read the *aggregate* curves (`summary.csv`) and per-word
  trajectories, where the noise averages out.
