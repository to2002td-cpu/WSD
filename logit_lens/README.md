# Logit-lens probe

Companion to `twosampletest.ipynb`'s kernel two-sample test. That notebook asks
*"are two senses of a word separable in embedding space?"* across checkpoints x
layers. This asks the complementary question at the same (word, checkpoint,
layer) coordinates: *"is the word itself what the model would have predicted
there?"* -- i.e. how much of a role does actual next-token predictability play
alongside separability, over training and depth.

For each SemCor occurrence of a target word, at every checkpoint and every
layer, the model is teacher-forced up to (not including) the target's
subtoken(s) and we read off, via the **logit lens** (apply the model's own
final layer norm + unembedding to that layer's residual stream instead of the
last one), whether the actual word was the argmax and what logprob it got.

## Why a separate folder

`src/` builds its own synthetic sentence corpus per WordNet sense (chat-API
generated) for the sense-cluster-purity analysis. This pipeline uses **real
SemCor sentences** instead (matching what `twosampletest.ipynb` analyzed) and
extracts **logprobs**, not pooled hidden states -- different data, different
output, so it's kept as its own modular pipeline rather than bolted onto
`src/extract.py`. It still follows the same conventions (config-driven,
resumable stages, thin `scripts/*.sh` OAR wrappers).

## Pipeline

```bash
uv run python -m logit_lens occurrences   # stage 0 (CPU, ~minutes): SemCor -> occ_df.pkl
uv run python -m logit_lens extract       # stage 1 (GPU, the long one): logprobs per checkpoint x layer
uv run python -m logit_lens aggregate     # stage 2 (CPU): tidy word x layer x step summary
```

or as OAR jobs (each stage is a separate launchable job, so e.g. `extract` can
run overnight on its own):

```bash
oarsub -S "./scripts/logit_lens_occurrences.sh"
oarsub -S "./scripts/logit_lens_extract.sh"       # after occurrences.sh finishes
oarsub -S "./scripts/logit_lens_aggregate.sh"     # after extract.sh finishes
./scripts/logit_lens_submit.sh                    # or chain all three via OAR -a
```

**Stage 0 -- `occurrences`.** Parses SemCor's sense-tagged sentences and
deterministically picks 60 polysemous target words (seed 1234, 20 per
frequency bin, `min_occ=5`, no cap on sense count, no lemma/POS exclusion),
reproducing `pythia_wsd_analysis_clean.ipynb`'s word-selection cells exactly --
the notebook that originally built `occ_df.pkl` (25,976 occurrences) for
`twosampletest.ipynb` and `perword_clustering.ipynb`. Re-running this from
scratch (rather than reading that pickle, which lives on Rennes-site storage
not reachable from every node) is intentional -- it's cheap and deterministic,
so the reconstructed word list matches: verified byte-for-byte against the
source notebook's own cached output (25,976 occurrences, 18,369 unique
sentences, identical per-word counts down to `be::v`'s 15,763). If you ever
want a different set, point
`word_selection.word_list_file` at a plain-text `lemma::pos`-per-line file to
use it instead of the algorithmic pick. Writes
`<storage>/occurrences/{targets_df.csv,occ_df.pkl}`.

**Stage 1 -- `extract`.** The GPU job. For each checkpoint in
`config.yaml`'s `checkpoints` (`[0, 8, 16, 64, 256, 1000, 4000, 8000, 16000,
32000, 64000, 128000, 143000]` -- note 1024/4096 aren't real Pythia
revisions, snapped to the nearest real ones, 1000/4000), loads
`EleutherAI/pythia-6.9b` at that revision, teacher-forces every SemCor
sentence containing a target word, and for each of `layers` (`[1, 8, 16, 24,
32]`, 1-indexed, same as the main pipeline) records the logprob, rank, and
top-1 flag of each target subtoken via the logit lens. **Resumable**: writes
one pickle per checkpoint (`<storage>/results/logprobs_step<N>.pkl`) as soon
as that checkpoint finishes, and skips any checkpoint whose file already
exists on the next run (`-f/--force` to redo). Each checkpoint's ~14GB HF
snapshot is purged after use -- the home directory quota here is small
(~24GB soft / ~95GB hard) and the HF cache defaults to node-local `/tmp`
anyway, never NFS home. Launch it, let it run overnight, and whatever
checkpoints finished are already on disk in the morning even if it didn't get
through all 13.

**Stage 2 -- `aggregate`.** Concatenates the per-checkpoint pickles, rolls
multi-subtoken words up to one joint logprob per occurrence, then to one row
per `(word_id, layer, step)`. Writes `<storage>/results/occurrence_logprobs.pkl`
(occurrence-level) and `<storage>/results/word_summary.csv` (word-level, with
`freq_bin` attached).

## Correlating with `twosampletest.ipynb`

`word_summary.csv`'s `(word_id, layer, step)` keys match the sweep's `prop`
table (`layer`, `step` index/columns) and `word_summary` dataframe there:

```python
sep = word_summary_from_twosampletest    # word_id, layer, step, mean_frac_rej, ...
pred = pd.read_csv("<storage>/results/word_summary.csv")   # word_id, layer, step, mean_logprob, top1_rate, ...
joined = sep.merge(pred, on=["word_id", "layer", "step"])
joined[["mean_frac_rej", "top1_rate"]].corr()
```

## Paths

Everything (`occurrences/`, `results/`) writes under `storage_root`:
`$WSD_STORAGE_ROOT` env var if set, else `storage_root` in `config.yaml`
(default `null`), else `<repo>/data/logit_lens` (git-ignored, on NFS `/home` --
fine, results here are tens of MB total, nowhere near the home quota).

gdelicat's team-storage folder
(`/srv/storage/linkmedia@storage2.rennes.grid5000.fr/gdelicat/logit_extraction`)
is available as an alternative but **only resolves on Rennes-site nodes** (the
storage server is on Rennes' internal network; confirmed unreachable from
Nancy). Set `export WSD_STORAGE_ROOT=/srv/storage/linkmedia@storage2.rennes.grid5000.fr/gdelicat/logit_extraction`
before submitting if/when running from Rennes instead.

The HF model cache is separate (`cache_dir` in config, default node-local
`/tmp`) since it holds full model snapshots to be discarded, not results, and
should never land on team storage or count against the home quota.
