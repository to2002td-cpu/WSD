# WSD probe

Probe sense separation in Pythia checkpoints. List lemmas in `lemmas.txt`; for
each we take every WordNet synset, generate `per_synset` sentences per sense
(chat API), then mean-pool the target token's hidden states across checkpoints.

Config: `config.yaml`. Prompt: `prompts/generate.txt`. Outputs go under
`storage_root` (override with `$WSD_STORAGE_ROOT`).

## Commands

```bash
./scripts/synsets.sh              # lemmas.txt -> synsets.json
./scripts/generate.sh             # stage 1 (CPU): synthesize sentences
./scripts/extract.sh              # stage 2 (GPU): pool hidden states
./scripts/plot.sh bank --pos n    # per-word UMAP grid
./scripts/submit.sh               # submit generate then extract as OAR jobs
```

`generate.sh`/`extract.sh` are OAR job scripts (`oarsub -S ...`); `submit.sh`
chains them. Run any stage directly with `uv run python -m src <stage>`.
