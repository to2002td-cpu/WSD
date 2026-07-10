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
./scripts/submit.sh                           # submit generate then extract as OAR jobs
```

`--only LEMMA.POS` restricts extraction (and the viz below) to given generated
files, scoping the dataset and embeddings under that tag; omit it for the whole
corpus. `generate.sh`/`extract.sh` are OAR job scripts (`oarsub -S ...`);
`submit.sh` chains them. Run any stage directly with `uv run python -m src <stage>`.

## Visualization

All four read one shared UMAP cache (computed once per word, then reused). Pass
the same `--only` used for extraction.

```bash
./scripts/plot.sh        bank --pos n --only bank.n   # scatter, coloured by sense
./scripts/density.sh     bank --pos n --only bank.n   # scatter + per-sense KDE
./scripts/heatmap.sh     bank --pos n --only bank.n   # thermal point-density (senses blended)
./scripts/interactive.sh bank --pos n --only bank.n   # interactive Plotly HTML
```

Outputs land in `<storage>/figures/`: `bank_umap.png`, `bank_density.png`,
`bank_thermal.png`, `bank_interactive.html`.
