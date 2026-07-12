"""Push the generated sense-annotated sentences (dataset.jsonl) to a HuggingFace
dataset repo. Token from $HF_TOKEN; target from config ``hub.repo_id`` or --repo-id."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

CARD = """\
---
license: cc-by-4.0
task_categories: [token-classification]
language: [en]
tags: [word-sense-disambiguation, wordnet]
---

# WSD probe — generated sense-annotated sentences

Synthetic sentences, one target word per sentence used in a specific WordNet
sense, generated with an instruction-tuned model across {n_styles} writing styles
(balanced). One JSON object per line.

Fields: `word`, `pos`, `sense` (WordNet synset id), `sentence`,
`target_start`/`target_end` (char span of the target), `surface`, `style`,
`model`, `seed`.

{n} sentences over {n_senses} senses of {n_words} words.
"""


def push_dataset(cfg: dict, repo_id: str | None = None, private: bool | None = None) -> None:
    from huggingface_hub import HfApi

    hub = cfg.get("hub", {})
    repo_id = repo_id or hub.get("repo_id")
    if not repo_id:
        raise SystemExit("Set hub.repo_id in config or pass --repo-id (e.g. your-org/wsd-sentences).")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("Set $HF_TOKEN (a write token) to push to the Hub.")
    private = hub.get("private", True) if private is None else private

    from .config import run_paths, store
    from .dataset import build_dataset

    records = build_dataset(cfg, None)
    data_path = run_paths(cfg, None)[0]
    words = {r["word"] for r in records}
    senses = {r["sense"] for r in records}
    n_styles = len({r.get("style") for r in records if r.get("style")})
    card = store(cfg, "DATASET_CARD.md")
    card.write_text(CARD.format(n=len(records), n_words=len(words), n_senses=len(senses),
                                n_styles=n_styles))

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(data_path), path_in_repo="dataset.jsonl",
                    repo_id=repo_id, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset")
    log.info("Pushed %d sentences -> https://huggingface.co/datasets/%s", len(records), repo_id)
