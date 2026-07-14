"""Publish the generated sense-annotated corpus to a HuggingFace dataset repo.

Rebuilds the cleaned dataset from the generated sentences (fresh each push, never
touching the extraction dataset), writes it as both JSONL and Parquet, and uploads
them with a dataset card. Repo from ``--repo-id`` > ``hub.repo_id`` > $HF_REPO_ID;
write token from $HF_TOKEN.
"""

from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

CARD = """\
---
license: cc-by-4.0
task_categories: [token-classification]
language: [en]
tags: [word-sense-disambiguation, wordnet]
configs:
- config_name: default
  data_files: dataset.parquet
---

# WSD probe — generated sense-annotated sentences

Synthetic sentences, one target word per sentence used in a specific WordNet
sense, generated with an instruction-tuned model across {n_styles} writing styles
(balanced).

Fields: `word`, `pos`, `sense` (WordNet synset id), `sentence`,
`target_start`/`target_end` (char span of the target), `surface`, `style`,
`model`, `seed`.

{n} sentences over {n_senses} senses of {n_words} words. Available as
`dataset.parquet` and `dataset.jsonl`.
"""


def _write_exports(records: "list[dict]", out_dir) -> "tuple":
    """Write the records as JSONL and Parquet; return both paths."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "dataset.jsonl"
    with jsonl_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    parquet_path = out_dir / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist(records), parquet_path)
    return jsonl_path, parquet_path


def push_dataset(cfg: dict, repo_id: str | None = None, private: bool | None = None) -> None:
    from huggingface_hub import HfApi

    from .config import store
    from .dataset import assemble_records

    hub = cfg.get("hub", {})
    repo_id = repo_id or hub.get("repo_id") or os.environ.get("HF_REPO_ID")
    if not repo_id:
        raise SystemExit("Set hub.repo_id in config, pass --repo-id, or export HF_REPO_ID "
                         "(e.g. your-org/wsd-sentences).")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("Set $HF_TOKEN (a write token) to push to the Hub.")
    private = hub.get("private", True) if private is None else private

    records = assemble_records(cfg, None)
    if not records:
        raise SystemExit("No records to push; generate sentences first.")
    words = {r["word"] for r in records}
    senses = {r["sense"] for r in records}
    n_styles = len({r.get("style") for r in records if r.get("style")})

    out_dir = store(cfg, "export")
    jsonl_path, parquet_path = _write_exports(records, out_dir)
    card = out_dir / "README.md"
    card.write_text(CARD.format(n=len(records), n_words=len(words), n_senses=len(senses),
                                n_styles=n_styles))

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    for path in (parquet_path, jsonl_path, card):
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                        repo_id=repo_id, repo_type="dataset")
    log.info("Pushed %d sentences (parquet + jsonl) -> https://huggingface.co/datasets/%s",
             len(records), repo_id)
