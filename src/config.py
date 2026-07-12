from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
DEFAULT_CONFIG = CONFIGS / "default.yaml"
MODELS_DIR = CONFIGS / "models"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_yaml(path: Path) -> dict:
    """Load a YAML config, deep-merging any ``include:`` fragments (relative to the
    file) first, so the file's own keys win over what it includes."""
    if not path.exists():
        raise SystemExit(f"Config {path} not found.")
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    merged: dict = {}
    for inc in cfg.pop("include", []):
        merged = _deep_merge(merged, _load_yaml((path.parent / inc).resolve()))
    return _deep_merge(merged, cfg)


def load_config(config: str | Path | None = None, model: str | None = None,
                lemmas: str | None = None) -> dict:
    """Compose a run config: the base (``configs/default.yaml`` or ``config``),
    then the model fragment ``configs/models/<model>.yaml``, then the lemma list."""
    cfg = _load_yaml(Path(config) if config else DEFAULT_CONFIG)
    if model:
        cfg = _deep_merge(cfg, _load_yaml(MODELS_DIR / f"{model}.yaml"))
    if lemmas:
        cfg["lemmas_file"] = lemmas if ("/" in lemmas or lemmas.endswith(".txt")) \
            else f"lemmas/{lemmas}.txt"
    return cfg


def resolve(p: str | Path) -> Path:
    """A repo-relative input path (API key, prompts, lemmas) made absolute."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def store(cfg: dict, p: str | Path) -> Path:
    """An output path under the storage root: $WSD_STORAGE_ROOT > config
    storage_root > repo. Point storage_root at a large disk on the cluster."""
    p = Path(p)
    if p.is_absolute():
        return p
    root = os.environ.get("WSD_STORAGE_ROOT") or cfg.get("storage_root")
    base = Path(root) if root else ROOT
    if not base.is_absolute():
        base = ROOT / base
    return base / p


def run_paths(cfg: dict, only: "list[str] | None" = None) -> "tuple[Path, Path]":
    """(dataset_path, emb_dir) for a run. ``only`` scopes the dataset and its
    embeddings under a tag; None -> the full corpus. emb_dir is namespaced by
    model, so models coexist under storage."""
    model = cfg["extract"]["model"].split("/")[-1]
    emb_dir = store(cfg, cfg["extract"]["emb_dir"]) / model
    if not only:
        return store(cfg, cfg["dataset"]["path"]), emb_dir
    tag = "_".join(sorted(s.removesuffix(".jsonl") for s in only))
    return store(cfg, "datasets") / f"{tag}.jsonl", emb_dir / tag
