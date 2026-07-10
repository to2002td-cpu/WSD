from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise SystemExit(f"Config {path} not found.")
    with path.open() as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    """An input path made absolute (relative paths hang off the repo root).

    For repo-committed inputs (API key, styles), not pipeline outputs.
    """
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def store(cfg: dict, p: str | Path) -> Path:
    """An output path under the storage root (embeddings, generated data, ...).

    Precedence for the root: $WSD_STORAGE_ROOT > config `storage_root` > repo.
    Point `storage_root` at a large disk (e.g. Grid'5000 group storage) so the
    generated corpus and float16 embeddings do not fill the NFS home quota.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    root = os.environ.get("WSD_STORAGE_ROOT") or cfg.get("storage_root")
    base = Path(root) if root else ROOT
    if not base.is_absolute():
        base = ROOT / base
    return base / p
