"""Config + path helpers for the logit-lens probe. Standalone from ``src/config.py``
(this pipeline has its own storage namespace) but follows the same conventions:
env var > config value > repo-relative default.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: "str | Path | None" = None) -> dict:
    with Path(path or DEFAULT_CONFIG).open() as f:
        return yaml.safe_load(f)


def storage_root(cfg: dict) -> Path:
    """$WSD_STORAGE_ROOT > config storage_root > <repo>/data/logit_lens. The first
    two are taken as-is (an explicit destination, e.g. the team-storage
    logit_extraction folder); only the repo-relative fallback gets a "logit_lens"
    namespace appended, to keep it out of the way of the main pipeline's own data/."""
    root = os.environ.get("WSD_STORAGE_ROOT") or cfg.get("storage_root")
    if root:
        base = Path(root)
        return base if base.is_absolute() else ROOT / base
    return ROOT / "data" / "logit_lens"


def store(cfg: dict, *parts: str) -> Path:
    """An output path under this pipeline's storage root; parent dirs are not
    created here (callers create just before writing)."""
    return storage_root(cfg).joinpath(*parts)


def hf_cache_dir(cfg: dict) -> Path:
    """Local-disk HF cache: node-local /tmp by default, never the NFS home (whose
    quota a single ~14GB Pythia checkpoint would eat into fast)."""
    d = cfg.get("cache_dir") or os.environ.get("TMPDIR") or "/tmp"
    return Path(d) / "wsd_logit_lens_hf_cache"
