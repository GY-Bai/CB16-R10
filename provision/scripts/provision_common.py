"""Shared helpers for CB16 Provisioner V1."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

PROVISION_ROOT = Path(__file__).resolve().parent.parent
WORKER_ROOT = Path(os.environ.get("CB16_CI_WORKER_ROOT", "/data/cb16_ci"))
VENV_ROOT = Path(os.environ.get("CB16_VENV_ROOT", WORKER_ROOT / "venvs"))
ASSET_REGISTRY_DIR = Path(os.environ.get("CB16_ASSET_REGISTRY_DIR", WORKER_ROOT / "asset_registry"))
LOCK_ROOT = Path(os.environ.get("CB16_LOCK_ROOT", WORKER_ROOT / "locks"))
JOB_ROOT = Path(os.environ.get("CB16_JOB_ROOT", WORKER_ROOT / "jobs"))
REGISTRY_PATH = ASSET_REGISTRY_DIR / "registry.json"


def repo_root() -> Path:
    return PROVISION_ROOT.parent


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry() -> dict[str, Any]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def save_registry(registry: dict[str, Any]) -> None:
    atomic_write_json(REGISTRY_PATH, registry)


def ensure_dirs() -> None:
    for d in (VENV_ROOT, ASSET_REGISTRY_DIR, LOCK_ROOT, JOB_ROOT):
        d.mkdir(parents=True, exist_ok=True)
