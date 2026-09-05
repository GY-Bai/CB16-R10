from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from provision.providers.base import Provider, ProviderError


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_frozen_directory_asset_verifies_relative_file(tmp_path: Path):
    data = b"cb16-frozen-asset-canary"
    model = tmp_path / "model.safetensors"
    model.write_bytes(data)
    manifest = {
        "integrity_mode": "FROZEN",
        "integrity": {
            "sha256": _sha(data),
            "size_bytes": len(data),
            "relative_path": "model.safetensors",
        },
    }
    Provider().verify(tmp_path, manifest)


def test_frozen_directory_asset_rejects_marker_only(tmp_path: Path):
    (tmp_path / "PROVIDER.txt").write_text("marker only\n")
    manifest = {
        "integrity_mode": "FROZEN",
        "integrity": {
            "sha256": _sha(b"expected"),
            "size_bytes": len(b"expected"),
            "relative_path": "model.safetensors",
        },
    }
    with pytest.raises(ProviderError, match="ASSET_INTEGRITY_TARGET_MISSING"):
        Provider().verify(tmp_path, manifest)


def test_frozen_directory_asset_rejects_wrong_hash(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"wrong")
    manifest = {
        "integrity_mode": "FROZEN",
        "integrity": {
            "sha256": _sha(b"right"),
            "size_bytes": len(b"wrong"),
            "relative_path": "model.safetensors",
        },
    }
    with pytest.raises(ProviderError, match="ASSET_SHA256_MISMATCH"):
        Provider().verify(tmp_path, manifest)


def test_integrity_relative_path_cannot_escape_asset_root(tmp_path: Path):
    manifest = {
        "integrity_mode": "FROZEN",
        "integrity": {
            "sha256": "0" * 64,
            "relative_path": "../secret",
        },
    }
    with pytest.raises(ProviderError, match="ASSET_INTEGRITY_RELATIVE_PATH_INVALID"):
        Provider().verify(tmp_path, manifest)
