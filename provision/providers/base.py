"""Base class for CB16 Provisioner providers."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Candidate:
    """A concrete fetch candidate resolved from a declarative source."""
    provider: str
    url: Optional[str] = None
    local_path: Optional[Path] = None
    repo_id: Optional[str] = None
    revision: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    pass


def integrity_target(destination: Path, manifest: dict[str, Any]) -> Path:
    integrity = manifest.get("integrity") or {}
    relative = integrity.get("relative_path")
    if not relative:
        return destination
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ProviderError("ASSET_INTEGRITY_RELATIVE_PATH_INVALID")
    return destination / rel


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


class Provider:
    name = "base"

    def can_handle(self, source: dict[str, Any]) -> bool:
        return False

    def resolve(self, source: dict[str, Any], env: dict[str, str]) -> Optional[Candidate]:
        raise NotImplementedError

    def download(self, candidate: Candidate, destination: Path, env: dict[str, str]) -> None:
        raise NotImplementedError

    def verify(self, destination: Path, manifest: dict[str, Any]) -> None:
        integrity = manifest.get("integrity") or {}
        sha256 = integrity.get("sha256")
        size = integrity.get("size_bytes")
        manifest_sha256 = integrity.get("manifest_sha256")
        mode = manifest.get("integrity_mode")

        if mode == "FROZEN" and not sha256 and not manifest_sha256:
            raise ProviderError("FROZEN_ASSET_MISSING_CANONICAL_INTEGRITY")

        if sha256 is not None or size is not None:
            target = integrity_target(destination, manifest)
            if not target.is_file():
                raise ProviderError("ASSET_INTEGRITY_TARGET_MISSING")
            if size is not None and target.stat().st_size != int(size):
                raise ProviderError(
                    f"ASSET_SIZE_MISMATCH:expected={size}:actual={target.stat().st_size}"
                )
            if sha256:
                actual = sha256_file(target)
                if actual != sha256:
                    raise ProviderError(
                        f"ASSET_SHA256_MISMATCH:expected={sha256}:actual={actual}"
                    )
