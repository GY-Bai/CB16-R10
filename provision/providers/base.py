"""Base class for CB16 Provisioner providers."""
from __future__ import annotations

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
        if sha256:
            import hashlib
            actual = hashlib.sha256(destination.read_bytes()).hexdigest()
            if actual != sha256:
                raise ProviderError(
                    f"ASSET_SHA256_MISMATCH:expected={sha256}:actual={actual}"
                )
        if size is not None:
            if destination.stat().st_size != size:
                raise ProviderError(
                    f"ASSET_SIZE_MISMATCH:expected={size}:actual={destination.stat().st_size}"
                )
