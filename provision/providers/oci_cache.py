"""OCI Relay cache provider (interface reserved; not enabled in V1)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .base import Candidate, Provider, ProviderError


class OciCacheProvider(Provider):
    name = "oci_cache"

    def can_handle(self, source: dict[str, Any]) -> bool:
        return source.get("type") == "oci_cache"

    def resolve(self, source: dict[str, Any], env: dict[str, str]) -> Optional[Candidate]:
        return None

    def download(self, candidate: Candidate, destination: Path, env: dict[str, str]) -> None:
        raise ProviderError("OCI_CACHE_PROVIDER_NOT_ENABLED_IN_V1")
