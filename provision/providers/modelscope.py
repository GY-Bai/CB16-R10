"""ModelScope provider adapter (V1 placeholder)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .base import Candidate, Provider, ProviderError


class ModelScopeProvider(Provider):
    name = "modelscope"

    def can_handle(self, source: dict[str, Any]) -> bool:
        return source.get("type") == "modelscope"

    def resolve(self, source: dict[str, Any], env: dict[str, str]) -> Optional[Candidate]:
        model_id = source.get("model_id")
        if not model_id:
            return None
        return Candidate(provider=self.name, repo_id=model_id, revision=source.get("revision"))

    def download(self, candidate: Candidate, destination: Path, env: dict[str, str]) -> None:
        raise ProviderError("MODELSCOPE_PROVIDER_NOT_ENABLED_IN_V1")
