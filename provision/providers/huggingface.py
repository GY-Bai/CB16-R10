"""Hugging Face Hub provider adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .base import Candidate, Provider, ProviderError


class HuggingFaceProvider(Provider):
    name = "huggingface"

    def can_handle(self, source: dict[str, Any]) -> bool:
        return source.get("type") == "huggingface"

    def resolve(self, source: dict[str, Any], env: dict[str, str]) -> Optional[Candidate]:
        repo_id = source.get("repo_id")
        if not repo_id:
            return None
        return Candidate(
            provider=self.name,
            repo_id=repo_id,
            revision=source.get("revision"),
            extra={
                "allow_patterns": source.get("allow_patterns"),
                "ignore_patterns": source.get("ignore_patterns"),
            },
        )

    def download(self, candidate: Candidate, destination: Path, env: dict[str, str]) -> None:
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise ProviderError(f"HUGGINGFACE_HUB_MISSING:{e}")
        if not candidate.repo_id:
            raise ProviderError("HF_CANDIDATE_HAS_NO_REPO_ID")
        destination.mkdir(parents=True, exist_ok=True)
        # Use the local cache; the resolved model directory is mirrored under destination.
        snapshot_download(
            repo_id=candidate.repo_id,
            revision=candidate.revision,
            allow_patterns=candidate.extra.get("allow_patterns"),
            ignore_patterns=candidate.extra.get("ignore_patterns"),
        )
        # The actual files remain under HF cache; we record a README marker.
        (destination / "PROVIDER.txt").write_text(
            f"provider=huggingface repo_id={candidate.repo_id} revision={candidate.revision or 'main'}\n"
        )
