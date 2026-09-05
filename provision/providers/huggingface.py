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
        except Exception:
            raise ProviderError("HUGGINGFACE_HUB_MISSING") from None
        if not candidate.repo_id:
            raise ProviderError("HF_CANDIDATE_HAS_NO_REPO_ID")
        if not candidate.revision:
            raise ProviderError("HF_CANDIDATE_REVISION_NOT_PINNED")

        destination.mkdir(parents=True, exist_ok=True)
        token = env.get("HF_TOKEN") or None
        endpoint = env.get("HF_ENDPOINT") or None
        try:
            max_workers = max(1, min(int(env.get("CB16_HF_MAX_WORKERS", "4")), 16))
        except ValueError:
            max_workers = 4
        try:
            etag_timeout = max(5.0, float(env.get("CB16_HF_ETAG_TIMEOUT_SECONDS", "30")))
        except ValueError:
            etag_timeout = 30.0

        try:
            snapshot_download(
                repo_id=candidate.repo_id,
                revision=candidate.revision,
                local_dir=str(destination),
                allow_patterns=candidate.extra.get("allow_patterns"),
                ignore_patterns=candidate.extra.get("ignore_patterns"),
                token=token,
                endpoint=endpoint,
                max_workers=max_workers,
                etag_timeout=etag_timeout,
            )
        except Exception as exc:
            # Never place URL/token/provider exception text into public provisioning evidence.
            raise ProviderError(f"HUGGINGFACE_DOWNLOAD_FAILED:{type(exc).__name__}") from None

        (destination / "PROVIDER.txt").write_text(
            f"provider=huggingface repo_id={candidate.repo_id} revision={candidate.revision}\n"
        )
