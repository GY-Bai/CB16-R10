"""HTTP(S) provider with .partial resume, retry and backoff."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import requests

from .base import Candidate, Provider, ProviderError


class HttpProvider(Provider):
    name = "http"

    def can_handle(self, source: dict[str, Any]) -> bool:
        return source.get("type") == "http"

    def _resolve_url(self, source: dict[str, Any], env: dict[str, str]) -> Optional[str]:
        if source.get("url"):
            return source["url"]
        url_env = source.get("url_env")
        if url_env:
            return env.get(url_env) or os.environ.get(url_env, "") or None
        return None

    def resolve(self, source: dict[str, Any], env: dict[str, str]) -> Optional[Candidate]:
        url = self._resolve_url(source, env)
        if not url:
            return None
        return Candidate(provider=self.name, url=url)

    def download(self, candidate: Candidate, destination: Path, env: dict[str, str]) -> None:
        if not candidate.url:
            raise ProviderError("HTTP_CANDIDATE_HAS_NO_URL")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        timeout = int(env.get("CB16_PROVISION_HTTP_TIMEOUT_SECONDS", "60"))
        retries = int(env.get("CB16_PROVISION_HTTP_RETRIES", "3"))
        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                self._download_once(candidate.url, partial, timeout)
                partial.replace(destination)
                return
            except Exception as e:
                last_error = e
                time.sleep(min(2 ** attempt, 15))
        raise ProviderError(f"HTTP_DOWNLOAD_FAILED after {retries} attempts: {last_error}")

    def _download_once(self, url: str, partial: Path, timeout: int) -> None:
        headers = {}
        if partial.exists():
            headers["Range"] = f"bytes={partial.stat().st_size}-"
        with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
            r.raise_for_status()
            mode = "ab" if partial.exists() and r.status_code == 206 else "wb"
            with partial.open(mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
