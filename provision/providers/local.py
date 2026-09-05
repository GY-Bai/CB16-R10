"""Local existing source provider."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .base import Candidate, Provider, ProviderError


class LocalProvider(Provider):
    name = "local"

    def can_handle(self, source: dict[str, Any]) -> bool:
        return source.get("type") in {"local", "manual_existing"}

    def resolve(self, source: dict[str, Any], env: dict[str, str]) -> Optional[Candidate]:
        hint_env = source.get("hint_env")
        path_hint = source.get("path")
        if path_hint:
            p = Path(path_hint).expanduser()
            if p.exists():
                return Candidate(provider=self.name, local_path=p)
        if hint_env:
            raw = env.get(hint_env) or os.environ.get(hint_env, "")
            if raw:
                p = Path(raw).expanduser()
                if p.exists():
                    return Candidate(provider=self.name, local_path=p)
        return None

    def download(self, candidate: Candidate, destination: Path, env: dict[str, str]) -> None:
        if candidate.local_path is None:
            raise ProviderError("LOCAL_CANDIDATE_HAS_NO_PATH")
        # For directories, we register rather than copy. For files, copy atomically.
        if candidate.local_path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(destination.suffix + ".partial")
        import shutil
        shutil.copy2(candidate.local_path, tmp)
        tmp.replace(destination)
