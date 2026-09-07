from __future__ import annotations

"""R2 frozen-authority runtime guard.

R2/WSS runs bind the package authority read-only.  Full SHA256 of large frozen assets
is therefore required at campaign start and end, not once per generation.  Between
those points a strict inode/size/mtime/ctime metadata guard detects ordinary mutation
immediately, while the final full hash remains authoritative.

This module is R2-only; legacy R10.2/R10.4 hashing behavior is unchanged.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


def _sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _metadata(path: Path) -> dict[str, int]:
    st = path.stat()
    return {
        "st_dev": int(st.st_dev),
        "st_ino": int(st.st_ino),
        "st_mode": int(st.st_mode),
        "st_size": int(st.st_size),
        "st_mtime_ns": int(st.st_mtime_ns),
        "st_ctime_ns": int(st.st_ctime_ns),
    }


@dataclass
class FrozenAuthorityGuardR2:
    root: Path
    relative_paths: tuple[str, ...]
    start_hashes: dict[str, str]
    start_metadata: dict[str, dict[str, int]]
    metadata_checks: int = 0

    @classmethod
    def capture(cls, root: str | Path, relative_paths: Iterable[str]):
        resolved = Path(root).resolve()
        rels = tuple(str(x) for x in relative_paths)
        hashes: dict[str, str] = {}
        meta: dict[str, dict[str, int]] = {}
        for rel in rels:
            p = resolved / rel
            if not p.is_file():
                raise FileNotFoundError(p)
            meta[rel] = _metadata(p)
            hashes[rel] = _sha256_file(p)
            # Hashing itself must not change any mutation-sensitive metadata.
            if _metadata(p) != meta[rel]:
                raise RuntimeError(f"R2_FROZEN_AUTHORITY_METADATA_CHANGED_DURING_START_HASH:{rel}")
        return cls(resolved, rels, hashes, meta)

    def metadata_unchanged(self) -> bool:
        self.metadata_checks += 1
        for rel in self.relative_paths:
            p = self.root / rel
            try:
                now = _metadata(p)
            except FileNotFoundError:
                return False
            if now != self.start_metadata[rel]:
                return False
        return True

    def finalize(self) -> dict[str, Any]:
        metadata_before_final_hash = self.metadata_unchanged()
        end_hashes: dict[str, str] = {}
        end_metadata: dict[str, dict[str, int]] = {}
        for rel in self.relative_paths:
            p = self.root / rel
            if not p.is_file():
                return {
                    "schema": "CB16_R2_FROZEN_AUTHORITY_RUNTIME_GUARD_V1",
                    "pass": False,
                    "failure": f"MISSING_AT_FINALIZE:{rel}",
                    "metadata_checks": self.metadata_checks,
                    "start_hashes": self.start_hashes,
                    "end_hashes": end_hashes,
                }
            end_hashes[rel] = _sha256_file(p)
            end_metadata[rel] = _metadata(p)
        hashes_equal = end_hashes == self.start_hashes
        metadata_equal = end_metadata == self.start_metadata
        passed = bool(metadata_before_final_hash and metadata_equal and hashes_equal)
        return {
            "schema": "CB16_R2_FROZEN_AUTHORITY_RUNTIME_GUARD_V1",
            "pass": passed,
            "policy": "FULL_SHA256_START_AND_END__STRICT_STAT_EACH_GENERATION",
            "metadata_checks": self.metadata_checks,
            "metadata_unchanged": metadata_equal,
            "full_hashes_unchanged": hashes_equal,
            "start_hashes": self.start_hashes,
            "end_hashes": end_hashes,
            "scientific_semantics_changed": False,
        }
