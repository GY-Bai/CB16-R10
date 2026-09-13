"""Generic byte-integrity manifest for qualification artifacts.

This verifies exported artifact bytes only. It deliberately does not claim that
those bytes semantically prove a stage capability; stage-specific verifiers own
that stronger judgment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


class QualificationArtifactError(ValueError):
    pass


def _safe_relative_path_v1(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise QualificationArtifactError(f"UNSAFE_ARTIFACT_PATH:{value}")
    return path


def _sha256_file_v1(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_manifest_v1(
    root: str | Path,
    required_relative_paths: Sequence[str],
) -> Mapping[str, Any]:
    base = Path(root)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in required_relative_paths:
        relative = _safe_relative_path_v1(str(raw))
        key = relative.as_posix()
        if key in seen:
            raise QualificationArtifactError(f"DUPLICATE_ARTIFACT_PATH:{key}")
        seen.add(key)
        full = base / relative
        if not full.is_file():
            raise QualificationArtifactError(f"REQUIRED_ARTIFACT_MISSING:{key}")
        entries.append({
            "path": key,
            "size_bytes": int(full.stat().st_size),
            "sha256": _sha256_file_v1(full),
        })
    return {
        "schema": "CB16_QUALIFICATION_ARTIFACT_BYTE_MANIFEST_V1",
        "entries": entries,
    }


def verify_artifact_manifest_v1(
    root: str | Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    if manifest.get("schema") != "CB16_QUALIFICATION_ARTIFACT_BYTE_MANIFEST_V1":
        raise QualificationArtifactError("ARTIFACT_MANIFEST_SCHEMA_MISMATCH")
    base = Path(root)
    violations: list[str] = []
    seen: set[str] = set()
    checked = 0
    for raw_entry in manifest.get("entries", []):
        entry = dict(raw_entry)
        relative = _safe_relative_path_v1(str(entry.get("path", "")))
        key = relative.as_posix()
        if key in seen:
            violations.append(f"DUPLICATE_ARTIFACT_PATH:{key}")
            continue
        seen.add(key)
        full = base / relative
        if not full.is_file():
            violations.append(f"ARTIFACT_MISSING:{key}")
            continue
        checked += 1
        observed_size = int(full.stat().st_size)
        observed_sha = _sha256_file_v1(full)
        if observed_size != int(entry.get("size_bytes", -1)):
            violations.append(f"ARTIFACT_SIZE_MISMATCH:{key}")
        if observed_sha != str(entry.get("sha256", "")):
            violations.append(f"ARTIFACT_SHA256_MISMATCH:{key}")
    if not manifest.get("entries"):
        violations.append("ARTIFACT_MANIFEST_EMPTY")
    return {
        "schema": "CB16_QUALIFICATION_ARTIFACT_BYTE_VERIFICATION_V1",
        "checked_files": checked,
        "contract_violations": sorted(set(violations)),
        "all_bytes_match": not violations,
    }
