#!/usr/bin/env python3
"""Read-only legacy-root scan and zero-copy registration into R11 G0 authority.

The scanner hashes every regular file in the configured legacy mounts, never
follows symlinks, never mutates legacy roots, and deliberately excludes final
holdout-named paths from content access. The output is an infrastructure
availability/adoption receipt, not scientific Evidence and not a new verdict.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = "CB16_R11_LEGACY_DATA_ADOPTION_V1"
G0 = Path(os.environ.get("CB16_G0_ROOT", "/cb16/g0"))
AUTHORITY = G0 / "authority"
INVENTORY_DIR = AUTHORITY / "legacy_inventory"
INVENTORY = INVENTORY_DIR / "CB16_R11_LEGACY_FILE_INVENTORY_V1.ndjson.gz"
RECEIPT = AUTHORITY / "CB16_R11_LEGACY_DATA_ADOPTION_V1.json"

ROOTS = (
    ("r104", Path("/cb16/runtime/r104"), "IMMUTABLE_REFERENCE"),
    ("package", Path("/cb16/package"), "IMMUTABLE_REFERENCE"),
    ("parent_r101", Path("/cb16/parent-r101"), "IMMUTABLE_REFERENCE"),
    ("parents", Path("/cb16/parents"), "IMMUTABLE_REFERENCE"),
    ("r2_authority", Path("/cb16/r2-authority"), "IMMUTABLE_REFERENCE"),
    ("r2_native", Path("/cb16/r2-native"), "MUTABLE_REFERENCE_ONLY"),
)
FORBIDDEN_COMPONENTS = {"2025-09", "final_holdout"}
CHUNK = 8 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def forbidden(rel: Path) -> bool:
    return any(part.lower() in FORBIDDEN_COMPONENTS for part in rel.parts)


def iter_tree(root: Path) -> Iterator[tuple[Path, os.stat_result]]:
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as it:
            entries = sorted(it, key=lambda e: e.name, reverse=True)
        for entry in entries:
            p = Path(entry.path)
            st = entry.stat(follow_symlinks=False)
            yield p, st
            if stat.S_ISDIR(st.st_mode) and not entry.is_symlink():
                stack.append(p)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main() -> int:
    started = time.monotonic()
    started_at = now()
    failures: list[str] = []
    root_summaries: list[dict] = []

    AUTHORITY.mkdir(parents=True, exist_ok=True)
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    inv_tmp = INVENTORY.with_name(INVENTORY.name + f".tmp.{os.getpid()}")
    inventory_hash = hashlib.sha256()

    with gzip.GzipFile(filename=str(inv_tmp), mode="wb", compresslevel=6, mtime=0) as gz:
        for name, root, role in ROOTS:
            summary = {
                "name": name,
                "path": str(root),
                "role": role,
                "exists": root.is_dir(),
                "writable": os.access(root, os.W_OK) if root.exists() else None,
                "files": 0,
                "directories": 0,
                "symlinks": 0,
                "other_nodes": 0,
                "bytes": 0,
                "hashed_files": 0,
                "forbidden_payload_files_skipped": 0,
                "forbidden_payload_bytes_skipped": 0,
                "scan_digest_sha256": None,
                "registration": "REFERENCE_ONLY" if role == "MUTABLE_REFERENCE_ONLY" else "ZERO_COPY_REFERENCE_REGISTERED",
            }
            if not root.is_dir():
                failures.append(f"LEGACY_ROOT_MISSING:{root}")
                root_summaries.append(summary)
                continue

            root_hash = hashlib.sha256()
            try:
                for path, st in iter_tree(root):
                    rel = path.relative_to(root)
                    mode = st.st_mode
                    if stat.S_ISDIR(mode):
                        summary["directories"] += 1
                        continue
                    if stat.S_ISLNK(mode):
                        summary["symlinks"] += 1
                        target = os.readlink(path)
                        row = {"root": name, "path": rel.as_posix(), "type": "symlink", "target": target}
                    elif stat.S_ISREG(mode):
                        summary["files"] += 1
                        summary["bytes"] += st.st_size
                        if forbidden(rel):
                            summary["forbidden_payload_files_skipped"] += 1
                            summary["forbidden_payload_bytes_skipped"] += st.st_size
                            row = {
                                "root": name,
                                "path": rel.as_posix(),
                                "type": "regular_excluded_final_holdout_name",
                                "size": st.st_size,
                                "content_opened": False,
                            }
                        else:
                            digest = sha256_file(path)
                            summary["hashed_files"] += 1
                            row = {"root": name, "path": rel.as_posix(), "type": "regular", "size": st.st_size, "sha256": digest}
                    else:
                        summary["other_nodes"] += 1
                        row = {"root": name, "path": rel.as_posix(), "type": "other", "mode": stat.S_IFMT(mode)}

                    line = canonical(row)
                    gz.write(line)
                    inventory_hash.update(line)
                    root_hash.update(line)
            except Exception as exc:
                failures.append(f"SCAN_FAILED:{name}:{type(exc).__name__}:{exc}")

            summary["scan_digest_sha256"] = root_hash.hexdigest()
            root_summaries.append(summary)

    os.replace(inv_tmp, INVENTORY)
    inventory_file_sha = sha256_file(INVENTORY)

    raw_seal = AUTHORITY / "CB16_R11_G0_DATASET_SEAL.json"
    raw_seal_summary = {"present": raw_seal.is_file(), "path": str(raw_seal), "payload_opened": False}
    if raw_seal.is_file():
        try:
            seal = json.loads(raw_seal.read_text(encoding="utf-8"))
            raw_seal_summary["canonical_seal_sha256"] = seal.get("canonical_seal_sha256")
        except Exception as exc:
            failures.append(f"RAW_SEAL_RECEIPT_UNREADABLE:{type(exc).__name__}:{exc}")

    receipt = {
        "schema": SCHEMA,
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "mode": "ZERO_COPY_EXISTING_MOUNT_REGISTRATION",
        "started_at_utc": started_at,
        "completed_at_utc": now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "roots": root_summaries,
        "inventory": {
            "path": str(INVENTORY),
            "compressed_sha256": inventory_file_sha,
            "canonical_rows_sha256": inventory_hash.hexdigest(),
            "format": "gzip_ndjson",
        },
        "existing_r11_raw_dataset_seal": raw_seal_summary,
        "semantics": {
            "legacy_bytes_copied": 0,
            "legacy_roots_mutated": False,
            "symlinks_followed": False,
            "new_scientific_evidence_created": False,
            "new_scientific_verdict": False,
            "training_started": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
        },
        "failures": failures,
    }
    atomic_write(RECEIPT, canonical(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if not failures else 78


if __name__ == "__main__":
    sys.exit(main())
