#!/usr/bin/env python3
"""Read-only legacy-root scan and zero-copy registration into R11 G0 authority.

Every readable regular file is SHA256 hashed. Symlinks are never followed and
final-holdout-named paths are metadata-only. A small explicit set of historically
private external-model runtime weights may remain opaque references when the R11
runner cannot read them; they are never promoted to scientific authority.
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

SCHEMA = "CB16_R11_LEGACY_DATA_ADOPTION_V2"
G0 = Path(os.environ.get("CB16_G0_ROOT", "/cb16/g0"))
AUTHORITY = G0 / "authority"
INVENTORY_DIR = AUTHORITY / "legacy_inventory"
INVENTORY = INVENTORY_DIR / "CB16_R11_LEGACY_FILE_INVENTORY_V2.ndjson.gz"
RECEIPT = AUTHORITY / "CB16_R11_LEGACY_DATA_ADOPTION_V2.json"

ROOTS = (
    ("r104", Path("/cb16/runtime/r104"), "IMMUTABLE_REFERENCE"),
    ("package", Path("/cb16/package"), "IMMUTABLE_REFERENCE"),
    ("parent_r101", Path("/cb16/parent-r101"), "IMMUTABLE_REFERENCE"),
    ("parents", Path("/cb16/parents"), "IMMUTABLE_REFERENCE"),
    ("r2_authority", Path("/cb16/r2-authority"), "IMMUTABLE_REFERENCE"),
    ("r2_native", Path("/cb16/r2-native"), "MUTABLE_REFERENCE_ONLY"),
)
FORBIDDEN_COMPONENTS = {"2025-09", "final_holdout"}
OPAQUE_PRIVATE_ALLOWLIST = {
    ("parent_r101", "assets/medium/runtime/timesfm_layer3.safetensors"): "PRIVATE_EXTERNAL_MODEL_REFERENCE",
    ("parent_r101", "assets/operator/runtime/kronos_tokenizer_encode.safetensors"): "PRIVATE_EXTERNAL_MODEL_REFERENCE",
    ("parent_r101", "assets/operator/runtime/kronos_model_l5.safetensors"): "PRIVATE_EXTERNAL_MODEL_REFERENCE",
}
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
    opaque_private: list[dict] = []
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
                "opaque_private_files": 0,
                "unapproved_unreadable_files": 0,
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
                iterator = iter_tree(root)
                for path, st in iterator:
                    rel = path.relative_to(root)
                    rel_s = rel.as_posix()
                    mode = st.st_mode
                    if stat.S_ISDIR(mode):
                        summary["directories"] += 1
                        continue
                    if stat.S_ISLNK(mode):
                        summary["symlinks"] += 1
                        row = {"root": name, "path": rel_s, "type": "symlink", "target": os.readlink(path)}
                    elif stat.S_ISREG(mode):
                        summary["files"] += 1
                        summary["bytes"] += st.st_size
                        if forbidden(rel):
                            summary["forbidden_payload_files_skipped"] += 1
                            summary["forbidden_payload_bytes_skipped"] += st.st_size
                            row = {"root": name, "path": rel_s, "type": "regular_excluded_final_holdout_name", "size": st.st_size, "content_opened": False}
                        else:
                            try:
                                digest = sha256_file(path)
                            except PermissionError as exc:
                                opaque_key = (name, rel_s)
                                metadata = {
                                    "root": name,
                                    "path": rel_s,
                                    "size": st.st_size,
                                    "mode_octal": oct(stat.S_IMODE(st.st_mode)),
                                    "uid": st.st_uid,
                                    "gid": st.st_gid,
                                    "errno": exc.errno,
                                    "content_opened": False,
                                }
                                if opaque_key in OPAQUE_PRIVATE_ALLOWLIST:
                                    summary["opaque_private_files"] += 1
                                    summary["registration"] = "ZERO_COPY_REFERENCE_WITH_OPAQUE_PRIVATE_MODEL"
                                    metadata["classification"] = OPAQUE_PRIVATE_ALLOWLIST[opaque_key]
                                    opaque_private.append(dict(metadata))
                                    row = dict(metadata, type="opaque_private_model_reference")
                                else:
                                    summary["unapproved_unreadable_files"] += 1
                                    failures.append(f"UNAPPROVED_UNREADABLE_FILE:{name}:{rel_s}:errno={exc.errno}")
                                    row = dict(metadata, type="unapproved_unreadable_regular")
                            else:
                                summary["hashed_files"] += 1
                                row = {"root": name, "path": rel_s, "type": "regular", "size": st.st_size, "sha256": digest}
                    else:
                        summary["other_nodes"] += 1
                        row = {"root": name, "path": rel_s, "type": "other", "mode": stat.S_IFMT(mode)}

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

    if failures:
        status = "FAIL_CLOSED"
    elif opaque_private:
        status = "PASS_WITH_OPAQUE_PRIVATE_MODEL_REFERENCE"
    else:
        status = "PASS"

    receipt = {
        "schema": SCHEMA,
        "status": status,
        "mode": "ZERO_COPY_EXISTING_MOUNT_REGISTRATION",
        "started_at_utc": started_at,
        "completed_at_utc": now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "roots": root_summaries,
        "opaque_private_model_references": opaque_private,
        "inventory": {"path": str(INVENTORY), "compressed_sha256": inventory_file_sha, "canonical_rows_sha256": inventory_hash.hexdigest(), "format": "gzip_ndjson"},
        "existing_r11_raw_dataset_seal": raw_seal_summary,
        "semantics": {
            "legacy_bytes_copied": 0,
            "legacy_roots_mutated": False,
            "symlinks_followed": False,
            "opaque_private_model_promoted_to_authority": False,
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
