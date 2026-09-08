#!/usr/bin/env python3
"""Metadata-only inventory for the frozen Shanxi 10-asset 1m dataset.

This script MUST NOT open market payload bytes.  It performs only directory
enumeration and lstat/stat metadata reads so the unopened 2025-09+ holdout
remains unopened.  Its output is engineering evidence for constructing the
R11 historical-data seal; it is not scientific Evidence and not a verdict.
"""
from __future__ import annotations

import json
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = REPO_ROOT / "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"
EXPECTED_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"
DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-_/]?([01]\d)(?!\d)")


def main() -> int:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    ds = freeze["immutable"]["historical_market_dataset"]
    root = Path(ds["expected_root"])
    symbols = tuple(ds["symbols"])

    if ds["storage_class"] != "SHANXI_LOCAL_READ_ONLY":
        raise SystemExit("DATASET_STORAGE_CLASS_MISMATCH")
    if ds["mutation"] != "FORBIDDEN":
        raise SystemExit("DATASET_MUTATION_POLICY_NOT_FORBIDDEN")
    if ds["network_repair_or_redownload_during_scientific_runtime"] != "FORBIDDEN":
        raise SystemExit("NETWORK_REPAIR_POLICY_NOT_FORBIDDEN")
    if ds["unopened_holdout_start"] != "2025-09-01T00:00:00Z":
        raise SystemExit("HOLDOUT_BOUNDARY_MISMATCH")
    if ds["kline_interval"] != "1m" or len(symbols) != 10:
        raise SystemExit("DATASET_GEOMETRY_MISMATCH")
    if not root.is_dir():
        raise SystemExit(f"DATASET_ROOT_MISSING:{root}")
    if root.is_symlink():
        raise SystemExit("DATASET_ROOT_SYMLINK_FORBIDDEN")

    records: list[dict] = []
    suffix_counts: Counter[str] = Counter()
    month_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    suspicious_symlinks: list[str] = []
    total_file_bytes = 0

    # os.walk/lstat only.  NEVER open(), read_bytes(), mmap(), hash payloads, etc.
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        base = Path(dirpath)
        for name in filenames:
            path = base / name
            rel = path.relative_to(root).as_posix()
            st = path.lstat()
            if stat.S_ISLNK(st.st_mode):
                suspicious_symlinks.append(rel)
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            size = int(st.st_size)
            total_file_bytes += size
            suffix_counts[path.suffix.lower() or "<none>"] += 1
            for symbol in symbols:
                if symbol in rel.upper():
                    symbol_counts[symbol] += 1
            for y, m in DATE_RE.findall(rel):
                if 1 <= int(m) <= 12:
                    month_counts[f"{y}-{m}"] += 1
            records.append({"path": rel, "size": size})

    if suspicious_symlinks:
        raise SystemExit("DATASET_SYMLINK_PAYLOAD_FORBIDDEN:" + ",".join(suspicious_symlinks[:20]))
    if not records:
        raise SystemExit("DATASET_EMPTY")

    holdout_named = [r for r in records if any(k >= "2025-09" for k in DATE_RE_MONTHS(r["path"]))]
    pre_holdout_named = [r for r in records if any(k < "2025-09" for k in DATE_RE_MONTHS(r["path"]))]
    sidecars = [r for r in records if r["path"].lower().endswith((".sha256", ".sha256sum", ".checksum", ".checksums")) or "sha256" in Path(r["path"]).name.lower()]

    result = {
        "schema": "CB16_R11_SCIENCE_G0_DATASET_METADATA_INVENTORY_V1",
        "status": "METADATA_ONLY_INVENTORY_PASS",
        "scientific_evidence": False,
        "scientific_verdict_created": False,
        "payload_files_opened": 0,
        "holdout_payload_files_opened": 0,
        "network_reads": 0,
        "writes_under_dataset_root": 0,
        "dataset_root": str(root),
        "dataset_root_realpath": str(root.resolve()),
        "expected_discovered_size_bytes": int(ds["discovered_size_bytes"]),
        "regular_file_count": len(records),
        "total_regular_file_bytes": total_file_bytes,
        "symbols_expected": list(symbols),
        "symbol_path_occurrence_counts": {s: symbol_counts[s] for s in symbols},
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "month_token_counts": dict(sorted(month_counts.items())),
        "pre_holdout_named_file_count": len(pre_holdout_named),
        "holdout_or_later_named_file_count": len(holdout_named),
        "checksum_sidecar_count": len(sidecars),
        "checksum_sidecar_paths": [r["path"] for r in sidecars],
        "file_metadata": records,
        "next_step": "BUILD_CANONICAL_SEAL_WITH_PAYLOAD_HASHES_ONLY_FOR_PRE_2025_09_AND_SIDECAR_BINDINGS_ONLY_FOR_2025_09_PLUS",
    }

    out = Path(os.environ.get("CB16_G0_INVENTORY_OUT", "/tmp/cb16_r11_science_g0_inventory.json"))
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "file_metadata"}, indent=2, sort_keys=True))
    return 0


def DATE_RE_MONTHS(text: str) -> list[str]:
    months: list[str] = []
    for y, m in DATE_RE.findall(text):
        if 1 <= int(m) <= 12:
            months.append(f"{y}-{m}")
    return months


if __name__ == "__main__":
    raise SystemExit(main())
