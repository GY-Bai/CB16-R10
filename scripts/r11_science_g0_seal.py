#!/usr/bin/env python3
"""Materialize the R11 canonical seal for the frozen Shanxi historical dataset.

Safety law from CB16_SEMANTIC_FREEZE_V1:
- archives strictly before 2025-09: hash archive bytes and verify official sidecar;
- archives at/after 2025-09: NEVER open payload; bind path, size, and official sidecar only;
- never mutate, repair, download, or append market data.

Docker migration note: CB16_RAW_ROOT is the physical access path. The frozen
semantic contract's expected_root remains the canonical identity path embedded
in the seal, so relocating access through /cb16/raw cannot rewrite seal identity.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = REPO_ROOT / "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"
MONTH_RE = re.compile(r"-(20\d{2})-([01]\d)\.zip$")
HEX64_RE = re.compile(r"\b([0-9a-fA-F]{64})\b")
HOLDOUT_MONTH = "2025-09"
ALLOWED_PAYLOAD_ROOTS = {"klines_1m", "fundingRate"}


def sha256_file(path: Path, *, allow_payload_month: str) -> str:
    if allow_payload_month >= HOLDOUT_MONTH:
        raise RuntimeError(f"HOLDOUT_PAYLOAD_OPEN_BLOCKED:{path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_month(path: Path) -> str:
    m = MONTH_RE.search(path.name)
    if not m:
        raise RuntimeError(f"ARCHIVE_MONTH_UNRESOLVED:{path}")
    month = f"{m.group(1)}-{m.group(2)}"
    if not (1 <= int(m.group(2)) <= 12):
        raise RuntimeError(f"ARCHIVE_MONTH_INVALID:{path}")
    return month


def parse_sidecar(sidecar: Path) -> tuple[str, str]:
    raw = sidecar.read_bytes()  # sidecars are metadata, never market payload.
    text = raw.decode("utf-8", errors="strict")
    m = HEX64_RE.search(text)
    if not m:
        raise RuntimeError(f"SIDECAR_SHA256_MISSING:{sidecar}")
    return m.group(1).lower(), hashlib.sha256(raw).hexdigest()


def canonical_bytes(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def main() -> int:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    ds = freeze["immutable"]["historical_market_dataset"]
    identity_root = Path(ds["expected_root"])
    root = Path(os.environ.get("CB16_RAW_ROOT", str(identity_root)))
    symbols = set(ds["symbols"])

    if not root.is_dir() or root.is_symlink():
        raise SystemExit("DATASET_ROOT_INVALID")
    if ds["mutation"] != "FORBIDDEN":
        raise SystemExit("MUTATION_POLICY_NOT_FROZEN")
    if ds["network_repair_or_redownload_during_scientific_runtime"] != "FORBIDDEN":
        raise SystemExit("NETWORK_REPAIR_POLICY_NOT_FROZEN")
    if ds["unopened_holdout_start"] != "2025-09-01T00:00:00Z":
        raise SystemExit("HOLDOUT_BOUNDARY_MISMATCH")

    all_regular: dict[str, tuple[Path, int]] = {}
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort(); filenames.sort()
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            rel = p.relative_to(root).as_posix()
            st = p.lstat()
            if stat.S_ISLNK(st.st_mode):
                raise SystemExit(f"DATASET_SYMLINK_FORBIDDEN:{rel}")
            if stat.S_ISREG(st.st_mode):
                all_regular[rel] = (p, int(st.st_size))
                total_bytes += int(st.st_size)

    if total_bytes != int(ds["discovered_size_bytes"]):
        raise SystemExit(f"DATASET_SIZE_IDENTITY_MISMATCH:{total_bytes}")

    archives = sorted(rel for rel in all_regular if rel.endswith(".zip"))
    sidecars = {rel for rel in all_regular if rel.endswith(".zip.CHECKSUM")}
    expected_sidecars = {rel + ".CHECKSUM" for rel in archives}
    if sidecars != expected_sidecars:
        missing = sorted(expected_sidecars - sidecars)
        extra = sorted(sidecars - expected_sidecars)
        raise SystemExit(f"ARCHIVE_SIDECAR_TOPOLOGY_MISMATCH:missing={missing[:5]}:extra={extra[:5]}")

    entries: list[dict] = []
    history_count = 0
    holdout_count = 0
    history_bytes_read = 0
    holdout_payload_files_opened = 0

    for rel in archives:
        p, size = all_regular[rel]
        parts = Path(rel).parts
        if len(parts) < 3 or parts[0] not in ALLOWED_PAYLOAD_ROOTS or parts[1] not in symbols:
            raise SystemExit(f"ARCHIVE_PATH_OUTSIDE_FROZEN_GEOMETRY:{rel}")
        month = parse_month(p)
        side_rel = rel + ".CHECKSUM"
        side_path, side_size = all_regular[side_rel]
        official_expected, sidecar_sha = parse_sidecar(side_path)
        entry = {
            "archive_path": rel,
            "archive_size": size,
            "month": month,
            "symbol": parts[1],
            "data_class": parts[0],
            "sidecar_path": side_rel,
            "sidecar_size": side_size,
            "sidecar_sha256": sidecar_sha,
            "official_expected_archive_sha256": official_expected,
        }
        if month < HOLDOUT_MONTH:
            actual = sha256_file(p, allow_payload_month=month)
            history_bytes_read += size
            if actual != official_expected:
                raise SystemExit(f"HISTORICAL_ARCHIVE_OFFICIAL_CHECKSUM_MISMATCH:{rel}")
            entry["authority_class"] = "CONSUMED_HISTORY"
            entry["archive_sha256"] = actual
            history_count += 1
        else:
            # Deliberately no open/read/hash of p here.
            entry["authority_class"] = "UNOPENED_FINAL_HOLDOUT"
            entry["archive_sha256"] = None
            holdout_count += 1
        entries.append(entry)

    paired_paths = set(archives) | expected_sidecars
    support_files = [
        {"path": rel, "size": all_regular[rel][1]}
        for rel in sorted(set(all_regular) - paired_paths)
    ]

    # Preserve the frozen authority identity even when Docker exposes the same bytes
    # through a different physical access path.
    identity_realpath = str(identity_root.resolve(strict=False))
    seal_core = {
        "schema": "CB16_R11_HISTORICAL_MARKET_DATASET_SEAL_V1",
        "status": "SEALED",
        "scientific_verdict_created": False,
        "dataset_root": str(identity_root),
        "dataset_root_realpath": identity_realpath,
        "storage_class": ds["storage_class"],
        "kline_interval": ds["kline_interval"],
        "symbols": ds["symbols"],
        "unopened_holdout_start": ds["unopened_holdout_start"],
        "provision_manifest_sha256": ds["provision_manifest_sha256"],
        "total_regular_file_bytes": total_bytes,
        "archive_count": len(archives),
        "historical_archive_count": history_count,
        "unopened_holdout_archive_count": holdout_count,
        "historical_payload_bytes_hashed": history_bytes_read,
        "holdout_payload_files_opened": holdout_payload_files_opened,
        "network_reads": 0,
        "writes_under_dataset_root": 0,
        "archive_entries": entries,
        "non_archive_support_files_metadata_only": support_files,
    }
    digest = hashlib.sha256(canonical_bytes(seal_core)).hexdigest()
    result = dict(seal_core)
    result["canonical_seal_sha256"] = digest

    out = Path(os.environ.get("CB16_G0_SEAL_OUT", "/tmp/cb16_r11_science_g0_dataset_seal.json"))
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "archive_entries"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
