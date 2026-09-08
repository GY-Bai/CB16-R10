#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from cb16_local_opt.binance_archive_input_r10 import BinanceUSDMArchiveSourceR10
from cb16_local_opt.r102_common import ALL_SUPPORTED_SYMBOLS_R102, sha256_file
from cb16_local_opt.r102_market import build_symbol_market_cache

EXPECTED_DATASET_SEAL = "94bc3288a7e6097c4301c93d836607edee207ac3b2c7cf3e7833c85ea0f9546a"
EXPECTED_BLOBS = {
    "cb16_local_opt/r102_market.py": "f3ea90f99565a36135da123ef0034057534eb40e",
    "cb16_local_opt/binance_archive_input_r10.py": "b0d6dc8599d47b8697d5775aabcf5f4029d6b775",
    "cb16_local_opt/r102_common.py": "87a28afe7a42aa3fadd999d1dc0acecd35c03a39",
}
RAW_ROOT = Path("/data/cb16_hdd/binance_usdm_1m_funding_2020_2026")
R11_ROOT = Path(os.environ.get("CB16_R11_G0_ROOT", "/home/bgy/cb16_ssd/runtime/R11/G0"))
SEAL_IN = Path(os.environ.get("CB16_G0_SEAL_IN", "/tmp/cb16_r11_science_g0_dataset_seal.json"))
STRIDE_HOURS = 256
PREHISTORY_HOURS = 96


def canonical_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob(path: str) -> str:
    return subprocess.check_output(["git", "hash-object", path], text=True).strip()


def publish_exact_no_replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != raw:
            raise RuntimeError(f"R11_G0_AUTHORITY_CONFLICT:{path}")
        return
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise RuntimeError(f"R11_G0_AUTHORITY_RACE_CONFLICT:{path}")
    finally:
        tmp.unlink(missing_ok=True)


def validate_seal() -> tuple[dict[str, Any], bytes]:
    obj = json.loads(SEAL_IN.read_text(encoding="utf-8"))
    if obj.get("schema") != "CB16_R11_HISTORICAL_MARKET_DATASET_SEAL_V1":
        raise RuntimeError("R11_G0_DATASET_SEAL_SCHEMA_MISMATCH")
    if obj.get("canonical_seal_sha256") != EXPECTED_DATASET_SEAL:
        raise RuntimeError(f"R11_G0_DATASET_SEAL_IDENTITY_MISMATCH:{obj.get('canonical_seal_sha256')}")
    if obj.get("dataset_root") != str(RAW_ROOT):
        raise RuntimeError("R11_G0_DATASET_ROOT_MISMATCH")
    if obj.get("unopened_holdout_start") != "2025-09-01T00:00:00Z":
        raise RuntimeError("R11_G0_HOLDOUT_BOUNDARY_MISMATCH")
    if obj.get("holdout_payload_files_opened") != 0 or obj.get("network_reads") != 0 or obj.get("writes_under_dataset_root") != 0:
        raise RuntimeError("R11_G0_DATASET_SEAL_SAFETY_VIOLATION")
    return obj, canonical_bytes(obj)


def main() -> int:
    _seal, seal_raw = validate_seal()
    for path, expected in EXPECTED_BLOBS.items():
        actual = git_blob(path)
        if actual != expected:
            raise SystemExit(f"R11_G0_BUILDER_BLOB_MISMATCH:{path}:{actual}:{expected}")

    authority_dir = R11_ROOT / "authority"
    cache_target = R11_ROOT / "market_cache"
    dataset_receipt = authority_dir / "CB16_R11_G0_DATASET_SEAL.json"
    lineage_receipt = authority_dir / "CB16_R11_G0_DERIVED_CACHE_LINEAGE.json"
    publish_exact_no_replace(dataset_receipt, seal_raw)

    if cache_target.exists():
        if not lineage_receipt.is_file():
            raise SystemExit("R11_G0_EXISTING_CACHE_WITHOUT_LINEAGE_RECEIPT")
        existing = json.loads(lineage_receipt.read_text(encoding="utf-8"))
        if existing.get("dataset_seal_sha256") != EXPECTED_DATASET_SEAL:
            raise SystemExit("R11_G0_EXISTING_CACHE_LINEAGE_SEAL_CONFLICT")
        print(json.dumps({"status": "ALREADY_MATERIALIZED", "lineage_receipt": str(lineage_receipt), "lineage_identity": existing.get("lineage_identity_sha256")}, indent=2))
        return 0

    R11_ROOT.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".g0-market-cache-build-", dir=R11_ROOT))
    source = BinanceUSDMArchiveSourceR10(RAW_ROOT)
    symbols = list(ALL_SUPPORTED_SYMBOLS_R102)
    if source.symbols() and not set(symbols).issubset(set(source.symbols())):
        raise SystemExit("R11_G0_RAW_SYMBOL_SET_INCOMPLETE")

    per_asset: list[dict[str, Any]] = []
    try:
        for symbol in symbols:
            paths = build_symbol_market_cache(
                source=source,
                symbol=symbol,
                out_dir=stage,
                stride_hours=STRIDE_HOURS,
                prehistory_hours=PREHISTORY_HOURS,
                verify_checksums=False,
            )
            manifest_path = Path(paths.manifest_json)
            hourly = Path(paths.hourly_npz)
            anchors = Path(paths.frames_npz)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("forbidden_month_opened") is not False:
                raise RuntimeError(f"R11_G0_HOLDOUT_GUARD_FAILED:{symbol}")
            per_asset.append({
                "symbol": symbol,
                "hourly_file": hourly.name,
                "hourly_sha256": sha256_file(hourly),
                "anchors_file": anchors.name,
                "anchors_sha256": sha256_file(anchors),
                "builder_manifest_file": manifest_path.name,
                "builder_manifest_sha256": sha256_file(manifest_path),
                "first_hour": manifest.get("first_hour"),
                "last_hour": manifest.get("last_hour"),
                "first_anchor": manifest.get("first_anchor"),
                "last_anchor": manifest.get("last_anchor"),
                "hourly_rows": manifest.get("hourly_rows"),
                "anchors": manifest.get("anchors"),
            })

        core = {
            "schema": "CB16_R11_G0_DERIVED_CACHE_LINEAGE_V1",
            "status": "STRONG_LINEAGE",
            "scientific_verdict_created": False,
            "raw_dataset_root": str(RAW_ROOT),
            "dataset_seal_sha256": EXPECTED_DATASET_SEAL,
            "dataset_seal_receipt": str(dataset_receipt),
            "builder_git_blobs": EXPECTED_BLOBS,
            "builder_entrypoint": "cb16_local_opt.r102_market.build_symbol_market_cache",
            "symbols": symbols,
            "stride_hours": STRIDE_HOURS,
            "prehistory_hours": PREHISTORY_HOURS,
            "final_holdout_start": "2025-09-01T00:00:00Z",
            "final_holdout_payload_opened": False,
            "network_reads": 0,
            "source_repairs_or_downloads": 0,
            "per_asset": per_asset,
        }
        lineage_id = sha256_bytes(canonical_bytes(core))
        receipt = {**core, "lineage_identity_sha256": lineage_id}
        (stage / "CB16_R11_G0_DERIVED_CACHE_LINEAGE.json").write_bytes(canonical_bytes(receipt))

        os.rename(stage, cache_target)
        publish_exact_no_replace(lineage_receipt, canonical_bytes(receipt))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(json.dumps({
        "status": "MATERIALIZED",
        "dataset_seal_sha256": EXPECTED_DATASET_SEAL,
        "lineage_identity_sha256": lineage_id,
        "cache_root": str(cache_target),
        "asset_count": len(per_asset),
        "final_holdout_payload_opened": False,
        "network_reads": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
