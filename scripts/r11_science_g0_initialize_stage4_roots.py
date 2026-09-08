#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from cb16_local_opt.stage4_state_roots_r11 import (
    SEMANTIC_FREEZE_BLOB_SHA,
    Stage4RootSetR11,
    Stage4StateRootsR11,
)

G0_ROOT = Path(os.environ.get("CB16_R11_G0_ROOT", "/cb16_r11_runtime/G0"))
RAW_ROOT = Path("/data/cb16_hdd/binance_usdm_1m_funding_2020_2026")
DATASET_SEAL_SHA256 = "94bc3288a7e6097c4301c93d836607edee207ac3b2c7cf3e7833c85ea0f9546a"
ADOPTION_CANONICAL_HASH = "2e9fed060ff29160197bf4a0ab391575fe15c4bb6e21f3ce368cfddd67ffab81"
GENESIS_IDENTITY_SHA256 = "dce5f5dc41b9b6ff712269fd0e3b87a861d2296147bd87ee62ff54e6813c798d"
SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"


def canonical_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def publish_exact_no_replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != raw:
            raise RuntimeError(f"R11_G0_STAGE4_ROOT_RECEIPT_CONFLICT:{path}")
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
                raise RuntimeError(f"R11_G0_STAGE4_ROOT_RECEIPT_RACE_CONFLICT:{path}")
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    authority = G0_ROOT / "authority"
    genesis_path = authority / "CB16_R11_G0_GENESIS_SOURCE_RECORD.json"
    adoption_path = authority / "CB16_R11_G0_AUTHORITY_ADOPTION_RECEIPT.json"
    dataset_seal_path = authority / "CB16_R11_G0_DATASET_SEAL.json"
    lineage_path = authority / "CB16_R11_G0_DERIVED_CACHE_LINEAGE.json"
    for p in (genesis_path, adoption_path, dataset_seal_path, lineage_path):
        if not p.is_file():
            raise SystemExit(f"R11_G0_REQUIRED_AUTHORITY_RECORD_MISSING:{p}")

    genesis = json.loads(genesis_path.read_text(encoding="utf-8"))
    adoption = json.loads(adoption_path.read_text(encoding="utf-8"))
    seal = json.loads(dataset_seal_path.read_text(encoding="utf-8"))
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if genesis.get("genesis_identity_sha256") != GENESIS_IDENTITY_SHA256:
        raise SystemExit("R11_G0_GENESIS_IDENTITY_MISMATCH")
    if adoption.get("canonical_content_hash") != ADOPTION_CANONICAL_HASH:
        raise SystemExit("R11_G0_ADOPTION_IDENTITY_MISMATCH")
    if adoption.get("target", {}).get("adoption_generation") != 0:
        raise SystemExit("R11_G0_ADOPTION_GENERATION_NOT_ZERO")
    if adoption.get("source", {}).get("semantic_freeze_identity") != SEMANTIC_FREEZE_BLOB_SHA:
        raise SystemExit("R11_G0_ADOPTION_FREEZE_MISMATCH")
    if seal.get("canonical_seal_sha256") != DATASET_SEAL_SHA256:
        raise SystemExit("R11_G0_DATASET_SEAL_MISMATCH")
    if lineage.get("dataset_seal_sha256") != DATASET_SEAL_SHA256 or lineage.get("status") != "STRONG_LINEAGE":
        raise SystemExit("R11_G0_STRONG_LINEAGE_MISMATCH")

    mode = stat.S_IMODE(RAW_ROOT.lstat().st_mode)
    write_bits = bool(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    control = G0_ROOT / "stage4" / "control"
    data = G0_ROOT / "stage4" / "data"
    out = Path("/tmp/cb16_r11_g0_stage4_roots.json")

    if write_bits:
        result = {
            "schema": "CB16_R11_G0_STAGE4_ROOT_INITIALIZATION_V1",
            "status": "BLOCKED_FROZEN_RAW_MODE_BITS",
            "raw_root": str(RAW_ROOT),
            "raw_root_mode_octal": oct(mode),
            "required_condition": "NO_WRITE_PERMISSION_BITS_ON_FROZEN_RAW_ROOT",
            "control_root_created": control.exists(),
            "data_root_created": data.exists(),
            "raw_files_modified": 0,
            "training_started": False,
            "final_holdout_payload_opened": False,
            "new_evidence_created": False,
            "new_scientific_verdict": False,
        }
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    roots = Stage4StateRootsR11(Stage4RootSetR11.from_paths(
        control_root=control,
        data_root=data,
        frozen_raw_root=RAW_ROOT,
        frozen_raw_identity=DATASET_SEAL_SHA256,
    ))
    receipt = roots.initialize()
    verified = roots.verify_startup(
        expected_control_id=receipt.control_root_id,
        expected_data_id=receipt.data_root_id,
    )
    if verified != receipt:
        raise SystemExit("R11_G0_STAGE4_ROOT_RESTART_VERIFICATION_MISMATCH")

    result = {
        "schema": "CB16_R11_G0_STAGE4_ROOT_INITIALIZATION_V1",
        "status": "R11_G0_STAGE4_ROOTS_READY",
        "r11_generation": 0,
        "target_authority": "CB16_R11_CANONICAL_RUNTIME",
        "semantic_freeze_git_blob": SEMANTIC_FREEZE_BLOB_SHA,
        "scientific_status": SCIENTIFIC_STATUS,
        "genesis_identity_sha256": GENESIS_IDENTITY_SHA256,
        "adoption_canonical_content_hash": ADOPTION_CANONICAL_HASH,
        "frozen_raw_root": str(RAW_ROOT),
        "frozen_raw_identity": DATASET_SEAL_SHA256,
        "control_root": str(control),
        "control_root_id": receipt.control_root_id,
        "data_root": str(data),
        "data_root_id": receipt.data_root_id,
        "sealed_object_count": receipt.sealed_object_count,
        "sealed_objects": [asdict(x) for x in receipt.sealed_objects],
        "raw_files_modified": 0,
        "training_started": False,
        "final_holdout_payload_opened": False,
        "new_evidence_created": False,
        "new_scientific_verdict": False,
        "generation_advanced": False,
    }
    raw = canonical_bytes(result)
    publish_exact_no_replace(authority / "CB16_R11_G0_STAGE4_ROOTS_RECEIPT.json", raw)
    out.write_bytes(raw)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
