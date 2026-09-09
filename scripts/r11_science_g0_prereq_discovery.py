#!/usr/bin/env python3
"""Metadata-only discovery for R11 G0 prerequisites in the Docker runner.

The Docker runner must not enumerate Shanxi host directories. Discovery is
restricted to explicitly mounted /cb16/* roots. Raw market ZIP payloads are
never opened; only path metadata and small authority/cache manifests are read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from cb16_local_opt.stage4_authority_adoption_r11 import load_adoption_receipt

RAW_ROOT = Path(os.environ.get("CB16_RAW_ROOT", "/cb16/raw"))
G0_ROOT = Path(os.environ.get("CB16_G0_ROOT", os.environ.get("CB16_R11_G0_ROOT", "/cb16/g0")))
R104_ROOT = Path(os.environ.get("CB16_R104_ROOT", "/cb16/runtime/r104"))
EXPECTED_RAW_SEAL = "94bc3288a7e6097c4301c93d836607edee207ac3b2c7cf3e7833c85ea0f9546a"
SYMBOL = "BTCUSDT"


def cache_probe(root: Path) -> dict | None:
    candidates = (root, root / "market_cache")
    for c in candidates:
        manifest = c / f"{SYMBOL}.manifest_r102.json"
        hourly = c / f"{SYMBOL}.hourly_r102.npz"
        anchors = c / f"{SYMBOL}.anchors_r102.npz"
        if manifest.is_file() and hourly.is_file() and anchors.is_file():
            obj = json.loads(manifest.read_text(encoding="utf-8"))
            return {
                "cache_root": str(c),
                "manifest_path": str(manifest),
                "hourly_path": str(hourly),
                "anchors_path": str(anchors),
                "manifest_schema": obj.get("schema"),
                "manifest_status": obj.get("status"),
                "forbidden_month_opened": obj.get("forbidden_month_opened"),
                "archive_boundary": obj.get("archive_boundary"),
                "hourly_sha256": obj.get("hourly_sha256"),
                "frames_sha256": obj.get("frames_sha256"),
                "stride_hours": obj.get("stride_hours"),
                "prehistory_hours": obj.get("prehistory_hours"),
                "first_hour": obj.get("first_hour"),
                "last_hour": obj.get("last_hour"),
                "first_anchor": obj.get("first_anchor"),
                "last_anchor": obj.get("last_anchor"),
            }
    return None


def adoption_probe() -> dict | None:
    receipt = G0_ROOT / "authority" / "CB16_R11_G0_AUTHORITY_ADOPTION_RECEIPT.json"
    if not receipt.is_file() or receipt.is_symlink():
        return None
    try:
        obj = load_adoption_receipt(receipt)
        source = obj["source"]
        return {
            "receipt_path": str(receipt),
            "verified": True,
            "canonical_content_hash": obj["canonical_content_hash"],
            "source_repo": source["source_repo"],
            "source_sha": source["source_sha"],
            "semantic_freeze_identity": source["semantic_freeze_identity"],
            "source_generation": source["source_generation"],
            "champion_identity": source["champion"]["identity"],
            "champion_sha256": source["champion"]["sha256"],
            "checkpoint_identity": source["checkpoint"]["identity"],
            "checkpoint_sha256": source["checkpoint"]["sha256"],
            "evidence_root_identity": source["evidence_root_identity"],
            "journal_head_identity": source["journal_head_identity"],
            "checkpoint_root_identity": source["checkpoint_root_identity"],
            "target_authority_identity": obj["target"]["r11_authority_identity"],
            "adoption_generation": obj["target"]["adoption_generation"],
        }
    except Exception as exc:
        return {"receipt_path": str(receipt), "verified": False, "error": str(exc)}


def main() -> int:
    for path, code in (
        (RAW_ROOT, "FROZEN_RAW_ROOT_INVALID"),
        (G0_ROOT, "R11_G0_ROOT_INVALID"),
        (R104_ROOT, "R104_REFERENCE_ROOT_INVALID"),
    ):
        if not path.is_dir() or path.is_symlink():
            raise SystemExit(f"{code}:{path}")
        if not str(path).startswith("/cb16/"):
            raise SystemExit(f"NON_DOCKER_ABSTRACT_ROOT:{path}")

    cache_roots = [G0_ROOT]
    explicit_cache = os.environ.get("CB16_G0_EXISTING_CACHE_ROOT")
    if explicit_cache:
        cache_roots.insert(0, Path(explicit_cache))

    cache_hits = []
    for root in cache_roots:
        if not str(root).startswith("/cb16/"):
            raise SystemExit(f"NON_DOCKER_CACHE_ROOT:{root}")
        hit = cache_probe(root)
        if hit and hit["cache_root"] not in {x["cache_root"] for x in cache_hits}:
            cache_hits.append(hit)

    adoption = adoption_probe()
    adoption_hits = [adoption] if adoption is not None else []

    result = {
        "schema": "CB16_R11_SCIENCE_G0_PREREQ_DISCOVERY_V2",
        "status": "DISCOVERED_FROM_DOCKER_VOLUMES",
        "scientific_verdict_created": False,
        "raw_payload_files_opened": 0,
        "network_reads": 0,
        "writes_under_raw_root": 0,
        "frozen_raw_access_root": str(RAW_ROOT),
        "r11_g0_root": str(G0_ROOT),
        "r104_reference_root": str(R104_ROOT),
        "expected_raw_dataset_seal_sha256": EXPECTED_RAW_SEAL,
        "cache_hits": cache_hits,
        "adoption_hits": adoption_hits,
        "cache_ready_for_lineage_canary": len(cache_hits) > 0,
        "verified_live_adoption_found": any(x.get("verified") is True for x in adoption_hits),
        "host_directory_enumeration_performed": False,
        "git_annex_used": False,
    }
    out = Path(os.environ.get("CB16_G0_PREREQ_OUT", "/tmp/cb16_r11_science_g0_prereq_discovery.json"))
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
