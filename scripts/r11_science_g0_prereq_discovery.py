#!/usr/bin/env python3
"""Metadata-only discovery for R11 G0 prerequisites on the Shanxi host.

This script never opens raw market ZIP payloads. It performs only bounded path checks,
reads derived-cache manifests and, if present, verifies a Stage-4 adoption receipt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from cb16_local_opt.stage4_authority_adoption_r11 import load_adoption_receipt

RAW_ROOT = Path("/data/cb16_hdd/binance_usdm_1m_funding_2020_2026")
SSD_ROOT = Path("/data/cb16_ssd")
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
                "cache_root": str(c.resolve()),
                "manifest_path": str(manifest.resolve()),
                "hourly_path": str(hourly.resolve()),
                "anchors_path": str(anchors.resolve()),
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


def bounded_ssd_children() -> list[Path]:
    out: list[Path] = []
    if not SSD_ROOT.is_dir():
        return out
    for p in sorted(SSD_ROOT.iterdir(), key=lambda x: x.name):
        if p.is_dir() and not p.is_symlink():
            out.append(p)
    return out


def adoption_candidates(children: list[Path]) -> list[Path]:
    explicit = []
    for key in ("CB16_STAGE4_CONTROL_ROOT", "CB16_R11_CONTROL_ROOT", "CB16_CONTROL_ROOT"):
        value = os.environ.get(key)
        if value:
            explicit.append(Path(value))
    roots = explicit + children
    names = (
        Path("stage4_authority_adoption_r11.json"),
        Path("control/stage4_authority_adoption_r11.json"),
        Path("control/adoption_control_metadata/stage4_authority_adoption_r11.json"),
        Path("adoption_control_metadata/stage4_authority_adoption_r11.json"),
    )
    seen: set[str] = set()
    result: list[Path] = []
    for root in roots:
        for rel in names:
            p = root / rel
            s = str(p)
            if s not in seen:
                seen.add(s)
                result.append(p)
    return result


def main() -> int:
    if not RAW_ROOT.is_dir() or RAW_ROOT.is_symlink():
        raise SystemExit("FROZEN_RAW_ROOT_INVALID")

    children = bounded_ssd_children()
    cache_hits = []
    env_cache = os.environ.get("CB16_G0_EXISTING_CACHE_ROOT")
    probe_roots = ([Path(env_cache)] if env_cache else []) + children
    for root in probe_roots:
        hit = cache_probe(root)
        if hit and hit["cache_root"] not in {x["cache_root"] for x in cache_hits}:
            cache_hits.append(hit)

    adoption_hits = []
    for p in adoption_candidates(children):
        if not p.is_file() or p.is_symlink():
            continue
        try:
            obj = load_adoption_receipt(p)
            source = obj["source"]
            adoption_hits.append({
                "receipt_path": str(p.resolve()),
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
            })
        except Exception as exc:
            adoption_hits.append({"receipt_path": str(p.resolve()), "verified": False, "error": str(exc)})

    result = {
        "schema": "CB16_R11_SCIENCE_G0_PREREQ_DISCOVERY_V1",
        "status": "DISCOVERED",
        "scientific_verdict_created": False,
        "raw_payload_files_opened": 0,
        "network_reads": 0,
        "writes_under_raw_root": 0,
        "frozen_raw_root": str(RAW_ROOT),
        "expected_raw_dataset_seal_sha256": EXPECTED_RAW_SEAL,
        "ssd_top_level_directory_names": [p.name for p in children],
        "cache_hits": cache_hits,
        "adoption_hits": adoption_hits,
        "cache_ready_for_lineage_canary": len(cache_hits) > 0,
        "verified_live_adoption_found": any(x.get("verified") is True for x in adoption_hits),
    }
    out = Path(os.environ.get("CB16_G0_PREREQ_OUT", "/tmp/cb16_r11_science_g0_prereq_discovery.json"))
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
