from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

from cb16_local_opt.stage4_authority_inventory_r11 import (
    Classification,
    GATEWORK_BASE_SHA,
    SCIENTIFIC_STATUS,
    SEMANTIC_FREEZE_BLOB_SHA,
    SEMANTIC_FREEZE_REL,
    audit_registry,
    discover_authority_surfaces,
    load_registry,
)
from cb16_local_opt.stage4_authority_registry_builder_r11 import (
    build_registry,
    classification_for_path,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "authority/rearchitecture_r11/CB16_R11_STAGE4_AUTHORITY_WRITER_REGISTRY_V1.json"


def _entry(registry: dict, classification: str) -> dict:
    return next(x for x in registry["entries"] if x["classification"] == classification)


def test_complete_registry_matches_current_discovery_and_passes_fail_closed_audit() -> None:
    registry = load_registry(REGISTRY)
    discovered = discover_authority_surfaces(ROOT)
    keys = [(x["path"], x["symbol"]) for x in registry["entries"]]
    assert len(keys) == len(set(keys))
    assert set(keys) == {x.key for x in discovered}
    assert registry == build_registry(ROOT)
    report = audit_registry(ROOT, registry, verify_git_guards=True)
    assert report["pass"], report["problems"]
    assert report["registered_writer_count"] == report["discovered_writer_count"]
    assert report["unknown_authority_count"] == 0


def test_unknown_authority_blocks_pass_for_missing_or_unreviewed_surface() -> None:
    registry = load_registry(REGISTRY)
    broken = copy.deepcopy(registry)
    broken["entries"][0]["classification"] = Classification.UNKNOWN_AUTHORITY.value
    report = audit_registry(ROOT, broken, verify_git_guards=False)
    assert not report["pass"]
    assert any(x.startswith("UNKNOWN_AUTHORITY:") for x in report["problems"])
    assert classification_for_path("cb16_local_opt/new_unreviewed_writer.py") == Classification.UNKNOWN_AUTHORITY.value


def test_qualification_and_test_writers_cannot_escalate_to_production_authority() -> None:
    registry = load_registry(REGISTRY)
    for classification in (Classification.QUALIFICATION_ONLY.value, Classification.TEST_ONLY.value):
        broken = copy.deepcopy(registry)
        target = _entry(broken, classification)
        target["current_role"] = "CANONICAL_RUNTIME_AUTHORITY"
        target["eligible_for_canonical_authority"] = True
        report = audit_registry(ROOT, broken, verify_git_guards=False)
        assert not report["pass"]
        assert any("NONPRODUCTION_WRITER_ESCALATION" in x for x in report["problems"])


def test_legacy_reachability_never_upgrades_legacy_to_authority() -> None:
    registry = load_registry(REGISTRY)
    legacy = _entry(registry, Classification.LEGACY_REFERENCE_ONLY.value)
    assert legacy["caller_callsite_evidence"]
    broken = copy.deepcopy(registry)
    target = next(
        x
        for x in broken["entries"]
        if (x["path"], x["symbol"]) == (legacy["path"], legacy["symbol"])
    )
    target["current_role"] = "PRODUCTION_AUTHORITY"
    target["eligible_for_canonical_authority"] = True
    report = audit_registry(ROOT, broken, verify_git_guards=False)
    assert not report["pass"]
    assert any("LEGACY_REACHABILITY_ESCALATION" in x for x in report["problems"])


def test_candidate_classification_is_not_a_stage4_canonical_authority_grant() -> None:
    registry = load_registry(REGISTRY)
    assert all(
        x["current_role"] != "CANONICAL_RUNTIME_AUTHORITY"
        for x in registry["entries"]
        if x["classification"] == Classification.R11_CANONICAL_CANDIDATE.value
    )
    broken = copy.deepcopy(registry)
    target = _entry(broken, Classification.R11_CANONICAL_CANDIDATE.value)
    target["current_role"] = "CANONICAL_RUNTIME_AUTHORITY"
    report = audit_registry(ROOT, broken, verify_git_guards=False)
    assert not report["pass"]
    assert any("PREMATURE_STAGE4_CANONICAL_CLAIM" in x for x in report["problems"])


def test_semantic_freeze_identity_is_byte_tree_unchanged() -> None:
    head_blob = subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{SEMANTIC_FREEZE_REL}"], cwd=ROOT, text=True
    ).strip()
    base_blob = subprocess.check_output(
        ["git", "rev-parse", f"{GATEWORK_BASE_SHA}:{SEMANTIC_FREEZE_REL}"], cwd=ROOT, text=True
    ).strip()
    assert head_blob == base_blob == SEMANTIC_FREEZE_BLOB_SHA


def test_inventory_is_administrative_only_and_touches_no_scientific_boundary() -> None:
    registry = load_registry(REGISTRY)
    report = audit_registry(ROOT, registry, verify_git_guards=True)
    assert report["scientific_status"] == SCIENTIFIC_STATUS
    assert report["scientific_semantics_changed"] is False
    assert report["new_scientific_verdict"] is False
    assert report["new_scientific_evidence_created"] is False
    assert report["final_holdout_opened"] is False
    assert report["fresh_market_data_downloaded"] is False
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{GATEWORK_BASE_SHA}...HEAD"], cwd=ROOT, text=True
    ).splitlines()
    assert changed
    assert all("stage4" in path.lower() for path in changed)
    assert all("2025-09" not in path.lower() and "final_holdout" not in path.lower() for path in changed)


def test_every_registered_surface_has_required_machine_readable_evidence() -> None:
    registry = load_registry(REGISTRY)
    for entry in registry["entries"]:
        assert entry["path"]
        assert entry["symbol"]
        assert entry["authority_domains"]
        assert entry["mutation_capability"]
        assert entry["caller_callsite_evidence"]
        assert entry["current_role"]
        assert entry["classification"] in {x.value for x in Classification}
        assert entry["classification"] != Classification.UNKNOWN_AUTHORITY.value
