from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "authority/rearchitecture_r11/CB16_R11_STAGE4_GATEWORK_V1.json"
SCHEMA = ROOT / "authority/rearchitecture_r11/CB16_R11_STAGE4_TASK_RECEIPT_SCHEMA_V1.json"
FREEZE = "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"
EXPECTED_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"


def test_stage4_gatework_manifest_is_fail_closed_and_parallel():
    m = json.loads(MANIFEST.read_text())
    assert m["schema"] == "CB16_R11_STAGE4_GATEWORK_V1"
    assert m["status"] == "FROZEN_FOR_PARALLEL_TASKS"
    assert m["upstream"]["head_sha"] == "35d6dccd85fe053f4d14d1d43b3110496947cdc1"
    assert m["upstream"]["stage3_integrated_smoke_run"] == 34184028833
    assert m["scientific_authority"]["new_scientific_verdict_allowed"] is False
    assert m["scientific_authority"]["final_holdout_opening_allowed"] is False
    assert m["scientific_authority"]["fresh_market_data_allowed"] is False
    assert m["parallelism_contract"]["wave1_sibling_branch_reads_for_implementation"] is False
    assert m["parallelism_contract"]["wave1_sibling_cherry_pick_allowed"] is False
    assert m["parallelism_contract"]["wave1_sibling_new_module_import_allowed"] is False
    assert m["parallelism_contract"]["each_task_must_be_independently_testable"] is True
    ids = [x["id"] for x in m["wave1_tasks"]]
    branches = [x["branch"] for x in m["wave1_tasks"]]
    assert ids == ["S4A", "S4B", "S4C", "S4D", "S4E", "S4F", "S4G", "S4H", "S4I"]
    assert len(branches) == len(set(branches))
    assert "2h or 6h endurance qualification" in m["scope"]["out_of_scope"]
    assert "distributed or cross-machine authority" in m["scope"]["out_of_scope"]


def test_stage4_receipt_schema_requires_scientific_guards():
    s = json.loads(SCHEMA.read_text())
    assert s["properties"]["schema"]["const"] == "CB16_R11_STAGE4_TASK_RECEIPT_V1"
    guards = s["properties"]["semantic_guards"]
    required = set(guards["required"])
    for key in {
        "semantic_freeze_unchanged",
        "final_holdout_untouched",
        "fresh_market_data_downloaded",
        "historical_market_data_mutated",
        "new_scientific_verdict",
        "scientific_semantics_changed",
        "replay_reinterpreted_as_new_evidence",
        "sibling_dependency_used",
    }:
        assert key in required


def test_stage4_gatework_keeps_semantic_freeze_blob_exact():
    blob = subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{FREEZE}"], cwd=ROOT, text=True
    ).strip()
    assert blob == EXPECTED_FREEZE_BLOB
