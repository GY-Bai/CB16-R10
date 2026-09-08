from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from cb16_local_opt.stage4_hostile_cutover_r11 import (
    INVARIANT_KEYS,
    REQUIRED_CASES,
    SCIENTIFIC_STATUS,
    SEMANTIC_FREEZE_BLOB,
    matrix_manifest,
    run_hostile_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "authority/rearchitecture_r11/CB16_R11_STAGE4_HOSTILE_CUTOVER_REPORT_SCHEMA_V1.json"
MODULE = ROOT / "cb16_local_opt/stage4_hostile_cutover_r11.py"
FREEZE = "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"


def test_stage4_hostile_matrix_contains_all_required_cases_once():
    ids = [x.case_id for x in REQUIRED_CASES]
    assert ids == [f"H{i:02d}" for i in range(1, 21)]
    assert len(ids) == len(set(ids)) == 20
    manifest = matrix_manifest()
    assert manifest["integrated_runtime_qualification_claimed"] is False
    assert manifest["scientific_status"] == SCIENTIFIC_STATUS


def test_stage4_reference_hostile_matrix_passes_all_invariants():
    report = run_hostile_matrix()
    assert report["status"] == "PASS"
    assert report["harness_qualified"] is True
    assert report["integrated_runtime_qualified"] is False
    assert report["case_count"] == report["passed_cases"] == 20
    assert report["failed_cases"] == 0
    for case in report["cases"]:
        assert case["passed"] is True, case
        assert set(case["invariants"]) == set(INVARIANT_KEYS)
        assert all(case["invariants"].values()), case


def test_stage4_hostile_matrix_is_deterministic():
    assert run_hostile_matrix() == run_hostile_matrix()


def test_stage4_hostile_matrix_rejects_expected_attack_classes():
    report = run_hostile_matrix()
    by_id = {row["case_id"]: row for row in report["cases"]}
    expected = {
        "H01": "DUPLICATE_CANONICAL_RUNTIME",
        "H03": "STALE_OR_INVALID_FENCE",
        "H04": "CONFLICTING_AUTHORITY_ADOPTION",
        "H06": "LEGACY_OR_NONCANONICAL_AUTHORITY_DENIED",
        "H13": "STALE_CHAMPION_WRITER",
        "H14": "PHYSICS_PERMISSION_REQUIRED",
        "H16": "REPLAY_IS_ENGINEERING_ONLY_NOT_NEW_EVIDENCE",
        "H20": "AUTHORITY_METADATA_CORRUPT",
    }
    for case_id, code in expected.items():
        assert code in by_id[case_id]["observed"]


def test_stage4_harness_does_not_import_sibling_stage4_modules():
    source = MODULE.read_text(encoding="utf-8")
    assert "from .stage4_" not in source
    assert "from cb16_local_opt.stage4_" not in source
    assert "import cb16_local_opt.stage4_" not in source


def test_stage4_report_schema_is_fail_closed_about_final_qualification():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    required = set(schema["required"])
    for key in {
        "status",
        "harness_qualified",
        "integrated_runtime_qualified",
        "scientific_status",
        "cases",
    }:
        assert key in required
    assert schema["properties"]["integrated_runtime_qualified"]["const"] is False
    assert schema["properties"]["scientific_status"]["const"] == SCIENTIFIC_STATUS


def test_stage4_cli_runs_reference_matrix(tmp_path: Path):
    output = tmp_path / "stage4h-report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_r11_stage4_hostile_cutover_matrix.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["integrated_runtime_qualified"] is False


def test_stage4_semantic_freeze_identity_constant_matches_repository_when_git_available():
    completed = subprocess.run(
        ["git", "rev-parse", f"HEAD:{FREEZE}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        assert completed.stdout.strip() == SEMANTIC_FREEZE_BLOB
