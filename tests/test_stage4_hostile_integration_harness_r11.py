from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from cb16_local_opt.stage4_hostile_cutover_r11 import REQUIRED_CASES
from cb16_local_opt.stage4_hostile_integration_harness_r11 import (
    FAULT_BOUNDARIES,
    INTEGRATION_SEED_SHA,
    NoopHostileFaultHooks,
    ReferenceProductionHostileAdapter,
    SEMANTIC_FREEZE_BLOB,
    SCIENTIFIC_STATUS,
    assert_s4h_contract_exact,
    integration_binding_manifest,
    run_integrated_hostile_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "cb16_local_opt/stage4_hostile_integration_harness_r11.py"
SCRIPT = ROOT / "scripts/run_r11_stage4_integrated_hostile_matrix.py"
SCHEMA = ROOT / "authority/rearchitecture_r11/CB16_R11_STAGE4_HOSTILE_CUTOVER_REPORT_SCHEMA_V1.json"
FREEZE = "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"


class RecordingHooks(NoopHostileFaultHooks):
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def before_boundary(self, spec, adapter) -> None:
        del adapter
        self.events.append(("before", spec.case_id))

    def after_boundary(self, spec, adapter) -> None:
        del adapter
        self.events.append(("after", spec.case_id))


class ReplayViolationAdapter(ReferenceProductionHostileAdapter):
    def snapshot(self):
        snap = super().snapshot()
        return replace(snap, replay_evidence_admitted=1)


def test_intg_preserves_exact_qualified_s4h_case_meanings_and_order():
    assert_s4h_contract_exact()
    assert [x.case_id for x in REQUIRED_CASES] == [f"H{i:02d}" for i in range(1, 21)]
    assert [x.case_id for x in FAULT_BOUNDARIES] == [x.case_id for x in REQUIRED_CASES]
    assert len(FAULT_BOUNDARIES) == 20


def test_intg_reference_bridge_passes_all_twenty_without_claiming_runtime_qualification():
    report = run_integrated_hostile_matrix()
    assert report["status"] == "PASS"
    assert report["harness_qualified"] is True
    assert report["integrated_runtime_qualified"] is False
    assert report["case_count"] == report["passed_cases"] == 20
    assert report["failed_cases"] == 0
    assert report["scientific_status"] == SCIENTIFIC_STATUS
    assert report["semantic_freeze_blob"] == SEMANTIC_FREEZE_BLOB
    assert all(row["passed"] for row in report["cases"])


def test_intg_reference_bridge_is_deterministic():
    assert run_integrated_hostile_matrix() == run_integrated_hostile_matrix()


def test_intg_expected_fail_closed_attack_codes_remain_visible():
    by_id = {row["case_id"]: row for row in run_integrated_hostile_matrix()["cases"]}
    expected = {
        "H01": "DUPLICATE_CANONICAL_RUNTIME",
        "H02": "RUNTIME_NOT_ACTIVE",
        "H03": "STALE_OR_INVALID_FENCE",
        "H04": "CONFLICTING_AUTHORITY_ADOPTION",
        "H06": "LEGACY_OR_NONCANONICAL_AUTHORITY_DENIED",
        "H09": "NO_PENDING_ADOPTION",
        "H10": "NO_PENDING_JOURNAL_TRANSITION",
        "H11": "DUPLICATE_COMMIT_CONFLICT",
        "H13": "STALE_CHAMPION_WRITER",
        "H14": "PHYSICS_PERMISSION_REQUIRED",
        "H15": "PHYSICS_PERMISSION_REQUIRED",
        "H16": "REPLAY_IS_ENGINEERING_ONLY_NOT_NEW_EVIDENCE",
        "H17": "STATE_OBJECT_NOT_SEALED_AUTHORITATIVE",
        "H19": "STALE_OR_INVALID_FENCE",
        "H20": "AUTHORITY_METADATA_CORRUPT",
    }
    for case_id, code in expected.items():
        assert code in by_id[case_id]["observed"]


def test_intg_hooks_wrap_every_h01_h20_boundary():
    hooks = RecordingHooks()
    report = run_integrated_hostile_matrix(hooks_factory=lambda: hooks)
    assert report["status"] == "PASS"
    expected = []
    for case in REQUIRED_CASES:
        expected.extend([("before", case.case_id), ("after", case.case_id)])
    assert hooks.events == expected


def test_intg_common_invariants_fail_closed_on_replay_laundering_observation():
    report = run_integrated_hostile_matrix(adapter_factory=ReplayViolationAdapter)
    assert report["status"] == "FAIL"
    assert report["integrated_runtime_qualified"] is False
    assert report["failed_cases"] == 20
    assert all(
        row["invariants"]["no_new_scientific_evidence_from_replay"] is False
        for row in report["cases"]
    )


def test_intg_machine_report_is_exactly_s4h_top_level_compatible():
    report = run_integrated_hostile_matrix()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert set(report) == set(schema["required"])
    assert report["schema"] == schema["properties"]["schema"]["const"]
    assert report["matrix_schema"] == schema["properties"]["matrix_schema"]["const"]
    assert report["integrated_runtime_qualified"] is schema["properties"]["integrated_runtime_qualified"]["const"]
    assert report["scientific_status"] == schema["properties"]["scientific_status"]["const"]


def test_intg_binding_manifest_names_all_real_provider_surfaces_and_refuses_claim():
    manifest = integration_binding_manifest()
    assert manifest["integration_seed_sha"] == INTEGRATION_SEED_SHA
    assert manifest["integrated_runtime_qualification_claimed"] is False
    assert manifest["sibling_integration_dependency_used"] is False
    assert manifest["adapter_contract"] == "ProductionHostileAdapter"
    assert manifest["fault_hook_contract"] == "HostileFaultHooks"
    assert len(manifest["fault_boundaries"]) == 20
    assert [row["case_id"] for row in manifest["fault_boundaries"]] == [f"H{i:02d}" for i in range(1, 21)]
    for required in {
        "acquire_authority",
        "crash",
        "recover",
        "append_journal",
        "issue_permission",
        "physics_transition",
        "admit_evidence",
        "corrupt_authority_metadata",
    }:
        assert required in manifest["required_adapter_methods"]


def test_intg_has_no_sibling_integration_import_dependency():
    source = MODULE.read_text(encoding="utf-8") + "\n" + SCRIPT.read_text(encoding="utf-8")
    forbidden = {
        "stage4_runtime_spine_integration_r11",
        "stage4_fenced_persistence_r11",
        "stage4_authoritative_engines_r11",
        "stage4_execution_integration_r11",
        "stage4_recovery_integration_r11",
        "stage4_orchestration_provider_r11",
        "stage4_integration_gate_compiler_r11",
    }
    for name in forbidden:
        assert name not in source


def test_intg_cli_emits_reference_report_and_binding_manifest(tmp_path: Path):
    output = tmp_path / "intg-report.json"
    manifest = tmp_path / "intg-manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--manifest-output",
            str(manifest),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    binding = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["integrated_runtime_qualified"] is False
    assert binding["integrated_runtime_qualification_claimed"] is False
    assert "INTEGRATED_RUNTIME_QUALIFICATION_CLAIMED=NO" in completed.stdout


def test_intg_semantic_freeze_identity_constant_matches_repository_when_git_available():
    completed = subprocess.run(
        ["git", "rev-parse", f"HEAD:{FREEZE}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        assert completed.stdout.strip() == SEMANTIC_FREEZE_BLOB
