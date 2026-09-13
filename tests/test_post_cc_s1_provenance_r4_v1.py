"""R4 counterexamples for the remaining S1-026 artifact-only provenance blocker."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from cb16_local_opt import post_cc_s1_qualification_v1 as Q
from cb16_local_opt import post_cc_s1_tasks_v1 as T
from cb16_local_opt.post_cc_s1_execution_manifest_v1 import build_s1_execution_manifest_v1
from cb16_local_opt.post_cc_s1_provenance_audit_r4_v1 import (
    FIXED_BEHAVIOR_NA_REASON,
    audit_exported_update_journal_r4_v1,
    run_job_with_r4_provenance_v1,
)

MANIFEST = build_s1_execution_manifest_v1()


def _minimal_passing_compiled() -> dict:
    return {
        "schema": "CB16_R11_POST_CC_S1_RESULT_V1",
        "compiler_id": "CB16_R11_POST_CC_S1_DETERMINISTIC_GATE_COMPILER_V1",
        "classification": "PASS",
        "status": "PASS",
        "gates": {"R4_FIXTURE": True},
        "contract_violations": [],
        "evidence_insufficient": [],
        "scientific_failures": [],
    }


def _run_r4_job(tmp: Path, *, task_id: str, max_units: int) -> tuple[dict, Path]:
    run_root = tmp / "scratch" / task_id.lower()
    provenance_staging = tmp / "provenance_staged"
    checkpoint_staging = tmp / "checkpoints_staged"
    outcome = run_job_with_r4_provenance_v1(
        {
            "task_id": task_id,
            "seed": 1701,
            "control_id": None,
            "unit_size": 2,
            "max_units": max_units,
            "evaluation_population": 4,
            "mode": "smoke",
            "manifest_sha256": MANIFEST["manifest_sha256"],
            "run_root": str(run_root),
            "checkpoint_staging_root": str(checkpoint_staging),
            "provenance_staging_root": str(provenance_staging),
            "cleanup_run_root": False,
        }
    )
    assert outcome["status"] == "OK", outcome
    return outcome, provenance_staging


def _write_with_r4_audit(
    *,
    output_root: Path,
    outcome: dict,
    mode: str = "qualification",
) -> dict:
    original = Q.audit_exported_update_journal_v1
    Q.audit_exported_update_journal_v1 = audit_exported_update_journal_r4_v1
    try:
        Q._write_artifacts_v1(
            output_root=output_root,
            manifest=MANIFEST,
            outcomes=[outcome],
            integrity={"attacks": {"a": {"rejected": True}}, "all_rejected": True},
            objective_audit={"all_checks_pass": True},
            fabricated_audit={"all_checks_pass": True},
            failure_fact_audit={"all_checks_pass": True},
            mode=mode,
            compiled=_minimal_passing_compiled() if mode == "qualification" else None,
        )
    finally:
        Q.audit_exported_update_journal_v1 = original
    if mode == "qualification":
        return json.loads((output_root / "S1_RESULT.json").read_text(encoding="utf-8"))
    return json.loads((output_root / "SMOKE_RESULT.json").read_text(encoding="utf-8"))


def _staging_run_dir(staging: Path, task_id: str) -> Path:
    return staging / task_id / "positive" / "seed_1701"


def test_r4_clean_artifact_graph_verifies_every_checkpoint_and_s1_026_field():
    with tempfile.TemporaryDirectory(prefix="cb16-s1-r4-clean-") as tmp:
        tmp_path = Path(tmp)
        outcome, _staging = _run_r4_job(
            tmp_path,
            task_id=T.TASK_DELAYED_CONSEQUENCE_CREDIT,
            max_units=2,
        )
        output = tmp_path / "artifacts" / "post_cc_s1"
        result = _write_with_r4_audit(output_root=output, outcome=outcome)
        assert result["classification"] == "PASS", result
        audit = audit_exported_update_journal_r4_v1(output)
        assert audit["all_checks_pass"] is True, audit["failures"]
        assert audit["traced_update_count"] == 2
        assert audit["checks"]["checkpoint_chain_verified"] is True
        assert audit["checks"]["mandatory_s1_026_fields_verified"] is True
        assert audit["checks"]["generation_switch_verified"] is True


def test_r4_intermediate_child_tamper_forces_contract_mismatch():
    with tempfile.TemporaryDirectory(prefix="cb16-s1-r4-checkpoint-tamper-") as tmp:
        tmp_path = Path(tmp)
        outcome, staging = _run_r4_job(
            tmp_path,
            task_id=T.TASK_DELAYED_CONSEQUENCE_CREDIT,
            max_units=2,
        )
        run_stage = _staging_run_dir(staging, T.TASK_DELAYED_CONSEQUENCE_CREDIT)
        children = sorted((run_stage / "child_checkpoints").glob("*.json"))
        assert len(children) == 2
        children[0].write_text('{"tampered":true}\n', encoding="utf-8")

        output = tmp_path / "artifacts" / "post_cc_s1"
        result = _write_with_r4_audit(output_root=output, outcome=outcome)
        assert result["classification"] == "CONTRACT_MISMATCH"
        assert result["gates"]["ARTIFACT_ONLY_UPDATE_TRACE"] is False
        assert "ARTIFACT_ONLY_UPDATE_TRACE" in result["contract_violations"]


def test_r4_transition_provenance_tamper_forces_contract_mismatch():
    with tempfile.TemporaryDirectory(prefix="cb16-s1-r4-transition-tamper-") as tmp:
        tmp_path = Path(tmp)
        outcome, staging = _run_r4_job(
            tmp_path,
            task_id=T.TASK_DELAYED_CONSEQUENCE_CREDIT,
            max_units=2,
        )
        run_stage = _staging_run_dir(staging, T.TASK_DELAYED_CONSEQUENCE_CREDIT)
        path = run_stage / "s1_026_transitions.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows
        rows[0]["behavior_log_mu"] = float(rows[0]["behavior_log_mu"]) + 0.25
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

        output = tmp_path / "artifacts" / "post_cc_s1"
        result = _write_with_r4_audit(output_root=output, outcome=outcome)
        assert result["classification"] == "CONTRACT_MISMATCH"
        assert result["gates"]["ARTIFACT_ONLY_UPDATE_TRACE"] is False


def test_r4_generation_switch_receipt_tamper_forces_contract_mismatch():
    with tempfile.TemporaryDirectory(prefix="cb16-s1-r4-switch-tamper-") as tmp:
        tmp_path = Path(tmp)
        outcome, staging = _run_r4_job(
            tmp_path,
            task_id=T.TASK_DELAYED_CONSEQUENCE_CREDIT,
            max_units=2,
        )
        run_stage = _staging_run_dir(staging, T.TASK_DELAYED_CONSEQUENCE_CREDIT)
        receipts = sorted((run_stage / "generation_switch_receipts").glob("*.json"))
        assert len(receipts) == 2
        payload = json.loads(receipts[0].read_text(encoding="utf-8"))
        payload["account_truth_hash_after"] = "0" * 64
        receipts[0].write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

        output = tmp_path / "artifacts" / "post_cc_s1"
        result = _write_with_r4_audit(output_root=output, outcome=outcome)
        assert result["classification"] == "CONTRACT_MISMATCH"
        assert result["gates"]["ARTIFACT_ONLY_UPDATE_TRACE"] is False


def test_r4_fixed_distinct_behavior_has_explicit_generation_switch_na_reason():
    with tempfile.TemporaryDirectory(prefix="cb16-s1-r4-fixed-behavior-") as tmp:
        tmp_path = Path(tmp)
        outcome, _staging = _run_r4_job(
            tmp_path,
            task_id=T.TASK_OFF_POLICY_VTRACE_CORRECTION,
            max_units=1,
        )
        proofs = outcome["result"]["s1_026_generation_switch_proof"]
        assert proofs
        assert all(
            proof["mode"] == FIXED_BEHAVIOR_NA_REASON
            and proof["reason"] == FIXED_BEHAVIOR_NA_REASON
            for proof in proofs.values()
        )
        output = tmp_path / "artifacts" / "post_cc_s1"
        result = _write_with_r4_audit(output_root=output, outcome=outcome)
        assert result["classification"] == "PASS", result
        audit = audit_exported_update_journal_r4_v1(output)
        assert audit["all_checks_pass"] is True, audit["failures"]


def test_canonical_cli_installs_r4_provenance_hooks():
    text = Path("scripts/run_r11_post_cc_s1_learnability.py").read_text(encoding="utf-8")
    assert "install_r4_provenance_hooks_v1" in text
    assert "post_cc_s1_provenance_audit_r4_v1" in text
