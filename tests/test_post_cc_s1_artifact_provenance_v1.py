"""R2-C4: every qualified update must be artifact-only traceable and audit-fail-closed."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from cb16_local_opt import post_cc_s1_tasks_v1 as T
from cb16_local_opt.post_cc_s1_qualification_v1 import (
    _write_artifacts_v1,
    audit_exported_update_journal_v1,
    export_run_provenance_v1,
)
from cb16_local_opt.post_cc_s1_execution_manifest_v1 import build_s1_execution_manifest_v1
from cb16_local_opt.post_cc_s1_training_loop_v1 import S1SeedRunConfigV1, run_seed_v1

MANIFEST = build_s1_execution_manifest_v1()


def _run_and_stage(tmp: Path):
    spec = T.build_task_specs_v1()[T.TASK_DELAYED_CONSEQUENCE_CREDIT]
    run_root = tmp / "scratch" / "run"
    result = run_seed_v1(
        S1SeedRunConfigV1(
            spec=spec,
            seed=1701,
            run_root=str(run_root),
            mode="smoke",
            control_id=None,
            unit_size=2,
            max_units=2,
            evaluation_population=4,
            manifest_sha256=MANIFEST["manifest_sha256"],
        )
    )
    provenance_staging = tmp / "provenance_staged"
    export_info = export_run_provenance_v1(
        run_root,
        provenance_staging / spec.task_id / "positive" / "seed_1701",
        result=result,
    )
    final_child_sha = result["unit_evidence"][-1]["child_checkpoint_sha256"]
    checkpoint_staging = tmp / "checkpoints_staged" / spec.task_id / "positive"
    checkpoint_staging.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        run_root / "updates" / "checkpoints" / f"{final_child_sha}.json",
        checkpoint_staging / "seed_1701_final_child.json",
    )
    outcome = {
        "status": "OK",
        "job": {
            "task_id": spec.task_id,
            "seed": 1701,
            "control_id": None,
            "run_root": str(run_root),
            "checkpoint_staging_root": str(tmp / "checkpoints_staged"),
            "provenance_staging_root": str(provenance_staging),
        },
        "result": result,
    }
    return run_root, provenance_staging, outcome, export_info


def _minimal_passing_compiled() -> dict:
    return {
        "schema": "CB16_R11_POST_CC_S1_RESULT_V1",
        "compiler_id": "CB16_R11_POST_CC_S1_DETERMINISTIC_GATE_COMPILER_V1",
        "classification": "PASS",
        "status": "PASS",
        "gates": {"POSITIVE:DELAYED_CONSEQUENCE_CREDIT": True},
        "contract_violations": [],
        "evidence_insufficient": [],
        "scientific_failures": [],
    }


def test_every_committed_update_is_traceable_after_scratch_deletion():
    with tempfile.TemporaryDirectory(prefix="cb16-s1-c4-") as tmp:
        tmp_path = Path(tmp)
        run_root, _staging, outcome, export_info = _run_and_stage(tmp_path)
        assert export_info["journal_records_exported"] == 2
        assert export_info["update_resolution_rows_exported"] == 2
        assert export_info["sequence_rows_exported"] > 0
        output_root = tmp_path / "artifacts" / "post_cc_s1"
        _write_artifacts_v1(
            output_root=output_root,
            manifest=MANIFEST,
            outcomes=[outcome],
            integrity={"attacks": {"a": {"rejected": True}}, "all_rejected": True},
            objective_audit={"all_checks_pass": True},
            fabricated_audit={"all_checks_pass": True},
            failure_fact_audit={"all_checks_pass": True},
            mode="smoke",
            compiled=None,
        )
        shutil.rmtree(run_root)
        assert not run_root.exists()
        audit = audit_exported_update_journal_v1(output_root)
        assert audit["all_checks_pass"] is True, audit.get("failures")
        assert audit["runs_audited"] == 1
        assert audit["traced_update_count"] == 2
        assert audit["checks"]["checkpoint_chain_verified"] is True
        assert audit["checks"]["durable_resolution_complete"] is True


def test_deleted_exported_link_forces_contract_mismatch_in_qualification():
    with tempfile.TemporaryDirectory(prefix="cb16-s1-c4-neg-") as tmp:
        tmp_path = Path(tmp)
        run_root, staging, outcome, _export_info = _run_and_stage(tmp_path)
        resolution_path = staging / T.TASK_DELAYED_CONSEQUENCE_CREDIT / "positive" / "seed_1701" / "update_resolution.jsonl"
        rows = [json.loads(line) for line in resolution_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = rows[:-1]  # delete one exported update link
        resolution_path.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        output_root = tmp_path / "artifacts" / "post_cc_s1"
        _write_artifacts_v1(
            output_root=output_root,
            manifest=MANIFEST,
            outcomes=[outcome],
            integrity={"attacks": {"a": {"rejected": True}}, "all_rejected": True},
            objective_audit={"all_checks_pass": True},
            fabricated_audit={"all_checks_pass": True},
            failure_fact_audit={"all_checks_pass": True},
            mode="qualification",
            compiled=_minimal_passing_compiled(),
        )
        shutil.rmtree(run_root)
        result = json.loads((output_root / "S1_RESULT.json").read_text(encoding="utf-8"))
        assert result["classification"] == "CONTRACT_MISMATCH"
        assert result["status"] == "CONTRACT_MISMATCH"
        assert result["gates"]["ARTIFACT_ONLY_UPDATE_TRACE"] is False
        assert "ARTIFACT_ONLY_UPDATE_TRACE" in result["contract_violations"]
        audit = audit_exported_update_journal_v1(output_root)
        assert audit["all_checks_pass"] is False
        assert audit["failures"]
