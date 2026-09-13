"""B6: artifact-only full update tracing after scratch deletion."""

from __future__ import annotations

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


def test_update_journal_survives_scratch_deletion_and_is_auditable():
    spec = T.build_task_specs_v1()[T.TASK_DELAYED_CONSEQUENCE_CREDIT]
    with tempfile.TemporaryDirectory(prefix="cb16-s1-b6-") as tmp:
        run_root = Path(tmp) / "scratch" / "run"
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
        provenance_staging = Path(tmp) / "provenance_staged"
        export_info = export_run_provenance_v1(
            run_root, provenance_staging / spec.task_id / "positive" / "seed_1701"
        )
        assert export_info["journal_records_exported"] == 2
        final_child_sha = result["unit_evidence"][-1]["child_checkpoint_sha256"]
        checkpoint_staging = Path(tmp) / "checkpoints_staged" / spec.task_id / "positive"
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
                "checkpoint_staging_root": str(Path(tmp) / "checkpoints_staged"),
                "provenance_staging_root": str(provenance_staging),
            },
            "result": result,
        }
        output_root = Path(tmp) / "artifacts" / "post_cc_s1"
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
        shutil.rmtree(run_root)  # scratch gone
        assert not run_root.exists()
        journal_files = list((output_root / "provenance" / "update_journals").rglob("*.json"))
        assert len(journal_files) == 2
        audit = audit_exported_update_journal_v1(output_root)
        assert audit["all_checks_pass"] is True, audit
        assert audit["checks"]["update_linked_from_result_record"] is True
        assert audit["checks"]["final_child_checkpoint_bytes_match_journal"] is True
        assert audit["checks"]["durable_index_exported"] is True
        assert audit["details"]["traced_update_id"]
