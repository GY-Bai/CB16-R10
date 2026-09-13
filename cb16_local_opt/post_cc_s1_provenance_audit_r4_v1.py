"""R4 closure for the S1-026 artifact-only provenance audit.

This module is deliberately evidence-only.  It does not change task generators,
sampling, rewards, losses, optimizers, or checkpoint construction.  The
canonical S1 CLI installs these hooks so Shanxi CI uses the strengthened
exporter/auditor while the frozen scientific runtime remains unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from .cc_experience_wire_r0 import canonical_json_bytes
from .post_cc_generation_continuity_v1 import (
    GenerationContinuityReceiptV1,
    child_policy_identity_v1,
)
from .post_cc_joint_replay_v1 import DurableTransitionRecordV1
from .post_cc_replay_materializer_v1 import ReplayStoreV1
from .post_cc_s1_tasks_v1 import (
    BEHAVIOR_MODE_FIXED_DISTINCT_V1,
    build_task_specs_v1,
    stable_sha256_v1,
)
from .post_cc_update_transaction_v1 import decode_checkpoint_bundle_v1

FIXED_BEHAVIOR_NA_REASON = "NOT_APPLICABLE_FIXED_DISTINCT_BEHAVIOR"
R4_AUDIT_SCHEMA = "CB16_R11_POST_CC_S1_ARTIFACT_ONLY_UPDATE_TRACE_AUDIT_R4_V1"
R4_BINDING_SCHEMA = "CB16_R11_POST_CC_S1_UPDATE_PROVENANCE_BINDING_R4_V1"
R4_TRANSITION_SCHEMA = "CB16_R11_POST_CC_S1_TRANSITION_PROVENANCE_R4_V1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(bytes(payload)).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _write_durable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _write_durable_bytes(path, payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _receipt_sha256(payload: Mapping[str, Any]) -> str:
    return _canonical_sha256(dict(payload))


def _switch_id_from_receipt(payload: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "old_policy_generation": str(payload["old_policy_generation"]),
            "new_policy_generation": str(payload["new_policy_generation"]),
            "new_policy_id": str(payload["new_policy_id"]),
            "new_policy_sha256": str(payload["new_policy_sha256"]),
            "child_checkpoint_sha256": str(payload["child_checkpoint_sha256"]),
            "environment_time": int(payload["environment_time"]),
            "policy_decision_index": int(payload["policy_decision_index"]),
            "boundary_type": str(payload["boundary_type"]),
        }
    )


def _parent_sha_from_child_bundle(bundle: Mapping[str, Any]) -> str:
    """Re-encode a committed child state in the canonical parent codec.

    Child and parent checkpoint byte schemas intentionally differ.  The next
    update's parent identity therefore must be recomputed from the child state,
    not compared to the child bundle hash.
    """
    payload = {
        "schema": "CB16_R11_S0V2_PARENT_CHECKPOINT_V1",
        "optimizer_step": int(bundle["optimizer_step"]),
        "actor_lr": float(bundle["actor_lr"]),
        "critic_lr": float(bundle["critic_lr"]),
        "actor_state": bundle["actor_state"],
        "critic_state": bundle["critic_state"],
    }
    return _canonical_sha256(payload)


def _persist_generation_switch_receipt_wrapper(original, receipt_root: Path):
    def wrapped(*args, **kwargs):
        receipt = original(*args, **kwargs)
        update_id = str(kwargs.get("update_id") or "")
        if not update_id:
            raise RuntimeError("R4_GENERATION_SWITCH_UPDATE_ID_MISSING")
        payload = asdict(receipt)
        receipt.validate()
        _write_durable_bytes(
            receipt_root / f"{update_id}.json",
            canonical_json_bytes(payload),
        )
        return receipt

    return wrapped


def _generation_switch_proof_v1(
    *,
    result: Mapping[str, Any],
    spec: Any,
    receipt_root: Path,
) -> dict[str, Any]:
    proofs: dict[str, Any] = {}
    for unit in result.get("unit_evidence", []):
        update_id = str(unit.get("update_id") or "")
        if not update_id:
            continue
        receipt_path = receipt_root / f"{update_id}.json"
        if receipt_path.exists():
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            proof = {
                "mode": "COMMITTED_CHILD_SWITCH",
                "receipt": payload,
                "receipt_content_sha256": _receipt_sha256(payload),
            }
        elif str(spec.behavior_mode) == BEHAVIOR_MODE_FIXED_DISTINCT_V1:
            proof = {
                "mode": FIXED_BEHAVIOR_NA_REASON,
                "reason": FIXED_BEHAVIOR_NA_REASON,
            }
        else:
            proof = {
                "mode": "MISSING_REQUIRED_GENERATION_SWITCH_RECEIPT",
                "reason": "MISSING_REQUIRED_GENERATION_SWITCH_RECEIPT",
            }
        proofs[update_id] = proof
    return proofs


def run_job_with_r4_provenance_v1(job: Mapping[str, Any]) -> dict[str, Any]:
    """Worker entrypoint used by the canonical S1 CLI.

    It preserves the frozen run path and only persists already-created
    generation-switch receipts plus the complete S1-026 artifact graph before
    scratch cleanup.
    """
    import torch

    from . import post_cc_s1_training_loop_v1 as training_loop_v1
    from .post_cc_s1_training_loop_v1 import S1SeedRunConfigV1

    torch.set_num_threads(1)
    spec = build_task_specs_v1()[str(job["task_id"])]
    config = S1SeedRunConfigV1(
        spec=spec,
        seed=int(job["seed"]),
        run_root=str(job["run_root"]),
        mode=str(job["mode"]),
        control_id=job["control_id"],
        unit_size=int(job["unit_size"]),
        max_units=int(job["max_units"]),
        evaluation_population=int(job["evaluation_population"]),
        manifest_sha256=str(job["manifest_sha256"]),
    )
    receipt_root = Path(str(job["run_root"])) / "updates" / "generation_switch_receipts"
    original_switch = training_loop_v1.commit_child_generation_v1
    training_loop_v1.commit_child_generation_v1 = _persist_generation_switch_receipt_wrapper(
        original_switch, receipt_root
    )
    try:
        result = training_loop_v1.run_seed_v1(config)
        result = dict(result)
        result["s1_026_generation_switch_proof"] = _generation_switch_proof_v1(
            result=result,
            spec=spec,
            receipt_root=receipt_root,
        )

        staging_root = job.get("checkpoint_staging_root")
        unit_evidence = result.get("unit_evidence", [])
        final_child_sha = (
            unit_evidence[-1].get("child_checkpoint_sha256") if unit_evidence else None
        )
        if staging_root and final_child_sha:
            source = (
                Path(str(job["run_root"]))
                / "updates"
                / "checkpoints"
                / f"{final_child_sha}.json"
            )
            destination_dir = Path(str(staging_root)) / str(job["task_id"]) / (
                "positive"
                if job["control_id"] is None
                else f"control_{str(job['control_id']).lower()}"
            )
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"seed_{int(job['seed'])}_final_child.json"
            if source.exists():
                shutil.copyfile(source, destination)

        provenance_staging = job.get("provenance_staging_root")
        if provenance_staging:
            staging_dir = (
                Path(str(provenance_staging))
                / str(job["task_id"])
                / (
                    "positive"
                    if job["control_id"] is None
                    else f"control_{str(job['control_id']).lower()}"
                )
                / f"seed_{int(job['seed'])}"
            )
            export_run_provenance_r4_v1(
                str(job["run_root"]),
                staging_dir,
                result=result,
            )

        if job.get("cleanup_run_root") and Path(str(job["run_root"])).exists():
            shutil.rmtree(str(job["run_root"]), ignore_errors=True)
        return {"status": "OK", "job": dict(job), "result": result}
    except Exception as exc:  # execution failures remain explicit evidence
        return {
            "status": "EXECUTION_ERROR",
            "job": dict(job),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    finally:
        training_loop_v1.commit_child_generation_v1 = original_switch


def export_run_provenance_r4_v1(
    run_root: str | Path,
    destination: str | Path,
    *,
    result: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Export the complete S1-026 graph before scratch deletion."""
    from . import post_cc_s1_qualification_v1 as qualification_v1

    run_root_path = Path(run_root)
    staging_dir = Path(destination)
    base = dict(
        qualification_v1.export_run_provenance_v1(
            run_root_path,
            staging_dir,
            result=result,
        )
    )

    checkpoints_source = run_root_path / "updates" / "checkpoints"
    checkpoints_destination = staging_dir / "child_checkpoints"
    checkpoint_count = 0
    if checkpoints_source.exists():
        checkpoints_destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(checkpoints_source.glob("*.json")):
            shutil.copyfile(source, checkpoints_destination / source.name)
            checkpoint_count += 1

    receipt_source = run_root_path / "updates" / "generation_switch_receipts"
    receipt_destination = staging_dir / "generation_switch_receipts"
    receipt_count = 0
    if receipt_source.exists():
        receipt_destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(receipt_source.glob("*.json")):
            shutil.copyfile(source, receipt_destination / source.name)
            receipt_count += 1

    index_path = run_root_path / "durable_index.jsonl"
    index_rows = _read_jsonl(index_path)
    replay_store = ReplayStoreV1(run_root_path)
    transition_rows: list[dict[str, Any]] = []
    for index_row in index_rows:
        sequence = replay_store.get_sequence(str(index_row["sequence_id"]))
        for ordinal, transition_id in enumerate(sequence.transition_ids):
            transition = replay_store.get_transition(transition_id)
            row = {
                "schema": R4_TRANSITION_SCHEMA,
                "task_id": str(result["task_id"]),
                "task_spec_hash": str(result["task_spec_hash"]),
                "manifest_sha256": str(result["manifest_sha256"]),
                "seed": int(result["seed"]),
                "control_id": result.get("control_id"),
                "context_id": index_row.get("context_id"),
                "unit_index": int(index_row.get("unit_index", -1)),
                "transition_ordinal": int(ordinal),
                "sequence_record_content_sha256": sequence.content_sha256,
                "transition_content_sha256": transition.content_sha256,
                **asdict(transition),
            }
            transition_rows.append(row)
    _write_jsonl(staging_dir / "s1_026_transitions.jsonl", transition_rows)

    updates_dir = run_root_path / "updates" / "updates"
    committed_records: list[dict[str, Any]] = []
    if updates_dir.exists():
        for path in sorted(updates_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("commit_status") == "COMMITTED":
                committed_records.append(record)
    committed_records.sort(key=lambda row: int(row.get("optimizer_step_after") or 0))

    unit_by_update = {
        str(unit.get("update_id")): unit for unit in result.get("unit_evidence", [])
    }
    switch_proof = dict(result.get("s1_026_generation_switch_proof", {}))
    binding_rows: list[dict[str, Any]] = []
    for record in committed_records:
        update_id = str(record["update_id"])
        unit = unit_by_update.get(update_id, {})
        proof = switch_proof.get(
            update_id,
            {
                "mode": "MISSING_REQUIRED_GENERATION_SWITCH_RECEIPT",
                "reason": "MISSING_REQUIRED_GENERATION_SWITCH_RECEIPT",
            },
        )
        binding = {
            "schema": R4_BINDING_SCHEMA,
            "task_id": str(result["task_id"]),
            "task_spec_hash": str(result["task_spec_hash"]),
            "manifest_sha256": str(result["manifest_sha256"]),
            "seed": int(result["seed"]),
            "control_id": result.get("control_id"),
            "update_id": update_id,
            "unit_index": int(unit.get("unit_index", -1)),
            "sampled_sequence_ids": list(record.get("sampled_sequence_ids", [])),
            "materialization_manifest_sha256": str(
                record.get("materialization_manifest_sha256")
            ),
            "batch_content_sha256": str(record.get("batch_content_sha256")),
            "parent_checkpoint_sha256": str(record.get("parent_checkpoint_sha256")),
            "child_checkpoint_sha256": str(record.get("child_checkpoint_sha256")),
            "optimizer_step_before": int(record.get("optimizer_step_before") or 0),
            "optimizer_step_after": int(record.get("optimizer_step_after") or 0),
            "generation_switch_proof": proof,
            "generation_switch_proof_sha256": _canonical_sha256(proof),
        }
        binding_rows.append(binding)
    _write_jsonl(staging_dir / "s1_026_update_bindings.jsonl", binding_rows)

    base.update(
        {
            "r4_transition_rows_exported": len(transition_rows),
            "r4_update_bindings_exported": len(binding_rows),
            "r4_child_checkpoints_exported": checkpoint_count,
            "r4_generation_switch_receipts_exported": receipt_count,
        }
    )
    return base


def _result_path_for_run(
    root: Path, identity: Mapping[str, Any], relative: Path
) -> Path:
    task_id = str(identity.get("task_id") or relative.parts[2])
    seed = identity.get("seed")
    control_id = identity.get("control_id")
    if control_id is None:
        return root / "task_results" / task_id / f"{seed}.json"
    return root / "control_results" / str(control_id) / f"{task_id}__{seed}.json"


def _transition_from_export(row: Mapping[str, Any]) -> DurableTransitionRecordV1:
    names = {field.name for field in fields(DurableTransitionRecordV1)}
    payload = {name: row[name] for name in names}
    return DurableTransitionRecordV1(**payload).validate()


def _full_receipt_valid(
    *,
    proof: Mapping[str, Any],
    record: Mapping[str, Any],
    unit: Mapping[str, Any],
    persisted_receipt: Mapping[str, Any] | None,
) -> tuple[bool, str | None]:
    if proof.get("mode") != "COMMITTED_CHILD_SWITCH":
        return False, "GENERATION_SWITCH_MODE_INVALID"
    receipt_payload = proof.get("receipt")
    if not isinstance(receipt_payload, Mapping):
        return False, "GENERATION_SWITCH_RECEIPT_PAYLOAD_MISSING"
    try:
        receipt = GenerationContinuityReceiptV1(**dict(receipt_payload)).validate()
    except Exception:
        return False, "GENERATION_SWITCH_RECEIPT_INVALID"
    if persisted_receipt is None or dict(persisted_receipt) != dict(receipt_payload):
        return False, "GENERATION_SWITCH_PERSISTED_RECEIPT_MISMATCH"
    expected_receipt_sha = _receipt_sha256(receipt_payload)
    if str(proof.get("receipt_content_sha256")) != expected_receipt_sha:
        return False, "GENERATION_SWITCH_RECEIPT_HASH_MISMATCH"
    if str(receipt.child_checkpoint_sha256) != str(record.get("child_checkpoint_sha256")):
        return False, "GENERATION_SWITCH_CHILD_CHECKPOINT_MISMATCH"
    if receipt.account_truth_hash_before != receipt.account_truth_hash_after:
        return False, "GENERATION_SWITCH_ACCOUNT_TRUTH_CHANGED"
    if not receipt.account_fully_preserved or not receipt.authorized_boundary:
        return False, "GENERATION_SWITCH_CONTINUITY_FLAGS_INVALID"
    if receipt.parent_checkpoint_mutated_in_place:
        return False, "GENERATION_SWITCH_PARENT_MUTATED"
    if _switch_id_from_receipt(receipt_payload) != receipt.switch_id:
        return False, "GENERATION_SWITCH_ID_MISMATCH"
    expected_policy_sha = child_policy_identity_v1(
        child_checkpoint_sha256=receipt.child_checkpoint_sha256,
        policy_generation=receipt.new_policy_generation,
        policy_id=receipt.new_policy_id,
    )
    if expected_policy_sha != receipt.new_policy_sha256:
        return False, "GENERATION_SWITCH_POLICY_CHILD_BINDING_MISMATCH"
    expected_before = (
        f"{receipt.old_policy_id}:g{receipt.old_policy_generation}:{receipt.old_policy_sha256}"
    )
    expected_after = (
        f"{receipt.new_policy_id}:g{receipt.new_policy_generation}:{receipt.new_policy_sha256}"
    )
    if str(unit.get("behavior_identity_before")) != expected_before:
        return False, "GENERATION_SWITCH_OLD_BEHAVIOR_IDENTITY_MISMATCH"
    if str(unit.get("behavior_identity_after")) != expected_after:
        return False, "GENERATION_SWITCH_NEW_BEHAVIOR_IDENTITY_MISMATCH"
    subset = unit.get("generation_switch_receipt")
    if not isinstance(subset, Mapping):
        return False, "GENERATION_SWITCH_UNIT_RECEIPT_MISSING"
    for name in (
        "switch_id",
        "child_checkpoint_sha256",
        "new_policy_generation",
        "new_policy_sha256",
        "account_fully_preserved",
        "authorized_boundary",
        "parent_checkpoint_mutated_in_place",
    ):
        if subset.get(name) != receipt_payload.get(name):
            return False, f"GENERATION_SWITCH_UNIT_RECEIPT_MISMATCH:{name}"
    return True, None


def audit_exported_update_journal_r4_v1(
    artifact_root: str | Path,
) -> Mapping[str, Any]:
    """Artifact-only R4 audit for every mandatory S1-026 provenance edge."""
    root = Path(artifact_root)
    journal_root = root / "provenance" / "update_journals"
    run_dirs = sorted({path.parent for path in journal_root.rglob("update_journal/*.json")})
    failures: list[str] = []
    checks: dict[str, bool] = {"update_journal_exported": bool(run_dirs)}
    details: dict[str, Any] = {"runs": []}
    traced_updates = 0
    runs_audited = 0

    manifest_path = root / "execution_manifest.json"
    task_specs_path = root / "provenance" / "task_specs.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    task_specs_payload = (
        json.loads(task_specs_path.read_text(encoding="utf-8"))
        if task_specs_path.exists()
        else {}
    )
    task_specs = {
        str(item.get("task_id")): item
        for item in task_specs_payload.get("task_specs", [])
    }
    if not run_dirs:
        failures.append("NO_EXPORTED_UPDATE_JOURNAL")

    transition_field_names = {field.name for field in fields(DurableTransitionRecordV1)}

    for run_dir in run_dirs:
        relative = run_dir.relative_to(root)
        data_dir = root / "replay_manifests" / Path(*relative.parts[2:-1])
        r4_dir = root / "provenance" / "run_stores" / Path(*relative.parts[2:-1])
        run_failures: list[str] = []

        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(run_dir.glob("*.json"))
        ]
        committed = [
            record for record in records if record.get("commit_status") == "COMMITTED"
        ]
        committed.sort(key=lambda item: int(item.get("optimizer_step_after") or 0))
        resolution_rows = _read_jsonl(data_dir / "update_resolution.jsonl")
        transition_rows = _read_jsonl(r4_dir / "s1_026_transitions.jsonl")
        binding_rows = _read_jsonl(r4_dir / "s1_026_update_bindings.jsonl")
        identity_path = data_dir / "run_identity.json"
        identity = (
            json.loads(identity_path.read_text(encoding="utf-8"))
            if identity_path.exists()
            else {}
        )
        if not identity:
            run_failures.append("RUN_IDENTITY_MISSING")

        result_path = _result_path_for_run(root, identity, relative)
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else {}
        )
        if not result:
            run_failures.append("RESULT_RECORD_MISSING")

        task_id = str(identity.get("task_id") or result.get("task_id") or "")
        expected_spec = task_specs.get(task_id, {})
        expected_manifest_sha = str(manifest.get("manifest_sha256") or "")
        expected_spec_hash = str(expected_spec.get("spec_hash") or "")
        fixed_behavior = (
            expected_spec.get("payload", {}).get("behavior_mode")
            == BEHAVIOR_MODE_FIXED_DISTINCT_V1
        )

        if len(committed) != len(resolution_rows):
            run_failures.append("COMMITTED_UPDATE_RESOLUTION_CARDINALITY_MISMATCH")
        if len(committed) != len(binding_rows):
            run_failures.append("COMMITTED_UPDATE_BINDING_CARDINALITY_MISMATCH")
        if identity.get("committed_updates_count") is not None and int(
            identity.get("committed_updates_count", -1)
        ) != len(committed):
            run_failures.append("RUN_IDENTITY_COMMITTED_COUNT_MISMATCH")

        resolution_by_update = {
            str(row.get("update_id")): row for row in resolution_rows
        }
        binding_by_update = {str(row.get("update_id")): row for row in binding_rows}
        unit_by_update = {
            str(unit.get("update_id")): unit
            for unit in result.get("unit_evidence", [])
        }
        result_switch_proofs = dict(
            result.get("s1_026_generation_switch_proof", {})
        )
        transitions_by_sequence: dict[str, list[dict[str, Any]]] = {}
        for row in transition_rows:
            transitions_by_sequence.setdefault(str(row.get("sequence_id")), []).append(row)
        for rows in transitions_by_sequence.values():
            rows.sort(key=lambda row: int(row.get("transition_ordinal", 0)))

        checkpoint_bundles: dict[str, Mapping[str, Any]] = {}

        for record in committed:
            update_id = str(record.get("update_id"))
            binding = binding_by_update.get(update_id)
            resolution = resolution_by_update.get(update_id)
            unit = unit_by_update.get(update_id)
            prefix = f"{update_id}:"
            if binding is None:
                run_failures.append(prefix + "S1_026_BINDING_MISSING")
                continue
            if resolution is None:
                run_failures.append(prefix + "UPDATE_RESOLUTION_MISSING")
                continue
            if unit is None:
                run_failures.append(prefix + "UNIT_EVIDENCE_MISSING")
                continue

            if str(binding.get("schema")) != R4_BINDING_SCHEMA:
                run_failures.append(prefix + "S1_026_BINDING_SCHEMA_MISMATCH")
            for key, expected in (
                ("task_id", task_id),
                ("task_spec_hash", expected_spec_hash),
                ("manifest_sha256", expected_manifest_sha),
                ("seed", int(result.get("seed", -1))),
                ("control_id", result.get("control_id")),
            ):
                if binding.get(key) != expected:
                    run_failures.append(prefix + f"RUN_BINDING_MISMATCH:{key}")
            if str(result.get("task_spec_hash")) != expected_spec_hash:
                run_failures.append(prefix + "RESULT_TASK_SPEC_HASH_MISMATCH")
            if str(result.get("manifest_sha256")) != expected_manifest_sha:
                run_failures.append(prefix + "RESULT_MANIFEST_HASH_MISMATCH")

            for key in (
                "batch_content_sha256",
                "parent_checkpoint_sha256",
                "child_checkpoint_sha256",
                "materialization_manifest_sha256",
            ):
                if str(binding.get(key)) != str(record.get(key)):
                    run_failures.append(prefix + f"UPDATE_BINDING_MISMATCH:{key}")
            if list(binding.get("sampled_sequence_ids", [])) != list(
                record.get("sampled_sequence_ids", [])
            ):
                run_failures.append(prefix + "SAMPLED_SEQUENCE_BINDING_MISMATCH")
            if int(binding.get("optimizer_step_before", -1)) != int(
                record.get("optimizer_step_before", -2)
            ):
                run_failures.append(prefix + "OPTIMIZER_STEP_BEFORE_MISMATCH")
            if int(binding.get("optimizer_step_after", -1)) != int(
                record.get("optimizer_step_after", -2)
            ):
                run_failures.append(prefix + "OPTIMIZER_STEP_AFTER_MISMATCH")
            if str(unit.get("batch_content_sha256")) != str(
                record.get("batch_content_sha256")
            ):
                run_failures.append(prefix + "UNIT_BATCH_HASH_MISMATCH")
            if str(unit.get("child_checkpoint_sha256")) != str(
                record.get("child_checkpoint_sha256")
            ):
                run_failures.append(prefix + "UNIT_CHILD_CHECKPOINT_MISMATCH")

            entries = {
                str(entry.get("sequence_id")): entry
                for entry in resolution.get("entries", [])
            }
            manifest_hashes: list[str] = []
            flattened_sample_hashes: list[str] = []
            for sequence_id in record.get("sampled_sequence_ids", []):
                sequence_id = str(sequence_id)
                entry = entries.get(sequence_id)
                if entry is None:
                    run_failures.append(prefix + f"MANIFEST_ENTRY_MISSING:{sequence_id}")
                    continue
                manifest_hashes.append(str(entry.get("manifest_sha256")))
                flattened_sample_hashes.extend(
                    str(value) for value in entry.get("sample_hashes", [])
                )
                rows = transitions_by_sequence.get(sequence_id, [])
                if not rows:
                    run_failures.append(prefix + f"TRANSITION_PROVENANCE_MISSING:{sequence_id}")
                    continue

                expected_transition_hashes = [
                    str(value) for value in entry.get("transition_hashes", [])
                ]
                expected_observation_hashes = [
                    str(value) for value in entry.get("observation_hashes", [])
                ]
                actual_transition_hashes: list[str] = []
                actual_observation_hashes: list[str] = []
                sequence_lineages: set[str] = set()
                for row in rows:
                    missing = [
                        name for name in transition_field_names if name not in row
                    ]
                    if missing:
                        run_failures.append(
                            prefix
                            + f"TRANSITION_FIELDS_MISSING:{sequence_id}:{','.join(sorted(missing))}"
                        )
                        continue
                    try:
                        transition = _transition_from_export(row)
                    except Exception:
                        run_failures.append(
                            prefix + f"TRANSITION_RECORD_INVALID:{sequence_id}"
                        )
                        continue
                    recomputed = transition.content_sha256
                    if recomputed != str(row.get("transition_content_sha256")):
                        run_failures.append(
                            prefix + f"TRANSITION_CONTENT_HASH_MISMATCH:{sequence_id}"
                        )
                    if str(row.get("schema")) != R4_TRANSITION_SCHEMA:
                        run_failures.append(
                            prefix + f"TRANSITION_EXPORT_SCHEMA_MISMATCH:{sequence_id}"
                        )
                    for key, expected in (
                        ("task_id", task_id),
                        ("task_spec_hash", expected_spec_hash),
                        ("manifest_sha256", expected_manifest_sha),
                        ("seed", int(result.get("seed", -1))),
                        ("control_id", result.get("control_id")),
                    ):
                        if row.get(key) != expected:
                            run_failures.append(
                                prefix + f"TRANSITION_RUN_BINDING_MISMATCH:{sequence_id}:{key}"
                            )
                    if (
                        not isinstance(transition.raw_transition_content_sha256, str)
                        or len(transition.raw_transition_content_sha256) != 64
                    ):
                        run_failures.append(
                            prefix + f"RAW_CONSEQUENCE_HASH_MISSING:{sequence_id}"
                        )
                    sequence_lineages.add(str(transition.account_lineage_id))
                    actual_transition_hashes.append(recomputed)
                    actual_observation_hashes.append(
                        str(transition.observation_content_sha256)
                    )
                if len(sequence_lineages) != 1:
                    run_failures.append(
                        prefix + f"ACCOUNT_LINEAGE_DISCONTINUITY:{sequence_id}"
                    )
                if actual_transition_hashes != expected_transition_hashes:
                    run_failures.append(
                        prefix + f"TRANSITION_MANIFEST_BINDING_MISMATCH:{sequence_id}"
                    )
                if actual_observation_hashes != expected_observation_hashes:
                    run_failures.append(
                        prefix + f"OBSERVATION_MANIFEST_BINDING_MISMATCH:{sequence_id}"
                    )

            if flattened_sample_hashes != [
                str(value) for value in record.get("materialized_sample_hashes", [])
            ]:
                run_failures.append(prefix + "MATERIALIZED_SAMPLE_HASH_MISMATCH")
            if manifest_hashes:
                aggregate_manifest_sha = stable_sha256_v1(tuple(manifest_hashes))
                if aggregate_manifest_sha != str(
                    record.get("materialization_manifest_sha256")
                ):
                    run_failures.append(
                        prefix + "MATERIALIZATION_AGGREGATE_HASH_MISMATCH"
                    )
            if list(resolution.get("sampled_sequence_ids", [])) != list(
                record.get("sampled_sequence_ids", [])
            ):
                run_failures.append(prefix + "RESOLUTION_SEQUENCE_MISMATCH")
            if str(resolution.get("batch_content_sha256")) != str(
                record.get("batch_content_sha256")
            ):
                run_failures.append(prefix + "RESOLUTION_BATCH_HASH_MISMATCH")

            child_sha = str(record.get("child_checkpoint_sha256"))
            child_path = r4_dir / "child_checkpoints" / f"{child_sha}.json"
            if not child_path.exists():
                run_failures.append(prefix + "CHILD_CHECKPOINT_BYTES_MISSING")
            else:
                payload = child_path.read_bytes()
                if _sha256_bytes(payload) != child_sha:
                    run_failures.append(prefix + "CHILD_CHECKPOINT_SHA_MISMATCH")
                else:
                    try:
                        bundle = decode_checkpoint_bundle_v1(payload)
                    except Exception:
                        run_failures.append(prefix + "CHILD_CHECKPOINT_DECODE_FAILED")
                    else:
                        checkpoint_bundles[update_id] = bundle
                        if str(bundle.get("update_id")) != update_id:
                            run_failures.append(prefix + "CHILD_CHECKPOINT_UPDATE_ID_MISMATCH")
                        if str(bundle.get("parent_checkpoint_sha256")) != str(
                            record.get("parent_checkpoint_sha256")
                        ):
                            run_failures.append(prefix + "CHILD_CHECKPOINT_PARENT_MISMATCH")
                        if int(bundle.get("optimizer_step", -1)) != int(
                            record.get("optimizer_step_after", -2)
                        ):
                            run_failures.append(prefix + "CHILD_CHECKPOINT_STEP_MISMATCH")

            proof = binding.get("generation_switch_proof")
            if not isinstance(proof, Mapping):
                run_failures.append(prefix + "GENERATION_SWITCH_PROOF_MISSING")
            else:
                if str(binding.get("generation_switch_proof_sha256")) != _canonical_sha256(
                    proof
                ):
                    run_failures.append(prefix + "GENERATION_SWITCH_PROOF_HASH_MISMATCH")
                result_proof = result_switch_proofs.get(update_id)
                if result_proof != proof:
                    run_failures.append(prefix + "GENERATION_SWITCH_RESULT_BINDING_MISMATCH")
                if fixed_behavior:
                    if proof.get("mode") != FIXED_BEHAVIOR_NA_REASON:
                        run_failures.append(prefix + "FIXED_BEHAVIOR_NA_REASON_MISSING")
                    if proof.get("reason") != FIXED_BEHAVIOR_NA_REASON:
                        run_failures.append(prefix + "FIXED_BEHAVIOR_NA_REASON_INVALID")
                    if unit.get("generation_switch_receipt") is not None:
                        run_failures.append(prefix + "FIXED_BEHAVIOR_UNEXPECTED_SWITCH_RECEIPT")
                else:
                    receipt_path = r4_dir / "generation_switch_receipts" / f"{update_id}.json"
                    persisted_receipt = (
                        json.loads(receipt_path.read_text(encoding="utf-8"))
                        if receipt_path.exists()
                        else None
                    )
                    valid, failure = _full_receipt_valid(
                        proof=proof,
                        record=record,
                        unit=unit,
                        persisted_receipt=persisted_receipt,
                    )
                    if not valid:
                        run_failures.append(prefix + str(failure))

        if committed:
            initial_parent = identity.get("initial_parent_checkpoint_sha256")
            if str(committed[0].get("parent_checkpoint_sha256")) != str(initial_parent):
                run_failures.append("CHECKPOINT_CHAIN_INITIAL_PARENT_MISMATCH")
            for previous, current in zip(committed, committed[1:]):
                previous_id = str(previous.get("update_id"))
                bundle = checkpoint_bundles.get(previous_id)
                if bundle is None:
                    run_failures.append(
                        f"{previous_id}:CHECKPOINT_CHAIN_CHILD_BUNDLE_UNAVAILABLE"
                    )
                    continue
                derived_parent = _parent_sha_from_child_bundle(bundle)
                if derived_parent != str(current.get("parent_checkpoint_sha256")):
                    run_failures.append(
                        f"{previous_id}:CHECKPOINT_CHAIN_CHILD_TO_NEXT_PARENT_MISMATCH"
                    )
                if int(current.get("optimizer_step_before", -1)) != int(
                    previous.get("optimizer_step_after", -2)
                ):
                    run_failures.append(
                        f"{previous_id}:CHECKPOINT_CHAIN_OPTIMIZER_STATE_DISCONTINUITY"
                    )

        runs_audited += 1
        traced_updates += len(committed)
        details["runs"].append(
            {
                "run": str(relative),
                "committed_updates": len(committed),
                "failures": list(run_failures),
            }
        )
        failures.extend(run_failures)

    clean = failures == []
    checks["every_committed_update_traced"] = clean
    checks["checkpoint_chain_verified"] = clean
    checks["mandatory_s1_026_fields_verified"] = clean
    checks["generation_switch_verified"] = clean
    checks["durable_resolution_complete"] = clean
    checks["all_checks_pass"] = clean
    return {
        "schema": R4_AUDIT_SCHEMA,
        "artifact_root": str(root),
        "checks": checks,
        "details": {
            **details,
            "traced_update_count": traced_updates,
            "runs_audited": runs_audited,
        },
        "failures": sorted(set(failures)),
        "runs_audited": runs_audited,
        "traced_update_count": traced_updates,
        "all_checks_pass": clean,
    }


def install_r4_provenance_hooks_v1(qualification_module: Any) -> None:
    """Install evidence-only hooks into the canonical S1 qualification module."""
    qualification_module._run_job_v1 = run_job_with_r4_provenance_v1
    qualification_module.audit_exported_update_journal_v1 = (
        audit_exported_update_journal_r4_v1
    )
