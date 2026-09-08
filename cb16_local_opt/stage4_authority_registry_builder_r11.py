from __future__ import annotations

"""Explicit S4A classification policy for the authority-writer inventory.

This builder is intentionally fail closed. Every repository path observed by the
static discoverer must be named in exactly one set below. A newly discovered path
is emitted as UNKNOWN_AUTHORITY; no filename/call-reachability heuristic may
silently promote it to a canonical candidate.
"""

import json
from pathlib import Path
from typing import Any

from cb16_local_opt.stage4_authority_inventory_r11 import (
    Classification,
    GATEWORK_BASE_SHA,
    REGISTRY_SCHEMA,
    SCIENTIFIC_STATUS,
    SEMANTIC_FREEZE_BLOB_SHA,
    discover_authority_surfaces,
)

_CANONICAL_CANDIDATE_PATHS = frozenset(
    {
        "authority/account_physics_r0/CB16_ACCOUNT_PHYSICS_STATE_V1_R0/runtime/account_physics_runtime_r0.py",
        "authority/account_physics_r0/CB16_ACCOUNT_PHYSICS_STATE_V1_R0/runtime/v55/kernel/bar.py",
        "authority/account_physics_r0/CB16_ACCOUNT_PHYSICS_STATE_V1_R0/runtime/v55/kernel/engine.py",
        "authority/control_plane_r1/risk_supervisor_r1.py",
        "cb16_local_opt/checkpoint_store_r11.py",
        "cb16_local_opt/event_journal_r11.py",
        "cb16_local_opt/evidence_store_r11.py",
        "cb16_local_opt/integration_adapters_r11.py",
        "cb16_local_opt/io_runtime_r11.py",
        "cb16_local_opt/orchestrator_r11.py",
        "cb16_local_opt/training_runtime_r11.py",
    }
)

_QUALIFICATION_ONLY_PATHS = frozenset(
    {
        ".github/workflows/cb16-r11-stage2-burst-qualification.yml",
        ".github/workflows/cb16-r11-stage2-measurement-window-patch.yml",
        ".github/workflows/cb16-r11-stage2-overlap-calibration.yml",
        ".github/workflows/cb16-r11-stage2-performance-plane-patch.yml",
        ".github/workflows/cb16-r11-stage2-performance-smoke.yml",
        ".github/workflows/cb16-r11-stage2-pipeline-qualification.yml",
        ".github/workflows/cb16-r11-stage2-task-a-hosted.yml",
        ".github/workflows/cb16-r11-stage2-task-a-shanxi-micro.yml",
        ".github/workflows/cb16-r11-stage2-task-a-shanxi-trace-process.yml",
        ".github/workflows/cb16-r11-stage2-task-b-gpu-microbench.yml",
        ".github/workflows/cb16-r11-stage2-task-c-io-microbench.yml",
        ".github/workflows/cb16-r11-stage2-task-d-microbench.yml",
        ".github/workflows/cb16-r11-stage2-task-e-microbenchmark.yml",
        ".github/workflows/cb16-r11-stage3-e1-data-plane-soak.yml",
        ".github/workflows/cb16-r11-stage3-e3-crash-recovery.yml",
        ".github/workflows/cb16-r11-stage3-integrated-smoke.yml",
        ".github/workflows/cb16-r11-task-f-qualification-v2.yml",
        ".github/workflows/cb16-r11-task-f-qualification.yml",
        ".github/workflows/cb16-r11-teacher-vectorized-qualification.yml",
        ".github/workflows/guard.yml",
        "benchmarks/task_c_io_microbench_r11.py",
        "cb16_local_opt/burst_orchestrator_r11.py",
        "cb16_local_opt/stage3_e2_lineage_soak_r11.py",
        "cb16_local_opt/stage3_e3_crash_recovery_r11.py",
        "scripts/benchmark_burst_orchestrator_r11.py",
        "scripts/benchmark_r2_evidence_storage.py",
        "scripts/qualify_r2_legacy_migration.py",
        "scripts/run_r11_stage2_integration_burst.py",
        "scripts/run_r11_stage2_overlap_calibration.py",
        "scripts/run_r11_stage2_pipeline_qualification.py",
        "scripts/run_r11_stage2_task_e_microbenchmark.py",
        "scripts/run_r11_stage3_e1_data_plane_soak.py",
        "scripts/run_r11_stage3_integrated_e3_matrix.py",
        "scripts/run_r11_task_b_gpu_microbench.py",
        "scripts/run_r11_task_f_acceptance.py",
    }
)

_TEST_ONLY_PATHS = frozenset(
    {
        "tests/test_burst_orchestrator_r11.py",
        "tests/test_gpu_runtime_r11.py",
        "tests/test_provision_asset_integrity.py",
        "tests/test_provision_mirror_routing.py",
        "tests/test_r102_core.py",
        "tests/test_r102_evidence_incremental.py",
        "tests/test_r11_async_orchestrator.py",
        "tests/test_r11_checkpoint_store.py",
        "tests/test_r11_event_journal.py",
        "tests/test_r11_evidence_store.py",
        "tests/test_r11_io_runtime.py",
        "tests/test_r11_stage2_burst_qualification.py",
        "tests/test_r11_stage2_task_a_cpu_runtime.py",
        "tests/test_r11_storage_lifecycle.py",
        "tests/test_r11_task_f_integration.py",
        "tests/test_r2_evidence_storage.py",
        "tests/test_r2_evidence_storage_incremental.py",
        "tests/test_r2_frozen_guard.py",
        "tests/test_r2_legacy_migration.py",
        "tests/test_stage3_e2_release_boundary_r11.py",
        "tests/test_stage4_authority_inventory_r11.py",
        "tests/test_training_runtime_r11.py",
    }
)

_NON_AUTHORITATIVE_READ_ONLY_PATHS = frozenset(
    {
        "cb16_local_opt/runtime_protocols_r11.py",
    }
)

_LEGACY_REFERENCE_ONLY_PATHS = frozenset(
    {
        "cb16_local_opt/async_trajectory_pipeline.py",
        "cb16_local_opt/campaign_stress_r7.py",
        "cb16_local_opt/campaign_telemetry_r9.py",
        "cb16_local_opt/checkpoint_recovery.py",
        "cb16_local_opt/conformance_guards.py",
        "cb16_local_opt/data_factory_supervisor_r8.py",
        "cb16_local_opt/dataset_catalog_r8.py",
        "cb16_local_opt/dataset_snapshot_r9.py",
        "cb16_local_opt/event_source_contracts.py",
        "cb16_local_opt/experience_maintenance_r7.py",
        "cb16_local_opt/factory_queue_r8.py",
        "cb16_local_opt/final_control_examiner_r7.py",
        "cb16_local_opt/final_controls_r7.py",
        "cb16_local_opt/final_holdout_r6.py",
        "cb16_local_opt/generation_orchestrator.py",
        "cb16_local_opt/gpu_training_policy.py",
        "cb16_local_opt/h72_worker_scaling_r8.py",
        "cb16_local_opt/historical_campaign_plugins_r6.py",
        "cb16_local_opt/historical_campaign_plugins_r7.py",
        "cb16_local_opt/incoming_bundle_watcher_r8.py",
        "cb16_local_opt/long_campaign_stress_r7.py",
        "cb16_local_opt/long_run_controller.py",
        "cb16_local_opt/multi_generation_runner_r5.py",
        "cb16_local_opt/performance_qualification_r8.py",
        "cb16_local_opt/probabilistic_teacher_r5.py",
        "cb16_local_opt/probabilistic_teacher_r6.py",
        "cb16_local_opt/r102_campaign.py",
        "cb16_local_opt/r102_evidence_cache.py",
        "cb16_local_opt/r102_learning.py",
        "cb16_local_opt/r102_physics.py",
        "cb16_local_opt/r102_teacher_incremental.py",
        "cb16_local_opt/r10_ci_reporting.py",
        "cb16_local_opt/r10_run_root_lock.py",
        "cb16_local_opt/r2_campaign.py",
        "cb16_local_opt/r2_event_journal.py",
        "cb16_local_opt/r2_evidence_storage.py",
        "cb16_local_opt/r2_evidence_storage_incremental.py",
        "cb16_local_opt/r2_learning_bridge.py",
        "cb16_local_opt/r2_legacy_migration.py",
        "cb16_local_opt/r2_sequential_audit.py",
        "cb16_local_opt/r2_training_incremental.py",
        "cb16_local_opt/runtime_authority_r9.py",
        "cb16_local_opt/sharded_experience_lake.py",
        "cb16_local_opt/short_campaign_runner_r9.py",
        "cb16_local_opt/vectorized_physics.py",
        "ci/relay/main.py",
        "ci/run_r10_8w_phase_control.py",
        "ci/worker/worker.py",
        "provision/scripts/provision_assets.py",
    }
)

_SETS = {
    Classification.R11_CANONICAL_CANDIDATE.value: _CANONICAL_CANDIDATE_PATHS,
    Classification.QUALIFICATION_ONLY.value: _QUALIFICATION_ONLY_PATHS,
    Classification.TEST_ONLY.value: _TEST_ONLY_PATHS,
    Classification.NON_AUTHORITATIVE_READ_ONLY.value: _NON_AUTHORITATIVE_READ_ONLY_PATHS,
    Classification.LEGACY_REFERENCE_ONLY.value: _LEGACY_REFERENCE_ONLY_PATHS,
}


def _validate_policy_disjoint() -> None:
    seen: dict[str, str] = {}
    for classification, paths in _SETS.items():
        for path in paths:
            prior = seen.get(path)
            if prior is not None:
                raise RuntimeError(f"STAGE4_CLASSIFICATION_POLICY_OVERLAP:{path}:{prior}:{classification}")
            seen[path] = classification


def classification_for_path(path: str) -> str:
    _validate_policy_disjoint()
    hits = [classification for classification, paths in _SETS.items() if path in paths]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(f"STAGE4_CLASSIFICATION_POLICY_OVERLAP:{path}:{hits}")
    return Classification.UNKNOWN_AUTHORITY.value


def _role(classification: str, path: str) -> str:
    if classification == Classification.R11_CANONICAL_CANDIDATE.value:
        if path.startswith("authority/account_physics_r0/"):
            return "FROZEN_PHYSICS_IMPLEMENTATION__CONTRACT_GOVERNED_CANDIDATE_ONLY"
        if path == "authority/control_plane_r1/risk_supervisor_r1.py":
            return "FROZEN_PERMISSION_SUPERVISOR__CONTRACT_GOVERNED_CANDIDATE_ONLY"
        return "CURRENT_R11_IMPLEMENTATION__CANDIDATE_ONLY"
    if classification == Classification.QUALIFICATION_ONLY.value:
        return "QUALIFICATION_BENCHMARK_ACCEPTANCE_OR_CI_HARNESS"
    if classification == Classification.TEST_ONLY.value:
        return "TEST_FIXTURE_OR_HOSTILE_TEST_WRITER"
    if classification == Classification.NON_AUTHORITATIVE_READ_ONLY.value:
        return "PROTOCOL_OR_DECLARATION_ONLY"
    if classification == Classification.LEGACY_REFERENCE_ONLY.value:
        return "LEGACY_OR_NONCANONICAL_ENGINEERING_REFERENCE"
    return "UNRESOLVED_AUTHORITY_ROLE"


def build_registry(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    discoveries = discover_authority_surfaces(root)
    entries: list[dict[str, Any]] = []
    for surface in discoveries:
        classification = classification_for_path(surface.path)
        entries.append(
            {
                "path": surface.path,
                "symbol": surface.symbol,
                "authority_domains": list(surface.authority_domains),
                "mutation_capability": list(surface.mutation_capabilities),
                "caller_callsite_evidence": list(surface.caller_evidence),
                "discovery_evidence": list(surface.discovery_evidence),
                "current_role": _role(classification, surface.path),
                "classification": classification,
                "eligible_for_canonical_authority": classification == Classification.R11_CANONICAL_CANDIDATE.value,
                "classification_basis": "Explicit S4A path review against Gatework and Semantic Freeze; import/call reachability is not authority.",
            }
        )
    return {
        "schema": REGISTRY_SCHEMA,
        "task_id": "S4A",
        "gatework_base_sha": GATEWORK_BASE_SHA,
        "semantic_freeze_blob_sha": SEMANTIC_FREEZE_BLOB_SHA,
        "scientific_status": SCIENTIFIC_STATUS,
        "authority_rule": "SEMANTIC_CONTRACTS_ARE_AUTHORITY__LEGACY_PYTHON_IMPLEMENTATION_IS_NOT_AUTHORITY",
        "classification_semantics": {
            "R11_CANONICAL_CANDIDATE": "Current contract-governed writer surface eligible for later integrated cutover review; this registry does not grant canonical authority.",
            "LEGACY_REFERENCE_ONLY": "Legacy/oracle/reference/compatibility surface; call/import reachability cannot upgrade it to authority.",
            "QUALIFICATION_ONLY": "Stage qualification, benchmark, acceptance or CI harness; may exercise writers but cannot own production authority.",
            "TEST_ONLY": "Test fixture/harness writer; cannot own production authority.",
            "NON_AUTHORITATIVE_READ_ONLY": "Protocol/declaration/read-only surface; no authoritative mutation role.",
            "UNKNOWN_AUTHORITY": "Fail-closed unresolved mutation surface; any occurrence makes the audit fail.",
        },
        "inventory_scope": {
            "scan_extensions": [".py", ".yml", ".yaml", ".sh"],
            "surface_identity": "(repository path, symbol/function/class/workflow)",
            "discovery_count": len(discoveries),
            "unknown_must_fail": True,
        },
        "entries": entries,
    }


def write_registry(repo_root: str | Path, output: str | Path) -> dict[str, Any]:
    registry = build_registry(repo_root)
    path = Path(output)
    if not path.is_absolute():
        path = Path(repo_root).resolve() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return registry


__all__ = ["build_registry", "classification_for_path", "write_registry"]
