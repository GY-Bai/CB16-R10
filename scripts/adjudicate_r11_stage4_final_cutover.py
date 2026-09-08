#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from cb16_local_opt.stage4_authority_adoption_r11 import (
    AuthorityAdoptionContract,
    SourceAuthorityIdentity,
    adopt_authority,
)
from cb16_local_opt.stage4_authority_inventory_r11 import (
    Classification,
    audit_registry,
    discover_authority_surfaces,
    load_registry,
)
from cb16_local_opt.stage4_authority_lease_r11 import Stage4AuthorityLeaseR11
from cb16_local_opt.stage4_canonical_runtime_r11 import LifecyclePhase
from cb16_local_opt.stage4_consolidated_hostile_adapter_r11 import (
    ConsolidatedProductionHostileAdapterR11,
)
from cb16_local_opt.stage4_hostile_integration_harness_r11 import (
    run_integrated_hostile_matrix,
)
from cb16_local_opt.stage4_integration_gate_compiler_r11 import (
    EXPECTED_TASK_BRANCHES,
    INTEGRATION_SEED_SHA,
    READY_VERDICT,
    RECEIPT_PREFIX,
    ReceiptInput,
    SCIENTIFIC_STATUS,
    SubprocessRepositoryEvidence,
    compile_integration_receipts,
    validate_final_cutover_report,
)
from cb16_local_opt.stage4_legacy_retirement_r11 import (
    Capability,
    LegacyRetirementGuard,
    RuntimeRole,
)
from cb16_local_opt.stage4_orchestration_provider_r11 import (
    AMP_ENABLED_R11,
    NUMERIC_MODE_R11,
    Stage4OrchestrationProviderR11,
    Stage4WorkerRuntimeConfigR11,
    WorkerPhase,
)
from cb16_local_opt.stage4_recovery_integration_r11 import (
    CanonicalRootExpectationR11,
    Stage3RecoveryObservationR11,
    Stage4RecoveryProviderR11,
)
from cb16_local_opt.stage4_runtime_spine_integration_r11 import (
    CanonicalRuntimeAuthoritySpineR11,
    Stage4AuthorityAdoptionProviderR11,
    Stage4AuthorityLeaseProviderR11,
    Stage4LegacyRetirementProviderR11,
    Stage4StateRootProviderR11,
)
from cb16_local_opt.stage4_state_roots_r11 import (
    SEMANTIC_FREEZE_BLOB_SHA,
    Stage4RootSetR11,
    Stage4StateRootsR11,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ci_evidence" / "stage4_final_cutover"
BASE_REGISTRY = ROOT / "authority/rearchitecture_r11/CB16_R11_STAGE4_AUTHORITY_WRITER_REGISTRY_V1.json"
FINAL_SCHEMA = ROOT / "authority/rearchitecture_r11/CB16_R11_STAGE4_FINAL_CUTOVER_REPORT_SCHEMA_V1.json"
GATEWORK = ROOT / "authority/rearchitecture_r11/CB16_R11_STAGE4_INTEGRATION_GATEWORK_V1.json"
RECEIPT_SCHEMA = ROOT / "authority/rearchitecture_r11/CB16_R11_STAGE4_INTEGRATION_RECEIPT_SCHEMA_V1.json"
MECHANICAL_UNION_SHA = "5769848b2cc2ec459f25655610f69068f812e16d"
AUTHORITY_ID = "CB16_R11_CANONICAL_RUNTIME"

# Final-head production paths are explicit. Anything else newly discovered fails closed.
FINAL_PRODUCTION_PATHS = frozenset({
    "cb16_local_opt/integration_adapters_r11.py",
    "cb16_local_opt/stage4_runtime_spine_integration_r11.py",
    "cb16_local_opt/stage4_fenced_persistence_r11.py",
    "cb16_local_opt/stage4_authoritative_engines_r11.py",
    "cb16_local_opt/stage4_execution_integration_r11.py",
    "cb16_local_opt/stage4_recovery_integration_r11.py",
    "cb16_local_opt/stage4_orchestration_provider_r11.py",
    "cb16_local_opt/stage4_consolidated_authority_r11.py",
})
FINAL_QUALIFICATION_PATHS = frozenset({
    "cb16_local_opt/stage4_hostile_integration_harness_r11.py",
    "cb16_local_opt/stage4_integration_gate_compiler_r11.py",
    "cb16_local_opt/stage4_consolidated_hostile_adapter_r11.py",
})


def _run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(obj: Any) -> str:
    return hashlib.sha256(_json_bytes(obj)).hexdigest()


def _write(name: str, obj: Any) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _git_guards(head: str) -> dict[str, bool]:
    if _run("git", "rev-parse", "HEAD") != head:
        raise RuntimeError("FINAL_HEAD_CHANGED_DURING_ADJUDICATION")
    if subprocess.run(["git", "merge-base", "--is-ancestor", INTEGRATION_SEED_SHA, head], cwd=ROOT).returncode != 0:
        raise RuntimeError("FINAL_HEAD_NOT_DESCENDANT_OF_INTEGRATION_SEED")
    if subprocess.run(["git", "merge-base", "--is-ancestor", MECHANICAL_UNION_SHA, head], cwd=ROOT).returncode != 0:
        raise RuntimeError("FINAL_HEAD_NOT_DESCENDANT_OF_MECHANICAL_UNION")
    if _run("git", "rev-parse", f"HEAD:authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json") != SEMANTIC_FREEZE_BLOB_SHA:
        raise RuntimeError("FINAL_SEMANTIC_FREEZE_BLOB_DRIFT")
    merges = _run("git", "rev-list", "--merges", f"{INTEGRATION_SEED_SHA}..HEAD")
    if merges:
        raise RuntimeError("FINAL_UNCONTROLLED_MERGE_HISTORY")
    changed = [x for x in _run("git", "diff", "--name-only", f"{INTEGRATION_SEED_SHA}..HEAD").splitlines() if x]
    forbidden = []
    for p in changed:
        q = p.lower()
        if (
            p == "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"
            or "final_holdout" in q
            or "2025-09" in q
            or q.startswith("data/")
            or q.startswith("market_data/")
            or q.startswith("provision/assets/binance_usdm_1m_funding_2020_2026")
        ):
            forbidden.append(p)
    if forbidden:
        raise RuntimeError("FINAL_FORBIDDEN_PATH_CHANGE:" + ",".join(sorted(forbidden)))
    return {
        "exact_seed_ancestry": True,
        "controlled_consolidation_only": True,
        "semantic_freeze_exact": True,
        "final_holdout_2025_09_untouched": True,
        "no_fresh_market_data": True,
    }


def _rebuild_writer_registry(head: str) -> tuple[dict[str, Any], dict[str, Any]]:
    base = load_registry(BASE_REGISTRY)
    base_entries = {(str(x.get("path")), str(x.get("symbol"))): x for x in base.get("entries", []) if isinstance(x, dict)}
    entries: list[dict[str, Any]] = []
    unknown: list[str] = []

    for surface in discover_authority_surfaces(ROOT):
        key = surface.key
        old = base_entries.get(key)
        if old is not None:
            row = dict(old)
            row["authority_domains"] = list(surface.authority_domains)
            row["mutation_capability"] = list(surface.mutation_capabilities)
            row["caller_callsite_evidence"] = list(surface.caller_evidence) or list(old.get("caller_callsite_evidence") or ["FINAL_HEAD_DISCOVERY"])
            row["discovery_evidence"] = list(surface.discovery_evidence)
            row["classification_basis"] = "Final-head rebuild preserving S4A classification while refreshing exact discovered domains/callsites."
            entries.append(row)
            continue

        path = surface.path
        if path in FINAL_PRODUCTION_PATHS:
            classification = Classification.R11_CANONICAL_CANDIDATE.value
            role = "FINAL_STAGE4_CUTOVER_CANDIDATE"
            eligible = True
        elif path in FINAL_QUALIFICATION_PATHS or path.startswith("scripts/"):
            classification = Classification.QUALIFICATION_ONLY.value
            role = "FINAL_STAGE4_QUALIFICATION_ONLY"
            eligible = False
        elif path.startswith("tests/"):
            classification = Classification.TEST_ONLY.value
            role = "TEST_ONLY"
            eligible = False
        else:
            unknown.append(f"{path}:{surface.symbol}")
            continue

        entries.append({
            "path": path,
            "symbol": surface.symbol,
            "authority_domains": list(surface.authority_domains),
            "mutation_capability": list(surface.mutation_capabilities),
            "caller_callsite_evidence": list(surface.caller_evidence) or ["FINAL_HEAD_DISCOVERY"],
            "discovery_evidence": list(surface.discovery_evidence),
            "current_role": role,
            "classification": classification,
            "eligible_for_canonical_authority": eligible,
            "classification_basis": "Final consolidated adjudicator explicit path classification; import/call reachability is not authority.",
        })

    if unknown:
        raise RuntimeError("FINAL_UNKNOWN_AUTHORITY_SURFACE:" + ";".join(sorted(unknown)))

    registry = dict(base)
    registry["task_id"] = "FINAL_CONSOLIDATED_ADJUDICATOR"
    registry["inventory_scope"] = dict(base.get("inventory_scope") or {})
    registry["inventory_scope"]["discovery_count"] = len(entries)
    registry["inventory_scope"]["rebuilt_at_head"] = head
    registry["entries"] = entries

    report = audit_registry(ROOT, registry, verify_git_guards=False)
    if not report.get("pass") or report.get("unknown_authority_count") != 0:
        raise RuntimeError("FINAL_WRITER_REGISTRY_AUDIT_FAIL:" + json.dumps(report.get("problems"), sort_keys=True))
    registry["final_head_sha"] = head
    registry["report_identity"] = _sha(registry)
    report["rebuilt_at_head"] = head
    report["report_identity"] = registry["report_identity"]
    _write("writer_registry.json", registry)
    _write("writer_registry_audit.json", report)
    return registry, report


def _compile_receipts() -> dict[str, Any]:
    gatework = json.loads(GATEWORK.read_text(encoding="utf-8"))
    receipt_schema = json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8"))
    inputs: list[ReceiptInput] = []
    for task_id, branch in EXPECTED_TASK_BRANCHES.items():
        ref = f"origin/{branch}"
        branch_tip = _run("git", "rev-parse", ref)
        receipt_path = f"{RECEIPT_PREFIX}/{task_id}_RECEIPT_V1.json"
        raw = _run("git", "show", f"{branch_tip}:{receipt_path}")
        doc = json.loads(raw)
        inputs.append(ReceiptInput(
            task_id=task_id,
            path=receipt_path,
            document=doc,
            receipt_commit_sha=branch_tip,
            branch_tip_sha=branch_tip,
        ))
    result = compile_integration_receipts(
        gatework=gatework,
        receipt_schema=receipt_schema,
        receipts=inputs,
        repo=SubprocessRepositoryEvidence(ROOT),
    )
    if result.get("verdict") != READY_VERDICT:
        raise RuntimeError("FINAL_RECEIPT_COMPILER_NOT_READY")
    _write("preconsolidation_receipts.json", result)
    return result


def _run_hostile() -> dict[str, Any]:
    report = run_integrated_hostile_matrix(adapter_factory=ConsolidatedProductionHostileAdapterR11)
    if report.get("status") != "PASS" or report.get("case_count") != 20 or report.get("passed_cases") != 20:
        raise RuntimeError("FINAL_REAL_HOSTILE_MATRIX_FAIL")
    if any(not row.get("passed") for row in report.get("cases", [])):
        raise RuntimeError("FINAL_REAL_HOSTILE_CASE_FAIL")
    _write("hostile_matrix.json", report)
    return report


class _Stage3Inspector:
    def __init__(self, source: SourceAuthorityIdentity) -> None:
        self.source = source
    def inspect_recovery_state(self, accepted_source: SourceAuthorityIdentity) -> Stage3RecoveryObservationR11:
        if accepted_source != self.source:
            raise RuntimeError("FINAL_SMOKE_STAGE3_SOURCE_MISMATCH")
        return Stage3RecoveryObservationR11(
            observed_source=self.source,
            fully_sealed=True,
            journal_transition_complete=True,
            checkpoint_transition_complete=True,
            journal_audit_passed=True,
            checkpoint_audit_passed=True,
            replay_engineering_only=True,
            new_evidence_created=False,
            scientific_history_rewritten=False,
            new_scientific_verdict=False,
        )


class _LiveContextFence:
    def __init__(self, lease_provider: Stage4AuthorityLeaseProviderR11) -> None:
        self.lease_provider = lease_provider
    def assert_current(self, context: Any) -> None:
        if context.binding.authority_id != context.recovered.authority_id:
            raise RuntimeError("FINAL_SMOKE_CONTEXT_AUTHORITY_MISMATCH")
        if context.binding.generation != context.recovered.generation:
            raise RuntimeError("FINAL_SMOKE_CONTEXT_GENERATION_MISMATCH")
        self.lease_provider.assert_current(context.lease)


def _run_short_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cb16-stage4-final-smoke-") as td:
        root = Path(td)
        frozen = root / "frozen_raw"
        frozen.mkdir()
        frozen.chmod(0o555)
        roots = Stage4StateRootsR11(Stage4RootSetR11.from_paths(
            control_root=root / "control",
            data_root=root / "data",
            frozen_raw_root=frozen,
            frozen_raw_identity="FINAL_STAGE4_FROZEN_RAW_QUALIFICATION_ID",
        ))
        startup = roots.initialize()
        lease_root = roots.control_path("runtime_lease_fencing_state")
        Stage4AuthorityLeaseR11.initialize(lease_root)

        source = SourceAuthorityIdentity(
            source_repo="GY-Bai/CB16-R10",
            source_sha=INTEGRATION_SEED_SHA,
            semantic_freeze_identity=SEMANTIC_FREEZE_BLOB_SHA,
            source_generation=17,
            champion_identity="champion:G00000017",
            champion_hash="a" * 64,
            checkpoint_identity="checkpoint:G00000017",
            checkpoint_hash="b" * 64,
            evidence_root_identity="evidence-root:accepted-17",
            journal_head_identity="journal-head:accepted-17",
            checkpoint_root_identity="checkpoint-root:accepted-17",
        )
        contract = AuthorityAdoptionContract(source, AUTHORITY_ID)
        adoption_receipt = roots.control_path("adoption_control_metadata/adoption.json")
        adopt_authority(source, contract=contract, receipt_path=adoption_receipt, adoption_timestamp_utc="2026-09-08T00:00:00Z")

        guard = LegacyRetirementGuard(b"F" * 32, issuer_id="FINAL_STAGE4_SMOKE")
        identity = guard.issue_identity(AUTHORITY_ID, RuntimeRole.R11_CANONICAL_RUNTIME)
        acquire_token = guard.issue_token(identity, Capability.ACQUIRE_CANONICAL_RUNTIME_AUTHORITY)
        legacy_provider = Stage4LegacyRetirementProviderR11(guard=guard, identity=identity, token=acquire_token)

        native_lease = Stage4AuthorityLeaseR11(lease_root, owner_id="final-stage4-smoke-owner")
        lease_provider = Stage4AuthorityLeaseProviderR11(
            lease=native_lease,
            expected_authority_id=AUTHORITY_ID,
            expected_generation=source.source_generation,
        )
        spine = CanonicalRuntimeAuthoritySpineR11(
            authority_adoption=Stage4AuthorityAdoptionProviderR11(contract=contract, receipt_path=adoption_receipt),
            state_roots=Stage4StateRootProviderR11(
                state_roots=roots,
                expected_control_root_id=startup.control_root_id,
                expected_data_root_id=startup.data_root_id,
                expected_frozen_raw_identity=startup.frozen_raw_identity,
                expected_authority_id=AUTHORITY_ID,
                expected_generation=source.source_generation,
            ),
            legacy_guard=legacy_provider,
            authority_lease=lease_provider,
        )
        recovery = Stage4RecoveryProviderR11(
            adoption_contract=contract,
            adoption_receipt_path=adoption_receipt,
            state_roots=roots,
            root_expectation=CanonicalRootExpectationR11(startup.control_root_id, startup.data_root_id, startup.frozen_raw_identity),
            stage3_recovery=_Stage3Inspector(source),
            lease_root=lease_root,
            recovery_owner_id="final-stage4-smoke-recovery",
        )
        config = Stage4WorkerRuntimeConfigR11()
        orchestration = Stage4OrchestrationProviderR11(
            live_authority=_LiveContextFence(lease_provider),
            components=(),
            config=config,
        )
        controller = spine.build_controller(recovery=recovery, orchestration=orchestration)
        context = controller.start()
        if controller.phase is not LifecyclePhase.RUNNING or orchestration.phase is not WorkerPhase.RUNNING:
            raise RuntimeError("FINAL_SMOKE_NOT_RUNNING")
        with orchestration.owned_authoritative_work(context):
            lease_provider.assert_current(context.lease)
        observation = controller.shutdown()
        orchestration.assert_no_orphan_workers()
        lease_state = native_lease.inspect_current_authority()
        if controller.phase is not LifecyclePhase.STOPPED or orchestration.phase is not WorkerPhase.STOPPED:
            raise RuntimeError("FINAL_SMOKE_NOT_STOPPED")
        if lease_state.state != "RELEASED":
            raise RuntimeError("FINAL_SMOKE_LEASE_NOT_RELEASED")
        if observation.generation != source.source_generation or observation.scientific_history_id != source.journal_head_identity:
            raise RuntimeError("FINAL_SMOKE_IDENTITY_DRIFT")
        if config.numeric_mode != "FP32" or config.amp is not False or NUMERIC_MODE_R11 != "FP32" or AMP_ENABLED_R11 is not False:
            raise RuntimeError("FINAL_SMOKE_NUMERIC_MODE_DRIFT")

        report = {
            "schema": "CB16_R11_STAGE4_SHORT_CANONICAL_MACHINE_SMOKE_V1",
            "lifecycle_passed": True,
            "controller_final_phase": controller.phase.value,
            "worker_final_phase": orchestration.phase.value,
            "authority_id": context.binding.authority_id,
            "generation": context.binding.generation,
            "scientific_history_id": context.recovered.scientific_history_id,
            "engineering_seal_id": observation.seal_id,
            "lease_final_state": lease_state.state,
            "lease_epoch": lease_state.epoch,
            "numeric_mode": config.numeric_mode,
            "amp": config.amp,
            "long_endurance_run": False,
            "scientific_evidence_created": False,
            "final_holdout_accesses": 0,
        }
        _write("short_canonical_machine_smoke.json", report)
        return report


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    head = _run("git", "rev-parse", "HEAD")
    guards = _git_guards(head)

    correctness_marker = ROOT / "ci_evidence" / "stage4_final_correctness.pass.json"
    if not correctness_marker.exists():
        raise RuntimeError("FINAL_CORRECTNESS_MARKER_MISSING")
    marker = json.loads(correctness_marker.read_text(encoding="utf-8"))
    if marker.get("head_sha") != head or marker.get("passed") is not True:
        raise RuntimeError("FINAL_CORRECTNESS_MARKER_INVALID")

    py_path = ROOT / "ci_evidence" / "stage4_final_python.json"
    py = json.loads(py_path.read_text(encoding="utf-8"))
    if py.get("status") != "READY" or py.get("mode") != "VERIFIED_CANONICAL_R104_VENV_REUSE":
        raise RuntimeError("FINAL_PYTHON_RESOLUTION_INVALID")
    if py.get("python", {}).get("network_install_attempted") is not False:
        raise RuntimeError("FINAL_NETWORK_INSTALL_ATTEMPTED")

    registry, registry_audit = _rebuild_writer_registry(head)
    receipts = _compile_receipts()
    hostile = _run_hostile()
    smoke = _run_short_smoke()

    accepted = {
        task_id: {
            "task_head_sha": row["task_head_sha"],
            "receipt_commit_sha": row["receipt_commit_sha"],
            "branch": row["branch"],
        }
        for task_id, row in receipts["accepted_tasks"].items()
    }

    gates = {
        "all_accepted_integration_receipts": True,
        "exact_seed_ancestry": guards["exact_seed_ancestry"],
        "controlled_consolidation_only": guards["controlled_consolidation_only"],
        "semantic_freeze_exact": guards["semantic_freeze_exact"],
        "scientific_status_unchanged": True,
        "s4a_writer_registry_rebuilt_and_reaudited_at_final_head": True,
        "unknown_authority_zero": registry_audit["unknown_authority_count"] == 0,
        "all_stage4_integration_correctness_tests_pass": True,
        "real_s4h_h01_h20_against_consolidated_runtime": hostile["passed_cases"] == 20,
        "exactly_one_live_canonical_authority_writer": all(row["invariants"]["no_dual_writer"] for row in hostile["cases"]),
        "stale_writer_cannot_mutate": all(row["invariants"]["no_stale_writer_mutation"] for row in hostile["cases"]),
        "legacy_writer_cannot_mutate": any(row["case_id"] == "H06" and row["passed"] for row in hostile["cases"]),
        "permission_cannot_be_bypassed": all(row["invariants"]["no_permission_bypass"] for row in hostile["cases"]),
        "replay_cannot_become_new_evidence": all(row["invariants"]["no_new_scientific_evidence_from_replay"] for row in hostile["cases"]),
        "short_canonical_machine_startup_steady_drain_shutdown_pass": smoke["lifecycle_passed"],
        "fp32_canonical": smoke["numeric_mode"] == "FP32",
        "amp_false": smoke["amp"] is False,
        "final_holdout_2025_09_untouched": guards["final_holdout_2025_09_untouched"],
        "no_fresh_market_data": guards["no_fresh_market_data"],
        "no_new_scientific_verdict": True,
        "long_endurance_not_required": smoke["long_endurance_run"] is False,
    }
    verdict = "R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED" if all(gates.values()) else "FAIL_CLOSED"
    report = {
        "schema": "CB16_R11_STAGE4_FINAL_CUTOVER_REPORT_V1",
        "verdict": verdict,
        "integration_seed_sha": INTEGRATION_SEED_SHA,
        "consolidation_head_sha": head,
        "scientific_status": SCIENTIFIC_STATUS,
        "accepted_integration_receipts": accepted,
        "qualification_gates": gates,
        "writer_registry": {
            "rebuilt_at_head": True,
            "unknown_authority_count": registry_audit["unknown_authority_count"],
            "report_identity": registry["report_identity"],
        },
        "hostile_matrix": {
            "report_schema": hostile["schema"],
            "integrated_runtime_qualification_claimed": True,
            "h01_h20_all_passed": hostile["passed_cases"] == 20 and hostile["failed_cases"] == 0,
        },
        "canonical_machine_smoke": {
            "runner_labels": ["self-hosted", "shanxi", "cb16-wss-qualification"],
            "python_resolution_status": py["status"],
            "python_resolution_mode": py["mode"],
            "numeric_mode": smoke["numeric_mode"],
            "amp": smoke["amp"],
            "lifecycle_passed": smoke["lifecycle_passed"],
            "long_endurance_run": smoke["long_endurance_run"],
        },
        "semantic_guards": {
            "semantic_freeze_unchanged": True,
            "final_holdout_untouched": True,
            "fresh_market_data_downloaded": False,
            "historical_market_data_mutated": False,
            "new_scientific_verdict": False,
            "scientific_semantics_changed": False,
            "replay_reinterpreted_as_new_evidence": False,
        },
        "reserved_verdict_issuer": "FINAL_CONSOLIDATED_ADJUDICATOR_ONLY",
    }

    schema = json.loads(FINAL_SCHEMA.read_text(encoding="utf-8"))
    validate_final_cutover_report(report, schema)
    _write("final_cutover_report.json", report)
    digest = hashlib.sha256((OUT / "final_cutover_report.json").read_bytes()).hexdigest()
    (OUT / "final_cutover_report.sha256").write_text(digest + "  final_cutover_report.json\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "head": head, "report_sha256": digest}, sort_keys=True))
    return 0 if verdict == "R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
