from __future__ import annotations

"""CB16 R11 Stage-3 E3 deterministic crash/restart qualification harness.

This module is infrastructure-only.  It drives the already-qualified R11 control plane
with frozen synthetic/control receipts, kills a child process at explicit durability
boundaries, then reopens the concrete R11 stores and audits recovery.  It never opens
market data or the final holdout and never interprets replay as scientific evidence.
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from .checkpoint_store_r11 import CheckpointStoreR11
from .evidence_store_r11 import (
    EvidenceItemR11,
    EvidenceStoreR11,
    FAIL_AFTER_PAYLOAD_FSYNC_BEFORE_INDEX,
)
from .event_journal_r11 import (
    EventItemR11,
    EventJournalR11,
    FAIL_AFTER_EVENTS_BEFORE_BATCH_SEAL,
)
from .integration_adapters_r11 import (
    CheckpointStoreProtocolAdapterR11,
    EvidenceStoreProtocolAdapterR11,
    EventJournalProtocolAdapterR11,
)
from .orchestrator_r11 import IllegalTransition, LineageViolation, R11Orchestrator
from .runtime_events_r11 import (
    AuthorityStamp,
    EvidenceRef,
    EventKind,
    FrozenInputReceipt,
    GenerationState,
    PolicyResult,
    SnapshotSeal,
    TournamentDecision,
    TournamentResult,
    TrainingResult,
    ValidationResult,
)

SCHEMA = "CB16_R11_STAGE3_E3_CRASH_RECOVERY_REPORT_V1"
BASE_COMMIT = "92f1ca011aec27bbe9e89ac5ba9afe74817ee2de"
SEMANTIC_FREEZE_PATH = "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"
SEMANTIC_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"
BRANCH = "ai/r11-stage3-e3-crash-recovery-r0"
INJECTED_EXIT_CODE = 86
TEACHER_AUTHORITY_ID = "teacher-v7-qualified"
PHYSICS_AUTHORITY_ID = "frozen-physics-v1"


class CrashBoundary(str, Enum):
    BEFORE_SNAPSHOT_SEAL = "01_BEFORE_SNAPSHOT_SEAL"
    AFTER_SNAPSHOT_BEFORE_TRAINING_START = "02_AFTER_SNAPSHOT_SEAL_BEFORE_TRAINING_START"
    AFTER_TRAINING_BEFORE_VALIDATION = "03_AFTER_TRAINING_COMPLETION_BEFORE_VALIDATION"
    AFTER_VALIDATION_BEFORE_TOURNAMENT = "04_AFTER_VALIDATION_BEFORE_TOURNAMENT_DECISION"
    AFTER_TOURNAMENT_BEFORE_COMMIT = "05_AFTER_TOURNAMENT_DECISION_BEFORE_COMMIT"
    AFTER_COMMIT_BEFORE_CHECKPOINT_SEAL = "06_AFTER_COMMIT_BEFORE_CHECKPOINT_SEAL"
    AFTER_CHECKPOINT_BEFORE_RELEASE = "07_AFTER_CHECKPOINT_SEAL_BEFORE_NEXT_GENERATION_RELEASE"
    DURING_EVIDENCE_STORAGE_WRITE = "08_DURING_EVIDENCE_STORAGE_WRITE"
    DURING_JOURNAL_WRITE = "09_DURING_JOURNAL_WRITE"
    DURING_FINAL_DRAIN_BARRIER = "10_DURING_FINAL_DRAIN_BARRIER"


FAILURE_TAXONOMY = (
    "E3_FAULT_NOT_TRIGGERED",
    "E3_UNEXPECTED_CHILD_EXIT",
    "E3_RECOVERY_EXCEPTION",
    "E3_DUPLICATE_COMMIT",
    "E3_DUPLICATE_RELEASE",
    "E3_ORPHAN_PROMOTED_CHAMPION",
    "E3_REJECTED_CHALLENGER_BECAME_AUTHORITY",
    "E3_JOURNAL_AUDIT_FAILED",
    "E3_CHECKPOINT_AUDIT_FAILED",
    "E3_EVIDENCE_AUDIT_FAILED",
    "E3_CHECKPOINT_JOURNAL_MISMATCH",
    "E3_CONTENT_ADDRESSED_REPLAY_DUPLICATED_OBJECT",
    "E3_PARTIAL_WRITE_RECOVERY_FAILED",
    "E3_STALE_WORK_ACCEPTED_AFTER_RESTART",
    "E3_TEACHER_AUTHORITY_DRIFT",
    "E3_PHYSICS_AUTHORITY_DRIFT",
    "E3_FP32_CONTRACT_DRIFT",
    "E3_AMP_CONTRACT_DRIFT",
    "E3_FROZEN_AUTHORITY_DRIFT",
    "E3_FINAL_HOLDOUT_OR_FROZEN_PATH_CHANGED",
    "E3_UNSCOPED_REPOSITORY_CHANGE",
)

ALLOWED_CHANGED_PATHS = {
    "cb16_local_opt/stage3_e3_crash_recovery_r11.py",
    "tests/test_stage3_e3_crash_recovery_r11.py",
    ".github/workflows/cb16-r11-stage3-e3-crash-recovery.yml",
}


@dataclass(frozen=True)
class CaseConfig:
    parent_hash: str
    challenger_hash: str


@dataclass
class OpenedCase:
    evidence_impl: EvidenceStoreR11
    journal_impl: EventJournalR11
    checkpoint_impl: CheckpointStoreR11
    runtime: R11Orchestrator

    def close(self) -> None:
        self.checkpoint_impl.close()
        self.journal_impl.close()
        self.evidence_impl.close()


def _fsync_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _config_path(root: Path) -> Path:
    return root / "e3_case_config.json"


def _marker_path(root: Path) -> Path:
    return root / "e3_pre_crash.json"


def _read_config(root: Path) -> CaseConfig:
    raw = json.loads(_config_path(root).read_text())
    return CaseConfig(str(raw["parent_hash"]), str(raw["challenger_hash"]))


def _authority(cfg: CaseConfig) -> AuthorityStamp:
    return AuthorityStamp(
        generation=1,
        champion_id=cfg.parent_hash,
        champion_hash=cfg.parent_hash,
        teacher_authority_id=TEACHER_AUTHORITY_ID,
        physics_authority_id=PHYSICS_AUTHORITY_ID,
    )


def _open_case(root: Path, *, recover: bool) -> OpenedCase:
    cfg = _read_config(root)
    evidence_impl = EvidenceStoreR11(
        metadata_root=root / "evidence_meta",
        payload_roots=[root / "evidence_payload"],
        segment_target_bytes=4096,
        codec="none",
        read_only_source_roots=[root / "frozen_source_never_written"],
    )
    journal_impl = EventJournalR11(root / "journal")
    checkpoint_impl = CheckpointStoreR11(root / "checkpoints")
    runtime = R11Orchestrator(
        authority=_authority(cfg),
        evidence_store=EvidenceStoreProtocolAdapterR11(evidence_impl),
        journal=EventJournalProtocolAdapterR11(journal_impl),
        checkpoint_store=CheckpointStoreProtocolAdapterR11(checkpoint_impl),
        recover=recover,
    )
    return OpenedCase(evidence_impl, journal_impl, checkpoint_impl, runtime)


def _prepare_case(root: Path) -> CaseConfig:
    root.mkdir(parents=True, exist_ok=True)
    store = CheckpointStoreR11(root / "checkpoints")
    try:
        parent = store.put_state_dict({"w": torch.tensor([1.0], dtype=torch.float32)})
        challenger = store.put_state_dict({"w": torch.tensor([2.0], dtype=torch.float32)})
        cfg = CaseConfig(parent.semantic_sha256, challenger.semantic_sha256)
        _fsync_json(_config_path(root), asdict(cfg))
        return cfg
    finally:
        store.close()


def _teacher_evidence(cfg: CaseConfig) -> EvidenceRef:
    return EvidenceRef(
        evidence_id="teacher-e0",
        payload_hash="stage3-e3-frozen-teacher-replay-payload",
        source_generation=0,
        producer_champion_id="historical-champion-g0",
        teacher_authority_id=TEACHER_AUTHORITY_ID,
        physics_authority_id=PHYSICS_AUTHORITY_ID,
        teacher_generation=0,
    )


def _trace_evidence(cfg: CaseConfig) -> EvidenceRef:
    return EvidenceRef(
        evidence_id="trace-e1",
        payload_hash="stage3-e3-frozen-trace-replay-payload",
        source_generation=1,
        producer_champion_id=cfg.parent_hash,
        teacher_authority_id=TEACHER_AUTHORITY_ID,
        physics_authority_id=PHYSICS_AUTHORITY_ID,
        teacher_generation=1,
    )


def _snapshot(cfg: CaseConfig) -> SnapshotSeal:
    return SnapshotSeal(
        snapshot_id="stage3-e3-snapshot-g1",
        snapshot_hash="stage3-e3-snapshot-g1-hash",
        generation=1,
        parent_champion_id=cfg.parent_hash,
        evidence_ids=("teacher-e0",),
    )


def _training(cfg: CaseConfig) -> TrainingResult:
    return TrainingResult(
        work_id="stage3-e3-train-g1",
        generation=1,
        challenger_id=cfg.challenger_hash,
        challenger_hash=cfg.challenger_hash,
        parent_champion_id=cfg.parent_hash,
        snapshot_id="stage3-e3-snapshot-g1",
        payload_hash="stage3-e3-training-result",
    )


def _validation(cfg: CaseConfig) -> ValidationResult:
    return ValidationResult(
        work_id="stage3-e3-validation-g1",
        generation=1,
        challenger_id=cfg.challenger_hash,
        parent_champion_id=cfg.parent_hash,
        snapshot_id="stage3-e3-snapshot-g1",
        validation_id="stage3-e3-validation-id-g1",
        payload_hash="stage3-e3-validation-result",
    )


def _tournament(cfg: CaseConfig, decision: TournamentDecision) -> TournamentResult:
    return TournamentResult(
        work_id="stage3-e3-tournament-g1",
        generation=1,
        challenger_id=cfg.challenger_hash,
        parent_champion_id=cfg.parent_hash,
        validation_id="stage3-e3-validation-id-g1",
        decision=decision,
        payload_hash="stage3-e3-tournament-" + decision.value.lower(),
    )


def _mark_and_die(root: Path, boundary: CrashBoundary, state: str, phase: str) -> None:
    _fsync_json(
        _marker_path(root),
        {
            "boundary": boundary.value,
            "runtime_state": state,
            "phase": phase,
            "pid": os.getpid(),
        },
    )
    os._exit(INJECTED_EXIT_CODE)


def _drive_to_tournament(rt: R11Orchestrator, cfg: CaseConfig) -> None:
    rt.verify_frozen_inputs(FrozenInputReceipt("stage3-e3-frozen-input", "stage3-e3-frozen-seal", True))
    rt.accept_teacher_evidence(_teacher_evidence(cfg))
    rt.seal_training_snapshot(_snapshot(cfg))
    rt.accept_policy_result(PolicyResult("stage3-e3-policy-g1", 1, cfg.parent_hash, "stage3-e3-policy-result"))
    rt.start_trace("stage3-e3-trace-g1")
    rt.materialize_trace_evidence(_trace_evidence(cfg), work_id="stage3-e3-trace-complete-g1")
    rt.start_challenger_training(work_id="stage3-e3-train-g1")
    rt.complete_training(_training(cfg))
    rt.complete_validation(_validation(cfg))


def _child_control(root_s: str, boundary_s: str, decision_s: str) -> None:
    root = Path(root_s)
    boundary = CrashBoundary(boundary_s)
    decision = TournamentDecision(decision_s)
    cfg = _read_config(root)
    opened = _open_case(root, recover=False)
    rt = opened.runtime

    if boundary is CrashBoundary.DURING_EVIDENCE_STORAGE_WRITE:
        item = EvidenceItemR11(
            evidence_id="stage3-e3-storage-canary",
            parent_snapshot_hash="ENGINEERING_ONLY",
            lineage_hash="ENGINEERING_REPLAY_NOT_SCIENTIFIC_LEARNING",
            teacher_protocol_hash=TEACHER_AUTHORITY_ID,
            payload={"stage3": "E3", "kind": "frozen-engineering-replay", "value": 1},
        )

        def kill_storage(phase: str) -> None:
            if phase == FAIL_AFTER_PAYLOAD_FSYNC_BEFORE_INDEX:
                _mark_and_die(root, boundary, rt.record.state.value, phase)

        opened.evidence_impl.put_evidence([item], fail_hook=kill_storage)
        os._exit(91)

    if boundary is CrashBoundary.DURING_JOURNAL_WRITE:
        item = EventItemR11(
            event_id="stage3-e3-journal-canary",
            event_type="STAGE3_E3_ENGINEERING_CANARY",
            generation=1,
            policy_weight_hash=cfg.parent_hash,
            snapshot_hash="ENGINEERING_ONLY",
            lineage_hash="ENGINEERING_REPLAY_NOT_SCIENTIFIC_LEARNING",
            payload={"stage3": "E3", "kind": "journal-rollback-canary"},
        )

        def kill_journal(phase: str) -> None:
            if phase == FAIL_AFTER_EVENTS_BEFORE_BATCH_SEAL:
                _mark_and_die(root, boundary, rt.record.state.value, phase)

        opened.journal_impl.seal_trace_batch(
            [item], trace_batch_id="STAGE3-E3-JOURNAL-CANARY", fail_hook=kill_journal
        )
        os._exit(92)

    rt.verify_frozen_inputs(FrozenInputReceipt("stage3-e3-frozen-input", "stage3-e3-frozen-seal", True))
    rt.accept_teacher_evidence(_teacher_evidence(cfg))
    if boundary is CrashBoundary.BEFORE_SNAPSHOT_SEAL:
        _mark_and_die(root, boundary, rt.record.state.value, "BEFORE_SNAPSHOT_SEAL")

    rt.seal_training_snapshot(_snapshot(cfg))
    if boundary is CrashBoundary.AFTER_SNAPSHOT_BEFORE_TRAINING_START:
        _mark_and_die(root, boundary, rt.record.state.value, "SNAPSHOT_DURABLE")

    rt.accept_policy_result(PolicyResult("stage3-e3-policy-g1", 1, cfg.parent_hash, "stage3-e3-policy-result"))
    rt.start_trace("stage3-e3-trace-g1")
    rt.materialize_trace_evidence(_trace_evidence(cfg), work_id="stage3-e3-trace-complete-g1")
    rt.start_challenger_training(work_id="stage3-e3-train-g1")
    rt.complete_training(_training(cfg))
    if boundary is CrashBoundary.AFTER_TRAINING_BEFORE_VALIDATION:
        _mark_and_die(root, boundary, rt.record.state.value, "TRAINING_COMPLETION_DURABLE")

    rt.complete_validation(_validation(cfg))
    if boundary is CrashBoundary.AFTER_VALIDATION_BEFORE_TOURNAMENT:
        _mark_and_die(root, boundary, rt.record.state.value, "VALIDATION_DURABLE")

    rt.decide_tournament(_tournament(cfg, decision))
    if boundary is CrashBoundary.AFTER_TOURNAMENT_BEFORE_COMMIT:
        _mark_and_die(root, boundary, rt.record.state.value, "TOURNAMENT_DECISION_DURABLE")

    if boundary is CrashBoundary.AFTER_COMMIT_BEFORE_CHECKPOINT_SEAL:
        # Deliberately create the strict atomic-store/journal-ack gap.  Recovery must
        # discover the durable store commit and journal it exactly once.
        adapter = rt.checkpoint_store
        proposal = rt.record.tournament
        from .runtime_events_r11 import TournamentCommitProposal
        adapter.atomic_commit(
            TournamentCommitProposal(
                generation=1,
                decision=decision,
                parent_champion_id=cfg.parent_hash,
                challenger_id=cfg.challenger_hash,
                challenger_hash=cfg.challenger_hash,
            )
        )
        _mark_and_die(root, boundary, rt.record.state.value, "COMMIT_STORE_DURABLE_JOURNAL_ACK_PENDING")

    rt.commit_tournament()

    if boundary is CrashBoundary.AFTER_CHECKPOINT_BEFORE_RELEASE:
        # As above, make the durable checkpoint-store / journal-ack gap explicit.
        commit = rt.record.commit
        assert commit is not None
        from .runtime_events_r11 import CheckpointSeal
        seal = CheckpointSeal(
            generation=1,
            checkpoint_id=commit.next_champion_hash,
            state_digest=rt._state_digest(),  # fault harness observation only
            commit_id=commit.commit_id,
        )
        rt.checkpoint_store.seal_checkpoint(seal)
        _mark_and_die(root, boundary, rt.record.state.value, "CHECKPOINT_STORE_DURABLE_JOURNAL_ACK_PENDING")

    assert rt.record.commit is not None
    rt.seal_checkpoint(rt.record.commit.next_champion_hash)

    if boundary is CrashBoundary.DURING_FINAL_DRAIN_BARRIER:
        rt.release_next_generation()
        # Release is already durable.  Kill between durability barrier legs; restart
        # must not release again or change authority.
        opened.evidence_impl.checkpoint("FULL")
        _mark_and_die(root, boundary, rt.record.state.value, "FINAL_DRAIN_AFTER_EVIDENCE_BARRIER")

    # No requested boundary should reach here.
    os._exit(93)


def _resume_to_release(rt: R11Orchestrator, cfg: CaseConfig, decision: TournamentDecision) -> AuthorityStamp:
    if rt.record.state is GenerationState.PREPARING:
        if not rt.record.frozen_inputs_verified:
            rt.verify_frozen_inputs(FrozenInputReceipt("stage3-e3-frozen-input", "stage3-e3-frozen-seal", True))
        if "teacher-e0" not in rt.record.accepted_evidence:
            rt.accept_teacher_evidence(_teacher_evidence(cfg))
        rt.seal_training_snapshot(_snapshot(cfg))

    if rt.record.state is GenerationState.SNAPSHOT_SEALED:
        if rt.record.policy_work_id is None:
            rt.accept_policy_result(
                PolicyResult("stage3-e3-policy-g1", 1, cfg.parent_hash, "stage3-e3-policy-result")
            )
        rt.start_trace("stage3-e3-trace-g1")

    if rt.record.state is GenerationState.TRACE_RUNNING:
        if "trace-e1" not in rt.record.trace_evidence_ids:
            rt.materialize_trace_evidence(_trace_evidence(cfg), work_id="stage3-e3-trace-complete-g1")
        rt.start_challenger_training(work_id="stage3-e3-train-g1")

    if rt.record.state is GenerationState.CHALLENGER_TRAINING:
        rt.complete_training(_training(cfg))
    if rt.record.state is GenerationState.VALIDATING:
        rt.complete_validation(_validation(cfg))
    if rt.record.state is GenerationState.TOURNAMENT_PENDING:
        rt.decide_tournament(_tournament(cfg, decision))
    if rt.record.state is GenerationState.COMMITTING:
        commit = rt.commit_tournament()
        if rt.record.checkpoint is None:
            rt.seal_checkpoint(commit.next_champion_hash)
    if rt.record.state is not GenerationState.COMMITTED:
        raise RuntimeError(f"E3_UNEXPECTED_RESUME_STATE:{rt.record.state.value}")

    if rt.record.next_generation_released:
        commit = rt.record.commit
        assert commit is not None
        return AuthorityStamp(
            generation=2,
            champion_id=commit.next_champion_id,
            champion_hash=commit.next_champion_hash,
            teacher_authority_id=TEACHER_AUTHORITY_ID,
            physics_authority_id=PHYSICS_AUTHORITY_ID,
        )
    return rt.release_next_generation()


def _persisted_objects(opened: OpenedCase) -> dict[str, Any]:
    event_counts = {
        str(kind): int(count)
        for kind, count in opened.journal_impl.conn.execute(
            "SELECT event_type,COUNT(*) FROM events GROUP BY event_type ORDER BY event_type"
        )
    }
    def count(conn: Any, table: str) -> int:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    return {
        "journal_events": count(opened.journal_impl.conn, "events"),
        "journal_trace_batches": count(opened.journal_impl.conn, "trace_batches"),
        "event_counts": event_counts,
        "orchestrator_commits": count(opened.checkpoint_impl.conn, "orchestrator_commits"),
        "orchestrator_checkpoint_seals": count(opened.checkpoint_impl.conn, "orchestrator_checkpoint_seals"),
        "checkpoint_objects": count(opened.checkpoint_impl.conn, "checkpoint_objects"),
        "orchestrator_evidence_refs": count(opened.evidence_impl.conn, "orchestrator_evidence_refs"),
        "orchestrator_snapshots": count(opened.evidence_impl.conn, "orchestrator_snapshots"),
        "content_addressed_payloads": count(opened.evidence_impl.conn, "payloads"),
        "content_addressed_evidence": count(opened.evidence_impl.conn, "evidence_catalog"),
    }


def _journal_commit_release_counts(opened: OpenedCase) -> tuple[int, int, int]:
    def c(kind: EventKind) -> int:
        return int(
            opened.journal_impl.conn.execute(
                "SELECT COUNT(*) FROM events WHERE event_type=?", ("ORCH:" + kind.value,)
            ).fetchone()[0]
        )
    return c(EventKind.TOURNAMENT_COMMITTED), c(EventKind.CHECKPOINT_SEALED), c(EventKind.NEXT_GENERATION_RELEASED)


def _replay_storage_canary(opened: OpenedCase) -> dict[str, Any]:
    item = EvidenceItemR11(
        evidence_id="stage3-e3-storage-canary",
        parent_snapshot_hash="ENGINEERING_ONLY",
        lineage_hash="ENGINEERING_REPLAY_NOT_SCIENTIFIC_LEARNING",
        teacher_protocol_hash=TEACHER_AUTHORITY_ID,
        payload={"stage3": "E3", "kind": "frozen-engineering-replay", "value": 1},
    )
    recovered_before = int(opened.evidence_impl.startup_stats["recovered_payloads"])
    _, first = opened.evidence_impl.put_evidence([item])
    _, second = opened.evidence_impl.put_evidence([item])
    return {
        "startup_recovered_payloads": recovered_before,
        "first_replay": asdict(first),
        "second_replay": asdict(second),
        "exactly_one_payload": int(opened.evidence_impl.conn.execute("SELECT COUNT(*) FROM payloads WHERE content_hash=?", (item.content_hash,)).fetchone()[0]) == 1,
        "exactly_one_engineering_evidence_row": int(opened.evidence_impl.conn.execute("SELECT COUNT(*) FROM evidence_catalog WHERE evidence_id=?", (item.evidence_id,)).fetchone()[0]) == 1,
    }


def _replay_journal_canary(opened: OpenedCase, cfg: CaseConfig) -> dict[str, Any]:
    item = EventItemR11(
        event_id="stage3-e3-journal-canary",
        event_type="STAGE3_E3_ENGINEERING_CANARY",
        generation=1,
        policy_weight_hash=cfg.parent_hash,
        snapshot_hash="ENGINEERING_ONLY",
        lineage_hash="ENGINEERING_REPLAY_NOT_SCIENTIFIC_LEARNING",
        payload={"stage3": "E3", "kind": "journal-rollback-canary"},
    )
    before = int(opened.journal_impl.conn.execute("SELECT COUNT(*) FROM events WHERE event_id=?", (item.event_id,)).fetchone()[0])
    first = opened.journal_impl.seal_trace_batch([item], trace_batch_id="STAGE3-E3-JOURNAL-CANARY")
    second = opened.journal_impl.seal_trace_batch([item], trace_batch_id="STAGE3-E3-JOURNAL-CANARY")
    after = int(opened.journal_impl.conn.execute("SELECT COUNT(*) FROM events WHERE event_id=?", (item.event_id,)).fetchone()[0])
    return {
        "rows_after_crash_before_replay": before,
        "first_replay_reused_batch": bool(first.reused_batch),
        "second_replay_reused_batch": bool(second.reused_batch),
        "rows_after_two_replays": after,
    }


def _case_failures(
    opened: OpenedCase,
    cfg: CaseConfig,
    decision: TournamentDecision,
    next_authority: AuthorityStamp,
    storage_replay: dict[str, Any] | None,
    journal_replay: dict[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    journal_audit = opened.journal_impl.audit()
    checkpoint_audit = opened.checkpoint_impl.full_forensic_audit()
    evidence_audit = opened.evidence_impl.full_forensic_audit()
    if not journal_audit["pass"]:
        failures.append("E3_JOURNAL_AUDIT_FAILED")
    if not checkpoint_audit["pass"]:
        failures.append("E3_CHECKPOINT_AUDIT_FAILED")
    if not evidence_audit["pass"]:
        failures.append("E3_EVIDENCE_AUDIT_FAILED")

    commit_events, checkpoint_events, release_events = _journal_commit_release_counts(opened)
    if commit_events != 1 or int(opened.checkpoint_impl.conn.execute("SELECT COUNT(*) FROM orchestrator_commits WHERE generation=1").fetchone()[0]) != 1:
        failures.append("E3_DUPLICATE_COMMIT")
    if release_events != 1:
        failures.append("E3_DUPLICATE_RELEASE")
    if checkpoint_events != 1 or int(opened.checkpoint_impl.conn.execute("SELECT COUNT(*) FROM orchestrator_checkpoint_seals WHERE generation=1").fetchone()[0]) != 1:
        failures.append("E3_CHECKPOINT_JOURNAL_MISMATCH")

    cp_adapter = CheckpointStoreProtocolAdapterR11(opened.checkpoint_impl)
    commit = cp_adapter.read_commit(1)
    checkpoint = cp_adapter.read_checkpoint(1)
    if commit is None or checkpoint is None or checkpoint.commit_id != commit.commit_id:
        failures.append("E3_CHECKPOINT_JOURNAL_MISMATCH")
    else:
        expected = cfg.challenger_hash if decision is TournamentDecision.PROMOTE else cfg.parent_hash
        if commit.next_champion_id != expected or next_authority.champion_id != expected:
            failures.append(
                "E3_ORPHAN_PROMOTED_CHAMPION"
                if decision is TournamentDecision.PROMOTE
                else "E3_REJECTED_CHALLENGER_BECAME_AUTHORITY"
            )
        if decision is TournamentDecision.REJECT and next_authority.champion_id == cfg.challenger_hash:
            failures.append("E3_REJECTED_CHALLENGER_BECAME_AUTHORITY")

    if next_authority.teacher_authority_id != TEACHER_AUTHORITY_ID:
        failures.append("E3_TEACHER_AUTHORITY_DRIFT")
    if next_authority.physics_authority_id != PHYSICS_AUTHORITY_ID:
        failures.append("E3_PHYSICS_AUTHORITY_DRIFT")
    if torch.get_default_dtype() != torch.float32:
        failures.append("E3_FP32_CONTRACT_DRIFT")

    if storage_replay is not None:
        second = storage_replay["second_replay"]
        if (
            storage_replay["startup_recovered_payloads"] < 1
            or not storage_replay["exactly_one_payload"]
            or not storage_replay["exactly_one_engineering_evidence_row"]
            or int(second["created_payload_count"]) != 0
            or int(second["created_evidence_count"]) != 0
        ):
            failures.append("E3_CONTENT_ADDRESSED_REPLAY_DUPLICATED_OBJECT")
    if journal_replay is not None:
        if journal_replay["rows_after_crash_before_replay"] != 0 or journal_replay["rows_after_two_replays"] != 1 or not journal_replay["second_replay_reused_batch"]:
            failures.append("E3_PARTIAL_WRITE_RECOVERY_FAILED")

    audit = {
        "journal": journal_audit,
        "checkpoint": checkpoint_audit,
        "evidence": evidence_audit,
        "commit_event_count": commit_events,
        "checkpoint_event_count": checkpoint_events,
        "release_event_count": release_events,
    }
    return sorted(set(failures)), audit


def run_case(root: Path, boundary: CrashBoundary, decision: TournamentDecision) -> dict[str, Any]:
    cfg = _prepare_case(root)
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_child_control, args=(str(root), boundary.value, decision.value))
    started = time.monotonic()
    proc.start()
    proc.join(timeout=90)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=10)
        return {
            "boundary": boundary.value,
            "decision": decision.value,
            "pass": False,
            "failures": ["E3_FAULT_NOT_TRIGGERED"],
            "child_exit_code": proc.exitcode,
        }
    if proc.exitcode != INJECTED_EXIT_CODE:
        return {
            "boundary": boundary.value,
            "decision": decision.value,
            "pass": False,
            "failures": ["E3_UNEXPECTED_CHILD_EXIT"],
            "child_exit_code": proc.exitcode,
        }

    marker = json.loads(_marker_path(root).read_text()) if _marker_path(root).is_file() else {}
    opened: OpenedCase | None = None
    try:
        opened = _open_case(root, recover=True)
        restart_state = opened.runtime.record.state.value
        restart_released = bool(opened.runtime.record.next_generation_released)
        persisted_after_restart = _persisted_objects(opened)
        initial_authoritative_generation = 2 if restart_released else 1
        if restart_released and opened.runtime.record.commit is not None:
            initial_authoritative_champion = opened.runtime.record.commit.next_champion_id
        else:
            initial_authoritative_champion = cfg.parent_hash

        storage_replay = None
        journal_replay = None
        if boundary is CrashBoundary.DURING_EVIDENCE_STORAGE_WRITE:
            storage_replay = _replay_storage_canary(opened)
        if boundary is CrashBoundary.DURING_JOURNAL_WRITE:
            journal_replay = _replay_journal_canary(opened, cfg)

        next_authority = _resume_to_release(opened.runtime, cfg, decision)

        stale_rejected = False
        try:
            opened.runtime.accept_policy_result(
                PolicyResult("stage3-e3-stale-future-policy", 2, cfg.parent_hash, "stale")
            )
        except (LineageViolation, IllegalTransition):
            stale_rejected = True

        final_persisted = _persisted_objects(opened)
        failures, audits = _case_failures(
            opened, cfg, decision, next_authority, storage_replay, journal_replay
        )
        if not stale_rejected:
            failures.append("E3_STALE_WORK_ACCEPTED_AFTER_RESTART")

        # A second reopen must reconstruct the exact same terminal authority and must
        # not create another release/commit as a side effect.
        opened.close()
        opened = _open_case(root, recover=True)
        if not opened.runtime.record.next_generation_released:
            failures.append("E3_DUPLICATE_RELEASE")
        commit_events, checkpoint_events, release_events = _journal_commit_release_counts(opened)
        if commit_events != 1:
            failures.append("E3_DUPLICATE_COMMIT")
        if checkpoint_events != 1 or release_events != 1:
            failures.append("E3_DUPLICATE_RELEASE")

        replay_decision = {
            CrashBoundary.DURING_EVIDENCE_STORAGE_WRITE: "RECOVER_CONTENT_ADDRESSED_TAIL_THEN_IDEMPOTENT_ENGINEERING_REPLAY",
            CrashBoundary.DURING_JOURNAL_WRITE: "ROLLBACK_UNCOMMITTED_JOURNAL_TXN_THEN_IDEMPOTENT_ENGINEERING_REPLAY",
            CrashBoundary.AFTER_COMMIT_BEFORE_CHECKPOINT_SEAL: "RECONCILE_DURABLE_COMMIT_THEN_RESUME",
            CrashBoundary.AFTER_CHECKPOINT_BEFORE_RELEASE: "RECONCILE_DURABLE_CHECKPOINT_THEN_RELEASE_ONCE",
            CrashBoundary.DURING_FINAL_DRAIN_BARRIER: "NO_AUTHORITY_REPLAY_RELEASE_ALREADY_DURABLE",
        }.get(boundary, "RESUME_FROM_LAST_DURABLE_R11_BOUNDARY")

        failures = sorted(set(failures))
        return {
            "boundary": boundary.value,
            "decision": decision.value,
            "pre_crash_state": marker,
            "persisted_objects_after_restart": persisted_after_restart,
            "restart_state": {
                "generation_record": restart_state,
                "next_generation_released": restart_released,
            },
            "recovered_champion": initial_authoritative_champion,
            "recovered_generation": initial_authoritative_generation,
            "replay_decision": replay_decision,
            "engineering_storage_replay": storage_replay,
            "engineering_journal_replay": journal_replay,
            "final_authority": asdict(next_authority),
            "final_persisted_objects": final_persisted,
            "stale_work_rejected": stale_rejected,
            "audit_result": audits,
            "child_exit_code": proc.exitcode,
            "elapsed_seconds": time.monotonic() - started,
            "failures": failures,
            "pass": not failures,
        }
    except Exception as exc:
        return {
            "boundary": boundary.value,
            "decision": decision.value,
            "pre_crash_state": marker,
            "child_exit_code": proc.exitcode,
            "elapsed_seconds": time.monotonic() - started,
            "failures": ["E3_RECOVERY_EXCEPTION"],
            "error": repr(exc),
            "pass": False,
        }
    finally:
        if opened is not None:
            try:
                opened.close()
            except Exception:
                pass


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def repository_guard(repo_root: Path) -> dict[str, Any]:
    ancestor = _git(repo_root, "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD")
    blob = _git(repo_root, "rev-parse", f"HEAD:{SEMANTIC_FREEZE_PATH}")
    diff = _git(repo_root, "diff", "--name-only", f"{BASE_COMMIT}...HEAD")
    changed = [line.strip() for line in diff.stdout.splitlines() if line.strip()]
    unscoped = sorted(set(changed) - ALLOWED_CHANGED_PATHS)
    forbidden = sorted(
        p for p in changed
        if p.startswith("authority/") or "final_holdout" in p.lower() or "2025-09" in p
    )
    failures: list[str] = []
    if ancestor.returncode != 0:
        failures.append("E3_UNSCOPED_REPOSITORY_CHANGE")
    if blob.returncode != 0 or blob.stdout.strip() != SEMANTIC_FREEZE_BLOB:
        failures.append("E3_FROZEN_AUTHORITY_DRIFT")
    if forbidden:
        failures.append("E3_FINAL_HOLDOUT_OR_FROZEN_PATH_CHANGED")
    if unscoped:
        failures.append("E3_UNSCOPED_REPOSITORY_CHANGE")
    return {
        "base_commit": BASE_COMMIT,
        "base_is_ancestor": ancestor.returncode == 0,
        "semantic_freeze_path": SEMANTIC_FREEZE_PATH,
        "semantic_freeze_expected_blob": SEMANTIC_FREEZE_BLOB,
        "semantic_freeze_actual_blob": blob.stdout.strip(),
        "changed_paths": changed,
        "unscoped_changed_paths": unscoped,
        "forbidden_changed_paths": forbidden,
        "failures": sorted(set(failures)),
        "pass": not failures,
    }


def run_matrix(repo_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    guard = repository_guard(repo_root)
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cb16-r11-stage3-e3-") as td:
        matrix_root = Path(td)
        for decision in (TournamentDecision.PROMOTE, TournamentDecision.REJECT):
            for boundary in CrashBoundary:
                case_root = matrix_root / decision.value.lower() / boundary.value.lower()
                cases.append(run_case(case_root, boundary, decision))

    failures = list(guard["failures"])
    for case in cases:
        failures.extend(case.get("failures", ()))
    failures = sorted(set(failures))
    passed = sum(bool(case.get("pass")) for case in cases)
    return {
        "schema": SCHEMA,
        "stage": "STAGE-3",
        "task": "E3_CRASH_RESTART_MATRIX",
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "scientific_interpretation": "INFRASTRUCTURE_RUNTIME_QUALIFICATION_ONLY",
        "replay_interpretation": "FROZEN_ENGINEERING_REPLAY_NOT_NEW_SCIENTIFIC_EVIDENCE",
        "correctness_identity": {
            "teacher_authority_id": TEACHER_AUTHORITY_ID,
            "physics_authority_id": PHYSICS_AUTHORITY_ID,
            "canonical_arithmetic": "FP32",
            "amp": False,
            "scientific_semantics_changed": False,
            "gradient_authority_changed": False,
            "champion_challenger_semantics_changed": False,
            "requested_risk_semantics_changed": False,
            "new_scientific_verdict_created": False,
            "fresh_market_data_downloaded": False,
            "final_holdout_access": "FORBIDDEN_AND_NOT_REFERENCED_BY_HARNESS",
        },
        "repository_guard": guard,
        "runtime_result": {
            "fault_mode": "DETERMINISTIC_CHILD_PROCESS_OS_EXIT",
            "injected_exit_code": INJECTED_EXIT_CODE,
            "cases_total": len(cases),
            "cases_passed": passed,
            "cases_failed": len(cases) - passed,
            "long_soak_run": False,
            "elapsed_seconds": time.monotonic() - started,
        },
        "failure_taxonomy": list(FAILURE_TAXONOMY),
        "cases": cases,
        "failures": failures,
        "status": "PASS" if not failures and passed == len(cases) else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CB16 R11 Stage-3 E3 crash/restart matrix")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_matrix(args.repo_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _fsync_json(args.out, report)
    print(json.dumps({
        "schema": report["schema"],
        "status": report["status"],
        "cases_total": report["runtime_result"]["cases_total"],
        "cases_passed": report["runtime_result"]["cases_passed"],
        "failures": report["failures"],
    }, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
