from __future__ import annotations

"""Fail-closed heterogeneous/asynchronous generation orchestrator for CB16 R11.

Only orchestration semantics live here. Teacher, Trace, Central-Brain training/eval,
Physics/Permission and storage implementations remain injected authorities.
"""

from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Iterable

from .runtime_events_r11 import (
    AuthorityStamp,
    CheckpointSeal,
    CommitReceipt,
    CompletionDisposition,
    EventKind,
    EvidenceRef,
    FrozenInputReceipt,
    GenerationState,
    PolicyResult,
    RetryMode,
    RuntimeEvent,
    SnapshotSeal,
    TournamentCommitProposal,
    TournamentDecision,
    TournamentResult,
    TrainingResult,
    ValidationResult,
    WorkerPool,
    WorkCompletion,
    WorkItem,
    WorkKind,
    semantic_hash,
)
from .runtime_protocols_r11 import CheckpointStore, EngineBundle, EventJournal, EvidenceStore


class R11OrchestrationError(RuntimeError):
    pass


class IllegalTransition(R11OrchestrationError):
    pass


class LineageViolation(R11OrchestrationError):
    pass


class DuplicateConflict(R11OrchestrationError):
    pass


class BackpressureRequired(R11OrchestrationError):
    pass


class WorkerPoolUnavailable(R11OrchestrationError):
    pass


class RecoveryError(R11OrchestrationError):
    pass


@dataclass(frozen=True)
class QueueLimits:
    cpu_teacher: int = 8
    cpu_trace: int = 8
    gpu_brain: int = 2
    storage: int = 8
    io_prefetch: int = 4

    def for_pool(self, pool: WorkerPool) -> int:
        return {
            WorkerPool.CPU_TEACHER: self.cpu_teacher,
            WorkerPool.CPU_TRACE: self.cpu_trace,
            WorkerPool.GPU_BRAIN: self.gpu_brain,
            WorkerPool.STORAGE: self.storage,
            WorkerPool.IO_PREFETCH: self.io_prefetch,
        }[pool]


class BoundedWorkQueues:
    """Bounded, lossless mailboxes. A full mailbox backpressures the producer."""

    def __init__(self, limits: QueueLimits = QueueLimits()):
        self._queues = {pool: Queue(maxsize=limits.for_pool(pool)) for pool in WorkerPool}
        self._available = {pool: True for pool in WorkerPool}

    def set_available(self, pool: WorkerPool, available: bool) -> None:
        self._available[pool] = bool(available)

    def enqueue(self, work: WorkItem, *, block: bool = False, timeout: float | None = None) -> None:
        try:
            if block:
                self._queues[work.pool].put(work, block=True, timeout=timeout)
            else:
                self._queues[work.pool].put(work, block=False)
        except Full as exc:
            raise BackpressureRequired(f"R11_BACKPRESSURE:{work.pool.value}:{work.work_id}") from exc

    def claim(self, pool: WorkerPool, *, block: bool = False, timeout: float | None = None) -> WorkItem | None:
        if not self._available[pool]:
            raise WorkerPoolUnavailable(f"R11_POOL_UNAVAILABLE:{pool.value}")
        try:
            if block:
                return self._queues[pool].get(block=True, timeout=timeout)
            return self._queues[pool].get(block=False)
        except Empty:
            return None

    def claim_any(self, pools: Iterable[WorkerPool]) -> WorkItem | None:
        """Non-blocking work stealing over an explicitly compatible pool set."""
        candidates = tuple(pools)
        if not candidates:
            return None
        available = False
        for pool in candidates:
            if not self._available[pool]:
                continue
            available = True
            work = self.claim(pool)
            if work is not None:
                return work
        if not available:
            raise WorkerPoolUnavailable(
                "R11_POOLS_UNAVAILABLE:" + ",".join(pool.value for pool in candidates)
            )
        return None

    def task_done(self, pool: WorkerPool) -> None:
        self._queues[pool].task_done()

    def depth(self, pool: WorkerPool) -> int:
        return self._queues[pool].qsize()


@dataclass
class GenerationRecord:
    authority: AuthorityStamp
    state: GenerationState = GenerationState.PREPARING
    frozen_inputs_verified: bool = False
    accepted_evidence: dict[str, EvidenceRef] = field(default_factory=dict)
    snapshot: SnapshotSeal | None = None
    policy_work_id: str | None = None
    policy_payload_hash: str | None = None
    trace_started: bool = False
    trace_evidence_ids: set[str] = field(default_factory=set)
    challenger_id: str | None = None
    challenger_hash: str | None = None
    training_complete: bool = False
    validation_id: str | None = None
    tournament: TournamentResult | None = None
    commit: CommitReceipt | None = None
    checkpoint: CheckpointSeal | None = None
    next_generation_released: bool = False
    completion_hashes: dict[str, str] = field(default_factory=dict)
    scheduled_work: dict[str, WorkItem] = field(default_factory=dict)
    completed_work_ids: set[str] = field(default_factory=set)

    @property
    def generation(self) -> int:
        return self.authority.generation


_ALLOWED_MAINLINE = {
    GenerationState.PREPARING: {GenerationState.SNAPSHOT_SEALED},
    GenerationState.SNAPSHOT_SEALED: {GenerationState.TRACE_RUNNING},
    GenerationState.TRACE_RUNNING: {GenerationState.CHALLENGER_TRAINING},
    GenerationState.CHALLENGER_TRAINING: {GenerationState.VALIDATING},
    GenerationState.VALIDATING: {GenerationState.TOURNAMENT_PENDING},
    GenerationState.TOURNAMENT_PENDING: {GenerationState.COMMITTING},
    GenerationState.COMMITTING: {GenerationState.COMMITTED},
    GenerationState.COMMITTED: set(),
}

_WORK_POOL = {
    WorkKind.TEACHER_ACQUIRE: WorkerPool.CPU_TEACHER,
    WorkKind.POLICY_INFER: WorkerPool.GPU_BRAIN,
    WorkKind.TRACE_EXECUTE: WorkerPool.CPU_TRACE,
    WorkKind.MATERIALIZE_EVIDENCE: WorkerPool.STORAGE,
    WorkKind.TRAIN_CHALLENGER: WorkerPool.GPU_BRAIN,
    WorkKind.VALIDATE: WorkerPool.GPU_BRAIN,
    WorkKind.TOURNAMENT: WorkerPool.GPU_BRAIN,
    WorkKind.CHECKPOINT_SEAL: WorkerPool.STORAGE,
    WorkKind.PREFETCH_IMMUTABLE: WorkerPool.IO_PREFETCH,
}

_WORK_ALLOWED_STATES = {
    WorkKind.TEACHER_ACQUIRE: {GenerationState.PREPARING},
    WorkKind.POLICY_INFER: {GenerationState.SNAPSHOT_SEALED},
    WorkKind.TRACE_EXECUTE: {GenerationState.SNAPSHOT_SEALED, GenerationState.TRACE_RUNNING},
    WorkKind.MATERIALIZE_EVIDENCE: {
        GenerationState.TRACE_RUNNING, GenerationState.CHALLENGER_TRAINING,
        GenerationState.VALIDATING, GenerationState.TOURNAMENT_PENDING,
        GenerationState.COMMITTING, GenerationState.COMMITTED,
    },
    WorkKind.TRAIN_CHALLENGER: {GenerationState.TRACE_RUNNING, GenerationState.CHALLENGER_TRAINING},
    WorkKind.VALIDATE: {GenerationState.VALIDATING},
    WorkKind.TOURNAMENT: {GenerationState.TOURNAMENT_PENDING},
    WorkKind.CHECKPOINT_SEAL: {GenerationState.COMMITTING},
    WorkKind.PREFETCH_IMMUTABLE: set(GenerationState),
}


STAGE_DAG_R11 = (
    "1_FROZEN_INPUT_CACHE_VERIFICATION",
    "2_TEACHER_EVIDENCE_ACQUISITION_LOOKUP",
    "3_TRAINING_SNAPSHOT_SEAL",
    "4_CHAMPION_POLICY_INFERENCE",
    "5_ON_POLICY_TRACE_EXECUTION",
    "6_MATURED_OUTCOME_EVIDENCE_MATERIALIZATION",
    "7_CHALLENGER_TRAINING",
    "8_VALIDATION",
    "9_TOURNAMENT",
    "10_PROMOTION_REJECTION_ATOMIC_COMMIT",
    "11_CHECKPOINT_JOURNAL_SEAL",
    "12_NEXT_GENERATION_RELEASE",
)

DURABLE_STATE_R11 = (
    "frozen input verification receipt",
    "accepted evidence identities and lineage",
    "immutable training snapshot seal",
    "scientific work schedule/completion identities",
    "matured trace evidence materialization receipts",
    "challenger lineage/result identity",
    "validation and tournament result identity",
    "atomic PROMOTE/REJECT commit receipt",
    "checkpoint seal and generation-release event",
)

EPHEMERAL_STATE_R11 = (
    "bounded queue memory image (rebuilt from durable pending work)",
    "worker leases/process/thread state",
    "prefetch and pinned-memory buffers",
    "decompressed/mmap/performance caches",
    "GPU residency/CUDA graph capture",
    "retry timers and scheduler heuristics",
)


class R11Orchestrator:
    def __init__(
        self,
        *,
        authority: AuthorityStamp,
        evidence_store: EvidenceStore,
        journal: EventJournal,
        checkpoint_store: CheckpointStore,
        engines: EngineBundle | None = None,
        queues: BoundedWorkQueues | None = None,
        recover: bool = False,
    ):
        self.evidence_store = evidence_store
        self.journal = journal
        self.checkpoint_store = checkpoint_store
        self.engines = engines or EngineBundle()
        self.queues = queues or BoundedWorkQueues()
        self.record = GenerationRecord(authority=authority)
        self._queued_work_ids: set[str] = set()
        self._leased_work_ids: set[str] = set()
        if recover:
            self._recover()
        else:
            self._persist(
                EventKind.GENERATION_PREPARED,
                payload=authority,
                metadata={
                    "champion_id": authority.champion_id,
                    "champion_hash": authority.champion_hash,
                    "teacher_authority_id": authority.teacher_authority_id,
                    "physics_authority_id": authority.physics_authority_id,
                },
            )

    def _persist(
        self,
        kind: EventKind,
        *,
        payload: object,
        work_id: str | None = None,
        metadata=None,
        state_after: GenerationState | None = None,
    ) -> None:
        event = RuntimeEvent.build(
            kind=kind,
            generation=self.record.generation,
            state_after=state_after or self.record.state,
            payload=payload,
            work_id=work_id,
            metadata=metadata,
        )
        receipt = self.journal.append_once(event)
        if not receipt.durable or receipt.event_id != event.event_id:
            raise RecoveryError(f"R11_JOURNAL_NOT_DURABLE:{kind.value}")

    def _ensure_state(self, *states: GenerationState) -> None:
        if self.record.state not in states:
            allowed = ",".join(s.value for s in states)
            raise IllegalTransition(f"R11_STATE_REQUIRED:{allowed}:GOT:{self.record.state.value}")

    def _assert_transition(self, target: GenerationState) -> None:
        if target not in _ALLOWED_MAINLINE[self.record.state]:
            raise IllegalTransition(f"R11_ILLEGAL_TRANSITION:{self.record.state.value}->{target.value}")

    def _check_poison(self, poison_bits: Iterable[str], context: str) -> None:
        bits = tuple(poison_bits)
        if bits:
            raise LineageViolation(f"R11_POISON_OR_FUTURE_TAINT:{context}:{'|'.join(bits)}")

    def _check_generation(self, generation: int, context: str) -> None:
        if generation != self.record.generation:
            raise LineageViolation(
                f"R11_GENERATION_MIXING:{context}:EXPECTED:{self.record.generation}:GOT:{generation}"
            )

    def _check_parent(self, parent_champion_id: str, context: str) -> None:
        if parent_champion_id != self.record.authority.champion_id:
            raise LineageViolation(
                f"R11_STALE_POLICY_LINEAGE:{context}:EXPECTED:{self.record.authority.champion_id}:GOT:{parent_champion_id}"
            )

    def verify_frozen_inputs(self, receipt: FrozenInputReceipt) -> None:
        self._ensure_state(GenerationState.PREPARING)
        if not receipt.verified:
            raise LineageViolation("R11_FROZEN_INPUT_VERIFICATION_FAILED")
        self._persist(EventKind.INPUTS_VERIFIED, payload=receipt)
        self.record.frozen_inputs_verified = True

    def accept_teacher_evidence(self, evidence: EvidenceRef) -> None:
        self._ensure_state(GenerationState.PREPARING)
        if not self.record.frozen_inputs_verified:
            raise IllegalTransition("R11_INPUTS_MUST_BE_VERIFIED_BEFORE_EVIDENCE")
        self._check_poison(evidence.poison_bits, evidence.evidence_id)
        if not evidence.detached_from_autograd:
            raise LineageViolation(f"R11_TEACHER_STUDENT_AUTOGRAD_PATH:{evidence.evidence_id}")
        if evidence.source_generation >= self.record.generation:
            raise LineageViolation(f"R11_FUTURE_EVIDENCE_TAINT:{evidence.evidence_id}")
        if evidence.teacher_generation > evidence.source_generation:
            raise LineageViolation(f"R11_FUTURE_TEACHER_TAINT:{evidence.evidence_id}")
        if evidence.teacher_authority_id != self.record.authority.teacher_authority_id:
            raise LineageViolation(f"R11_TEACHER_AUTHORITY_MIXING:{evidence.evidence_id}")
        if evidence.physics_authority_id != self.record.authority.physics_authority_id:
            raise LineageViolation(f"R11_PHYSICS_AUTHORITY_MIXING:{evidence.evidence_id}")
        prior = self.record.accepted_evidence.get(evidence.evidence_id)
        if prior is not None:
            if prior == evidence:
                return
            raise DuplicateConflict(f"R11_DUPLICATE_EVIDENCE_CONFLICT:{evidence.evidence_id}")
        stored = self.evidence_store.put_once(evidence)
        if not stored.durable or stored.payload_hash != evidence.payload_hash:
            raise RecoveryError(f"R11_EVIDENCE_STORE_NOT_DURABLE:{evidence.evidence_id}")
        if not stored.inserted and self.evidence_store.get(evidence.evidence_id) != evidence:
            raise DuplicateConflict(f"R11_DUPLICATE_EVIDENCE_STORE_CONFLICT:{evidence.evidence_id}")
        self._persist(
            EventKind.TEACHER_EVIDENCE_ACCEPTED,
            payload=evidence,
            metadata={
                "evidence_id": evidence.evidence_id,
                "payload_hash": evidence.payload_hash,
                "source_generation": evidence.source_generation,
                "producer_champion_id": evidence.producer_champion_id,
                "teacher_generation": evidence.teacher_generation,
            },
        )
        self.record.accepted_evidence[evidence.evidence_id] = evidence

    def seal_training_snapshot(self, seal: SnapshotSeal) -> None:
        self._ensure_state(GenerationState.PREPARING)
        self._assert_transition(GenerationState.SNAPSHOT_SEALED)
        if not self.record.frozen_inputs_verified:
            raise IllegalTransition("R11_INPUTS_MUST_BE_VERIFIED_BEFORE_SNAPSHOT")
        self._check_generation(seal.generation, "SNAPSHOT")
        self._check_parent(seal.parent_champion_id, "SNAPSHOT")
        if not seal.immutable:
            raise LineageViolation("R11_SNAPSHOT_NOT_IMMUTABLE")
        if len(set(seal.evidence_ids)) != len(seal.evidence_ids):
            raise DuplicateConflict("R11_SNAPSHOT_DUPLICATE_EVIDENCE")
        if set(seal.evidence_ids) != set(self.record.accepted_evidence):
            raise LineageViolation("R11_SNAPSHOT_EVIDENCE_SET_MISMATCH")
        stored = self.evidence_store.seal_snapshot(seal)
        if stored != seal or not stored.immutable:
            raise RecoveryError("R11_SNAPSHOT_SEAL_NOT_DURABLE_IMMUTABLE")
        self._persist(
            EventKind.SNAPSHOT_SEALED,
            payload=seal,
            metadata={
                "snapshot_id": seal.snapshot_id,
                "snapshot_hash": seal.snapshot_hash,
                "evidence_ids": ",".join(seal.evidence_ids),
            },
            state_after=GenerationState.SNAPSHOT_SEALED,
        )
        self.record.snapshot = seal
        self.record.state = GenerationState.SNAPSHOT_SEALED

    def accept_policy_result(self, result: PolicyResult) -> CompletionDisposition:
        self._check_generation(result.generation, "POLICY_RESULT")
        self._check_parent(result.champion_id, "POLICY_RESULT")
        self._check_poison(result.poison_bits, result.work_id)
        duplicate = self._duplicate_disposition(result.work_id, result.payload_hash)
        if duplicate is not None:
            return duplicate
        self._ensure_state(GenerationState.SNAPSHOT_SEALED)
        self._persist(
            EventKind.POLICY_INFERENCE_COMPLETED,
            payload=result,
            work_id=result.work_id,
            metadata={"result_payload_hash": result.payload_hash, "champion_id": result.champion_id},
        )
        self.record.policy_work_id = result.work_id
        self.record.policy_payload_hash = result.payload_hash
        self._mark_completion(result.work_id, result.payload_hash)
        return CompletionDisposition(True, False, "ACCEPTED")

    def start_trace(self, work_id: str) -> None:
        self._ensure_state(GenerationState.SNAPSHOT_SEALED)
        self._assert_transition(GenerationState.TRACE_RUNNING)
        if self.record.snapshot is None:
            raise IllegalTransition("R11_SNAPSHOT_REQUIRED_FOR_TRACE")
        if self.record.policy_work_id is None:
            raise IllegalTransition("R11_POLICY_INFERENCE_REQUIRED_BEFORE_TRACE")
        self._persist(
            EventKind.TRACE_STARTED,
            payload={"work_id": work_id},
            work_id=work_id,
            state_after=GenerationState.TRACE_RUNNING,
        )
        self.record.trace_started = True
        self.record.state = GenerationState.TRACE_RUNNING

    def materialize_trace_evidence(self, evidence: EvidenceRef, *, work_id: str) -> CompletionDisposition:
        self._check_poison(evidence.poison_bits, evidence.evidence_id)
        if not evidence.detached_from_autograd:
            raise LineageViolation(f"R11_TEACHER_STUDENT_AUTOGRAD_PATH:{evidence.evidence_id}")
        if evidence.source_generation != self.record.generation:
            raise LineageViolation(f"R11_TRACE_SOURCE_GENERATION_MISMATCH:{evidence.evidence_id}")
        self._check_parent(evidence.producer_champion_id, "TRACE_EVIDENCE")
        if evidence.teacher_generation > self.record.generation:
            raise LineageViolation(f"R11_FUTURE_TEACHER_TAINT:{evidence.evidence_id}")
        if evidence.teacher_authority_id != self.record.authority.teacher_authority_id:
            raise LineageViolation(f"R11_TEACHER_AUTHORITY_MIXING:{evidence.evidence_id}")
        if evidence.physics_authority_id != self.record.authority.physics_authority_id:
            raise LineageViolation(f"R11_PHYSICS_AUTHORITY_MIXING:{evidence.evidence_id}")
        duplicate = self._duplicate_disposition(work_id, evidence.payload_hash)
        if duplicate is not None:
            return duplicate
        self._ensure_state(
            GenerationState.TRACE_RUNNING,
            GenerationState.CHALLENGER_TRAINING,
            GenerationState.VALIDATING,
            GenerationState.TOURNAMENT_PENDING,
            GenerationState.COMMITTING,
            GenerationState.COMMITTED,
        )
        stored = self.evidence_store.put_once(evidence)
        if not stored.durable or stored.payload_hash != evidence.payload_hash:
            raise RecoveryError(f"R11_TRACE_EVIDENCE_NOT_DURABLE:{evidence.evidence_id}")
        if not stored.inserted and self.evidence_store.get(evidence.evidence_id) != evidence:
            raise DuplicateConflict(f"R11_DUPLICATE_TRACE_EVIDENCE_CONFLICT:{evidence.evidence_id}")
        self._persist(
            EventKind.TRACE_EVIDENCE_MATERIALIZED,
            payload=evidence,
            work_id=work_id,
            metadata={
                "evidence_id": evidence.evidence_id,
                "payload_hash": evidence.payload_hash,
                "result_payload_hash": evidence.payload_hash,
            },
        )
        self.record.trace_evidence_ids.add(evidence.evidence_id)
        self._mark_completion(work_id, evidence.payload_hash)
        return CompletionDisposition(True, False, "ACCEPTED")

    def start_challenger_training(self, *, work_id: str) -> None:
        self._ensure_state(GenerationState.TRACE_RUNNING)
        self._assert_transition(GenerationState.CHALLENGER_TRAINING)
        if not self.record.trace_started or self.record.snapshot is None:
            raise IllegalTransition("R11_TRACE_AND_SNAPSHOT_REQUIRED_BEFORE_TRAINING")
        self._persist(
            EventKind.TRAINING_STARTED,
            payload={"work_id": work_id},
            work_id=work_id,
            state_after=GenerationState.CHALLENGER_TRAINING,
        )
        self.record.state = GenerationState.CHALLENGER_TRAINING

    def complete_training(self, result: TrainingResult) -> CompletionDisposition:
        self._check_generation(result.generation, "TRAINING_RESULT")
        self._check_parent(result.parent_champion_id, "TRAINING_RESULT")
        self._check_poison(result.poison_bits, result.work_id)
        duplicate = self._duplicate_disposition(result.work_id, result.payload_hash)
        if duplicate is not None:
            return duplicate
        self._ensure_state(GenerationState.CHALLENGER_TRAINING)
        self._assert_transition(GenerationState.VALIDATING)
        if self.record.snapshot is None or result.snapshot_id != self.record.snapshot.snapshot_id:
            raise LineageViolation("R11_TRAINING_SNAPSHOT_MISMATCH")
        self._persist(
            EventKind.TRAINING_COMPLETED,
            payload=result,
            work_id=result.work_id,
            metadata={
                "challenger_id": result.challenger_id,
                "challenger_hash": result.challenger_hash,
                "snapshot_id": result.snapshot_id,
                "result_payload_hash": result.payload_hash,
            },
            state_after=GenerationState.VALIDATING,
        )
        self.record.challenger_id = result.challenger_id
        self.record.challenger_hash = result.challenger_hash
        self.record.training_complete = True
        self.record.state = GenerationState.VALIDATING
        self._mark_completion(result.work_id, result.payload_hash)
        return CompletionDisposition(True, False, "ACCEPTED")

    def complete_validation(self, result: ValidationResult) -> CompletionDisposition:
        self._check_generation(result.generation, "VALIDATION_RESULT")
        self._check_parent(result.parent_champion_id, "VALIDATION_RESULT")
        self._check_poison(result.poison_bits, result.work_id)
        duplicate = self._duplicate_disposition(result.work_id, result.payload_hash)
        if duplicate is not None:
            return duplicate
        self._ensure_state(GenerationState.VALIDATING)
        self._assert_transition(GenerationState.TOURNAMENT_PENDING)
        if result.challenger_id != self.record.challenger_id:
            raise LineageViolation("R11_VALIDATION_CHALLENGER_MISMATCH")
        if self.record.snapshot is None or result.snapshot_id != self.record.snapshot.snapshot_id:
            raise LineageViolation("R11_VALIDATION_SNAPSHOT_MISMATCH")
        self._persist(
            EventKind.VALIDATION_COMPLETED,
            payload=result,
            work_id=result.work_id,
            metadata={
                "validation_id": result.validation_id,
                "challenger_id": result.challenger_id,
                "result_payload_hash": result.payload_hash,
            },
            state_after=GenerationState.TOURNAMENT_PENDING,
        )
        self.record.validation_id = result.validation_id
        self.record.state = GenerationState.TOURNAMENT_PENDING
        self._mark_completion(result.work_id, result.payload_hash)
        return CompletionDisposition(True, False, "ACCEPTED")

    def decide_tournament(self, result: TournamentResult) -> CompletionDisposition:
        self._check_generation(result.generation, "TOURNAMENT_RESULT")
        self._check_parent(result.parent_champion_id, "TOURNAMENT_RESULT")
        self._check_poison(result.poison_bits, result.work_id)
        duplicate = self._duplicate_disposition(result.work_id, result.payload_hash)
        if duplicate is not None:
            return duplicate
        self._ensure_state(GenerationState.TOURNAMENT_PENDING)
        self._assert_transition(GenerationState.COMMITTING)
        if result.challenger_id != self.record.challenger_id:
            raise LineageViolation("R11_TOURNAMENT_CHALLENGER_MISMATCH")
        if result.validation_id != self.record.validation_id:
            raise LineageViolation("R11_TOURNAMENT_VALIDATION_MISMATCH")
        self._persist(
            EventKind.TOURNAMENT_DECIDED,
            payload=result,
            work_id=result.work_id,
            metadata={
                "decision": result.decision.value,
                "challenger_id": result.challenger_id,
                "result_payload_hash": result.payload_hash,
            },
            state_after=GenerationState.COMMITTING,
        )
        self.record.tournament = result
        self.record.state = GenerationState.COMMITTING
        self._mark_completion(result.work_id, result.payload_hash)
        return CompletionDisposition(True, False, "ACCEPTED")

    def commit_tournament(self) -> CommitReceipt:
        self._ensure_state(GenerationState.COMMITTING)
        if self.record.tournament is None or self.record.challenger_id is None or self.record.challenger_hash is None:
            raise IllegalTransition("R11_TOURNAMENT_RESULT_REQUIRED_FOR_COMMIT")
        if self.record.commit is not None:
            return self.record.commit
        proposal = TournamentCommitProposal(
            generation=self.record.generation,
            decision=self.record.tournament.decision,
            parent_champion_id=self.record.authority.champion_id,
            challenger_id=self.record.challenger_id,
            challenger_hash=self.record.challenger_hash,
        )
        receipt = self.checkpoint_store.atomic_commit(proposal)
        self._validate_commit_receipt(receipt)
        # Atomic store authority may succeed before journal acknowledgement. Keep it in
        # memory and recover by read_commit() if this append fails/crashes.
        self.record.commit = receipt
        self._persist(
            EventKind.TOURNAMENT_COMMITTED,
            payload=receipt,
            metadata={
                "decision": receipt.decision.value,
                "commit_id": receipt.commit_id,
                "next_champion_id": receipt.next_champion_id,
                "next_champion_hash": receipt.next_champion_hash,
            },
        )
        return receipt

    def _validate_commit_receipt(self, receipt: CommitReceipt) -> None:
        if not receipt.atomic or not receipt.durable:
            raise RecoveryError("R11_TOURNAMENT_COMMIT_NOT_ATOMIC_DURABLE")
        self._check_generation(receipt.generation, "COMMIT_RECEIPT")
        self._check_parent(receipt.parent_champion_id, "COMMIT_RECEIPT")
        if self.record.tournament is None or receipt.decision != self.record.tournament.decision:
            raise LineageViolation("R11_COMMIT_DECISION_MISMATCH")
        expected_id = (
            self.record.challenger_id
            if receipt.decision is TournamentDecision.PROMOTE
            else self.record.authority.champion_id
        )
        expected_hash = (
            self.record.challenger_hash
            if receipt.decision is TournamentDecision.PROMOTE
            else self.record.authority.champion_hash
        )
        if receipt.next_champion_id != expected_id or receipt.next_champion_hash != expected_hash:
            raise LineageViolation("R11_CHALLENGER_SELF_LAUNDERING_OR_COMMIT_MISMATCH")

    def seal_checkpoint(self, checkpoint_id: str) -> CheckpointSeal:
        self._ensure_state(GenerationState.COMMITTING)
        self._assert_transition(GenerationState.COMMITTED)
        if self.record.commit is None:
            raise IllegalTransition("R11_ATOMIC_COMMIT_REQUIRED_BEFORE_CHECKPOINT")
        seal = CheckpointSeal(
            generation=self.record.generation,
            checkpoint_id=checkpoint_id,
            state_digest=self._state_digest(),
            commit_id=self.record.commit.commit_id,
        )
        stored = self.checkpoint_store.seal_checkpoint(seal)
        if stored != seal or not stored.durable:
            raise RecoveryError("R11_CHECKPOINT_SEAL_NOT_DURABLE")
        self._persist(
            EventKind.CHECKPOINT_SEALED,
            payload=seal,
            metadata={"checkpoint_id": seal.checkpoint_id, "commit_id": seal.commit_id},
            state_after=GenerationState.COMMITTED,
        )
        self.record.checkpoint = seal
        self.record.state = GenerationState.COMMITTED
        return seal

    def release_next_generation(self) -> AuthorityStamp:
        self._ensure_state(GenerationState.COMMITTED)
        if self.record.commit is None or self.record.checkpoint is None:
            raise IllegalTransition("R11_COMMIT_AND_CHECKPOINT_REQUIRED_FOR_RELEASE")
        if self.record.next_generation_released:
            raise IllegalTransition("R11_NEXT_GENERATION_ALREADY_RELEASED")
        next_authority = AuthorityStamp(
            generation=self.record.generation + 1,
            champion_id=self.record.commit.next_champion_id,
            champion_hash=self.record.commit.next_champion_hash,
            teacher_authority_id=self.record.authority.teacher_authority_id,
            physics_authority_id=self.record.authority.physics_authority_id,
        )
        self._persist(
            EventKind.NEXT_GENERATION_RELEASED,
            payload=next_authority,
            metadata={
                "next_generation": next_authority.generation,
                "next_champion_id": next_authority.champion_id,
                "next_champion_hash": next_authority.champion_hash,
            },
        )
        self.record.next_generation_released = True
        return next_authority

    @staticmethod
    def _work_metadata(work: WorkItem) -> dict[str, object]:
        return {
            "kind": work.kind.value,
            "pool": work.pool.value,
            "parent_champion_id": work.parent_champion_id,
            "snapshot_id": work.snapshot_id or "",
            "payload_ref": work.payload_ref or "",
            "attempt": work.attempt,
            "retry_mode": work.retry_mode.value,
        }

    @staticmethod
    def _work_from_event(event: RuntimeEvent) -> WorkItem:
        md = event.meta()
        work = WorkItem(
            work_id=event.work_id or "",
            kind=WorkKind(md["kind"]),
            generation=event.generation,
            pool=WorkerPool(md["pool"]),
            parent_champion_id=md["parent_champion_id"],
            snapshot_id=md.get("snapshot_id") or None,
            payload_ref=md.get("payload_ref") or None,
            attempt=int(md.get("attempt", "0")),
            retry_mode=RetryMode(md.get("retry_mode", RetryMode.IDEMPOTENT_RECOMPUTE.value)),
        )
        if semantic_hash(work) != event.payload_hash:
            raise RecoveryError(f"R11_RECOVERY_WORK_IDENTITY_HASH_MISMATCH:{work.work_id}")
        return work

    def _validate_work(self, work: WorkItem) -> None:
        self._check_generation(work.generation, "WORK_ITEM")
        self._check_parent(work.parent_champion_id, "WORK_ITEM")
        expected_pool = _WORK_POOL[work.kind]
        if work.pool is not expected_pool:
            raise LineageViolation(
                f"R11_WORK_POOL_MISMATCH:{work.kind.value}:{work.pool.value}:{expected_pool.value}"
            )
        if self.record.state not in _WORK_ALLOWED_STATES[work.kind]:
            raise IllegalTransition(
                f"R11_WORK_STAGE_VIOLATION:{work.kind.value}:{self.record.state.value}"
            )

    def enqueue(self, work: WorkItem, *, block: bool = False, timeout: float | None = None) -> None:
        self._validate_work(work)
        if work.snapshot_id is not None:
            if self.record.snapshot is None or work.snapshot_id != self.record.snapshot.snapshot_id:
                raise LineageViolation("R11_WORK_SNAPSHOT_MISMATCH")
        prior = self.record.scheduled_work.get(work.work_id)
        if prior is not None and prior != work:
            raise DuplicateConflict(f"R11_DUPLICATE_WORK_ID_CONFLICT:{work.work_id}")
        if prior is None:
            self._persist(
                EventKind.WORK_SCHEDULED,
                payload=work,
                work_id=work.work_id,
                metadata=self._work_metadata(work),
            )
            self.record.scheduled_work[work.work_id] = work
        if (
            work.work_id in self.record.completed_work_ids
            or work.work_id in self._queued_work_ids
            or work.work_id in self._leased_work_ids
        ):
            return
        self.queues.enqueue(work, block=block, timeout=timeout)
        self._queued_work_ids.add(work.work_id)

    def claim_work(self, pool: WorkerPool, *, block: bool = False, timeout: float | None = None) -> WorkItem | None:
        work = self.queues.claim(pool, block=block, timeout=timeout)
        if work is not None:
            self._queued_work_ids.discard(work.work_id)
            self._leased_work_ids.add(work.work_id)
        return work

    def claim_any(self, pools: Iterable[WorkerPool]) -> WorkItem | None:
        work = self.queues.claim_any(pools)
        if work is not None:
            self._queued_work_ids.discard(work.work_id)
            self._leased_work_ids.add(work.work_id)
        return work

    def pending_work_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                work_id
                for work_id in self.record.scheduled_work
                if work_id not in self.record.completed_work_ids
            )
        )

    def pump_pending_work(self) -> int:
        """Rebuild durable incomplete work into bounded queues without dropping it."""
        queued = 0
        for work_id in self.pending_work_ids():
            if work_id in self._queued_work_ids or work_id in self._leased_work_ids:
                continue
            work = self.record.scheduled_work[work_id]
            try:
                self.queues.enqueue(work, block=False)
            except BackpressureRequired:
                continue
            self._queued_work_ids.add(work_id)
            queued += 1
        return queued

    def worker_crashed(self, work: WorkItem) -> WorkItem:
        self._check_generation(work.generation, "WORKER_CRASH")
        if work.work_id in self.record.completed_work_ids:
            raise IllegalTransition(f"R11_COMPLETED_WORK_CANNOT_RETRY:{work.work_id}")
        retried = WorkItem(
            work_id=work.work_id,
            kind=work.kind,
            generation=work.generation,
            pool=work.pool,
            parent_champion_id=work.parent_champion_id,
            snapshot_id=work.snapshot_id,
            payload_ref=work.payload_ref,
            attempt=work.attempt + 1,
            retry_mode=work.retry_mode,
        )
        self._persist(
            EventKind.WORK_REQUEUED,
            payload=retried,
            work_id=retried.work_id,
            metadata=self._work_metadata(retried),
        )
        self.record.scheduled_work[retried.work_id] = retried
        self._queued_work_ids.discard(retried.work_id)
        self._leased_work_ids.discard(retried.work_id)
        try:
            self.queues.enqueue(retried, block=False)
        except BackpressureRequired:
            return retried
        self._queued_work_ids.add(retried.work_id)
        return retried

    def execute_claimed(self, work: WorkItem) -> WorkCompletion:
        """Route immutable work to injected adapters; no concrete engine import occurs."""
        self._validate_work(work)
        engine = None
        if work.kind is WorkKind.TEACHER_ACQUIRE:
            engine = self.engines.teacher
        elif work.kind is WorkKind.TRACE_EXECUTE:
            engine = self.engines.trace
        elif work.kind is WorkKind.POLICY_INFER:
            engine = self.engines.inference
        elif work.kind is WorkKind.TRAIN_CHALLENGER:
            engine = self.engines.training
        elif work.kind is WorkKind.VALIDATE:
            engine = self.engines.validation
        elif work.kind is WorkKind.TOURNAMENT:
            engine = self.engines.tournament
        if engine is None:
            raise WorkerPoolUnavailable(f"R11_ENGINE_UNAVAILABLE:{work.kind.value}")
        completion = engine.execute(work)
        self._check_generation(completion.generation, "ENGINE_COMPLETION")
        self._check_poison(completion.poison_bits, completion.work_id)
        if completion.work_id != work.work_id or completion.kind != work.kind:
            raise LineageViolation("R11_ENGINE_COMPLETION_WORK_IDENTITY_MISMATCH")
        return completion

    def _duplicate_disposition(self, work_id: str, payload_hash: str) -> CompletionDisposition | None:
        prior = self.record.completion_hashes.get(work_id)
        if prior is None:
            return None
        if prior == payload_hash:
            return CompletionDisposition(True, True, "IDEMPOTENT_DUPLICATE")
        raise DuplicateConflict(f"R11_DUPLICATE_COMPLETION_CONFLICT:{work_id}")

    def _mark_completion(self, work_id: str, payload_hash: str) -> None:
        self.record.completion_hashes[work_id] = payload_hash
        self.record.completed_work_ids.add(work_id)
        self._queued_work_ids.discard(work_id)
        self._leased_work_ids.discard(work_id)

    def _state_digest(self) -> str:
        return semantic_hash(
            {
                "generation": self.record.generation,
                "state": self.record.state.value,
                "champion_id": self.record.authority.champion_id,
                "snapshot_id": self.record.snapshot.snapshot_id if self.record.snapshot else None,
                "accepted_evidence": sorted(self.record.accepted_evidence),
                "trace_evidence": sorted(self.record.trace_evidence_ids),
                "challenger_id": self.record.challenger_id,
                "validation_id": self.record.validation_id,
                "decision": self.record.tournament.decision.value if self.record.tournament else None,
                "commit_id": self.record.commit.commit_id if self.record.commit else None,
            }
        )

    def _recover(self) -> None:
        events = list(self.journal.read_generation(self.record.generation))
        if not events:
            raise RecoveryError(f"R11_NO_DURABLE_GENERATION:{self.record.generation}")
        self.record = GenerationRecord(authority=self.record.authority)
        self._queued_work_ids.clear()
        self._leased_work_ids.clear()
        for event in events:
            self._replay(event)

        # Crash gap 1: atomic tournament commit succeeded but its journal event did not.
        if self.record.state is GenerationState.COMMITTING and self.record.commit is None:
            durable_commit = self.checkpoint_store.read_commit(self.record.generation)
            if durable_commit is not None:
                self._validate_commit_receipt(durable_commit)
                self.record.commit = durable_commit
                self._persist(
                    EventKind.TOURNAMENT_COMMITTED,
                    payload=durable_commit,
                    metadata={
                        "decision": durable_commit.decision.value,
                        "commit_id": durable_commit.commit_id,
                        "next_champion_id": durable_commit.next_champion_id,
                        "next_champion_hash": durable_commit.next_champion_hash,
                    },
                )

        # Crash gap 2: checkpoint store sealed after commit but journal ack was lost.
        if self.record.state is GenerationState.COMMITTING and self.record.commit is not None:
            durable_checkpoint = self.checkpoint_store.read_checkpoint(self.record.generation)
            if durable_checkpoint is not None:
                if durable_checkpoint.commit_id != self.record.commit.commit_id:
                    raise RecoveryError("R11_RECOVERY_CHECKPOINT_COMMIT_MISMATCH")
                self._persist(
                    EventKind.CHECKPOINT_SEALED,
                    payload=durable_checkpoint,
                    metadata={
                        "checkpoint_id": durable_checkpoint.checkpoint_id,
                        "commit_id": durable_checkpoint.commit_id,
                    },
                    state_after=GenerationState.COMMITTED,
                )
                self.record.checkpoint = durable_checkpoint
                self.record.state = GenerationState.COMMITTED

        self.pump_pending_work()

    def _replay(self, event: RuntimeEvent) -> None:
        if event.generation != self.record.generation:
            raise RecoveryError("R11_RECOVERY_GENERATION_MIXING")
        md = event.meta()
        k = event.kind

        if event.work_id and k in {
            EventKind.POLICY_INFERENCE_COMPLETED,
            EventKind.TRACE_EVIDENCE_MATERIALIZED,
            EventKind.TRAINING_COMPLETED,
            EventKind.VALIDATION_COMPLETED,
            EventKind.TOURNAMENT_DECIDED,
        }:
            self.record.completion_hashes[event.work_id] = md.get(
                "result_payload_hash", md.get("payload_hash", event.payload_hash)
            )
            self.record.completed_work_ids.add(event.work_id)

        if k in {EventKind.WORK_SCHEDULED, EventKind.WORK_REQUEUED}:
            work = self._work_from_event(event)
            self.record.scheduled_work[work.work_id] = work
        elif k is EventKind.GENERATION_PREPARED:
            if md.get("champion_id") != self.record.authority.champion_id:
                raise RecoveryError("R11_RECOVERY_CHAMPION_AUTHORITY_MISMATCH")
            if md.get("champion_hash") != self.record.authority.champion_hash:
                raise RecoveryError("R11_RECOVERY_CHAMPION_HASH_MISMATCH")
        elif k is EventKind.INPUTS_VERIFIED:
            self.record.frozen_inputs_verified = True
        elif k is EventKind.TEACHER_EVIDENCE_ACCEPTED:
            evidence_id = md["evidence_id"]
            evidence = self.evidence_store.get(evidence_id)
            if evidence is None or evidence.payload_hash != md["payload_hash"]:
                raise RecoveryError(f"R11_RECOVERY_EVIDENCE_MISSING:{evidence_id}")
            self.record.accepted_evidence[evidence_id] = evidence
        elif k is EventKind.SNAPSHOT_SEALED:
            ids = tuple(filter(None, md.get("evidence_ids", "").split(",")))
            snapshot = self.evidence_store.get_snapshot(md["snapshot_id"])
            if snapshot is None or snapshot.snapshot_hash != md["snapshot_hash"]:
                raise RecoveryError("R11_RECOVERY_SNAPSHOT_MISSING")
            if snapshot.evidence_ids != ids:
                raise RecoveryError("R11_RECOVERY_SNAPSHOT_EVIDENCE_MISMATCH")
            self.record.snapshot = snapshot
            self.record.state = GenerationState.SNAPSHOT_SEALED
        elif k is EventKind.POLICY_INFERENCE_COMPLETED:
            self.record.policy_work_id = event.work_id
            self.record.policy_payload_hash = md.get("result_payload_hash")
        elif k is EventKind.TRACE_STARTED:
            self.record.trace_started = True
            self.record.state = GenerationState.TRACE_RUNNING
        elif k is EventKind.TRACE_EVIDENCE_MATERIALIZED:
            self.record.trace_evidence_ids.add(md["evidence_id"])
        elif k is EventKind.TRAINING_STARTED:
            self.record.state = GenerationState.CHALLENGER_TRAINING
        elif k is EventKind.TRAINING_COMPLETED:
            self.record.challenger_id = md["challenger_id"]
            self.record.challenger_hash = md["challenger_hash"]
            self.record.training_complete = True
            self.record.state = GenerationState.VALIDATING
        elif k is EventKind.VALIDATION_COMPLETED:
            self.record.validation_id = md["validation_id"]
            self.record.state = GenerationState.TOURNAMENT_PENDING
        elif k is EventKind.TOURNAMENT_DECIDED:
            if self.record.challenger_id is None or self.record.validation_id is None:
                raise RecoveryError("R11_RECOVERY_TOURNAMENT_WITHOUT_LINEAGE")
            self.record.tournament = TournamentResult(
                work_id=event.work_id or "recovered-tournament",
                generation=self.record.generation,
                challenger_id=self.record.challenger_id,
                parent_champion_id=self.record.authority.champion_id,
                validation_id=self.record.validation_id,
                decision=TournamentDecision(md["decision"]),
                payload_hash=md.get("result_payload_hash", event.payload_hash),
            )
            self.record.state = GenerationState.COMMITTING
        elif k is EventKind.TOURNAMENT_COMMITTED:
            receipt = self.checkpoint_store.read_commit(self.record.generation)
            if receipt is None or receipt.commit_id != md["commit_id"]:
                raise RecoveryError("R11_RECOVERY_COMMIT_MISSING")
            self.record.commit = receipt
        elif k is EventKind.CHECKPOINT_SEALED:
            seal = self.checkpoint_store.read_checkpoint(self.record.generation)
            if seal is None or seal.checkpoint_id != md["checkpoint_id"]:
                raise RecoveryError("R11_RECOVERY_CHECKPOINT_MISSING")
            self.record.checkpoint = seal
            self.record.state = GenerationState.COMMITTED
        elif k is EventKind.NEXT_GENERATION_RELEASED:
            self.record.next_generation_released = True
