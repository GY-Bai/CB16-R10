from __future__ import annotations

"""Dependency-free event and lineage types for the R11 asynchronous orchestrator.

These objects carry scientific identity; they do not implement Teacher, Trace,
Training, Physics, Permission, or Storage semantics.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Mapping


class GenerationState(str, Enum):
    PREPARING = "PREPARING"
    SNAPSHOT_SEALED = "SNAPSHOT_SEALED"
    TRACE_RUNNING = "TRACE_RUNNING"
    CHALLENGER_TRAINING = "CHALLENGER_TRAINING"
    VALIDATING = "VALIDATING"
    TOURNAMENT_PENDING = "TOURNAMENT_PENDING"
    COMMITTING = "COMMITTING"
    COMMITTED = "COMMITTED"


class WorkerPool(str, Enum):
    CPU_TEACHER = "CPU_TEACHER"
    CPU_TRACE = "CPU_TRACE"
    GPU_BRAIN = "GPU_BRAIN"
    STORAGE = "STORAGE"
    IO_PREFETCH = "IO_PREFETCH"


class StorageTier(str, Enum):
    SSD_METADATA = "SSD_METADATA"
    HDD_IMMUTABLE_PAYLOAD = "HDD_IMMUTABLE_PAYLOAD"
    MEMORY_EPHEMERAL = "MEMORY_EPHEMERAL"
    GPU_EPHEMERAL = "GPU_EPHEMERAL"


class RetryMode(str, Enum):
    IDEMPOTENT_RECOMPUTE = "IDEMPOTENT_RECOMPUTE"
    RESUME_FROM_DURABLE_ENGINE_CHECKPOINT = "RESUME_FROM_DURABLE_ENGINE_CHECKPOINT"


class WorkKind(str, Enum):
    TEACHER_ACQUIRE = "TEACHER_ACQUIRE"
    POLICY_INFER = "POLICY_INFER"
    TRACE_EXECUTE = "TRACE_EXECUTE"
    MATERIALIZE_EVIDENCE = "MATERIALIZE_EVIDENCE"
    TRAIN_CHALLENGER = "TRAIN_CHALLENGER"
    VALIDATE = "VALIDATE"
    TOURNAMENT = "TOURNAMENT"
    CHECKPOINT_SEAL = "CHECKPOINT_SEAL"
    PREFETCH_IMMUTABLE = "PREFETCH_IMMUTABLE"


class EventKind(str, Enum):
    GENERATION_PREPARED = "GENERATION_PREPARED"
    INPUTS_VERIFIED = "INPUTS_VERIFIED"
    TEACHER_EVIDENCE_ACCEPTED = "TEACHER_EVIDENCE_ACCEPTED"
    SNAPSHOT_SEALED = "SNAPSHOT_SEALED"
    POLICY_INFERENCE_COMPLETED = "POLICY_INFERENCE_COMPLETED"
    TRACE_STARTED = "TRACE_STARTED"
    TRACE_EVIDENCE_MATERIALIZED = "TRACE_EVIDENCE_MATERIALIZED"
    TRAINING_STARTED = "TRAINING_STARTED"
    TRAINING_COMPLETED = "TRAINING_COMPLETED"
    VALIDATION_COMPLETED = "VALIDATION_COMPLETED"
    TOURNAMENT_DECIDED = "TOURNAMENT_DECIDED"
    TOURNAMENT_COMMITTED = "TOURNAMENT_COMMITTED"
    CHECKPOINT_SEALED = "CHECKPOINT_SEALED"
    NEXT_GENERATION_RELEASED = "NEXT_GENERATION_RELEASED"
    WORK_SCHEDULED = "WORK_SCHEDULED"
    WORK_REQUEUED = "WORK_REQUEUED"


class TournamentDecision(str, Enum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=lambda value: value.value if isinstance(value, Enum) else asdict(value),
    ).encode("utf-8")


def semantic_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


@dataclass(frozen=True)
class AuthorityStamp:
    generation: int
    champion_id: str
    champion_hash: str
    teacher_authority_id: str
    physics_authority_id: str


@dataclass(frozen=True)
class FrozenInputReceipt:
    input_identity: str
    seal_hash: str
    verified: bool = True


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    payload_hash: str
    source_generation: int
    producer_champion_id: str
    teacher_authority_id: str
    physics_authority_id: str
    teacher_generation: int
    detached_from_autograd: bool = True
    poison_bits: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoreReceipt:
    object_id: str
    payload_hash: str
    durable: bool = True
    inserted: bool = True


@dataclass(frozen=True)
class SnapshotSeal:
    snapshot_id: str
    snapshot_hash: str
    generation: int
    parent_champion_id: str
    evidence_ids: tuple[str, ...]
    immutable: bool = True


@dataclass(frozen=True)
class PolicyResult:
    work_id: str
    generation: int
    champion_id: str
    payload_hash: str
    poison_bits: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingResult:
    work_id: str
    generation: int
    challenger_id: str
    challenger_hash: str
    parent_champion_id: str
    snapshot_id: str
    payload_hash: str
    poison_bits: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationResult:
    work_id: str
    generation: int
    challenger_id: str
    parent_champion_id: str
    snapshot_id: str
    validation_id: str
    payload_hash: str
    poison_bits: tuple[str, ...] = ()


@dataclass(frozen=True)
class TournamentResult:
    work_id: str
    generation: int
    challenger_id: str
    parent_champion_id: str
    validation_id: str
    decision: TournamentDecision
    payload_hash: str
    poison_bits: tuple[str, ...] = ()


@dataclass(frozen=True)
class TournamentCommitProposal:
    generation: int
    decision: TournamentDecision
    parent_champion_id: str
    challenger_id: str
    challenger_hash: str


@dataclass(frozen=True)
class CommitReceipt:
    generation: int
    decision: TournamentDecision
    parent_champion_id: str
    next_champion_id: str
    next_champion_hash: str
    commit_id: str
    atomic: bool = True
    durable: bool = True


@dataclass(frozen=True)
class CheckpointSeal:
    generation: int
    checkpoint_id: str
    state_digest: str
    commit_id: str
    durable: bool = True


@dataclass(frozen=True)
class WorkItem:
    work_id: str
    kind: WorkKind
    generation: int
    pool: WorkerPool
    parent_champion_id: str
    snapshot_id: str | None = None
    payload_ref: str | None = None
    attempt: int = 0
    retry_mode: RetryMode = RetryMode.IDEMPOTENT_RECOMPUTE


@dataclass(frozen=True)
class WorkCompletion:
    work_id: str
    kind: WorkKind
    generation: int
    payload_hash: str
    result: Any
    poison_bits: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    kind: EventKind
    generation: int
    state_after: GenerationState
    payload_hash: str
    work_id: str | None = None
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @staticmethod
    def build(
        *,
        kind: EventKind,
        generation: int,
        state_after: GenerationState,
        payload: Any,
        work_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RuntimeEvent":
        payload_hash = semantic_hash(payload)
        md = tuple(sorted((str(k), str(v)) for k, v in (metadata or {}).items()))
        event_id = semantic_hash(
            {
                "kind": kind.value,
                "generation": generation,
                "state_after": state_after.value,
                "payload_hash": payload_hash,
                "work_id": work_id,
                "metadata": md,
            }
        )
        return RuntimeEvent(event_id, kind, generation, state_after, payload_hash, work_id, md)

    def meta(self) -> dict[str, str]:
        return dict(self.metadata)


@dataclass(frozen=True)
class JournalReceipt:
    event_id: str
    sequence: int
    durable: bool = True


@dataclass(frozen=True)
class CompletionDisposition:
    accepted: bool
    idempotent_duplicate: bool = False
    detail: str = "ACCEPTED"
