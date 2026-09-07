from __future__ import annotations

"""Dependency-injection protocols for heterogeneous R11 runtime engines.

Concrete Task A/B/C/D implementations are intentionally not imported here.
Adapters only need to satisfy these structural Protocols.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from .runtime_events_r11 import (
    CheckpointSeal,
    CommitReceipt,
    EvidenceRef,
    JournalReceipt,
    RuntimeEvent,
    SnapshotSeal,
    StoreReceipt,
    TournamentCommitProposal,
    WorkCompletion,
    WorkItem,
)


@runtime_checkable
class TeacherEngine(Protocol):
    def execute(self, work: WorkItem) -> WorkCompletion: ...


@runtime_checkable
class TraceEngine(Protocol):
    def execute(self, work: WorkItem) -> WorkCompletion: ...


@runtime_checkable
class InferenceEngine(Protocol):
    def execute(self, work: WorkItem) -> WorkCompletion: ...


@runtime_checkable
class TrainingEngine(Protocol):
    def execute(self, work: WorkItem) -> WorkCompletion: ...


@runtime_checkable
class ValidationEngine(Protocol):
    def execute(self, work: WorkItem) -> WorkCompletion: ...


@runtime_checkable
class TournamentEngine(Protocol):
    def execute(self, work: WorkItem) -> WorkCompletion: ...


@runtime_checkable
class EvidenceStore(Protocol):
    """Scientific evidence persistence. No method may silently overwrite or drop."""

    def put_once(self, evidence: EvidenceRef) -> StoreReceipt: ...
    def get(self, evidence_id: str) -> EvidenceRef | None: ...
    def seal_snapshot(self, seal: SnapshotSeal) -> SnapshotSeal: ...
    def get_snapshot(self, snapshot_id: str) -> SnapshotSeal | None: ...


@runtime_checkable
class EventJournal(Protocol):
    """Append-only durable generation journal used as recovery authority."""

    def append_once(self, event: RuntimeEvent) -> JournalReceipt: ...
    def read_generation(self, generation: int) -> Sequence[RuntimeEvent]: ...


@runtime_checkable
class CheckpointStore(Protocol):
    """Atomic Champion/Challenger commit and checkpoint authority."""

    def atomic_commit(self, proposal: TournamentCommitProposal) -> CommitReceipt: ...
    def read_commit(self, generation: int) -> CommitReceipt | None: ...
    def seal_checkpoint(self, seal: CheckpointSeal) -> CheckpointSeal: ...
    def read_checkpoint(self, generation: int) -> CheckpointSeal | None: ...


@dataclass(frozen=True)
class EngineBundle:
    teacher: TeacherEngine | None = None
    trace: TraceEngine | None = None
    inference: InferenceEngine | None = None
    training: TrainingEngine | None = None
    validation: ValidationEngine | None = None
    tournament: TournamentEngine | None = None
