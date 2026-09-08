from __future__ import annotations

"""Task F adapters binding the R11 orchestrator control plane to Task D durable stores.

The adapters do not change scientific evidence, Teacher, Physics, training, or
Champion/Challenger meaning. They persist only control-plane references and barriers.

Stage-4 integration may inject a fail-closed mutation guard. The guard is optional so
existing qualification/reference behavior remains unchanged. When supplied it is called
again at the concrete protocol-adapter mutation boundary immediately before the write.
"""

import json
import time
from typing import Any, Callable

from .runtime_events_r11 import (
    CheckpointSeal,
    CommitReceipt,
    EvidenceRef,
    EventKind,
    GenerationState,
    JournalReceipt,
    RuntimeEvent,
    SnapshotSeal,
    StoreReceipt,
    TournamentCommitProposal,
    TournamentDecision,
    semantic_hash,
)
from .event_journal_r11 import EventItemR11

MutationGuardR11 = Callable[[str], None]


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _evidence_obj(x: EvidenceRef) -> dict[str, Any]:
    return {
        "evidence_id": x.evidence_id,
        "payload_hash": x.payload_hash,
        "source_generation": int(x.source_generation),
        "producer_champion_id": x.producer_champion_id,
        "teacher_authority_id": x.teacher_authority_id,
        "physics_authority_id": x.physics_authority_id,
        "teacher_generation": int(x.teacher_generation),
        "detached_from_autograd": bool(x.detached_from_autograd),
        "poison_bits": list(x.poison_bits),
    }


def _evidence_from_obj(x: dict[str, Any]) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=str(x["evidence_id"]),
        payload_hash=str(x["payload_hash"]),
        source_generation=int(x["source_generation"]),
        producer_champion_id=str(x["producer_champion_id"]),
        teacher_authority_id=str(x["teacher_authority_id"]),
        physics_authority_id=str(x["physics_authority_id"]),
        teacher_generation=int(x["teacher_generation"]),
        detached_from_autograd=bool(x["detached_from_autograd"]),
        poison_bits=tuple(str(v) for v in x.get("poison_bits", ())),
    )


def _snapshot_obj(x: SnapshotSeal) -> dict[str, Any]:
    return {
        "snapshot_id": x.snapshot_id,
        "snapshot_hash": x.snapshot_hash,
        "generation": int(x.generation),
        "parent_champion_id": x.parent_champion_id,
        "evidence_ids": list(x.evidence_ids),
        "immutable": bool(x.immutable),
    }


def _snapshot_from_obj(x: dict[str, Any]) -> SnapshotSeal:
    return SnapshotSeal(
        snapshot_id=str(x["snapshot_id"]),
        snapshot_hash=str(x["snapshot_hash"]),
        generation=int(x["generation"]),
        parent_champion_id=str(x["parent_champion_id"]),
        evidence_ids=tuple(str(v) for v in x["evidence_ids"]),
        immutable=bool(x["immutable"]),
    )


def _runtime_event_obj(x: RuntimeEvent) -> dict[str, Any]:
    return {
        "event_id": x.event_id,
        "kind": x.kind.value,
        "generation": int(x.generation),
        "state_after": x.state_after.value,
        "payload_hash": x.payload_hash,
        "work_id": x.work_id,
        "metadata": [list(v) for v in x.metadata],
    }


def _runtime_event_from_obj(x: dict[str, Any]) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=str(x["event_id"]),
        kind=EventKind(str(x["kind"])),
        generation=int(x["generation"]),
        state_after=GenerationState(str(x["state_after"])),
        payload_hash=str(x["payload_hash"]),
        work_id=None if x.get("work_id") is None else str(x["work_id"]),
        metadata=tuple((str(a), str(b)) for a, b in x.get("metadata", ())),
    )


def _commit_obj(x: CommitReceipt) -> dict[str, Any]:
    return {
        "generation": int(x.generation),
        "decision": x.decision.value,
        "parent_champion_id": x.parent_champion_id,
        "next_champion_id": x.next_champion_id,
        "next_champion_hash": x.next_champion_hash,
        "commit_id": x.commit_id,
        "atomic": bool(x.atomic),
        "durable": bool(x.durable),
    }


def _commit_from_obj(x: dict[str, Any]) -> CommitReceipt:
    return CommitReceipt(
        generation=int(x["generation"]),
        decision=TournamentDecision(str(x["decision"])),
        parent_champion_id=str(x["parent_champion_id"]),
        next_champion_id=str(x["next_champion_id"]),
        next_champion_hash=str(x["next_champion_hash"]),
        commit_id=str(x["commit_id"]),
        atomic=bool(x.get("atomic", True)),
        durable=bool(x.get("durable", True)),
    )


def _checkpoint_obj(x: CheckpointSeal) -> dict[str, Any]:
    return {
        "generation": int(x.generation),
        "checkpoint_id": x.checkpoint_id,
        "state_digest": x.state_digest,
        "commit_id": x.commit_id,
        "durable": bool(x.durable),
    }


def _checkpoint_from_obj(x: dict[str, Any]) -> CheckpointSeal:
    return CheckpointSeal(
        generation=int(x["generation"]),
        checkpoint_id=str(x["checkpoint_id"]),
        state_digest=str(x["state_digest"]),
        commit_id=str(x["commit_id"]),
        durable=bool(x.get("durable", True)),
    )


class _MutationGuardMixinR11:
    _mutation_guard: MutationGuardR11 | None

    def _assert_mutation(self, operation: str) -> None:
        if self._mutation_guard is not None:
            self._mutation_guard(operation)


class EvidenceStoreProtocolAdapterR11(_MutationGuardMixinR11):
    """Persist E's scientific references inside Task D's metadata SQLite.

    The underlying payload hash remains an external content address. This adapter
    does not manufacture, transform, or relabel the evidence payload.
    """

    def __init__(self, store: Any, *, mutation_guard: MutationGuardR11 | None = None):
        self.store = store
        self.conn = store.conn
        self._mutation_guard = mutation_guard
        self._assert_mutation("evidence.adapter_schema")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS orchestrator_evidence_refs(
              evidence_id TEXT PRIMARY KEY,
              payload_hash TEXT NOT NULL,
              semantic_hash TEXT NOT NULL,
              payload_json BLOB NOT NULL,
              created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS orchestrator_snapshots(
              snapshot_id TEXT PRIMARY KEY,
              snapshot_hash TEXT NOT NULL,
              semantic_hash TEXT NOT NULL,
              payload_json BLOB NOT NULL,
              created_at REAL NOT NULL
            );
            """
        )

    def put_once(self, evidence: EvidenceRef) -> StoreReceipt:
        obj = _evidence_obj(evidence)
        raw = _json_bytes(obj)
        sem = semantic_hash(obj)
        self._assert_mutation(f"evidence.put_once.begin:{evidence.evidence_id}")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            old = self.conn.execute(
                "SELECT payload_hash,semantic_hash,payload_json FROM orchestrator_evidence_refs WHERE evidence_id=?",
                (evidence.evidence_id,),
            ).fetchone()
            if old is not None:
                self.conn.execute("COMMIT")
                if str(old[1]) != sem or bytes(old[2]) != raw:
                    return StoreReceipt(evidence.evidence_id, str(old[0]), True, False)
                return StoreReceipt(evidence.evidence_id, evidence.payload_hash, True, False)
            self._assert_mutation(f"evidence.put_once.insert:{evidence.evidence_id}")
            self.conn.execute(
                "INSERT INTO orchestrator_evidence_refs VALUES(?,?,?,?,?)",
                (evidence.evidence_id, evidence.payload_hash, sem, raw, time.time()),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        return StoreReceipt(evidence.evidence_id, evidence.payload_hash, True, True)

    def get(self, evidence_id: str) -> EvidenceRef | None:
        row = self.conn.execute(
            "SELECT payload_json FROM orchestrator_evidence_refs WHERE evidence_id=?",
            (str(evidence_id),),
        ).fetchone()
        return None if row is None else _evidence_from_obj(json.loads(bytes(row[0])))

    def seal_snapshot(self, seal: SnapshotSeal) -> SnapshotSeal:
        if not seal.immutable:
            raise RuntimeError("R11_ADAPTER_REFUSES_MUTABLE_SNAPSHOT")
        for evidence_id in seal.evidence_ids:
            if self.get(evidence_id) is None:
                raise RuntimeError(f"R11_ADAPTER_SNAPSHOT_EVIDENCE_MISSING:{evidence_id}")
        obj = _snapshot_obj(seal)
        raw = _json_bytes(obj)
        sem = semantic_hash(obj)
        self._assert_mutation(f"evidence.seal_snapshot.begin:{seal.snapshot_id}")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            old = self.conn.execute(
                "SELECT semantic_hash,payload_json FROM orchestrator_snapshots WHERE snapshot_id=?",
                (seal.snapshot_id,),
            ).fetchone()
            if old is not None:
                if str(old[0]) != sem or bytes(old[1]) != raw:
                    raise RuntimeError(f"R11_ADAPTER_SNAPSHOT_CONFLICT:{seal.snapshot_id}")
                self.conn.execute("COMMIT")
                return seal
            self._assert_mutation(f"evidence.seal_snapshot.insert:{seal.snapshot_id}")
            self.conn.execute(
                "INSERT INTO orchestrator_snapshots VALUES(?,?,?,?,?)",
                (seal.snapshot_id, seal.snapshot_hash, sem, raw, time.time()),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        return seal

    def get_snapshot(self, snapshot_id: str) -> SnapshotSeal | None:
        row = self.conn.execute(
            "SELECT payload_json FROM orchestrator_snapshots WHERE snapshot_id=?",
            (str(snapshot_id),),
        ).fetchone()
        return None if row is None else _snapshot_from_obj(json.loads(bytes(row[0])))


class EventJournalProtocolAdapterR11(_MutationGuardMixinR11):
    """Persist RuntimeEvent exactly once through Task D's SQLite event journal."""

    EVENT_TYPE_PREFIX = "ORCH:"

    def __init__(self, journal: Any, *, mutation_guard: MutationGuardR11 | None = None):
        self.journal = journal
        self.conn = journal.conn
        self._mutation_guard = mutation_guard

    def append_once(self, event: RuntimeEvent) -> JournalReceipt:
        payload = _runtime_event_obj(event)
        item = EventItemR11(
            event_id=event.event_id,
            event_type=self.EVENT_TYPE_PREFIX + event.kind.value,
            generation=int(event.generation),
            policy_weight_hash="R11_ORCHESTRATOR_CONTROL_PLANE",
            snapshot_hash=event.meta().get("snapshot_hash", event.state_after.value),
            lineage_hash=event.payload_hash,
            payload=payload,
        )
        self._assert_mutation(f"journal.append_once:{event.kind.value}:{event.event_id}")
        self.journal.seal_trace_batch([item], trace_batch_id=f"ORCH-EVENT:{event.event_id}")
        row = self.conn.execute("SELECT rowid FROM events WHERE event_id=?", (event.event_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"R11_ADAPTER_EVENT_NOT_DURABLE:{event.event_id}")
        return JournalReceipt(event.event_id, int(row[0]), True)

    def read_generation(self, generation: int) -> list[RuntimeEvent]:
        rows = self.conn.execute(
            "SELECT payload_json FROM events WHERE generation=? AND event_type LIKE ? ORDER BY rowid",
            (int(generation), self.EVENT_TYPE_PREFIX + "%"),
        ).fetchall()
        out = [_runtime_event_from_obj(json.loads(bytes(row[0]))) for row in rows]
        if len({e.event_id for e in out}) != len(out):
            raise RuntimeError("R11_ADAPTER_DUPLICATE_RUNTIME_EVENT")
        return out


class CheckpointStoreProtocolAdapterR11(_MutationGuardMixinR11):
    """Bind tournament control commits to real content-addressed tensor objects.

    Production invariant: Champion/Challenger IDs are their semantic tensor hashes.
    This removes a second mutable naming layer from the promotion barrier.
    """

    def __init__(self, store: Any, *, mutation_guard: MutationGuardR11 | None = None):
        self.store = store
        self.conn = store.conn
        self._mutation_guard = mutation_guard
        self._assert_mutation("checkpoint.adapter_schema")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS orchestrator_commits(
              generation INTEGER PRIMARY KEY,
              semantic_hash TEXT NOT NULL,
              payload_json BLOB NOT NULL,
              created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS orchestrator_checkpoint_seals(
              generation INTEGER PRIMARY KEY,
              semantic_hash TEXT NOT NULL,
              payload_json BLOB NOT NULL,
              created_at REAL NOT NULL
            );
            """
        )

    @staticmethod
    def _require_hash_id(policy_id: str, policy_hash: str) -> None:
        if policy_id != policy_hash:
            raise RuntimeError(f"R11_POLICY_ID_MUST_EQUAL_CONTENT_HASH:{policy_id}:{policy_hash}")

    def atomic_commit(self, proposal: TournamentCommitProposal) -> CommitReceipt:
        self._require_hash_id(proposal.challenger_id, proposal.challenger_hash)
        if not self.store.has_object(proposal.parent_champion_id):
            raise RuntimeError(f"R11_PARENT_CHAMPION_OBJECT_MISSING:{proposal.parent_champion_id}")
        if not self.store.has_object(proposal.challenger_hash):
            raise RuntimeError(f"R11_CHALLENGER_OBJECT_MISSING:{proposal.challenger_hash}")
        next_hash = proposal.challenger_hash if proposal.decision is TournamentDecision.PROMOTE else proposal.parent_champion_id
        body = {
            "generation": int(proposal.generation),
            "decision": proposal.decision.value,
            "parent_champion_id": proposal.parent_champion_id,
            "challenger_id": proposal.challenger_id,
            "challenger_hash": proposal.challenger_hash,
            "next_champion_hash": next_hash,
        }
        receipt = CommitReceipt(
            generation=int(proposal.generation),
            decision=proposal.decision,
            parent_champion_id=proposal.parent_champion_id,
            next_champion_id=next_hash,
            next_champion_hash=next_hash,
            commit_id=semantic_hash(body),
            atomic=True,
            durable=True,
        )
        raw = _json_bytes(_commit_obj(receipt))
        sem = semantic_hash(_commit_obj(receipt))
        self._assert_mutation(f"checkpoint.atomic_commit.begin:{proposal.generation}")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            old = self.conn.execute(
                "SELECT semantic_hash,payload_json FROM orchestrator_commits WHERE generation=?",
                (int(proposal.generation),),
            ).fetchone()
            if old is not None:
                existing = _commit_from_obj(json.loads(bytes(old[1])))
                if str(old[0]) != sem or existing != receipt:
                    raise RuntimeError(f"R11_ADAPTER_TOURNAMENT_COMMIT_CONFLICT:{proposal.generation}")
                self.conn.execute("COMMIT")
                return existing
            self._assert_mutation(f"checkpoint.atomic_commit.insert:{proposal.generation}")
            self.conn.execute(
                "INSERT INTO orchestrator_commits VALUES(?,?,?,?)",
                (int(proposal.generation), sem, raw, time.time()),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        return receipt

    def read_commit(self, generation: int) -> CommitReceipt | None:
        row = self.conn.execute(
            "SELECT payload_json FROM orchestrator_commits WHERE generation=?",
            (int(generation),),
        ).fetchone()
        return None if row is None else _commit_from_obj(json.loads(bytes(row[0])))

    def seal_checkpoint(self, seal: CheckpointSeal) -> CheckpointSeal:
        commit = self.read_commit(seal.generation)
        if commit is None:
            raise RuntimeError(f"R11_ADAPTER_COMMIT_REQUIRED:{seal.generation}")
        if seal.commit_id != commit.commit_id:
            raise RuntimeError("R11_ADAPTER_CHECKPOINT_COMMIT_ID_MISMATCH")
        if seal.checkpoint_id != commit.next_champion_hash:
            raise RuntimeError("R11_ADAPTER_CHECKPOINT_MUST_REFERENCE_COMMITTED_CHAMPION_OBJECT")
        if not self.store.has_object(seal.checkpoint_id):
            raise RuntimeError(f"R11_ADAPTER_CHECKPOINT_OBJECT_MISSING:{seal.checkpoint_id}")
        obj = _checkpoint_obj(seal)
        raw = _json_bytes(obj)
        sem = semantic_hash(obj)
        self._assert_mutation(f"checkpoint.seal_checkpoint.begin:{seal.generation}")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            old = self.conn.execute(
                "SELECT semantic_hash,payload_json FROM orchestrator_checkpoint_seals WHERE generation=?",
                (int(seal.generation),),
            ).fetchone()
            if old is not None:
                existing = _checkpoint_from_obj(json.loads(bytes(old[1])))
                if str(old[0]) != sem or existing != seal:
                    raise RuntimeError(f"R11_ADAPTER_CHECKPOINT_CONFLICT:{seal.generation}")
                self.conn.execute("COMMIT")
                return existing
            self._assert_mutation(f"checkpoint.seal_checkpoint.insert:{seal.generation}")
            self.conn.execute(
                "INSERT INTO orchestrator_checkpoint_seals VALUES(?,?,?,?)",
                (int(seal.generation), sem, raw, time.time()),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        return seal

    def read_checkpoint(self, generation: int) -> CheckpointSeal | None:
        row = self.conn.execute(
            "SELECT payload_json FROM orchestrator_checkpoint_seals WHERE generation=?",
            (int(generation),),
        ).fetchone()
        return None if row is None else _checkpoint_from_obj(json.loads(bytes(row[0])))


__all__ = [
    "MutationGuardR11",
    "EvidenceStoreProtocolAdapterR11",
    "EventJournalProtocolAdapterR11",
    "CheckpointStoreProtocolAdapterR11",
]
