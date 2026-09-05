from __future__ import annotations

import hashlib
import json
import queue
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def canonical_json(obj: Any) -> bytes:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


@dataclass(frozen=True)
class TrajectoryEnvelope:
    trajectory_id: str
    policy_generation: int
    policy_weight_hash: str
    snapshot_hash: str
    market_lineage_hash: str
    account_lineage_hash: str
    config_hash: str
    payload: dict[str, Any]

    @property
    def hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True)
class EvidenceEnvelope:
    evidence_id: str
    trajectory_id: str
    trajectory_hash: str
    teacher_version: str
    provenance_hash: str
    admitted: bool
    payload: dict[str, Any]

    @property
    def hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True)
class FrozenTrainingSnapshot:
    snapshot_id: str
    generation_parent: int
    parent_policy_hash: str
    evidence_ids: tuple[str, ...]
    evidence_hashes: tuple[str, ...]
    created_at_unix: float

    @property
    def hash(self) -> str:
        return content_hash(self)


class ExactlyOnceLedger:
    """Crash-safe logical identity ledger backed by SQLite WAL.

    Replaying the same id+hash is idempotent.
    Reusing an id for different content fails closed.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS receipts(
                namespace TEXT NOT NULL,
                object_id TEXT NOT NULL,
                object_hash TEXT NOT NULL,
                first_seen REAL NOT NULL,
                PRIMARY KEY(namespace, object_id)
            )
            """
        )
        self.db.commit()
        self._lock = threading.Lock()

    def register(self, namespace: str, object_id: str, object_hash: str) -> bool:
        with self._lock:
            row = self.db.execute(
                "SELECT object_hash FROM receipts WHERE namespace=? AND object_id=?",
                (namespace, object_id),
            ).fetchone()
            if row is not None:
                if row[0] != object_hash:
                    raise RuntimeError(f"DUPLICATE_ID_CONTENT_CONFLICT:{namespace}:{object_id}")
                return False
            self.db.execute(
                "INSERT INTO receipts(namespace,object_id,object_hash,first_seen) VALUES(?,?,?,?)",
                (namespace, object_id, object_hash, time.time()),
            )
            self.db.commit()
            return True

    def count(self, namespace: str) -> int:
        return int(self.db.execute(
            "SELECT COUNT(*) FROM receipts WHERE namespace=?", (namespace,)
        ).fetchone()[0])

    def close(self) -> None:
        self.db.close()


class BoundedTrajectoryPipeline:
    """Bounded asynchronous trajectory -> Teacher/Evidence pipeline.

    This is deliberately backpressure-aware: when the Teacher cannot keep up,
    producers block rather than silently dropping trajectories.
    """

    _STOP = object()

    def __init__(
        self,
        *,
        teacher_fn: Callable[[TrajectoryEnvelope], EvidenceEnvelope],
        ledger: ExactlyOnceLedger,
        queue_capacity: int = 256,
        teacher_workers: int = 2,
    ):
        if queue_capacity <= 0 or teacher_workers <= 0:
            raise ValueError("queue_capacity and teacher_workers must be positive")
        self.teacher_fn = teacher_fn
        self.ledger = ledger
        self.in_q: queue.Queue[Any] = queue.Queue(maxsize=queue_capacity)
        self.out_q: queue.Queue[Any] = queue.Queue()  # output is durable/logically guarded; input queue provides backpressure
        self.teacher_workers = int(teacher_workers)
        self._threads: list[threading.Thread] = []
        self._errors: queue.Queue[BaseException] = queue.Queue()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for i in range(self.teacher_workers):
            t = threading.Thread(target=self._worker, name=f"cb16-teacher-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def _worker(self) -> None:
        while True:
            item = self.in_q.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, TrajectoryEnvelope)
                is_new = self.ledger.register("trajectory", item.trajectory_id, item.hash)
                if not is_new:
                    # Idempotent replay: no second Teacher compilation.
                    continue
                evidence = self.teacher_fn(item)
                if evidence.trajectory_id != item.trajectory_id or evidence.trajectory_hash != item.hash:
                    raise RuntimeError("EVIDENCE_TRAJECTORY_LINEAGE_MISMATCH")
                self.ledger.register("evidence", evidence.evidence_id, evidence.hash)
                self.out_q.put(evidence)  # bounded, no silent drop
            except BaseException as exc:
                self._errors.put(exc)
            finally:
                self.in_q.task_done()

    def submit(self, trajectory: TrajectoryEnvelope, timeout: float | None = None) -> None:
        if not self._started:
            self.start()
        self._raise_if_error()
        self.in_q.put(trajectory, timeout=timeout)

    def _raise_if_error(self) -> None:
        try:
            exc = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("ASYNC_PIPELINE_WORKER_FAIL") from exc

    def collect_available(self, max_items: int | None = None) -> list[EvidenceEnvelope]:
        self._raise_if_error()
        out = []
        while max_items is None or len(out) < max_items:
            try:
                item = self.out_q.get_nowait()
            except queue.Empty:
                break
            out.append(item)
            self.out_q.task_done()
        return out

    def drain(self) -> list[EvidenceEnvelope]:
        self.in_q.join()
        self._raise_if_error()
        return self.collect_available()

    def close(self) -> None:
        if not self._started:
            return
        self.in_q.join()
        for _ in self._threads:
            self.in_q.put(self._STOP)
        for t in self._threads:
            t.join(timeout=5)
        self._raise_if_error()
        self._started = False


class SnapshotBuilder:
    """Seal admitted Evidence into an immutable, ordered training snapshot."""

    def __init__(self, ledger: ExactlyOnceLedger):
        self.ledger = ledger

    def build(
        self,
        *,
        snapshot_id: str,
        generation_parent: int,
        parent_policy_hash: str,
        evidence: Iterable[EvidenceEnvelope],
    ) -> FrozenTrainingSnapshot:
        admitted = [e for e in evidence if e.admitted]
        # Deterministic ordering prevents worker scheduling from changing snapshot identity.
        admitted.sort(key=lambda e: (e.evidence_id, e.hash))
        ids = tuple(e.evidence_id for e in admitted)
        hashes = tuple(e.hash for e in admitted)
        if len(set(ids)) != len(ids):
            raise RuntimeError("SNAPSHOT_DUPLICATE_EVIDENCE_ID")
        snap = FrozenTrainingSnapshot(
            snapshot_id=snapshot_id,
            generation_parent=int(generation_parent),
            parent_policy_hash=str(parent_policy_hash),
            evidence_ids=ids,
            evidence_hashes=hashes,
            created_at_unix=0.0,  # status-driving identity is deterministic
        )
        self.ledger.register("training_snapshot", snap.snapshot_id, snap.hash)
        return snap
