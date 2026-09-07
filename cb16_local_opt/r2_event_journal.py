from __future__ import annotations

"""R2 hot event journal for generation-specific decision/outcome events.

Unlike immutable teacher evidence, on-policy events are generation-specific and low-volume.
They belong on the SSD metadata tier and are committed in generation batches.  The journal
preserves exactly-once event identity and hard-fails same-ID/different-content conflicts.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class R2EventItem:
    event_id: str
    event_type: str
    generation: int
    policy_weight_hash: str
    snapshot_hash: str
    lineage_hash: str
    payload: Mapping[str, Any]

    @property
    def payload_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload)

    @property
    def payload_hash(self) -> str:
        return sha256_bytes(self.payload_bytes)

    @property
    def identity_hash(self) -> str:
        return sha256_bytes(canonical_json_bytes({
            "event_id": self.event_id,
            "event_type": self.event_type,
            "generation": int(self.generation),
            "policy_weight_hash": self.policy_weight_hash,
            "snapshot_hash": self.snapshot_hash,
            "lineage_hash": self.lineage_hash,
            "payload_hash": self.payload_hash,
        }))


@dataclass(frozen=True)
class R2EventRef:
    event_id: str
    event_type: str
    generation: int
    identity_hash: str
    payload_hash: str


@dataclass(frozen=True)
class R2EventBatchReceipt:
    schema: str
    generation: int
    input_count: int
    created_count: int
    reused_count: int
    transaction_seconds: float


class R2EventJournal:
    def __init__(
        self,
        root: str | Path,
        *,
        synchronous: str = "FULL",
        busy_timeout_ms: int = 30_000,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        sync = synchronous.upper()
        if sync not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError("invalid synchronous mode")
        self.conn = sqlite3.connect(
            self.root / "r2_events.sqlite",
            isolation_level=None,
            timeout=busy_timeout_ms / 1000.0,
        )
        self.conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(f"PRAGMA synchronous={sync}")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS events(
          event_id TEXT PRIMARY KEY,
          event_type TEXT NOT NULL,
          generation INTEGER NOT NULL,
          policy_weight_hash TEXT NOT NULL,
          snapshot_hash TEXT NOT NULL,
          lineage_hash TEXT NOT NULL,
          identity_hash TEXT NOT NULL,
          payload_hash TEXT NOT NULL,
          payload_json BLOB NOT NULL,
          created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_r2_events_generation
          ON events(generation,event_type,event_id);
        CREATE INDEX IF NOT EXISTS idx_r2_events_policy
          ON events(policy_weight_hash,generation,event_id);
        """)

    def close(self) -> None:
        self.conn.close()

    def checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        mode = mode.upper()
        if mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError("invalid checkpoint mode")
        return tuple(int(x) for x in self.conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone())

    def put_events(self, events: Iterable[R2EventItem]) -> tuple[list[R2EventRef], R2EventBatchReceipt]:
        rows = list(events)
        if not rows:
            return [], R2EventBatchReceipt(
                schema="CB16_R2_EVENT_BATCH_RECEIPT_V1", generation=-1,
                input_count=0, created_count=0, reused_count=0, transaction_seconds=0.0,
            )
        generations = {int(x.generation) for x in rows}
        if len(generations) != 1:
            raise RuntimeError("R2_EVENT_BATCH_MUST_BE_SINGLE_GENERATION")
        if len({x.event_id for x in rows}) != len(rows):
            raise RuntimeError("R2_EVENT_BATCH_DUPLICATE_EVENT_ID")

        started = time.perf_counter()
        created = 0
        refs: list[R2EventRef] = []
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for item in rows:
                old = self.conn.execute(
                    "SELECT event_type,generation,identity_hash,payload_hash FROM events WHERE event_id=?",
                    (item.event_id,),
                ).fetchone()
                if old is not None:
                    if old[2] != item.identity_hash:
                        raise RuntimeError(f"R2_EVENT_ID_CONTENT_CONFLICT:{item.event_id}")
                    refs.append(R2EventRef(item.event_id, str(old[0]), int(old[1]), str(old[2]), str(old[3])))
                    continue
                self.conn.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        item.event_id, item.event_type, int(item.generation),
                        item.policy_weight_hash, item.snapshot_hash, item.lineage_hash,
                        item.identity_hash, item.payload_hash, item.payload_bytes, time.time(),
                    ),
                )
                created += 1
                refs.append(R2EventRef(
                    item.event_id, item.event_type, int(item.generation),
                    item.identity_hash, item.payload_hash,
                ))
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        wall = time.perf_counter() - started
        return refs, R2EventBatchReceipt(
            schema="CB16_R2_EVENT_BATCH_RECEIPT_V1",
            generation=next(iter(generations)), input_count=len(rows),
            created_count=created, reused_count=len(rows)-created,
            transaction_seconds=wall,
        )

    def count(self, *, generation: int | None = None, event_type: str | None = None) -> int:
        clauses, args = [], []
        if generation is not None:
            clauses.append("generation=?"); args.append(int(generation))
        if event_type is not None:
            clauses.append("event_type=?"); args.append(event_type)
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        return int(self.conn.execute("SELECT COUNT(*) FROM events" + where, args).fetchone()[0])

    def audit(self) -> dict[str, Any]:
        total = self.count()
        conflicts = []
        for event_id, identity_hash, payload_hash, payload_json, event_type, generation, policy, snapshot, lineage in self.conn.execute(
            "SELECT event_id,identity_hash,payload_hash,payload_json,event_type,generation,policy_weight_hash,snapshot_hash,lineage_hash FROM events"
        ):
            raw = bytes(payload_json)
            if sha256_bytes(raw) != payload_hash:
                conflicts.append({"event_id": event_id, "error": "PAYLOAD_HASH_MISMATCH"})
                continue
            expected = sha256_bytes(canonical_json_bytes({
                "event_id": event_id, "event_type": event_type, "generation": int(generation),
                "policy_weight_hash": policy, "snapshot_hash": snapshot,
                "lineage_hash": lineage, "payload_hash": payload_hash,
            }))
            if expected != identity_hash:
                conflicts.append({"event_id": event_id, "error": "IDENTITY_HASH_MISMATCH"})
        return {
            "schema": "CB16_R2_EVENT_JOURNAL_AUDIT_V1",
            "events": total,
            "corrupt": conflicts,
            "pass": not conflicts,
        }


class R2BufferedEventSink:
    """Compatibility sink for run_real_on_policy_trace().

    ``put`` is intentionally non-durable until ``flush``.  The campaign writes the
    generation receipt only after flush succeeds, so a crash before flush simply causes
    deterministic trace regeneration on recovery.  Flush commits the full generation in
    one SSD transaction.
    """

    def __init__(self, journal: R2EventJournal):
        self.journal = journal
        self._items: list[R2EventItem] = []
        self._ids: dict[str, str] = {}

    def put(self, obj) -> tuple[R2EventRef, bool]:
        item = R2EventItem(
            event_id=str(obj.object_id), event_type=str(obj.object_type),
            generation=int(obj.generation), policy_weight_hash=str(obj.policy_weight_hash),
            snapshot_hash=str(obj.snapshot_hash), lineage_hash=str(obj.lineage_hash),
            payload=obj.payload,
        )
        old = self._ids.get(item.event_id)
        if old is not None and old != item.identity_hash:
            raise RuntimeError(f"R2_EVENT_ID_CONTENT_CONFLICT:{item.event_id}")
        self._ids[item.event_id] = item.identity_hash
        self._items.append(item)
        return R2EventRef(item.event_id, item.event_type, item.generation, item.identity_hash, item.payload_hash), True

    def flush(self) -> tuple[list[R2EventRef], R2EventBatchReceipt]:
        refs, receipt = self.journal.put_events(self._items)
        self._items.clear(); self._ids.clear()
        return refs, receipt

    @property
    def buffered_count(self) -> int:
        return len(self._items)
