from __future__ import annotations

"""R11 exactly-once event journal and generation transaction seals.

The journal is metadata/control-plane persistence only.  It does not change event or
evidence scientific meaning.  Event completion order is normalized before a trace batch
is sealed, and the batch seal is committed in the same SQLite transaction as new events.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

FAIL_AFTER_EVENTS_BEFORE_BATCH_SEAL = "AFTER_EVENTS_BEFORE_BATCH_SEAL"
FAIL_AFTER_JOURNAL_COMMIT_BEFORE_SNAPSHOT_SEAL = "AFTER_JOURNAL_COMMIT_BEFORE_SNAPSHOT_SEAL"
FailHook = Callable[[str], None]


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


@dataclass(frozen=True)
class EventItemR11:
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
        return sha256_obj({
            "event_id": self.event_id,
            "event_type": self.event_type,
            "generation": int(self.generation),
            "policy_weight_hash": self.policy_weight_hash,
            "snapshot_hash": self.snapshot_hash,
            "lineage_hash": self.lineage_hash,
            "payload_hash": self.payload_hash,
        })


@dataclass(frozen=True)
class TraceBatchSealR11:
    trace_batch_id: str
    generation: int
    batch_hash: str
    event_count: int
    created_event_count: int
    reused_event_count: int
    reused_batch: bool


@dataclass(frozen=True)
class GenerationOutcomeR11:
    generation: int
    parent_champion: str
    challenger: str
    decision: str
    champion_after: str
    trace_batch_id: str
    snapshot_hash: str
    identity_hash: str


class EventJournalR11:
    def __init__(self, root: str | Path, *, synchronous: str = "FULL", busy_timeout_ms: int = 30_000):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        sync = synchronous.upper()
        if sync not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError("invalid synchronous mode")
        self.conn = sqlite3.connect(
            self.root / "r11_events.sqlite", isolation_level=None, timeout=busy_timeout_ms / 1000.0
        )
        self.conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(f"PRAGMA synchronous={sync}")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS events(
          event_id TEXT PRIMARY KEY,event_type TEXT NOT NULL,generation INTEGER NOT NULL,
          policy_weight_hash TEXT NOT NULL,snapshot_hash TEXT NOT NULL,lineage_hash TEXT NOT NULL,
          identity_hash TEXT NOT NULL,payload_hash TEXT NOT NULL,payload_json BLOB NOT NULL,
          created_at REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_r11_events_generation
          ON events(generation,event_type,event_id);
        CREATE TABLE IF NOT EXISTS trace_batches(
          trace_batch_id TEXT PRIMARY KEY,generation INTEGER NOT NULL,batch_hash TEXT NOT NULL,
          event_count INTEGER NOT NULL,sealed_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS trace_batch_members(
          trace_batch_id TEXT NOT NULL,event_id TEXT NOT NULL,ordinal INTEGER NOT NULL,
          event_identity_hash TEXT NOT NULL,
          PRIMARY KEY(trace_batch_id,event_id),UNIQUE(trace_batch_id,ordinal));
        CREATE TABLE IF NOT EXISTS generation_outcomes(
          generation INTEGER PRIMARY KEY,parent_champion TEXT NOT NULL,challenger TEXT NOT NULL,
          decision TEXT NOT NULL,champion_after TEXT NOT NULL,trace_batch_id TEXT NOT NULL,
          snapshot_hash TEXT NOT NULL,identity_hash TEXT NOT NULL,sealed_at REAL NOT NULL);
        """)

    def close(self) -> None:
        self.conn.close()

    def checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        mode = mode.upper()
        if mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError("invalid checkpoint mode")
        return tuple(int(x) for x in self.conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone())

    @staticmethod
    def _normalize_events(events: Iterable[EventItemR11]) -> tuple[list[EventItemR11], int]:
        rows = list(events)
        if not rows:
            raise RuntimeError("R11_TRACE_BATCH_EMPTY")
        generations = {int(x.generation) for x in rows}
        if len(generations) != 1:
            raise RuntimeError("R11_TRACE_BATCH_MUST_BE_SINGLE_GENERATION")
        by_id: dict[str, EventItemR11] = {}
        for item in rows:
            old = by_id.get(item.event_id)
            if old is not None and old.identity_hash != item.identity_hash:
                raise RuntimeError(f"R11_EVENT_ID_CONTENT_CONFLICT:{item.event_id}")
            by_id[item.event_id] = item
        return sorted(by_id.values(), key=lambda x: (x.event_id, x.identity_hash)), len(rows)

    def seal_trace_batch(
        self,
        events: Iterable[EventItemR11],
        *,
        trace_batch_id: str | None = None,
        fail_hook: FailHook | None = None,
    ) -> TraceBatchSealR11:
        rows, input_count = self._normalize_events(events)
        generation = int(rows[0].generation)
        batch_body = {
            "schema": "CB16_R11_TRACE_BATCH_SEAL_V1",
            "generation": generation,
            "events": [[x.event_id, x.identity_hash] for x in rows],
        }
        batch_hash = sha256_obj(batch_body)
        batch_id = str(trace_batch_id or batch_hash)
        created = 0
        reused = 0
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            existing_batch = self.conn.execute(
                "SELECT generation,batch_hash,event_count FROM trace_batches WHERE trace_batch_id=?",
                (batch_id,),
            ).fetchone()
            if existing_batch is not None:
                if (int(existing_batch[0]), str(existing_batch[1]), int(existing_batch[2])) != (
                    generation, batch_hash, len(rows)
                ):
                    raise RuntimeError(f"R11_TRACE_BATCH_ID_CONTENT_CONFLICT:{batch_id}")
                members = self.conn.execute(
                    "SELECT event_id,event_identity_hash FROM trace_batch_members "
                    "WHERE trace_batch_id=? ORDER BY ordinal", (batch_id,)
                ).fetchall()
                expected = [(x.event_id, x.identity_hash) for x in rows]
                actual = [(str(a), str(b)) for a, b in members]
                if actual != expected:
                    raise RuntimeError(f"R11_TRACE_BATCH_MEMBERSHIP_CONFLICT:{batch_id}")
                self.conn.execute("COMMIT")
                return TraceBatchSealR11(
                    batch_id, generation, batch_hash, len(rows), 0, len(rows), True
                )

            for item in rows:
                old = self.conn.execute(
                    "SELECT event_type,generation,identity_hash,payload_hash FROM events WHERE event_id=?",
                    (item.event_id,),
                ).fetchone()
                if old is not None:
                    if str(old[2]) != item.identity_hash:
                        raise RuntimeError(f"R11_EVENT_ID_CONTENT_CONFLICT:{item.event_id}")
                    reused += 1
                    continue
                self.conn.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (item.event_id, item.event_type, generation, item.policy_weight_hash,
                     item.snapshot_hash, item.lineage_hash, item.identity_hash, item.payload_hash,
                     item.payload_bytes, time.time()),
                )
                created += 1
            if fail_hook is not None:
                fail_hook(FAIL_AFTER_EVENTS_BEFORE_BATCH_SEAL)
            self.conn.execute(
                "INSERT INTO trace_batches VALUES(?,?,?,?,?)",
                (batch_id, generation, batch_hash, len(rows), time.time()),
            )
            self.conn.executemany(
                "INSERT INTO trace_batch_members VALUES(?,?,?,?)",
                [(batch_id, item.event_id, ordinal, item.identity_hash)
                 for ordinal, item in enumerate(rows)],
            )
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        if fail_hook is not None:
            fail_hook(FAIL_AFTER_JOURNAL_COMMIT_BEFORE_SNAPSHOT_SEAL)
        # input_count can exceed event_count because exact duplicate inputs collapse by identity.
        return TraceBatchSealR11(
            batch_id, generation, batch_hash, len(rows), created,
            reused + (input_count - len(rows)), False
        )

    def _last_outcome(self) -> tuple[int, str] | None:
        row = self.conn.execute(
            "SELECT generation,champion_after FROM generation_outcomes ORDER BY generation DESC LIMIT 1"
        ).fetchone()
        return None if row is None else (int(row[0]), str(row[1]))

    def seal_generation_outcome(
        self,
        *,
        generation: int,
        parent_champion: str,
        challenger: str,
        decision: str,
        champion_after: str,
        trace_batch_id: str,
        snapshot_hash: str,
    ) -> GenerationOutcomeR11:
        generation = int(generation)
        decision = str(decision).upper()
        if decision not in {"PROMOTE", "REJECT"}:
            raise RuntimeError(f"R11_INVALID_PROMOTION_DECISION:{decision}")
        expected_after = challenger if decision == "PROMOTE" else parent_champion
        if champion_after != expected_after:
            raise RuntimeError(
                f"R11_CHAMPION_AFTER_BINDING_INVALID:{decision}:{champion_after}:{expected_after}"
            )
        batch = self.conn.execute(
            "SELECT generation FROM trace_batches WHERE trace_batch_id=?", (trace_batch_id,)
        ).fetchone()
        if batch is None:
            raise RuntimeError(f"R11_TRACE_BATCH_NOT_SEALED:{trace_batch_id}")
        if int(batch[0]) != generation:
            raise RuntimeError("R11_TRACE_BATCH_GENERATION_MISMATCH")
        body = {
            "schema": "CB16_R11_GENERATION_OUTCOME_V1",
            "generation": generation,
            "parent_champion": parent_champion,
            "challenger": challenger,
            "decision": decision,
            "champion_after": champion_after,
            "trace_batch_id": trace_batch_id,
            "snapshot_hash": snapshot_hash,
        }
        identity_hash = sha256_obj(body)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            old = self.conn.execute(
                "SELECT parent_champion,challenger,decision,champion_after,trace_batch_id,"
                "snapshot_hash,identity_hash FROM generation_outcomes WHERE generation=?",
                (generation,),
            ).fetchone()
            if old is not None:
                if str(old[6]) != identity_hash:
                    raise RuntimeError(f"R11_GENERATION_OUTCOME_CONFLICT:{generation}")
                self.conn.execute("COMMIT")
                return GenerationOutcomeR11(
                    generation, parent_champion, challenger, decision, champion_after,
                    trace_batch_id, snapshot_hash, identity_hash
                )
            last = self._last_outcome()
            if last is not None:
                last_generation, last_champion_after = last
                if generation != last_generation + 1:
                    raise RuntimeError(
                        f"R11_GENERATION_SEQUENCE_GAP:{last_generation}:{generation}"
                    )
                if parent_champion != last_champion_after:
                    raise RuntimeError(
                        f"R11_REJECTED_OR_STALE_CHALLENGER_PARENT_FORBIDDEN:"
                        f"expected={last_champion_after}:got={parent_champion}"
                    )
            self.conn.execute(
                "INSERT INTO generation_outcomes VALUES(?,?,?,?,?,?,?,?,?)",
                (generation, parent_champion, challenger, decision, champion_after,
                 trace_batch_id, snapshot_hash, identity_hash, time.time()),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        return GenerationOutcomeR11(
            generation, parent_champion, challenger, decision, champion_after,
            trace_batch_id, snapshot_hash, identity_hash
        )

    def pending_trace_batches(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT b.trace_batch_id,b.generation,b.batch_hash,b.event_count "
            "FROM trace_batches b LEFT JOIN generation_outcomes g "
            "ON g.trace_batch_id=b.trace_batch_id WHERE g.trace_batch_id IS NULL "
            "ORDER BY b.generation,b.trace_batch_id"
        ).fetchall()
        return [
            {"trace_batch_id": str(a), "generation": int(b), "batch_hash": str(c), "event_count": int(d)}
            for a, b, c, d in rows
        ]

    def canonical_events(self, trace_batch_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT e.event_id,e.event_type,e.generation,e.identity_hash,e.payload_json "
            "FROM trace_batch_members m JOIN events e ON e.event_id=m.event_id "
            "WHERE m.trace_batch_id=? ORDER BY m.ordinal", (trace_batch_id,)
        ).fetchall()
        return [
            {"event_id": str(a), "event_type": str(b), "generation": int(c),
             "identity_hash": str(d), "payload": json.loads(bytes(e))}
            for a, b, c, d, e in rows
        ]

    def recover(self) -> dict[str, Any]:
        """SQLite resolves uncommitted WAL txns; durable unbound batches are surfaced, never dropped."""
        audit = self.audit()
        if not audit["pass"]:
            raise RuntimeError(f"R11_EVENT_JOURNAL_CORRUPT:{audit['problems']}")
        pending = self.pending_trace_batches()
        return {
            "schema": "CB16_R11_EVENT_JOURNAL_RECOVERY_V1",
            "pending_trace_batches": pending,
            "pending_count": len(pending),
            "events_dropped": 0,
            "pass": True,
        }

    def audit(self) -> dict[str, Any]:
        problems: list[dict[str, Any]] = []
        for row in self.conn.execute(
            "SELECT event_id,event_type,generation,policy_weight_hash,snapshot_hash,lineage_hash,"
            "identity_hash,payload_hash,payload_json FROM events"
        ):
            event_id, event_type, generation, policy, snapshot, lineage, identity, payload_hash, payload_json = row
            raw = bytes(payload_json)
            if sha256_bytes(raw) != str(payload_hash):
                problems.append({"event_id": str(event_id), "error": "PAYLOAD_HASH_MISMATCH"})
                continue
            expected = sha256_obj({
                "event_id": str(event_id), "event_type": str(event_type), "generation": int(generation),
                "policy_weight_hash": str(policy), "snapshot_hash": str(snapshot),
                "lineage_hash": str(lineage), "payload_hash": str(payload_hash),
            })
            if expected != str(identity):
                problems.append({"event_id": str(event_id), "error": "IDENTITY_HASH_MISMATCH"})
        for batch_id, generation, batch_hash, event_count in self.conn.execute(
            "SELECT trace_batch_id,generation,batch_hash,event_count FROM trace_batches"
        ):
            members = self.conn.execute(
                "SELECT event_id,event_identity_hash FROM trace_batch_members "
                "WHERE trace_batch_id=? ORDER BY ordinal", (batch_id,)
            ).fetchall()
            if len(members) != int(event_count):
                problems.append({"trace_batch_id": str(batch_id), "error": "MEMBER_COUNT_MISMATCH"})
                continue
            expected_hash = sha256_obj({
                "schema": "CB16_R11_TRACE_BATCH_SEAL_V1",
                "generation": int(generation),
                "events": [[str(a), str(b)] for a, b in members],
            })
            if expected_hash != str(batch_hash):
                problems.append({"trace_batch_id": str(batch_id), "error": "BATCH_HASH_MISMATCH"})
        previous: tuple[int, str] | None = None
        for generation, parent, challenger, decision, champion_after, batch_id, snapshot_hash, identity_hash in self.conn.execute(
            "SELECT generation,parent_champion,challenger,decision,champion_after,trace_batch_id,"
            "snapshot_hash,identity_hash FROM generation_outcomes ORDER BY generation"
        ):
            expected_after = str(challenger) if str(decision) == "PROMOTE" else str(parent)
            if str(champion_after) != expected_after:
                problems.append({"generation": int(generation), "error": "CHAMPION_AFTER_BINDING_INVALID"})
            if previous is not None:
                if int(generation) != previous[0] + 1 or str(parent) != previous[1]:
                    problems.append({"generation": int(generation), "error": "PARENT_LINEAGE_DISCONTINUITY"})
            expected_identity = sha256_obj({
                "schema": "CB16_R11_GENERATION_OUTCOME_V1", "generation": int(generation),
                "parent_champion": str(parent), "challenger": str(challenger),
                "decision": str(decision), "champion_after": str(champion_after),
                "trace_batch_id": str(batch_id), "snapshot_hash": str(snapshot_hash),
            })
            if expected_identity != str(identity_hash):
                problems.append({"generation": int(generation), "error": "OUTCOME_IDENTITY_HASH_MISMATCH"})
            previous = (int(generation), str(champion_after))
        return {
            "schema": "CB16_R11_EVENT_JOURNAL_AUDIT_V1",
            "events": int(self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
            "trace_batches": int(self.conn.execute("SELECT COUNT(*) FROM trace_batches").fetchone()[0]),
            "generation_outcomes": int(self.conn.execute("SELECT COUNT(*) FROM generation_outcomes").fetchone()[0]),
            "problems": problems,
            "pass": not problems,
        }
