from __future__ import annotations

"""
Persistent sharded Experience Lake.

Implementation:
- N SQLite metadata shards, each in WAL mode;
- content-addressed compressed JSON payload files;
- deterministic shard routing by object_id hash;
- exactly-once identity with same-id/different-content conflict detection;
- immutable objects;
- generation/type indexes;
- explicit WAL checkpoints;
- snapshot sealing by immutable ordered object identities.

The Lake is local-filesystem only. SQLite WAL must not be placed on a network
filesystem because WAL coordination depends on same-host shared memory semantics.
"""

import hashlib
import json
import os
import sqlite3
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


def canonical_json_bytes(obj: Any) -> bytes:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_shard(object_id: str, shards: int) -> int:
    if shards <= 0:
        raise ValueError("shards must be positive")
    return int(hashlib.sha256(object_id.encode()).hexdigest()[:16], 16) % shards


@dataclass(frozen=True)
class ExperienceObject:
    object_id: str
    object_type: str
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
            "object_id": self.object_id,
            "object_type": self.object_type,
            "generation": self.generation,
            "policy_weight_hash": self.policy_weight_hash,
            "snapshot_hash": self.snapshot_hash,
            "lineage_hash": self.lineage_hash,
            "payload_hash": self.payload_hash,
        }))


@dataclass(frozen=True)
class ExperienceRef:
    object_id: str
    object_type: str
    generation: int
    shard: int
    identity_hash: str
    payload_hash: str
    payload_path: str
    bytes_raw: int
    bytes_stored: int


@dataclass(frozen=True)
class ExperienceSnapshot:
    snapshot_id: str
    parent_generation: int
    parent_policy_hash: str
    object_ids: tuple[str, ...]
    object_identity_hashes: tuple[str, ...]
    object_count: int

    @property
    def content_hash(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))


class _Shard:
    def __init__(
        self,
        db_path: Path,
        payload_root: Path,
        *,
        busy_timeout_ms: int,
        synchronous: str,
    ):
        self.db_path = db_path
        self.payload_root = payload_root
        db_path.parent.mkdir(parents=True, exist_ok=True)
        payload_root.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            db_path,
            timeout=busy_timeout_ms / 1000.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self.conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(f"PRAGMA synchronous={synchronous}")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA wal_autocheckpoint=2000")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS objects(
                object_id TEXT PRIMARY KEY,
                object_type TEXT NOT NULL,
                generation INTEGER NOT NULL,
                policy_weight_hash TEXT NOT NULL,
                snapshot_hash TEXT NOT NULL,
                lineage_hash TEXT NOT NULL,
                identity_hash TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_path TEXT NOT NULL,
                bytes_raw INTEGER NOT NULL,
                bytes_stored INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_objects_generation
                ON objects(generation, object_type, object_id);
            CREATE INDEX IF NOT EXISTS idx_objects_snapshot
                ON objects(snapshot_hash, object_id);
            CREATE INDEX IF NOT EXISTS idx_objects_policy
                ON objects(policy_weight_hash, generation, object_id);

            CREATE TABLE IF NOT EXISTS snapshots(
                snapshot_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                parent_generation INTEGER NOT NULL,
                parent_policy_hash TEXT NOT NULL,
                object_count INTEGER NOT NULL,
                payload_json BLOB NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )

    def close(self) -> None:
        self.conn.close()

    def checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        mode = mode.upper()
        if mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError("invalid WAL checkpoint mode")
        row = self.conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        return tuple(int(x) for x in row)

    def _payload_path(self, payload_hash: str) -> Path:
        return (
            self.payload_root
            / payload_hash[:2]
            / payload_hash[2:4]
            / f"{payload_hash}.json.zlib"
        )

    def _atomic_write_payload(self, payload_hash: str, raw: bytes) -> tuple[Path, int]:
        target = self._payload_path(payload_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            stored = target.read_bytes()
            try:
                if sha256_bytes(zlib.decompress(stored)) != payload_hash:
                    raise RuntimeError("EXISTING_CAS_PAYLOAD_CORRUPT")
            except zlib.error as exc:
                raise RuntimeError("EXISTING_CAS_PAYLOAD_CORRUPT") from exc
            return target, len(stored)

        compressed = zlib.compress(raw, level=3)
        fd, tmp_name = tempfile.mkstemp(
            prefix=target.name + ".",
            suffix=".partial",
            dir=str(target.parent),
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(compressed)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
            return target, len(compressed)
        finally:
            tmp.unlink(missing_ok=True)

    def put(self, obj: ExperienceObject) -> tuple[ExperienceRef, bool]:
        if not obj.object_id or not obj.object_type:
            raise ValueError("object id/type required")
        raw = obj.payload_bytes
        path, stored_bytes = self._atomic_write_payload(obj.payload_hash, raw)
        now = time.time()

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            old = self.conn.execute(
                """
                SELECT object_type,generation,policy_weight_hash,snapshot_hash,lineage_hash,
                       identity_hash,payload_hash,payload_path,bytes_raw,bytes_stored
                FROM objects WHERE object_id=?
                """,
                (obj.object_id,),
            ).fetchone()
            if old is not None:
                if old[5] != obj.identity_hash:
                    raise RuntimeError(f"EXPERIENCE_ID_CONTENT_CONFLICT:{obj.object_id}")
                self.conn.execute("COMMIT")
                return (
                    ExperienceRef(
                        object_id=obj.object_id,
                        object_type=old[0],
                        generation=int(old[1]),
                        shard=-1,
                        identity_hash=old[5],
                        payload_hash=old[6],
                        payload_path=old[7],
                        bytes_raw=int(old[8]),
                        bytes_stored=int(old[9]),
                    ),
                    False,
                )
            self.conn.execute(
                """
                INSERT INTO objects(
                    object_id,object_type,generation,policy_weight_hash,snapshot_hash,
                    lineage_hash,identity_hash,payload_hash,payload_path,
                    bytes_raw,bytes_stored,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    obj.object_id,
                    obj.object_type,
                    int(obj.generation),
                    obj.policy_weight_hash,
                    obj.snapshot_hash,
                    obj.lineage_hash,
                    obj.identity_hash,
                    obj.payload_hash,
                    str(path),
                    len(raw),
                    stored_bytes,
                    now,
                ),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

        return (
            ExperienceRef(
                object_id=obj.object_id,
                object_type=obj.object_type,
                generation=int(obj.generation),
                shard=-1,
                identity_hash=obj.identity_hash,
                payload_hash=obj.payload_hash,
                payload_path=str(path),
                bytes_raw=len(raw),
                bytes_stored=stored_bytes,
            ),
            True,
        )

    def get_row(self, object_id: str):
        return self.conn.execute(
            """
            SELECT object_id,object_type,generation,identity_hash,payload_hash,
                   payload_path,bytes_raw,bytes_stored
            FROM objects WHERE object_id=?
            """,
            (object_id,),
        ).fetchone()

    def iter_rows(
        self,
        *,
        generation: int | None = None,
        object_type: str | None = None,
    ):
        clauses = []
        args: list[Any] = []
        if generation is not None:
            clauses.append("generation=?")
            args.append(int(generation))
        if object_type is not None:
            clauses.append("object_type=?")
            args.append(object_type)
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        q = (
            "SELECT object_id,object_type,generation,identity_hash,payload_hash,"
            "payload_path,bytes_raw,bytes_stored FROM objects"
            + where
            + " ORDER BY generation,object_type,object_id"
        )
        yield from self.conn.execute(q, args)


class ShardedExperienceLake:
    def __init__(
        self,
        root: str | Path,
        *,
        shards: int = 4,
        busy_timeout_ms: int = 30_000,
        synchronous: str = "FULL",
    ):
        if shards <= 0:
            raise ValueError("shards must be positive")
        if synchronous.upper() not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError("invalid synchronous mode")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_count = int(shards)
        self._shards = [
            _Shard(
                self.root / "metadata" / f"experience_{i:02d}.sqlite",
                self.root / "objects" / f"shard_{i:02d}",
                busy_timeout_ms=busy_timeout_ms,
                synchronous=synchronous.upper(),
            )
            for i in range(shards)
        ]

    def close(self) -> None:
        for s in self._shards:
            s.close()

    def _index(self, object_id: str) -> int:
        return stable_shard(object_id, self.shard_count)

    def put(self, obj: ExperienceObject) -> tuple[ExperienceRef, bool]:
        idx = self._index(obj.object_id)
        ref, created = self._shards[idx].put(obj)
        return ExperienceRef(
            object_id=ref.object_id,
            object_type=ref.object_type,
            generation=ref.generation,
            shard=idx,
            identity_hash=ref.identity_hash,
            payload_hash=ref.payload_hash,
            payload_path=ref.payload_path,
            bytes_raw=ref.bytes_raw,
            bytes_stored=ref.bytes_stored,
        ), created

    def get(self, object_id: str, *, verify_payload: bool = True) -> tuple[ExperienceRef, dict[str, Any]] | None:
        idx = self._index(object_id)
        row = self._shards[idx].get_row(object_id)
        if row is None:
            return None
        ref = ExperienceRef(
            object_id=row[0],
            object_type=row[1],
            generation=int(row[2]),
            shard=idx,
            identity_hash=row[3],
            payload_hash=row[4],
            payload_path=row[5],
            bytes_raw=int(row[6]),
            bytes_stored=int(row[7]),
        )
        stored = Path(ref.payload_path).read_bytes()
        raw = zlib.decompress(stored)
        if verify_payload and sha256_bytes(raw) != ref.payload_hash:
            raise RuntimeError(f"EXPERIENCE_PAYLOAD_HASH_MISMATCH:{object_id}")
        return ref, json.loads(raw)

    def iter_refs(
        self,
        *,
        generation: int | None = None,
        object_type: str | None = None,
    ) -> Iterator[ExperienceRef]:
        all_rows = []
        for idx, shard in enumerate(self._shards):
            for row in shard.iter_rows(generation=generation, object_type=object_type):
                all_rows.append((int(row[2]), str(row[1]), str(row[0]), idx, row))
        all_rows.sort()
        for _, _, _, idx, row in all_rows:
            yield ExperienceRef(
                object_id=row[0],
                object_type=row[1],
                generation=int(row[2]),
                shard=idx,
                identity_hash=row[3],
                payload_hash=row[4],
                payload_path=row[5],
                bytes_raw=int(row[6]),
                bytes_stored=int(row[7]),
            )

    def count(self, *, generation: int | None = None, object_type: str | None = None) -> int:
        return sum(1 for _ in self.iter_refs(generation=generation, object_type=object_type))

    def seal_snapshot(
        self,
        *,
        snapshot_id: str,
        parent_generation: int,
        parent_policy_hash: str,
        refs: Iterable[ExperienceRef],
    ) -> ExperienceSnapshot:
        refs = sorted(refs, key=lambda r: (r.object_id, r.identity_hash))
        ids = tuple(r.object_id for r in refs)
        hashes = tuple(r.identity_hash for r in refs)
        if len(ids) != len(set(ids)):
            raise RuntimeError("SNAPSHOT_DUPLICATE_OBJECT_ID")
        snap = ExperienceSnapshot(
            snapshot_id=snapshot_id,
            parent_generation=int(parent_generation),
            parent_policy_hash=parent_policy_hash,
            object_ids=ids,
            object_identity_hashes=hashes,
            object_count=len(ids),
        )
        # Snapshot metadata itself is routed by snapshot_id.  Its identity is immutable.
        idx = self._index("SNAPSHOT:" + snapshot_id)
        payload = canonical_json_bytes(asdict(snap))
        conn = self._shards[idx].conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            old = conn.execute(
                "SELECT content_hash FROM snapshots WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
            if old is not None:
                if old[0] != snap.content_hash:
                    raise RuntimeError(f"SNAPSHOT_ID_CONTENT_CONFLICT:{snapshot_id}")
                conn.execute("COMMIT")
                return snap
            conn.execute(
                """
                INSERT INTO snapshots(
                    snapshot_id,content_hash,parent_generation,parent_policy_hash,
                    object_count,payload_json,created_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    snapshot_id,
                    snap.content_hash,
                    parent_generation,
                    parent_policy_hash,
                    snap.object_count,
                    payload,
                    time.time(),
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return snap

    def checkpoint_all(self, mode: str = "PASSIVE") -> list[tuple[int, int, int]]:
        return [s.checkpoint(mode) for s in self._shards]

    def audit(self, *, verify_payloads: bool = True) -> dict[str, Any]:
        count = 0
        raw = 0
        stored = 0
        corrupt = []
        for ref in self.iter_refs():
            count += 1
            raw += ref.bytes_raw
            stored += ref.bytes_stored
            if verify_payloads:
                try:
                    self.get(ref.object_id, verify_payload=True)
                except Exception as exc:
                    corrupt.append({"object_id": ref.object_id, "error": repr(exc)})
        return {
            "schema": "CB16_SHARDED_EXPERIENCE_LAKE_AUDIT_R3",
            "shards": self.shard_count,
            "objects": count,
            "bytes_raw": raw,
            "bytes_stored": stored,
            "compression_ratio": (stored / raw) if raw else 0.0,
            "corrupt": corrupt,
            "pass": not corrupt,
        }
