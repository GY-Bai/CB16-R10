"""S0-v2 immutable content-addressed observation store.

The store persists canonical ``PostCCObservationFactV1`` bytes on disk and
keeps a crash-safe logical-identity index.  Qualified replay truth is read from
this store; process-local dictionaries are never accepted as durable truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Mapping

from .post_cc_observation_contract_v1 import PostCCObservationFactV1
from .post_cc_observation_fact_v1 import (
    decode_observation_fact_v1,
    encode_observation_fact_v1,
    observation_content_sha256_v1,
    observation_logical_id_from_identity_v1,
    observation_logical_id_v1,
)


class ObservationStoreError(RuntimeError):
    """Base class for storage failures that must stay fail-closed."""


class ObservationSemanticConflict(ObservationStoreError):
    """The same logical identity was bound to different immutable content."""


class ObservationStoreCorruption(ObservationStoreError):
    """A stored object or index row failed content verification."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class ImmutableContentStore:
    """Namespaced logical-id -> immutable content-addressed bytes.

    Logical identities are explicit; their immutable binding is enforced by the
    index.  The same content can be referenced by multiple logical identities.
    """

    def __init__(self, root: str | Path, namespace: str):
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be non-empty text")
        if "/" in namespace or "\\" in namespace:
            raise ValueError("namespace must be a single path component")
        self.root = Path(root).resolve() / namespace
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.sqlite3"
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=30.0)

    def _ensure_schema(self) -> None:
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS entries ("
                "logical_id TEXT PRIMARY KEY, "
                "content_sha256 TEXT NOT NULL, "
                "byte_size INTEGER NOT NULL"
                ")"
            )
            db.commit()

    def _object_path(self, digest: str) -> Path:
        return self.objects / digest[:2] / f"{digest}.bin"

    def has(self, logical_id: str) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT 1 FROM entries WHERE logical_id=?", (logical_id,)).fetchone()
        return row is not None

    def content_sha256_for(self, logical_id: str) -> str:
        with self._connect() as db:
            row = db.execute("SELECT content_sha256 FROM entries WHERE logical_id=?", (logical_id,)).fetchone()
        if row is None:
            raise KeyError(logical_id)
        return str(row[0])

    def _read_and_verify(self, path: Path, expected_sha256: str, expected_size: int | None = None) -> bytes:
        if not path.is_file():
            raise ObservationStoreCorruption(f"CONTENT_OBJECT_MISSING:{path.name}")
        payload = path.read_bytes()
        if expected_size is not None and len(payload) != int(expected_size):
            raise ObservationStoreCorruption("CONTENT_OBJECT_SIZE_MISMATCH")
        digest = _sha256_bytes(payload)
        if digest != expected_sha256:
            raise ObservationStoreCorruption("CONTENT_OBJECT_HASH_MISMATCH")
        return payload

    def put_bytes(self, logical_id: str, payload: bytes, *, expected_sha256: str | None = None) -> str:
        if not isinstance(logical_id, str) or not logical_id:
            raise ValueError("logical_id must be non-empty text")
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes")
        raw = bytes(payload)
        digest = _sha256_bytes(raw)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ObservationStoreCorruption("SUPPLIED_CONTENT_HASH_MISMATCH")

        with self._connect() as db:
            row = db.execute(
                "SELECT content_sha256, byte_size FROM entries WHERE logical_id=?", (logical_id,)
            ).fetchone()
        if row is not None:
            stored_digest, stored_size = str(row[0]), int(row[1])
            if stored_digest != digest:
                raise ObservationSemanticConflict(
                    f"LOGICAL_ID_ALREADY_BOUND:{logical_id}"
                )
            self._read_and_verify(self._object_path(digest), stored_digest, stored_size)
            return stored_digest

        path = self._object_path(digest)
        if path.exists():
            self._read_and_verify(path, digest)
        else:
            _atomic_write_bytes(path, raw)
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO entries(logical_id, content_sha256, byte_size) VALUES(?,?,?)",
                    (logical_id, digest, len(raw)),
                )
                db.commit()
        except sqlite3.IntegrityError as exc:
            with self._connect() as db:
                row = db.execute(
                    "SELECT content_sha256 FROM entries WHERE logical_id=?", (logical_id,)
                ).fetchone()
            if row is None or str(row[0]) != digest:
                raise ObservationSemanticConflict(
                    f"CONCURRENT_LOGICAL_ID_REBOUND:{logical_id}"
                ) from exc
        return digest

    def get_bytes(self, logical_id: str, *, expected_sha256: str | None = None) -> bytes:
        with self._connect() as db:
            row = db.execute(
                "SELECT content_sha256, byte_size FROM entries WHERE logical_id=?", (logical_id,)
            ).fetchone()
        if row is None:
            raise KeyError(logical_id)
        digest, size = str(row[0]), int(row[1])
        if expected_sha256 is not None and digest != expected_sha256:
            raise ObservationStoreCorruption("CONTENT_IDENTITY_MISMATCH")
        return self._read_and_verify(self._object_path(digest), digest, size)

    def verify_all(self) -> int:
        with self._connect() as db:
            rows = db.execute("SELECT logical_id, content_sha256, byte_size FROM entries").fetchall()
        for logical_id, digest, size in rows:
            self._read_and_verify(self._object_path(str(digest)), str(digest), int(size))
        return len(rows)

    def count(self) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM entries").fetchone()[0])

    def delete_for_test_only(self, logical_id: str) -> None:
        """Test helper; production callers must never mutate committed content."""
        with self._connect() as db:
            row = db.execute(
                "SELECT content_sha256 FROM entries WHERE logical_id=?", (logical_id,)
            ).fetchone()
            if row is None:
                raise KeyError(logical_id)
            db.execute("DELETE FROM entries WHERE logical_id=?", (logical_id,))
            db.commit()
        path = self._object_path(str(row[0]))
        if path.exists():
            path.unlink()


@dataclass(frozen=True)
class ObservationStoreReceiptV1:
    logical_id: str
    observation_hash: str
    content_sha256: str
    byte_size: int
    account_lineage_id: str
    decision_index: int
    environment_time: str

    def validate(self) -> "ObservationStoreReceiptV1":
        for name in ("logical_id", "observation_hash", "content_sha256", "account_lineage_id", "environment_time"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if len(self.observation_hash) != 64 or len(self.content_sha256) != 64:
            raise ValueError("receipt hashes must be 64-hex text")
        if isinstance(self.decision_index, bool) or self.decision_index < 0:
            raise ValueError("decision_index must be >= 0")
        if self.byte_size <= 0:
            raise ValueError("byte_size must be positive")
        return self


class ObservationStoreV1:
    """Durable immutable store for canonical observation facts."""

    NAMESPACE = "observations"

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._content = ImmutableContentStore(self.root, self.NAMESPACE)

    def _receipt_from_fact(self, fact: PostCCObservationFactV1, digest: str) -> ObservationStoreReceiptV1:
        return ObservationStoreReceiptV1(
            logical_id=observation_logical_id_v1(fact),
            observation_hash=fact.observation_hash,
            content_sha256=digest,
            byte_size=len(encode_observation_fact_v1(fact)),
            account_lineage_id=fact.account_lineage_id,
            decision_index=int(fact.decision_index),
            environment_time=fact.environment_time,
        ).validate()

    def put(self, fact: PostCCObservationFactV1) -> ObservationStoreReceiptV1:
        if not isinstance(fact, PostCCObservationFactV1):
            raise TypeError("fact must be a PostCCObservationFactV1")
        fact.validate()
        logical_id = observation_logical_id_v1(fact)
        payload = encode_observation_fact_v1(fact)
        digest = self._content.put_bytes(logical_id, payload)
        stored = self._content.get_bytes(logical_id, expected_sha256=digest)
        if stored != payload:
            raise ObservationStoreCorruption("OBSERVATION_ROUNDTRIP_MISMATCH")
        decoded = decode_observation_fact_v1(stored)
        if decoded != fact:
            raise ObservationStoreCorruption("OBSERVATION_DECODE_MISMATCH")
        return self._receipt_from_fact(fact, digest)

    def get(self, logical_id: str) -> PostCCObservationFactV1:
        payload = self._content.get_bytes(logical_id)
        fact = decode_observation_fact_v1(payload)
        if observation_logical_id_v1(fact) != logical_id:
            raise ObservationStoreCorruption("OBSERVATION_LOGICAL_ID_MISMATCH")
        if observation_content_sha256_v1(fact) != _sha256_bytes(payload):
            raise ObservationStoreCorruption("OBSERVATION_CONTENT_HASH_MISMATCH")
        return fact

    def get_receipt(self, logical_id: str) -> ObservationStoreReceiptV1:
        fact = self.get(logical_id)
        digest = self._content.content_sha256_for(logical_id)
        return self._receipt_from_fact(fact, digest)

    def get_by_identity(
        self, *, account_lineage_id: str, decision_index: int, environment_time: str, observation_hash: str
    ) -> PostCCObservationFactV1:
        logical_id = observation_logical_id_from_identity_v1(
            account_lineage_id=account_lineage_id,
            decision_index=int(decision_index),
            environment_time=environment_time,
            observation_hash=observation_hash,
        )
        return self.get(logical_id)

    def count(self) -> int:
        return self._content.count()

    def verify_all(self) -> int:
        return self._content.verify_all()

    def close(self) -> None:
        """SQLite connections are per-call; explicit close is kept for callers."""
        return None
