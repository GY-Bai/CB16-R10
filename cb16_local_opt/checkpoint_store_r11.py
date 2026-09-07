from __future__ import annotations

"""R11 content-addressed checkpoint objects.

Object identity is the semantic tensor mapping (name, dtype, shape, exact tensor bytes),
not Python pickle serialization bytes.  ``torch.save`` is only a transport container.
Generation metadata is small and references immutable checkpoint objects by semantic hash.
"""

import hashlib
import json
import os
import sqlite3
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(obj) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(obj, tmp)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    finally:
        tmp.unlink(missing_ok=True)


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    t = tensor.detach().cpu().contiguous()
    if t.layout != torch.strided:
        raise RuntimeError(f"R11_CHECKPOINT_UNSUPPORTED_TENSOR_LAYOUT:{t.layout}")
    # uint8 view avoids NumPy dtype gaps such as bfloat16 while preserving exact storage bits.
    return t.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def tensor_mapping_semantic_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    h.update(b"CB16_R11_TENSOR_MAPPING_SEMANTIC_V1\0")
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"R11_CHECKPOINT_NON_TENSOR_VALUE:{name}:{type(tensor)!r}")
        name_b = str(name).encode("utf-8")
        dtype_b = str(tensor.dtype).encode("ascii")
        raw = _tensor_raw_bytes(tensor)
        h.update(struct.pack(">I", len(name_b)))
        h.update(name_b)
        h.update(struct.pack(">I", len(dtype_b)))
        h.update(dtype_b)
        h.update(struct.pack(">I", tensor.ndim))
        for dim in tensor.shape:
            h.update(struct.pack(">q", int(dim)))
        h.update(struct.pack(">Q", len(raw)))
        h.update(raw)
    return h.hexdigest()


def _canonical_cpu_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"R11_CHECKPOINT_NON_TENSOR_VALUE:{name}:{type(tensor)!r}")
        if tensor.layout != torch.strided:
            raise RuntimeError(f"R11_CHECKPOINT_UNSUPPORTED_TENSOR_LAYOUT:{tensor.layout}")
        out[str(name)] = tensor.detach().cpu().contiguous().clone()
    return out


@dataclass(frozen=True)
class CheckpointObjectRefR11:
    semantic_sha256: str
    object_path: str
    metadata_path: str
    tensor_count: int
    tensor_bytes: int
    created: bool


@dataclass(frozen=True)
class GenerationCheckpointRefR11:
    generation: int
    parent_champion: str
    challenger: str
    decision: str
    champion_after: str
    trace_batch_id: str
    trace_batch_hash: str
    snapshot_hash: str
    snapshot_path: str


class CheckpointStoreR11:
    def __init__(self, root: str | Path, *, synchronous: str = "FULL"):
        self.root = Path(root).resolve()
        self.object_root = self.root / "objects"
        self.object_metadata_root = self.root / "object_metadata"
        self.snapshot_root = self.root / "generation_snapshots"
        self.generation_root = self.root / "generations"
        for path in (self.object_root, self.object_metadata_root, self.snapshot_root, self.generation_root):
            path.mkdir(parents=True, exist_ok=True)
        sync = synchronous.upper()
        if sync not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError("invalid synchronous mode")
        self.conn = sqlite3.connect(self.root / "r11_checkpoints.sqlite", isolation_level=None, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(f"PRAGMA synchronous={sync}")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS checkpoint_objects(
          semantic_sha256 TEXT PRIMARY KEY,object_path TEXT NOT NULL,metadata_path TEXT NOT NULL,
          tensor_count INTEGER NOT NULL,tensor_bytes INTEGER NOT NULL,created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS generation_checkpoints(
          generation INTEGER PRIMARY KEY,parent_champion TEXT NOT NULL,challenger TEXT NOT NULL,
          decision TEXT NOT NULL,champion_after TEXT NOT NULL,trace_batch_id TEXT NOT NULL,
          trace_batch_hash TEXT NOT NULL,snapshot_hash TEXT NOT NULL,snapshot_path TEXT NOT NULL,
          sealed_at REAL NOT NULL);
        """)
        try:
            self.fast_startup_validation()
        except BaseException:
            self.conn.close()
            raise

    def close(self) -> None:
        self.conn.close()

    def checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        mode = mode.upper()
        if mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError("invalid checkpoint mode")
        return tuple(int(x) for x in self.conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone())

    def _paths(self, semantic_hash: str) -> tuple[Path, Path]:
        prefix = semantic_hash[:2]
        obj_dir = self.object_root / prefix
        meta_dir = self.object_metadata_root / prefix
        obj_dir.mkdir(exist_ok=True)
        meta_dir.mkdir(exist_ok=True)
        return obj_dir / f"{semantic_hash}.pt", meta_dir / f"{semantic_hash}.json"

    def has_object(self, semantic_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT object_path,metadata_path FROM checkpoint_objects WHERE semantic_sha256=?",
            (semantic_hash,),
        ).fetchone()
        return bool(row is not None and Path(row[0]).is_file() and Path(row[1]).is_file())

    def put_state_dict(self, state: Mapping[str, torch.Tensor]) -> CheckpointObjectRefR11:
        canonical_state = _canonical_cpu_state(state)
        semantic_hash = tensor_mapping_semantic_sha256(canonical_state)
        tensor_bytes = sum(len(_tensor_raw_bytes(t)) for t in canonical_state.values())
        tensor_count = len(canonical_state)
        row = self.conn.execute(
            "SELECT object_path,metadata_path,tensor_count,tensor_bytes FROM checkpoint_objects "
            "WHERE semantic_sha256=?", (semantic_hash,)
        ).fetchone()
        if row is not None:
            object_path, metadata_path = Path(row[0]), Path(row[1])
            if not object_path.is_file() or not metadata_path.is_file():
                raise RuntimeError(f"R11_CHECKPOINT_OBJECT_MISSING:{semantic_hash}")
            return CheckpointObjectRefR11(
                semantic_hash, str(object_path), str(metadata_path), int(row[2]), int(row[3]), False
            )

        object_path, metadata_path = self._paths(semantic_hash)
        if object_path.exists() or metadata_path.exists():
            # An unindexed object is not silently trusted/adopted during normal writes.
            raise RuntimeError(f"R11_UNINDEXED_CHECKPOINT_OBJECT_PRESENT:{semantic_hash}")
        payload = {
            "schema": "CB16_R11_CHECKPOINT_TENSOR_OBJECT_V1",
            "semantic_sha256": semantic_hash,
            "state_dict": canonical_state,
        }
        _atomic_torch_save(payload, object_path)
        metadata = {
            "schema": "CB16_R11_CHECKPOINT_OBJECT_METADATA_V1",
            "semantic_sha256": semantic_hash,
            "object_file": object_path.name,
            "object_path": str(object_path),
            "tensor_count": tensor_count,
            "tensor_bytes": tensor_bytes,
            "identity_basis": "TENSOR_NAME_DTYPE_SHAPE_EXACT_CONTENT",
            "pickle_byte_identity_is_authority": False,
        }
        _atomic_json(metadata_path, metadata)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            old = self.conn.execute(
                "SELECT 1 FROM checkpoint_objects WHERE semantic_sha256=?", (semantic_hash,)
            ).fetchone()
            if old is not None:
                raise RuntimeError(f"R11_CHECKPOINT_OBJECT_RACE:{semantic_hash}")
            self.conn.execute(
                "INSERT INTO checkpoint_objects VALUES(?,?,?,?,?,?)",
                (semantic_hash, str(object_path), str(metadata_path), tensor_count, tensor_bytes, time.time()),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        return CheckpointObjectRefR11(
            semantic_hash, str(object_path), str(metadata_path), tensor_count, tensor_bytes, True
        )

    def load_state_dict(self, semantic_hash: str) -> dict[str, torch.Tensor]:
        row = self.conn.execute(
            "SELECT object_path FROM checkpoint_objects WHERE semantic_sha256=?", (semantic_hash,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"R11_CHECKPOINT_OBJECT_NOT_INDEXED:{semantic_hash}")
        path = Path(row[0])
        if not path.is_file():
            raise RuntimeError(f"R11_CHECKPOINT_OBJECT_MISSING:{semantic_hash}")
        payload = _load_torch(path)
        if not isinstance(payload, Mapping) or payload.get("schema") != "CB16_R11_CHECKPOINT_TENSOR_OBJECT_V1":
            raise RuntimeError(f"R11_CHECKPOINT_OBJECT_SCHEMA_INVALID:{semantic_hash}")
        state = payload.get("state_dict")
        if not isinstance(state, Mapping):
            raise RuntimeError(f"R11_CHECKPOINT_STATE_DICT_MISSING:{semantic_hash}")
        actual = tensor_mapping_semantic_sha256(state)
        if actual != semantic_hash or payload.get("semantic_sha256") != semantic_hash:
            raise RuntimeError(f"R11_CHECKPOINT_SEMANTIC_HASH_MISMATCH:{semantic_hash}:{actual}")
        return _canonical_cpu_state(state)

    def seal_generation_checkpoint(
        self,
        *,
        generation: int,
        parent_champion: str,
        challenger: str,
        decision: str,
        champion_after: str,
        trace_batch_id: str,
        trace_batch_hash: str,
        evidence_set_hash: str | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> GenerationCheckpointRefR11:
        generation = int(generation)
        decision = str(decision).upper()
        if decision not in {"PROMOTE", "REJECT"}:
            raise RuntimeError(f"R11_INVALID_PROMOTION_DECISION:{decision}")
        expected_after = challenger if decision == "PROMOTE" else parent_champion
        if champion_after != expected_after:
            raise RuntimeError("R11_CHECKPOINT_CHAMPION_AFTER_BINDING_INVALID")
        for role, semantic_hash in (
            ("parent_champion", parent_champion),
            ("challenger", challenger),
            ("champion_after", champion_after),
        ):
            if not self.has_object(semantic_hash):
                raise RuntimeError(f"R11_{role.upper()}_CHECKPOINT_MISSING:{semantic_hash}")

        last = self.conn.execute(
            "SELECT generation,champion_after FROM generation_checkpoints ORDER BY generation DESC LIMIT 1"
        ).fetchone()
        existing = self.conn.execute(
            "SELECT snapshot_hash,snapshot_path FROM generation_checkpoints WHERE generation=?",
            (generation,),
        ).fetchone()
        identity = {
            "schema": "CB16_R11_GENERATION_CHECKPOINT_IDENTITY_V1",
            "generation": generation,
            "parent_champion": parent_champion,
            "challenger": challenger,
            "decision": decision,
            "champion_after": champion_after,
            "trace_batch_id": str(trace_batch_id),
            "trace_batch_hash": str(trace_batch_hash),
            "evidence_set_hash": evidence_set_hash,
        }
        # Runtime placement/scheduling annotations are carried beside, not inside, identity.
        body = {
            "schema": "CB16_R11_GENERATION_CHECKPOINT_V1",
            "identity": identity,
            "runtime_metadata": dict(extra_metadata or {}),
        }
        snapshot_hash = sha256_obj(identity)
        snapshot_path = self.snapshot_root / f"{snapshot_hash}.json"
        if existing is not None:
            if str(existing[0]) != snapshot_hash:
                raise RuntimeError(f"R11_GENERATION_CHECKPOINT_CONFLICT:{generation}")
            return GenerationCheckpointRefR11(
                generation, parent_champion, challenger, decision, champion_after,
                str(trace_batch_id), str(trace_batch_hash), snapshot_hash, str(existing[1])
            )
        if last is not None:
            last_generation, last_champion = int(last[0]), str(last[1])
            if generation != last_generation + 1:
                raise RuntimeError(f"R11_CHECKPOINT_GENERATION_SEQUENCE_GAP:{last_generation}:{generation}")
            if parent_champion != last_champion:
                raise RuntimeError(
                    f"R11_CHECKPOINT_STALE_OR_REJECTED_PARENT_FORBIDDEN:"
                    f"expected={last_champion}:got={parent_champion}"
                )
        if not snapshot_path.exists():
            _atomic_json(snapshot_path, body)
        pointer = {
            "schema": "CB16_R11_GENERATION_CHECKPOINT_POINTER_V1",
            "generation": generation,
            "snapshot_hash": snapshot_hash,
            "snapshot_path": str(snapshot_path),
        }
        pointer_path = self.generation_root / f"G{generation:08d}.json"
        if pointer_path.exists():
            old_pointer = json.loads(pointer_path.read_text())
            if old_pointer != pointer:
                raise RuntimeError(f"R11_GENERATION_POINTER_CONFLICT:{generation}")
        else:
            _atomic_json(pointer_path, pointer)
        self.conn.execute(
            "INSERT INTO generation_checkpoints VALUES(?,?,?,?,?,?,?,?,?,?)",
            (generation, parent_champion, challenger, decision, champion_after,
             str(trace_batch_id), str(trace_batch_hash), snapshot_hash, str(snapshot_path), time.time()),
        )
        return GenerationCheckpointRefR11(
            generation, parent_champion, challenger, decision, champion_after,
            str(trace_batch_id), str(trace_batch_hash), snapshot_hash, str(snapshot_path)
        )

    def fast_startup_validation(self) -> dict[str, Any]:
        """Validate checkpoint references with stat/metadata only; do not load tensor objects."""
        problems: list[dict[str, Any]] = []
        objects = self.conn.execute(
            "SELECT semantic_sha256,object_path,metadata_path,tensor_count,tensor_bytes FROM checkpoint_objects"
        ).fetchall()
        for semantic_hash, object_path, metadata_path, tensor_count, tensor_bytes in objects:
            op, mp = Path(object_path), Path(metadata_path)
            if not op.is_file():
                problems.append({"semantic_sha256": semantic_hash, "error": "OBJECT_MISSING"})
                continue
            if not mp.is_file():
                problems.append({"semantic_sha256": semantic_hash, "error": "OBJECT_METADATA_MISSING"})
                continue
            try:
                meta = json.loads(mp.read_text())
                if meta.get("semantic_sha256") != semantic_hash:
                    raise RuntimeError("METADATA_SEMANTIC_HASH_MISMATCH")
                if int(meta.get("tensor_count", -1)) != int(tensor_count):
                    raise RuntimeError("METADATA_TENSOR_COUNT_MISMATCH")
                if int(meta.get("tensor_bytes", -1)) != int(tensor_bytes):
                    raise RuntimeError("METADATA_TENSOR_BYTES_MISMATCH")
            except Exception as exc:
                problems.append({"semantic_sha256": semantic_hash, "error": repr(exc)})
        for generation, parent, challenger, champion_after, snapshot_hash, snapshot_path in self.conn.execute(
            "SELECT generation,parent_champion,challenger,champion_after,snapshot_hash,snapshot_path "
            "FROM generation_checkpoints ORDER BY generation"
        ):
            for role, semantic_hash in (("parent", parent), ("challenger", challenger), ("champion_after", champion_after)):
                if not self.has_object(str(semantic_hash)):
                    problems.append({"generation": int(generation), "error": f"{role.upper()}_OBJECT_MISSING"})
            sp = Path(snapshot_path)
            if not sp.is_file():
                problems.append({"generation": int(generation), "error": "SNAPSHOT_MISSING"})
            else:
                try:
                    body = json.loads(sp.read_text())
                    identity = body.get("identity")
                    if not isinstance(identity, Mapping) or sha256_obj(identity) != str(snapshot_hash):
                        raise RuntimeError("SNAPSHOT_HASH_MISMATCH")
                except Exception as exc:
                    problems.append({"generation": int(generation), "error": repr(exc)})
        if problems:
            raise RuntimeError(f"R11_CHECKPOINT_FAST_VALIDATION_FAILED:{problems}")
        return {
            "schema": "CB16_R11_CHECKPOINT_FAST_STARTUP_VALIDATION_V1",
            "objects": len(objects),
            "tensor_objects_loaded": 0,
            "generation_snapshots": int(self.conn.execute("SELECT COUNT(*) FROM generation_checkpoints").fetchone()[0]),
            "pass": True,
        }

    def full_forensic_audit(self) -> dict[str, Any]:
        problems: list[dict[str, Any]] = []
        checked = 0
        for (semantic_hash,) in self.conn.execute(
            "SELECT semantic_sha256 FROM checkpoint_objects ORDER BY semantic_sha256"
        ):
            try:
                self.load_state_dict(str(semantic_hash))
                checked += 1
            except Exception as exc:
                problems.append({"semantic_sha256": str(semantic_hash), "error": repr(exc)})
        try:
            self.fast_startup_validation()
        except Exception as exc:
            problems.append({"scope": "metadata", "error": repr(exc)})
        return {
            "schema": "CB16_R11_CHECKPOINT_FULL_FORENSIC_AUDIT_V1",
            "objects_checked": checked,
            "problems": problems,
            "pass": not problems,
        }
