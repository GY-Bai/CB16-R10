"""S0-v2 durable learning-update provenance and exactly-once transaction store.

The transaction is a three-phase durable journal:

``PREPARED`` -> update computed, no child authority;
``STAGED``   -> child checkpoint bytes are durable, authority not committed;
``COMMITTED`` -> the single logical update is durably authoritative.

Re-running a committed update only re-loads the already-staged child; gradients
are never applied a second time.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .cc_experience_wire_r0 import canonical_json_bytes

HEX64 = re.compile(r"^[0-9a-f]{64}$")
UPDATE_SCHEMA_VERSION = "CB16_R11_S0V2_DURABLE_LEARNING_UPDATE_V1"
STATUS_PREPARED = "PREPARED"
STATUS_STAGED = "STAGED"
STATUS_COMMITTED = "COMMITTED"
STATUSES = (STATUS_PREPARED, STATUS_STAGED, STATUS_COMMITTED)


class DurableUpdateError(RuntimeError):
    pass


class DurableUpdateCorruptionError(DurableUpdateError):
    pass


class DurableUpdateSemanticConflict(DurableUpdateError):
    pass


def _require_nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_hex64(name: str, value: Any) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64-hex text")
    return value


def _require_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> bytes:
    encoded = canonical_json_bytes(dict(payload))
    _atomic_write_bytes(path, encoded)
    return encoded


@contextmanager
def _journal_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl  # type: ignore
    except ImportError:  # pragma: no cover - Windows fallback
        yield
        return
    with open(path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def encode_tensor_state_v1(state_dict: Mapping[str, Tensor]) -> Mapping[str, Any]:
    rows: dict[str, Any] = {}
    for name, tensor in sorted(state_dict.items()):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"state_dict entry {name} is not a tensor")
        values = tensor.detach().cpu().reshape(-1).tolist()
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"NONFINITE_CHECKPOINT_TENSOR:{name}")
        rows[name] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "values": [float(value) for value in values],
        }
    return rows


def decode_tensor_state_v1(rows: Mapping[str, Any]) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int64": torch.int64,
        "int32": torch.int32,
        "bool": torch.bool,
    }
    for name, row in sorted(rows.items()):
        if not isinstance(row, Mapping):
            raise ValueError(f"CHECKPOINT_TENSOR_ROW_INVALID:{name}")
        dtype_name = str(row.get("dtype"))
        if dtype_name not in dtype_map:
            raise ValueError(f"CHECKPOINT_DTYPE_UNSUPPORTED:{name}")
        shape = tuple(int(value) for value in row.get("shape", ()))
        values = [torch.tensor(value, dtype=dtype_map[dtype_name]) for value in row.get("values", ())]
        if not values:
            tensor = torch.empty(shape, dtype=dtype_map[dtype_name])
        else:
            tensor = torch.stack(values).reshape(shape).to(dtype_map[dtype_name])
        state[name] = tensor
    return state


def checkpoint_bundle_bytes_v1(
    *,
    actor: nn.Module,
    critic: nn.Module,
    optimizer_step: int,
    actor_lr: float,
    critic_lr: float,
    update_id: str,
    parent_checkpoint_sha256: str,
) -> bytes:
    payload = {
        "schema": "CB16_R11_S0V2_CHILD_CHECKPOINT_V1",
        "update_id": _require_nonempty("update_id", update_id),
        "parent_checkpoint_sha256": _require_hex64("parent_checkpoint_sha256", parent_checkpoint_sha256),
        "optimizer_step": int(optimizer_step),
        "actor_lr": _require_finite("actor_lr", actor_lr),
        "critic_lr": _require_finite("critic_lr", critic_lr),
        "actor_state": encode_tensor_state_v1(actor.state_dict()),
        "critic_state": encode_tensor_state_v1(critic.state_dict()),
    }
    return canonical_json_bytes(payload)


def checkpoint_sha256_v1(payload: bytes) -> str:
    return hashlib.sha256(bytes(payload)).hexdigest()


def decode_checkpoint_bundle_v1(payload: bytes) -> Mapping[str, Any]:
    try:
        raw = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DurableUpdateCorruptionError("CHECKPOINT_BYTES_INVALID") from exc
    if not isinstance(raw, dict):
        raise DurableUpdateCorruptionError("CHECKPOINT_MUST_BE_OBJECT")
    if raw.get("schema") != "CB16_R11_S0V2_CHILD_CHECKPOINT_V1":
        raise DurableUpdateCorruptionError("CHECKPOINT_SCHEMA_MISMATCH")
    for name in ("update_id", "parent_checkpoint_sha256"):
        _require_nonempty(name, raw.get(name))
    _require_hex64("parent_checkpoint_sha256", raw["parent_checkpoint_sha256"])
    if not isinstance(raw.get("optimizer_step"), int) or int(raw["optimizer_step"]) < 0:
        raise DurableUpdateCorruptionError("CHECKPOINT_STEP_INVALID")
    actor_state = decode_tensor_state_v1(raw.get("actor_state", {}))
    critic_state = decode_tensor_state_v1(raw.get("critic_state", {}))
    if not actor_state or not critic_state:
        raise DurableUpdateCorruptionError("CHECKPOINT_STATE_EMPTY")
    raw["actor_state_tensors"] = actor_state
    raw["critic_state_tensors"] = critic_state
    return raw


@dataclass(frozen=True)
class DurableLearningUpdateV1:
    update_id: str
    schema: str
    science_semantic_version: str
    parent_checkpoint_sha256: str
    parent_policy_identity: str
    target_policy_identity: str
    materialization_manifest_sha256: str
    batch_content_sha256: str
    sampled_sequence_ids: tuple[str, ...]
    materialized_sample_hashes: tuple[str, ...]
    sampling_probabilities_or_weights: tuple[float, ...]
    behavior_policy_identities: tuple[str, ...]
    optimizer_step_before: int
    commit_status: str
    child_checkpoint_sha256: str | None = None
    actor_loss: float | None = None
    critic_loss: float | None = None
    vtrace_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    gradient_ownership_summary: Mapping[str, Any] = field(default_factory=dict)
    optimizer_step_after: int | None = None
    committed_receipt_sha256: str | None = None

    def validate(self) -> "DurableLearningUpdateV1":
        if self.schema != UPDATE_SCHEMA_VERSION:
            raise ValueError("UPDATE_SCHEMA_MISMATCH")
        _require_nonempty("update_id", self.update_id)
        for name in (
            "science_semantic_version",
            "parent_policy_identity",
            "target_policy_identity",
            "materialization_manifest_sha256",
            "batch_content_sha256",
        ):
            _require_nonempty(name, getattr(self, name))
        _require_hex64("parent_checkpoint_sha256", self.parent_checkpoint_sha256)
        _require_hex64("materialization_manifest_sha256", self.materialization_manifest_sha256)
        _require_hex64("batch_content_sha256", self.batch_content_sha256)
        if not self.sampled_sequence_ids:
            raise ValueError("sampled_sequence_ids required")
        if len(self.sampled_sequence_ids) != len(self.sampling_probabilities_or_weights):
            raise ValueError("sampling cardinality mismatch")
        if len(self.materialized_sample_hashes) == 0:
            raise ValueError("materialized_sample_hashes required")
        for item in self.materialized_sample_hashes:
            _require_hex64("materialized_sample_hashes[]", item)
        for weight in self.sampling_probabilities_or_weights:
            value = _require_finite("sampling_probability_or_weight", weight)
            if not 0.0 < value <= 1.0:
                raise ValueError("sampling weight outside (0,1]")
        if not self.behavior_policy_identities:
            raise ValueError("behavior_policy_identities required")
        if isinstance(self.optimizer_step_before, bool) or int(self.optimizer_step_before) < 0:
            raise ValueError("optimizer_step_before must be >= 0")
        if self.commit_status not in STATUSES:
            raise ValueError("commit_status invalid")
        if self.commit_status == STATUS_PREPARED:
            if any(
                value is not None
                for value in (self.child_checkpoint_sha256, self.actor_loss, self.critic_loss, self.optimizer_step_after)
            ):
                raise ValueError("PREPARED must not carry staged authority")
        else:
            _require_hex64("child_checkpoint_sha256", self.child_checkpoint_sha256)
            _require_finite("actor_loss", self.actor_loss)
            _require_finite("critic_loss", self.critic_loss)
            if self.optimizer_step_after is None or int(self.optimizer_step_after) < int(self.optimizer_step_before):
                raise ValueError("optimizer_step_after invalid")
        if self.committed_receipt_sha256 is not None:
            _require_hex64("committed_receipt_sha256", self.committed_receipt_sha256)
        return self

    @property
    def record_sha256(self) -> str:
        self.validate()
        return hashlib.sha256(canonical_json_bytes(asdict(self))).hexdigest()

    def request_fields(self) -> Mapping[str, Any]:
        return {
            "science_semantic_version": self.science_semantic_version,
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "parent_policy_identity": self.parent_policy_identity,
            "target_policy_identity": self.target_policy_identity,
            "materialization_manifest_sha256": self.materialization_manifest_sha256,
            "batch_content_sha256": self.batch_content_sha256,
            "sampled_sequence_ids": self.sampled_sequence_ids,
            "materialized_sample_hashes": self.materialized_sample_hashes,
            "sampling_probabilities_or_weights": self.sampling_probabilities_or_weights,
            "behavior_policy_identities": self.behavior_policy_identities,
            "optimizer_step_before": int(self.optimizer_step_before),
        }


class DurableUpdateStoreV1:
    """Filesystem-backed three-phase exactly-once update journal."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.updates_dir = self.root / "updates"
        self.checkpoints_dir = self.root / "checkpoints"
        self.lock_path = self.root / ".update.lock"
        self.updates_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _record_path(self, update_id: str) -> Path:
        return self.updates_dir / f"{update_id}.json"

    def _checkpoint_path(self, child_checkpoint_sha256: str) -> Path:
        return self.checkpoints_dir / f"{child_checkpoint_sha256}.json"

    def _load_record(self, update_id: str) -> DurableLearningUpdateV1 | None:
        path = self._record_path(update_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DurableUpdateCorruptionError("UPDATE_RECORD_BYTES_INVALID") from exc
        for name in ("sampled_sequence_ids", "materialized_sample_hashes", "sampling_probabilities_or_weights", "behavior_policy_identities"):
            raw[name] = tuple(raw.get(name, ()))
        record = DurableLearningUpdateV1(**raw)
        record.validate()
        return record

    def _write_record(self, record: DurableLearningUpdateV1) -> None:
        record.validate()
        _atomic_write_json(self._record_path(record.update_id), asdict(record))

    def get_record(self, update_id: str) -> DurableLearningUpdateV1 | None:
        return self._load_record(update_id)

    def begin(
        self,
        *,
        update_id: str,
        science_semantic_version: str,
        parent_checkpoint_sha256: str,
        parent_policy_identity: str,
        target_policy_identity: str,
        materialization_manifest_sha256: str,
        batch_content_sha256: str,
        sampled_sequence_ids: Sequence[str],
        materialized_sample_hashes: Sequence[str],
        sampling_probabilities_or_weights: Sequence[float],
        behavior_policy_identities: Sequence[str],
        optimizer_step_before: int,
    ) -> DurableLearningUpdateV1:
        record = DurableLearningUpdateV1(
            update_id=update_id,
            schema=UPDATE_SCHEMA_VERSION,
            science_semantic_version=science_semantic_version,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            parent_policy_identity=parent_policy_identity,
            target_policy_identity=target_policy_identity,
            materialization_manifest_sha256=materialization_manifest_sha256,
            batch_content_sha256=batch_content_sha256,
            sampled_sequence_ids=tuple(sampled_sequence_ids),
            materialized_sample_hashes=tuple(materialized_sample_hashes),
            sampling_probabilities_or_weights=tuple(float(x) for x in sampling_probabilities_or_weights),
            behavior_policy_identities=tuple(behavior_policy_identities),
            optimizer_step_before=int(optimizer_step_before),
            commit_status=STATUS_PREPARED,
        ).validate()
        with _journal_lock(self.lock_path):
            existing = self._load_record(update_id)
            if existing is not None:
                if existing.request_fields() != record.request_fields():
                    raise DurableUpdateSemanticConflict("UPDATE_ID_REBOUND")
                return existing
            self._write_record(record)
            return record

    def stage(
        self,
        update_id: str,
        *,
        child_checkpoint_bytes: bytes,
        actor_loss: float,
        critic_loss: float,
        vtrace_diagnostics: Mapping[str, Any],
        gradient_ownership_summary: Mapping[str, Any],
        optimizer_step_after: int,
    ) -> DurableLearningUpdateV1:
        child_sha = checkpoint_sha256_v1(child_checkpoint_bytes)
        with _journal_lock(self.lock_path):
            record = self._load_record(update_id)
            if record is None:
                raise DurableUpdateError("UPDATE_NOT_PREPARED")
            if record.commit_status == STATUS_COMMITTED:
                if record.child_checkpoint_sha256 != child_sha:
                    raise DurableUpdateSemanticConflict("COMMITTED_CHILD_MISMATCH")
                return record
            if record.commit_status == STATUS_STAGED:
                if record.child_checkpoint_sha256 != child_sha:
                    raise DurableUpdateSemanticConflict("STAGED_CHILD_MISMATCH")
                return record
            checkpoint_path = self._checkpoint_path(child_sha)
            if checkpoint_path.exists():
                existing_bytes = checkpoint_path.read_bytes()
                if checkpoint_sha256_v1(existing_bytes) != child_sha:
                    raise DurableUpdateCorruptionError("CHECKPOINT_FILE_HASH_MISMATCH")
            else:
                _atomic_write_bytes(checkpoint_path, bytes(child_checkpoint_bytes))
            staged = DurableLearningUpdateV1(
                **{
                    **asdict(record),
                    "commit_status": STATUS_STAGED,
                    "child_checkpoint_sha256": child_sha,
                    "actor_loss": float(actor_loss),
                    "critic_loss": float(critic_loss),
                    "vtrace_diagnostics": dict(vtrace_diagnostics),
                    "gradient_ownership_summary": dict(gradient_ownership_summary),
                    "optimizer_step_after": int(optimizer_step_after),
                }
            ).validate()
            self._write_record(staged)
            return staged

    def commit(self, update_id: str, child_checkpoint_sha256: str) -> DurableLearningUpdateV1:
        _require_hex64("child_checkpoint_sha256", child_checkpoint_sha256)
        with _journal_lock(self.lock_path):
            record = self._load_record(update_id)
            if record is None:
                raise DurableUpdateError("UPDATE_NOT_PREPARED")
            if record.commit_status == STATUS_COMMITTED:
                if record.child_checkpoint_sha256 != child_checkpoint_sha256:
                    raise DurableUpdateSemanticConflict("COMMITTED_CHILD_MISMATCH")
                return record
            if record.commit_status != STATUS_STAGED or record.child_checkpoint_sha256 != child_checkpoint_sha256:
                raise DurableUpdateSemanticConflict("COMMIT_REQUIRES_MATCHING_STAGED_CHILD")
            payload = asdict(record)
            payload["commit_status"] = STATUS_COMMITTED
            payload.pop("committed_receipt_sha256", None)
            committed = DurableLearningUpdateV1(**payload).validate()
            payload["committed_receipt_sha256"] = committed.record_sha256
            committed = DurableLearningUpdateV1(**payload).validate()
            self._write_record(committed)
            return committed

    def load_child_checkpoint(self, update_id: str) -> Mapping[str, Any]:
        record = self._load_record(update_id)
        if record is None:
            raise DurableUpdateError("UPDATE_NOT_PREPARED")
        if record.child_checkpoint_sha256 is None:
            raise DurableUpdateError("UPDATE_HAS_NO_CHILD_CHECKPOINT")
        path = self._checkpoint_path(record.child_checkpoint_sha256)
        if not path.exists():
            raise DurableUpdateCorruptionError("CHILD_CHECKPOINT_MISSING")
        payload = path.read_bytes()
        if checkpoint_sha256_v1(payload) != record.child_checkpoint_sha256:
            raise DurableUpdateCorruptionError("CHILD_CHECKPOINT_HASH_MISMATCH")
        bundle = decode_checkpoint_bundle_v1(payload)
        if bundle["update_id"] != update_id:
            raise DurableUpdateCorruptionError("CHILD_CHECKPOINT_UPDATE_ID_MISMATCH")
        if bundle["parent_checkpoint_sha256"] != record.parent_checkpoint_sha256:
            raise DurableUpdateCorruptionError("CHILD_CHECKPOINT_PARENT_MISMATCH")
        if int(bundle["optimizer_step"]) != int(record.optimizer_step_after):
            raise DurableUpdateCorruptionError("CHILD_CHECKPOINT_STEP_MISMATCH")
        return bundle

    def verify_all(self) -> Mapping[str, int]:
        records = 0
        checkpoints = 0
        for path in sorted(self.updates_dir.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            record = DurableLearningUpdateV1(**{**raw, "sampled_sequence_ids": tuple(raw["sampled_sequence_ids"]), "materialized_sample_hashes": tuple(raw["materialized_sample_hashes"]), "sampling_probabilities_or_weights": tuple(raw["sampling_probabilities_or_weights"]), "behavior_policy_identities": tuple(raw["behavior_policy_identities"])})
            record.validate()
            records += 1
            if record.child_checkpoint_sha256 is not None:
                self.load_child_checkpoint(record.update_id)
                checkpoints += 1
        return {"update_records": records, "child_checkpoints": checkpoints}
