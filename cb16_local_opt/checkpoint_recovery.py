from __future__ import annotations

"""
Identity-bound checkpoint/recovery for long local runs.

A checkpoint is resumable as the SAME run only when its scientific/runtime identity
matches the requested identity.  Model and optimizer state_dicts are saved rather
than pickling the nn.Module object.

Checkpoint layout:
  checkpoint_<generation>_<step>.pt
  checkpoint_<generation>_<step>.json
  LATEST.json

The .pt file contains tensors/state_dicts.
The .json sidecar contains identity, hashes, progress and Python/NumPy RNG state.
"""

import base64
import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


@dataclass(frozen=True)
class ResumeIdentity:
    experiment_version: str
    dataset_hash: str
    split_hash: str
    physics_hash: str
    supervisor_hash: str
    teacher_hash: str
    training_snapshot_hash: str
    parent_policy_hash: str
    architecture_hash: str
    training_config_hash: str

    @property
    def content_hash(self) -> str:
        return sha256_obj(asdict(self))


@dataclass(frozen=True)
class TrainingProgress:
    generation: int
    epoch: int
    global_step: int
    examples_seen: int
    evidence_cursor: str
    best_metric: float | None = None


@dataclass(frozen=True)
class CheckpointReceipt:
    checkpoint_id: str
    tensor_path: str
    metadata_path: str
    tensor_sha256: str
    metadata_sha256: str
    resume_identity_hash: str
    model_weight_hash: str
    generation: int
    global_step: int


def _state_dict_weight_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        t = tensor.detach().cpu().contiguous()
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _python_state_to_jsonable(state):
    if isinstance(state, tuple):
        return {"__tuple__": [_python_state_to_jsonable(x) for x in state]}
    if isinstance(state, list):
        return [_python_state_to_jsonable(x) for x in state]
    if isinstance(state, (int, float, str, bool)) or state is None:
        return state
    raise TypeError(f"unsupported Python RNG state type: {type(state)!r}")


def _python_state_from_jsonable(obj):
    if isinstance(obj, dict) and "__tuple__" in obj:
        return tuple(_python_state_from_jsonable(x) for x in obj["__tuple__"])
    if isinstance(obj, list):
        return [_python_state_from_jsonable(x) for x in obj]
    return obj


def capture_numpy_rng_state() -> dict[str, Any]:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "name": name,
        "keys": keys.astype(np.uint32).tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def restore_numpy_rng_state(obj: Mapping[str, Any]) -> None:
    np.random.set_state((
        str(obj["name"]),
        np.asarray(obj["keys"], dtype=np.uint32),
        int(obj["pos"]),
        int(obj["has_gauss"]),
        float(obj["cached_gaussian"]),
    ))


def _atomic_torch_save(payload: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".partial",
        dir=str(target.parent),
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(payload, tmp)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_json_save(payload: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".partial",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical_json_bytes(payload) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


class CheckpointManager:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        identity: ResumeIdentity,
        progress: TrainingProgress,
        scheduler: Any | None = None,
        extra_tensor_state: Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointReceipt:
        checkpoint_id = f"G{progress.generation:04d}_S{progress.global_step:012d}"
        tensor_path = self.root / f"checkpoint_{checkpoint_id}.pt"
        metadata_path = self.root / f"checkpoint_{checkpoint_id}.json"

        tensor_payload: dict[str, Any] = {
            "schema": "CB16_LOCAL_TRAINING_CHECKPOINT_TENSORS_R3",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        if scheduler is not None:
            tensor_payload["scheduler_state_dict"] = scheduler.state_dict()
        if extra_tensor_state:
            tensor_payload["extra_tensor_state"] = dict(extra_tensor_state)

        _atomic_torch_save(tensor_payload, tensor_path)
        tensor_hash = sha256_file(tensor_path)
        weight_hash = _state_dict_weight_hash(model)

        metadata = {
            "schema": "CB16_LOCAL_TRAINING_CHECKPOINT_METADATA_R3",
            "checkpoint_id": checkpoint_id,
            "tensor_file": tensor_path.name,
            "tensor_sha256": tensor_hash,
            "resume_identity": asdict(identity),
            "resume_identity_hash": identity.content_hash,
            "progress": asdict(progress),
            "model_weight_hash": weight_hash,
            "python_random_state": _python_state_to_jsonable(random.getstate()),
            "numpy_random_state": capture_numpy_rng_state(),
            "created_at_unix": time.time(),
            "extra_metadata": dict(extra_metadata or {}),
        }
        _atomic_json_save(metadata, metadata_path)
        metadata_hash = sha256_file(metadata_path)

        latest = {
            "schema": "CB16_LOCAL_LATEST_CHECKPOINT_R3",
            "checkpoint_id": checkpoint_id,
            "tensor_path": str(tensor_path),
            "metadata_path": str(metadata_path),
            "tensor_sha256": tensor_hash,
            "metadata_sha256": metadata_hash,
            "resume_identity_hash": identity.content_hash,
        }
        _atomic_json_save(latest, self.root / "LATEST.json")

        return CheckpointReceipt(
            checkpoint_id=checkpoint_id,
            tensor_path=str(tensor_path),
            metadata_path=str(metadata_path),
            tensor_sha256=tensor_hash,
            metadata_sha256=metadata_hash,
            resume_identity_hash=identity.content_hash,
            model_weight_hash=weight_hash,
            generation=progress.generation,
            global_step=progress.global_step,
        )

    def latest_receipt(self) -> dict[str, Any] | None:
        p = self.root / "LATEST.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        expected_identity: ResumeIdentity,
        checkpoint_id: str | None = None,
        scheduler: Any | None = None,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> tuple[TrainingProgress, dict[str, Any]]:
        if checkpoint_id is None:
            latest = self.latest_receipt()
            if latest is None:
                raise FileNotFoundError("NO_CHECKPOINT")
            metadata_path = Path(latest["metadata_path"])
            tensor_path = Path(latest["tensor_path"])
        else:
            metadata_path = self.root / f"checkpoint_{checkpoint_id}.json"
            tensor_path = self.root / f"checkpoint_{checkpoint_id}.pt"

        metadata = json.loads(metadata_path.read_text())
        if metadata["resume_identity_hash"] != expected_identity.content_hash:
            raise RuntimeError(
                "RESUME_IDENTITY_MISMATCH:"
                f"saved={metadata['resume_identity_hash']}:expected={expected_identity.content_hash}"
            )
        if sha256_file(tensor_path) != metadata["tensor_sha256"]:
            raise RuntimeError("CHECKPOINT_TENSOR_HASH_MISMATCH")

        # This checkpoint is created locally by this runtime and contains state_dicts +
        # primitive/tensor state. weights_only=True is preferred on modern PyTorch.
        try:
            payload = torch.load(tensor_path, map_location=map_location, weights_only=True)
        except TypeError:  # older PyTorch compatibility
            payload = torch.load(tensor_path, map_location=map_location)

        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])

        if _state_dict_weight_hash(model) != metadata["model_weight_hash"]:
            raise RuntimeError("RESTORED_MODEL_WEIGHT_HASH_MISMATCH")

        if restore_rng:
            random.setstate(_python_state_from_jsonable(metadata["python_random_state"]))
            restore_numpy_rng_state(metadata["numpy_random_state"])
            torch.set_rng_state(payload["torch_cpu_rng_state"])
            if torch.cuda.is_available() and payload.get("torch_cuda_rng_states"):
                torch.cuda.set_rng_state_all(payload["torch_cuda_rng_states"])

        return TrainingProgress(**metadata["progress"]), metadata

    def validate_all(self) -> dict[str, Any]:
        problems = []
        metas = sorted(self.root.glob("checkpoint_*.json"))
        for mp in metas:
            try:
                m = json.loads(mp.read_text())
                tp = self.root / m["tensor_file"]
                if not tp.is_file():
                    raise FileNotFoundError(tp)
                if sha256_file(tp) != m["tensor_sha256"]:
                    raise RuntimeError("TENSOR_HASH_MISMATCH")
            except Exception as exc:
                problems.append({"metadata": str(mp), "error": repr(exc)})
        return {
            "schema": "CB16_LOCAL_CHECKPOINT_AUDIT_R3",
            "checkpoints": len(metas),
            "problems": problems,
            "pass": not problems,
        }
