from __future__ import annotations

"""R11 FP32 training/evaluation runtime for the frozen TIER_1 Central Brain.

Scientific semantics are frozen by CB16_SEMANTIC_FREEZE_V1.json.  This module only
changes execution: evidence is packed/transferred once per split, batches reuse
static FP32 buffers, validation loss and behavior fingerprint share one forward,
and evaluation is content-addressed by policy/evidence hashes.

Legacy R10.2 training remains a qualification oracle, not runtime authority.
"""

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


TIER1_PARAMETER_COUNT_R11 = 189_052
SMOOTH_L1_BETA_R11 = 0.05
GRADIENT_CLIP_MAX_NORM_R11 = 10.0
GENERATION_BASE_SEED_R11 = 24_680

GRADIENT_OWNER_PREFIXES_R11: Mapping[str, tuple[str, ...]] = {
    "Operator Brain Stem": ("operator_encoder",),
    "Medium Brain Stem": ("medium_encoder",),
    "Account Brain Stem": ("account_encoder",),
    "Shared Decision Core": ("shared_core",),
    "Direction Head": ("direction_body", "direction_out"),
    "Requested-Risk Head": ("direction_embedding", "sizing_body", "sizing_out"),
}
AUTHORIZED_GRADIENT_OWNERS_R11 = frozenset(GRADIENT_OWNER_PREFIXES_R11)

# One contiguous FP32 tensor per admitted evidence split.
_OP = slice(0, 48)
_MED = slice(48, 96)
_ACC = slice(96, 102)
_DP = slice(102, 105)
_RT = 105
_W = 106
_PACK_WIDTH = 107


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256_obj(obj: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(obj)).hexdigest()


def policy_hash_r11(model: torch.nn.Module) -> str:
    """Serialization-independent policy hash compatible with R10.2 tensor semantics."""
    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8") + b"\0")
        h.update(str(tensor.dtype).encode("ascii") + b"\0")
        h.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        h.update(tensor.numpy().tobytes(order="C"))
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(_canonical_json_bytes(obj) + b"\n")
    os.replace(tmp, path)


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _state_l2_delta(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for key in before:
        d = after[key].detach().cpu().double() - before[key].detach().cpu().double()
        total += float(torch.sum(d * d))
    return math.sqrt(total)


def _device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("R11_CUDA_REQUESTED_BUT_UNAVAILABLE")
    return dev


def _disable_noncanonical_cuda_math() -> None:
    # GTX1060/sm_61 canonical path is explicit FP32.  These switches are harmless
    # on Pascal but fail closed against accidental TF32 assumptions on other GPUs.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def assert_tier1_fp32_runtime_r11(model: torch.nn.Module) -> None:
    params = list(model.named_parameters())
    count = sum(int(p.numel()) for _, p in params)
    if count != TIER1_PARAMETER_COUNT_R11:
        raise RuntimeError(f"R11_TIER1_PARAMETER_COUNT_DRIFT:{count}")
    if any(p.dtype != torch.float32 for _, p in params):
        raise RuntimeError("R11_NON_FP32_PARAMETER_FORBIDDEN")
    if any(not p.requires_grad for _, p in params):
        raise RuntimeError("R11_AUTHORIZED_BRAIN_PARAMETER_NOT_TRAINABLE")

    unmatched = [
        name
        for name, _ in params
        if not any(name.startswith(prefixes) for prefixes in GRADIENT_OWNER_PREFIXES_R11.values())
    ]
    if unmatched:
        raise RuntimeError(f"R11_UNCLASSIFIED_TRAINABLE_PARAMETER:{unmatched}")


def group_weights_r11(dependence_group_ids: Sequence[str]) -> np.ndarray:
    """Exact R10.2 dependence-group weighting: inverse rows/group, mean normalized."""
    if not dependence_group_ids:
        raise RuntimeError("R11_NO_ADMITTED_EVIDENCE")
    counts: dict[str, int] = {}
    for group_id in dependence_group_ids:
        counts[group_id] = counts.get(group_id, 0) + 1
    w = np.asarray([1.0 / counts[g] for g in dependence_group_ids], dtype=np.float32)
    return w / max(float(w.mean()), 1e-12)


@dataclass(frozen=True)
class R11TrainingConfig:
    epochs: int = 12
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = GRADIENT_CLIP_MAX_NORM_R11
    generation_base_seed: int = GENERATION_BASE_SEED_R11
    amp_enabled: bool = False
    dtype: torch.dtype = torch.float32

    def validate(self) -> None:
        if self.amp_enabled:
            raise RuntimeError("R11_AMP_FORBIDDEN_ON_CANONICAL_GTX1060_PATH")
        if self.dtype != torch.float32:
            raise RuntimeError("R11_NON_FP32_RUNTIME_DTYPE_FORBIDDEN")
        if self.max_grad_norm != GRADIENT_CLIP_MAX_NORM_R11:
            raise RuntimeError("R11_GRADIENT_CLIP_SEMANTIC_DRIFT")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("R11_INVALID_TRAINING_SHAPE")
        if self.lr <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("R11_INVALID_ADAMW_HYPERPARAMETER")


@dataclass(frozen=True)
class PreparedEvidenceR11:
    """Immutable admitted evidence resident on one device in one FP32 allocation."""

    parent_ids: tuple[str, ...]
    dependence_group_ids: tuple[str, ...]
    packed: torch.Tensor
    evidence_hash: str
    host_to_device_transfers: int

    @property
    def rows(self) -> int:
        return int(self.packed.shape[0])

    @property
    def operator48(self) -> torch.Tensor:
        return self.packed[:, _OP]

    @property
    def medium48(self) -> torch.Tensor:
        return self.packed[:, _MED]

    @property
    def account6(self) -> torch.Tensor:
        return self.packed[:, _ACC]

    @property
    def direction_target_probs(self) -> torch.Tensor:
        return self.packed[:, _DP]

    @property
    def requested_risk_target(self) -> torch.Tensor:
        return self.packed[:, _RT]

    @property
    def group_weight(self) -> torch.Tensor:
        return self.packed[:, _W]

    def validate(self) -> None:
        if self.packed.ndim != 2 or tuple(self.packed.shape)[1] != _PACK_WIDTH:
            raise RuntimeError("R11_PREPARED_EVIDENCE_PACK_SHAPE_DRIFT")
        if self.packed.dtype != torch.float32:
            raise RuntimeError("R11_PREPARED_EVIDENCE_NON_FP32")
        if self.packed.requires_grad:
            raise RuntimeError("R11_TEACHER_OR_FROZEN_INPUT_ENTERED_AUTOGRAD")
        if self.rows != len(self.parent_ids) or self.rows != len(self.dependence_group_ids):
            raise RuntimeError("R11_PREPARED_EVIDENCE_ID_LENGTH_DRIFT")
        if not torch.isfinite(self.packed).all().item():
            raise RuntimeError("R11_NONFINITE_PREPARED_EVIDENCE")
        sums = self.direction_target_probs.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0.0):
            raise RuntimeError("R11_DIRECTION_TARGET_NOT_DISTRIBUTION")
        rt = self.requested_risk_target
        if torch.any((rt < 0.0) | (rt > 1.0)).item():
            raise RuntimeError("R11_REQUESTED_RISK_TARGET_OUT_OF_RANGE")
        if torch.any(self.group_weight <= 0.0).item():
            raise RuntimeError("R11_NONPOSITIVE_DEPENDENCE_GROUP_WEIGHT")

    @classmethod
    def from_evidence(
        cls,
        evidence: Sequence[Any],
        parents: Mapping[str, Any],
        *,
        device: str | torch.device,
        pin_memory: bool = True,
    ) -> "PreparedEvidenceR11":
        admitted = [e for e in evidence if bool(e.admission.admitted)]
        if not admitted:
            raise RuntimeError("R11_NO_ADMITTED_EVIDENCE")

        parent_ids = tuple(str(e.parent_id) for e in admitted)
        if len(set(parent_ids)) != len(parent_ids):
            raise RuntimeError("R11_DUPLICATED_ADMITTED_PARENT")
        group_ids = tuple(str(e.target_dependence_group_id) for e in admitted)

        op = np.asarray([parents[e.parent_id].operator48 for e in admitted], dtype=np.float32)
        med = np.asarray([parents[e.parent_id].medium48 for e in admitted], dtype=np.float32)
        acc = np.asarray([parents[e.parent_id].account6 for e in admitted], dtype=np.float32)
        dp = np.asarray([e.direction_target_probs for e in admitted], dtype=np.float32)
        rt = np.asarray([e.requested_risk_target for e in admitted], dtype=np.float32)[:, None]
        w = group_weights_r11(group_ids)[:, None]

        if op.shape != (len(admitted), 48) or med.shape != (len(admitted), 48):
            raise RuntimeError("R11_FROZEN_SENSORY_DIMENSION_DRIFT")
        if acc.shape != (len(admitted), 6):
            raise RuntimeError("R11_ACCOUNT6_DIMENSION_DRIFT")
        if dp.shape != (len(admitted), 3):
            raise RuntimeError("R11_DIRECTION_TARGET_DIMENSION_DRIFT")

        packed_np = np.ascontiguousarray(
            np.concatenate([op, med, acc, dp, rt, w], axis=1), dtype=np.float32
        )
        if packed_np.shape[1] != _PACK_WIDTH or not np.isfinite(packed_np).all():
            raise RuntimeError("R11_INVALID_EVIDENCE_PACK")

        digest = hashlib.sha256()
        digest.update(b"CB16_R11_PREPARED_EVIDENCE_V1\0")
        for e, pid, gid in zip(admitted, parent_ids, group_ids):
            digest.update(pid.encode("utf-8") + b"\0" + gid.encode("utf-8") + b"\0")
            for attr in ("evidence_id", "teacher_protocol_hash", "content_hash"):
                value = getattr(e, attr, None)
                if value is not None:
                    digest.update(
                        attr.encode("ascii") + b"=" + str(value).encode("utf-8") + b"\0"
                    )
        digest.update(packed_np.tobytes(order="C"))

        host = torch.from_numpy(packed_np)
        dev = _device(device)
        transfers = 0
        if dev.type == "cuda":
            if pin_memory:
                host = host.pin_memory()
            packed = host.to(dev, dtype=torch.float32, non_blocking=bool(pin_memory))
            transfers = 1
        else:
            packed = host
        packed.requires_grad_(False)
        out = cls(
            parent_ids=parent_ids,
            dependence_group_ids=group_ids,
            packed=packed,
            evidence_hash=digest.hexdigest(),
            host_to_device_transfers=transfers,
        )
        out.validate()
        return out

    # Compatibility adapter for callers that still hold R10.2 evidence objects.
    # Only the frozen evidence-contract fields above are consumed.
    from_legacy_evidence = from_evidence


@dataclass(frozen=True)
class PreparedCampaignR11:
    train: PreparedEvidenceR11
    validation: PreparedEvidenceR11


def prepare_evidence_campaign_r11(
    *,
    train_evidence: Sequence[Any],
    validation_evidence: Sequence[Any],
    parents: Mapping[str, Any],
    device: str | torch.device,
    pin_memory: bool = True,
) -> PreparedCampaignR11:
    """Campaign-level prepare-once boundary; no generation-loop tensor rebuilding."""
    return PreparedCampaignR11(
        train=PreparedEvidenceR11.from_evidence(
            train_evidence, parents, device=device, pin_memory=pin_memory
        ),
        validation=PreparedEvidenceR11.from_evidence(
            validation_evidence, parents, device=device, pin_memory=pin_memory
        ),
    )


class StaticPreparedBatchR11:
    """Reusable device-resident FP32 batch buffer."""

    def __init__(self, *, batch_capacity: int, device: str | torch.device):
        if batch_capacity <= 0:
            raise ValueError("R11_INVALID_BATCH_CAPACITY")
        self.batch_capacity = int(batch_capacity)
        self.device = _device(device)
        self._packed = torch.empty(
            (self.batch_capacity, _PACK_WIDTH), dtype=torch.float32, device=self.device
        )
        self._active_rows = 0

    @property
    def packed(self) -> torch.Tensor:
        return self._packed[: self._active_rows]

    def load(self, source: PreparedEvidenceR11, ids: torch.Tensor) -> "StaticPreparedBatchR11":
        if ids.ndim != 1:
            raise ValueError("R11_BATCH_IDS_MUST_BE_1D")
        n = int(ids.numel())
        if n <= 0 or n > self.batch_capacity:
            raise ValueError("R11_BATCH_SIZE_OUT_OF_BUFFER_RANGE")
        if ids.device != source.packed.device:
            raise RuntimeError("R11_BATCH_IDS_DEVICE_DRIFT")
        if ids.dtype != torch.long:
            raise RuntimeError("R11_BATCH_IDS_DTYPE_DRIFT")
        torch.index_select(source.packed, 0, ids, out=self._packed[:n])
        self._active_rows = n
        return self

    @property
    def operator48(self) -> torch.Tensor:
        return self.packed[:, _OP]

    @property
    def medium48(self) -> torch.Tensor:
        return self.packed[:, _MED]

    @property
    def account6(self) -> torch.Tensor:
        return self.packed[:, _ACC]

    @property
    def direction_target_probs(self) -> torch.Tensor:
        return self.packed[:, _DP]

    @property
    def requested_risk_target(self) -> torch.Tensor:
        return self.packed[:, _RT]

    @property
    def group_weight(self) -> torch.Tensor:
        return self.packed[:, _W]


@dataclass(frozen=True)
class LossBreakdownR11:
    loss: torch.Tensor
    direction_loss: torch.Tensor
    sizing_loss: torch.Tensor


def student_loss_from_outputs_r11(
    outputs: Mapping[str, torch.Tensor], batch: Any
) -> LossBreakdownR11:
    """Frozen admitted Student objective: CE_soft + SmoothL1(beta=.05), group weighted."""
    target_p = batch.direction_target_probs.detach()
    risk_target = batch.requested_risk_target.detach()
    weight = batch.group_weight.detach()
    if target_p.requires_grad or risk_target.requires_grad or weight.requires_grad:
        raise RuntimeError("R11_TEACHER_TARGET_AUTOGRAD_TAINT")
    logp = F.log_softmax(outputs["direction_logits"], dim=-1)
    row_direction = -(target_p * logp).sum(dim=-1)
    row_sizing = F.smooth_l1_loss(
        outputs["requested_risk_raw"], risk_target, reduction="none", beta=SMOOTH_L1_BETA_R11
    )
    denom = weight.sum()
    if not torch.isfinite(denom) or float(denom.detach().cpu()) <= 0.0:
        raise RuntimeError("R11_INVALID_GROUP_WEIGHT_DENOMINATOR")
    direction = (row_direction * weight).sum() / denom
    sizing = (row_sizing * weight).sum() / denom
    # Preserve the legacy FP32 reduction order for differential equivalence.
    total = ((row_direction + row_sizing) * weight).sum() / denom
    return LossBreakdownR11(loss=total, direction_loss=direction, sizing_loss=sizing)


def gradient_group_norms_r11(model: torch.nn.Module) -> dict[str, float]:
    """One-device-reduction gradient guard; one small D2H sync for all six owners."""
    device = next(model.parameters()).device
    reductions: list[torch.Tensor] = []
    for _owner, prefixes in GRADIENT_OWNER_PREFIXES_R11.items():
        parts = [
            torch.sum(param.grad.detach() * param.grad.detach())
            for name, param in model.named_parameters()
            if name.startswith(prefixes) and param.grad is not None
        ]
        reductions.append(torch.stack(parts).sum() if parts else torch.zeros((), device=device))
    values = torch.stack(reductions).detach().cpu().numpy()
    return {
        owner: math.sqrt(max(float(value), 0.0))
        for owner, value in zip(GRADIENT_OWNER_PREFIXES_R11, values)
    }


def gradient_owner_set_r11(model: torch.nn.Module) -> frozenset[str]:
    norms = gradient_group_norms_r11(model)
    return frozenset(k for k, v in norms.items() if v > 0.0 and math.isfinite(v))


def _update_group_norms(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for owner, prefixes in GRADIENT_OWNER_PREFIXES_R11.items():
        total = 0.0
        for name, old in before.items():
            if name.startswith(prefixes):
                d = after[name].detach().cpu().double() - old.detach().cpu().double()
                total += float(torch.sum(d * d))
        out[owner] = math.sqrt(total)
    return out


def _forward_model(model: torch.nn.Module, batch: Any) -> Mapping[str, torch.Tensor]:
    return model(batch.operator48, batch.medium48, batch.account6)


def forward_prepared_r11(
    model: torch.nn.Module, prepared: PreparedEvidenceR11
) -> Mapping[str, torch.Tensor]:
    """Public differential/integration adapter for the frozen Central Brain forward."""
    prepared.validate()
    assert_tier1_fp32_runtime_r11(model)
    if next(model.parameters()).device != prepared.packed.device:
        raise RuntimeError("R11_MODEL_EVIDENCE_DEVICE_DRIFT")
    return model(prepared.operator48, prepared.medium48, prepared.account6)


@dataclass
class _CapturedEvalGraphR11:
    static_pack: torch.Tensor
    graph: Any
    outputs: Mapping[str, torch.Tensor]


class EvaluationRuntimeR11:
    """Single-forward validation + behavior fingerprint with policy/evidence cache."""

    def __init__(
        self,
        *,
        enable_cuda_graph: bool = True,
        cuda_graph_capability: Callable[[torch.device], bool] | None = None,
    ):
        self.enable_cuda_graph = bool(enable_cuda_graph)
        self._capability = cuda_graph_capability or self._default_graph_capability
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._graphs: dict[tuple[int, int, int | None], _CapturedEvalGraphR11] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _default_graph_capability(device: torch.device) -> bool:
        return bool(
            device.type == "cuda"
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "CUDAGraph")
            and hasattr(torch.cuda, "graph")
        )

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def _eager_forward(
        self, model: torch.nn.Module, prepared: PreparedEvidenceR11
    ) -> tuple[Mapping[str, torch.Tensor], str]:
        return model(prepared.operator48, prepared.medium48, prepared.account6), "EAGER_FP32"

    def _graph_forward(
        self, model: torch.nn.Module, prepared: PreparedEvidenceR11
    ) -> tuple[Mapping[str, torch.Tensor], str]:
        device = prepared.packed.device
        if not self.enable_cuda_graph or not self._capability(device):
            outputs, _ = self._eager_forward(model, prepared)
            return outputs, "EAGER_FP32_CUDA_GRAPH_FALLBACK"

        key = (id(model), prepared.rows, device.index)
        try:
            captured = self._graphs.get(key)
            if captured is None:
                static_pack = torch.empty_like(prepared.packed)
                static_pack.copy_(prepared.packed)
                # Warm up on a side stream so allocator state is stable before capture.
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        model(static_pack[:, _OP], static_pack[:, _MED], static_pack[:, _ACC])
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = model(static_pack[:, _OP], static_pack[:, _MED], static_pack[:, _ACC])
                captured = _CapturedEvalGraphR11(
                    static_pack=static_pack, graph=graph, outputs=outputs
                )
                self._graphs[key] = captured
            captured.static_pack.copy_(prepared.packed)
            captured.graph.replay()
            return captured.outputs, "CUDA_GRAPH_FP32"
        except Exception:
            # Graph support is a performance capability, never a scientific prerequisite.
            self._graphs.pop(key, None)
            outputs, _ = self._eager_forward(model, prepared)
            return outputs, "EAGER_FP32_CUDA_GRAPH_FALLBACK"

    @staticmethod
    def _behavior_fingerprint(
        model: torch.nn.Module, outputs: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        action = model.compose_action(outputs)
        arr = np.concatenate(
            [
                outputs["direction_probs"].detach().cpu().numpy(),
                outputs["requested_risk_raw"].detach().cpu().numpy()[:, None],
                action["direction"].detach().cpu().numpy().astype(np.float32)[:, None],
            ],
            axis=1,
        ).astype(np.float32)
        return {
            "rows": int(arr.shape[0]),
            "sha256": _sha256_obj(arr.tolist()),
            "mean_direction_probs": arr[:, :3].mean(0).tolist(),
            "mean_requested_risk": float(arr[:, 3].mean()),
            "long_rate": float(np.mean(arr[:, 4] == 1)),
            "flat_rate": float(np.mean(arr[:, 4] == 0)),
            "short_rate": float(np.mean(arr[:, 4] == -1)),
        }

    def evaluate(
        self,
        model: torch.nn.Module,
        prepared: PreparedEvidenceR11,
        *,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        prepared.validate()
        assert_tier1_fp32_runtime_r11(model)
        policy_hash = policy_hash_r11(model)
        key = (policy_hash, prepared.evidence_hash)
        if use_cache and key in self._cache:
            self.cache_hits += 1
            return copy.deepcopy(self._cache[key])
        self.cache_misses += 1

        was_training = bool(model.training)
        model.eval()
        try:
            with torch.inference_mode():
                outputs, mode = self._graph_forward(model, prepared)
                loss = student_loss_from_outputs_r11(outputs, prepared)
                behavior = self._behavior_fingerprint(model, outputs)
                result = {
                    "schema": "CB16_R11_EVALUATION_RESULT_V1",
                    "policy_hash": policy_hash,
                    "validation_evidence_hash": prepared.evidence_hash,
                    "rows": prepared.rows,
                    "loss": float(loss.loss.detach().cpu()),
                    "direction_loss": float(loss.direction_loss.detach().cpu()),
                    "sizing_loss": float(loss.sizing_loss.detach().cpu()),
                    "behavior_fingerprint": behavior,
                    "execution_mode": mode,
                    "dtype": "torch.float32",
                    "amp": False,
                }
        finally:
            model.train(was_training)
        if use_cache:
            self._cache[key] = copy.deepcopy(result)
        return result


@dataclass(frozen=True)
class TrainingStepResultR11:
    loss: float
    direction_loss: float
    sizing_loss: float
    gradient_group_norms: dict[str, float]
    gradient_owner_set: frozenset[str]
    pre_clip_grad_norm: float


class SnapshotConsumptionGuardR11:
    """Commit/recover a trained challenger without double-consuming a snapshot."""

    def __init__(self, receipt_dir: str | Path, generation: int):
        root = Path(receipt_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.consume_path = root / f"R11_SNAPSHOT_CONSUMPTION_G{int(generation)}.json"
        self.receipt_path = root / f"R11_CHALLENGER_TRAINING_RECEIPT_G{int(generation)}.json"
        self.recovery_path = root / f"R11_CHALLENGER_TRAINED_RECOVERY_G{int(generation)}.pt"

    def recover_if_consumed(
        self,
        *,
        model: torch.nn.Module,
        snapshot_hash: str,
        device: str | torch.device,
    ) -> dict[str, Any] | None:
        if not self.consume_path.exists():
            return None
        consume = json.loads(self.consume_path.read_text(encoding="utf-8"))
        if consume.get("snapshot_hash") != snapshot_hash:
            raise RuntimeError("R11_SNAPSHOT_CONSUMPTION_RECEIPT_CONFLICT")
        if not self.receipt_path.is_file() or not self.recovery_path.is_file():
            raise RuntimeError("R11_SNAPSHOT_CONSUMED_WITHOUT_RECOVERABLE_CHALLENGER")
        recovered = torch.load(self.recovery_path, map_location="cpu", weights_only=True)
        if not isinstance(recovered, dict) or recovered.get("snapshot_hash") != snapshot_hash:
            raise RuntimeError("R11_RECOVERY_SNAPSHOT_HASH_CONFLICT")
        state = recovered.get("state_dict")
        if not isinstance(state, Mapping):
            raise RuntimeError("R11_RECOVERY_STATE_MISSING")
        model.load_state_dict(state, strict=True)
        model.to(_device(device))
        return json.loads(self.receipt_path.read_text(encoding="utf-8"))

    def commit(
        self,
        *,
        model: torch.nn.Module,
        snapshot_hash: str,
        generation: int,
        receipt: Mapping[str, Any],
    ) -> None:
        if self.consume_path.exists():
            raise RuntimeError("R11_SNAPSHOT_ALREADY_CONSUMED")
        tmp = self.recovery_path.with_suffix(self.recovery_path.suffix + ".tmp")
        torch.save(
            {
                "schema": "CB16_R11_TRAINED_CHALLENGER_RECOVERY_V1",
                "snapshot_hash": snapshot_hash,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            },
            tmp,
        )
        os.replace(tmp, self.recovery_path)
        _atomic_write_json(self.receipt_path, dict(receipt))
        _atomic_write_json(
            self.consume_path,
            {
                "schema": "CB16_R11_SNAPSHOT_CONSUMPTION_V1",
                "snapshot_hash": snapshot_hash,
                "generation": int(generation),
                "status": "CONSUMED_EXACTLY_ONCE",
                "recovery_checkpoint": str(self.recovery_path),
                "training_receipt": str(self.receipt_path),
            },
        )


class TrainingRuntimeR11:
    """Prepared-evidence, static-buffer AdamW runtime for the canonical FP32 path."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        config: R11TrainingConfig | None = None,
        evaluation_runtime: EvaluationRuntimeR11 | None = None,
    ):
        self.device = _device(device)
        self.config = config or R11TrainingConfig()
        self.config.validate()
        _disable_noncanonical_cuda_math()
        self.evaluation_runtime = evaluation_runtime or EvaluationRuntimeR11(
            enable_cuda_graph=True
        )
        self._batch_buffer = StaticPreparedBatchR11(
            batch_capacity=self.config.batch_size, device=self.device
        )

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.AdamW:
        assert_tier1_fp32_runtime_r11(model)
        return torch.optim.AdamW(
            model.parameters(), lr=float(self.config.lr), weight_decay=float(self.config.weight_decay)
        )

    def train_one_step(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        prepared: PreparedEvidenceR11,
        ids: torch.Tensor,
    ) -> TrainingStepResultR11:
        assert_tier1_fp32_runtime_r11(model)
        if prepared.packed.device != self.device:
            raise RuntimeError("R11_PREPARED_EVIDENCE_DEVICE_DRIFT")
        if next(model.parameters()).device != self.device:
            raise RuntimeError("R11_MODEL_DEVICE_DRIFT")
        batch = self._batch_buffer.load(prepared, ids)
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward_model(model, batch)
        loss = student_loss_from_outputs_r11(outputs, batch)
        if not torch.isfinite(loss.loss).item():
            raise RuntimeError("R11_NONFINITE_TRAIN_LOSS")
        loss.loss.backward()
        norms = gradient_group_norms_r11(model)
        owners = frozenset(k for k, v in norms.items() if v > 0.0 and math.isfinite(v))
        if owners != AUTHORIZED_GRADIENT_OWNERS_R11:
            raise RuntimeError(f"R11_GRADIENT_OWNER_SET_DRIFT:{sorted(owners)}")
        if any(not math.isfinite(v) or v <= 0.0 for v in norms.values()):
            raise RuntimeError(f"R11_AUTHORIZED_GRADIENT_DISCONNECT:{norms}")
        pre_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float(self.config.max_grad_norm)
        )
        if not torch.isfinite(pre_clip).item():
            raise RuntimeError("R11_NONFINITE_PRECLIP_GRADIENT_NORM")
        optimizer.step()
        return TrainingStepResultR11(
            loss=float(loss.loss.detach().cpu()),
            direction_loss=float(loss.direction_loss.detach().cpu()),
            sizing_loss=float(loss.sizing_loss.detach().cpu()),
            gradient_group_norms=norms,
            gradient_owner_set=owners,
            pre_clip_grad_norm=float(pre_clip.detach().cpu()),
        )

    def train_challenger(
        self,
        *,
        model: torch.nn.Module,
        campaign: PreparedCampaignR11,
        generation: int,
        snapshot_hash: str,
        receipt_dir: str | Path,
    ) -> dict[str, Any]:
        self.config.validate()
        campaign.train.validate()
        campaign.validation.validate()
        if (
            campaign.train.packed.device != self.device
            or campaign.validation.packed.device != self.device
        ):
            raise RuntimeError("R11_CAMPAIGN_DEVICE_DRIFT")
        model.to(self.device)
        assert_tier1_fp32_runtime_r11(model)

        guard = SnapshotConsumptionGuardR11(receipt_dir, int(generation))
        recovered = guard.recover_if_consumed(
            model=model, snapshot_hash=snapshot_hash, device=self.device
        )
        if recovered is not None:
            return recovered

        independent_groups = len(set(campaign.train.dependence_group_ids))
        if independent_groups < 32:
            raise RuntimeError("INSUFFICIENT_INDEPENDENT_TRAIN_GROUPS_FOR_R11")

        before = _clone_state_dict(model)
        validation_before = self.evaluation_runtime.evaluate(
            model, campaign.validation, use_cache=True
        )
        model.train()
        optimizer = self.build_optimizer(model)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.generation_base_seed + int(generation))

        last_step: TrainingStepResultR11 | None = None
        steps = 0
        for _epoch in range(int(self.config.epochs)):
            # The permutation itself is generated on CPU exactly as R10.2.  It is moved
            # once per epoch instead of once per mini-batch.
            permutation_cpu = torch.randperm(campaign.train.rows, generator=generator)
            permutation = permutation_cpu.to(self.device)
            for start in range(0, campaign.train.rows, int(self.config.batch_size)):
                ids = permutation[start : start + int(self.config.batch_size)]
                last_step = self.train_one_step(
                    model=model, optimizer=optimizer, prepared=campaign.train, ids=ids
                )
                steps += 1

        if last_step is None:
            raise RuntimeError("R11_ZERO_OPTIMIZER_STEPS")
        after = _clone_state_dict(model)
        update_norms = _update_group_norms(before, after)
        if any(v <= 0.0 or not math.isfinite(v) for v in update_norms.values()):
            raise RuntimeError(f"R11_BRAIN_GROUP_NOT_UPDATED:{update_norms}")

        validation_after = self.evaluation_runtime.evaluate(
            model, campaign.validation, use_cache=True
        )
        receipt: dict[str, Any] = {
            "schema": "CB16_R11_CHALLENGER_TRAINING_RECEIPT_V1",
            "generation": int(generation),
            "snapshot_hash": snapshot_hash,
            "optimizer": "AdamW_FP32",
            "amp": False,
            "dtype": "torch.float32",
            "epochs": int(self.config.epochs),
            "batch_size": int(self.config.batch_size),
            "optimizer_steps": int(steps),
            "lr": float(self.config.lr),
            "weight_decay": float(self.config.weight_decay),
            "gradient_clip_max_norm": float(self.config.max_grad_norm),
            "generation_seed": int(self.config.generation_base_seed + int(generation)),
            "train_evidence_hash": campaign.train.evidence_hash,
            "validation_evidence_hash": campaign.validation.evidence_hash,
            "train_host_to_device_transfers": campaign.train.host_to_device_transfers,
            "validation_host_to_device_transfers": campaign.validation.host_to_device_transfers,
            "parameter_l2_delta": _state_l2_delta(before, after),
            "gradient_group_norms_last_step": last_step.gradient_group_norms,
            "gradient_owner_set_last_step": sorted(last_step.gradient_owner_set),
            "update_group_norms": update_norms,
            "challenger_semantic_sha256": policy_hash_r11(model),
            "validation_before": validation_before,
            "validation_after": validation_after,
            "external_frozen_organ_gradients": "NOT_IN_AUTOGRAD_GRAPH__INPUT_VALUES_DETACHED",
            "teacher_future_autograd": "TARGET_VALUES_ONLY_THROUGH_ADMITTED_STUDENT_LOSS",
        }
        guard.commit(
            model=model,
            snapshot_hash=snapshot_hash,
            generation=int(generation),
            receipt=receipt,
        )
        return receipt
