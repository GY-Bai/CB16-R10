from __future__ import annotations

"""R11 FP32 training/evaluation runtime for the frozen TIER_1 Central Brain.

Scientific semantics are frozen by CB16_SEMANTIC_FREEZE_V1.json. This module
only changes execution: admitted evidence is packed once, campaign tensors stay
resident, minibatch storage is reused, CPU permutations are transferred once per
generation, validation/fingerprint share one forward, and graph/telemetry paths
are explicitly optional. Legacy R10.2 remains an oracle, not runtime authority.
"""

import copy
import hashlib
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .gpu_runtime_r11 import (
    CUDA_GRAPH_DISABLED_FALLBACK,
    GPUExecutionProfileR11,
    H2DBenchmarkResultR11,
    RuntimeTelemetryR11,
    TelemetryConfigR11,
    inspect_gpu_execution_profile_r11,
    prepare_epoch_permutations_r11,
    transfer_host_tensor_r11,
)


TIER1_PARAMETER_COUNT_R11 = 189_052
SMOOTH_L1_BETA_R11 = 0.05
GRADIENT_CLIP_MAX_NORM_R11 = 10.0
GENERATION_BASE_SEED_R11 = 24_680
CANONICAL_EPOCHS_R11 = 12
CANONICAL_BATCH_SIZE_R11 = 512
CANONICAL_LR_R11 = 3e-4
CANONICAL_WEIGHT_DECAY_R11 = 1e-4

GRADIENT_OWNER_PREFIXES_R11: Mapping[str, tuple[str, ...]] = {
    "Operator Brain Stem": ("operator_encoder",),
    "Medium Brain Stem": ("medium_encoder",),
    "Account Brain Stem": ("account_encoder",),
    "Shared Decision Core": ("shared_core",),
    "Direction Head": ("direction_body", "direction_out"),
    "Requested-Risk Head": ("direction_embedding", "sizing_body", "sizing_out"),
}
AUTHORIZED_GRADIENT_OWNERS_R11 = frozenset(GRADIENT_OWNER_PREFIXES_R11)

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
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _disable_noncanonical_cuda_math() -> None:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def _assert_finite_scalar_r11(value: torch.Tensor, message: str) -> None:
    finite = torch.isfinite(value)
    if value.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(finite, message)
    elif not bool(finite.item()):
        raise RuntimeError(message)


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
    """Exact R10.2 weighting: inverse rows/group, normalized to mean one."""
    if not dependence_group_ids:
        raise RuntimeError("R11_NO_ADMITTED_EVIDENCE")
    counts: dict[str, int] = {}
    for group_id in dependence_group_ids:
        counts[group_id] = counts.get(group_id, 0) + 1
    w = np.asarray([1.0 / counts[g] for g in dependence_group_ids], dtype=np.float32)
    return w / max(float(w.mean()), 1e-12)


@dataclass(frozen=True)
class R11TrainingConfig:
    epochs: int = CANONICAL_EPOCHS_R11
    batch_size: int = CANONICAL_BATCH_SIZE_R11
    lr: float = CANONICAL_LR_R11
    weight_decay: float = CANONICAL_WEIGHT_DECAY_R11
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
        if self.generation_base_seed != GENERATION_BASE_SEED_R11:
            raise RuntimeError("R11_GENERATION_SEED_RULE_DRIFT")
        if (
            self.epochs != CANONICAL_EPOCHS_R11
            or self.batch_size != CANONICAL_BATCH_SIZE_R11
            or self.lr != CANONICAL_LR_R11
            or self.weight_decay != CANONICAL_WEIGHT_DECAY_R11
        ):
            raise RuntimeError("R11_ADAMW_TRAINING_RULE_DRIFT")


@dataclass(frozen=True)
class PreparedEvidenceR11:
    """Immutable admitted evidence in one contiguous FP32 device allocation."""

    parent_ids: tuple[str, ...]
    dependence_group_ids: tuple[str, ...]
    packed: torch.Tensor
    evidence_hash: str
    host_to_device_transfers: int
    h2d_strategy: str = "CPU_RESIDENT"
    h2d_non_blocking: bool = False
    h2d_latency_ms: float | None = 0.0
    h2d_benchmark: Mapping[str, Any] | None = None

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
        pin_memory: bool | None = None,
        h2d_benchmark_hook: (
            Callable[[torch.Tensor, torch.device], H2DBenchmarkResultR11] | None
        ) = None,
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
        packed, transfer = transfer_host_tensor_r11(
            host,
            dev,
            pin_memory=pin_memory,
            benchmark_hook=h2d_benchmark_hook,
        )
        packed.requires_grad_(False)
        out = cls(
            parent_ids=parent_ids,
            dependence_group_ids=group_ids,
            packed=packed,
            evidence_hash=digest.hexdigest(),
            host_to_device_transfers=1 if dev.type == "cuda" else 0,
            h2d_strategy=transfer.strategy,
            h2d_non_blocking=transfer.non_blocking,
            h2d_latency_ms=transfer.transfer_ms,
            h2d_benchmark=(
                None if transfer.benchmark is None else transfer.benchmark.as_dict()
            ),
        )
        out.validate()
        return out

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
    pin_memory: bool | None = None,
    h2d_benchmark_hook: (
        Callable[[torch.Tensor, torch.device], H2DBenchmarkResultR11] | None
    ) = None,
) -> PreparedCampaignR11:
    """Campaign-level prepare-once boundary; no generation-loop tensor rebuild."""
    return PreparedCampaignR11(
        train=PreparedEvidenceR11.from_evidence(
            train_evidence,
            parents,
            device=device,
            pin_memory=pin_memory,
            h2d_benchmark_hook=h2d_benchmark_hook,
        ),
        validation=PreparedEvidenceR11.from_evidence(
            validation_evidence,
            parents,
            device=device,
            pin_memory=pin_memory,
            h2d_benchmark_hook=h2d_benchmark_hook,
        ),
    )


class StaticPreparedBatchR11:
    """Reusable device-resident FP32 minibatch buffer."""

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
    def storage_data_ptr(self) -> int:
        return int(self._packed.data_ptr())

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


def _student_loss_impl_r11(
    outputs: Mapping[str, torch.Tensor], batch: Any, *, validate_weight_denom: bool
) -> LossBreakdownR11:
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
    if validate_weight_denom:
        if not torch.isfinite(denom).item() or float(denom.detach().cpu()) <= 0.0:
            raise RuntimeError("R11_INVALID_GROUP_WEIGHT_DENOMINATOR")
    direction = (row_direction * weight).sum() / denom
    sizing = (row_sizing * weight).sum() / denom
    total = ((row_direction + row_sizing) * weight).sum() / denom
    return LossBreakdownR11(loss=total, direction_loss=direction, sizing_loss=sizing)


def student_loss_from_outputs_r11(
    outputs: Mapping[str, torch.Tensor], batch: Any
) -> LossBreakdownR11:
    """Frozen objective: soft CE + SmoothL1(beta=.05), dependence-group weighted."""
    return _student_loss_impl_r11(outputs, batch, validate_weight_denom=True)


def gradient_group_norms_r11(model: torch.nn.Module) -> dict[str, float]:
    """Reduce all six owner norms on device, then perform one small D2H copy."""
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
    """Single-forward validation/fingerprint with bounded result/graph buffers."""

    def __init__(
        self,
        *,
        enable_cuda_graph: bool = True,
        cuda_graph_capability: Callable[[torch.device], bool] | None = None,
        max_cached_results: int = 32,
        max_cuda_graphs: int = 2,
        max_behavior_buffers: int = 2,
    ):
        self.enable_cuda_graph = bool(enable_cuda_graph)
        self._capability = cuda_graph_capability or self._default_graph_capability
        self.max_cached_results = max(1, int(max_cached_results))
        self.max_cuda_graphs = max(1, int(max_cuda_graphs))
        self.max_behavior_buffers = max(1, int(max_behavior_buffers))
        self._cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._graphs: OrderedDict[tuple[Any, ...], _CapturedEvalGraphR11] = OrderedDict()
        self._behavior_buffers: OrderedDict[tuple[str, int, int], torch.Tensor] = OrderedDict()
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

    @staticmethod
    def _put_bounded(store: OrderedDict, key: Any, value: Any, limit: int) -> None:
        if key in store:
            store.pop(key)
        store[key] = value
        while len(store) > limit:
            store.popitem(last=False)

    def clear_cache(self) -> None:
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_cuda_state(self) -> None:
        self._graphs.clear()
        self._behavior_buffers.clear()

    def resource_stats(self) -> dict[str, int]:
        return {
            "evaluation_cache_entries": len(self._cache),
            "cuda_graph_entries": len(self._graphs),
            "behavior_buffer_entries": len(self._behavior_buffers),
        }

    def _eager_forward(
        self, model: torch.nn.Module, prepared: PreparedEvidenceR11
    ) -> tuple[Mapping[str, torch.Tensor], str, str]:
        return (
            model(prepared.operator48, prepared.medium48, prepared.account6),
            "EAGER_FP32",
            CUDA_GRAPH_DISABLED_FALLBACK,
        )

    def _graph_forward(
        self, model: torch.nn.Module, prepared: PreparedEvidenceR11
    ) -> tuple[Mapping[str, torch.Tensor], str, str]:
        device = prepared.packed.device
        if not self.enable_cuda_graph or not self._capability(device):
            outputs, _, status = self._eager_forward(model, prepared)
            return outputs, "EAGER_FP32_CUDA_GRAPH_FALLBACK", status

        parameter_storage = tuple(int(p.data_ptr()) for p in model.parameters())
        key = (
            id(model),
            parameter_storage,
            prepared.evidence_hash,
            prepared.rows,
            device.index,
        )
        try:
            captured = self._graphs.get(key)
            if captured is None:
                static_pack = torch.empty_like(prepared.packed)
                static_pack.copy_(prepared.packed)
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
                self._put_bounded(self._graphs, key, captured, self.max_cuda_graphs)
            else:
                self._graphs.move_to_end(key)
            captured.graph.replay()
            return captured.outputs, "CUDA_GRAPH_FP32", "CUDA_GRAPH_ACTIVE"
        except Exception:
            self._graphs.pop(key, None)
            outputs, _, _ = self._eager_forward(model, prepared)
            return outputs, "EAGER_FP32_CUDA_GRAPH_FALLBACK", CUDA_GRAPH_DISABLED_FALLBACK

    def _behavior_buffer(self, outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        probs = outputs["direction_probs"]
        rows = int(probs.shape[0])
        device = probs.device
        key = (device.type, -1 if device.index is None else int(device.index), rows)
        buffer = self._behavior_buffers.get(key)
        if buffer is None:
            buffer = torch.empty((rows, 5), dtype=torch.float32, device=device)
            self._put_bounded(
                self._behavior_buffers, key, buffer, self.max_behavior_buffers
            )
        else:
            self._behavior_buffers.move_to_end(key)
        return buffer

    def _behavior_fingerprint(
        self, model: torch.nn.Module, outputs: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        action = model.compose_action(outputs)
        packed = self._behavior_buffer(outputs)
        packed[:, :3].copy_(outputs["direction_probs"])
        packed[:, 3].copy_(outputs["requested_risk_raw"])
        packed[:, 4].copy_(action["direction"].to(dtype=torch.float32))
        arr = packed.detach().cpu().numpy()
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
            self._cache.move_to_end(key)
            return copy.deepcopy(self._cache[key])
        self.cache_misses += 1

        was_training = bool(model.training)
        model.eval()
        try:
            with torch.inference_mode():
                outputs, mode, graph_status = self._graph_forward(model, prepared)
                loss = _student_loss_impl_r11(
                    outputs, prepared, validate_weight_denom=False
                )
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
                    "cuda_graph_status": graph_status,
                    "dtype": "torch.float32",
                    "amp": False,
                }
        finally:
            model.train(was_training)
        if use_cache:
            self._put_bounded(self._cache, key, copy.deepcopy(result), self.max_cached_results)
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
    """Commit/recover a challenger without consuming one snapshot twice."""

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
    """Prepared-evidence, static-buffer AdamW runtime for canonical FP32 execution."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        config: R11TrainingConfig | None = None,
        evaluation_runtime: EvaluationRuntimeR11 | None = None,
        telemetry_config: TelemetryConfigR11 | None = None,
    ):
        self.device = _device(device)
        self.config = config or R11TrainingConfig()
        self.config.validate()
        _disable_noncanonical_cuda_math()
        self.execution_profile: GPUExecutionProfileR11 = inspect_gpu_execution_profile_r11(
            self.device
        )
        self.evaluation_runtime = evaluation_runtime or EvaluationRuntimeR11(
            enable_cuda_graph=True
        )
        self.telemetry = RuntimeTelemetryR11(telemetry_config or TelemetryConfigR11())
        self._batch_buffer = StaticPreparedBatchR11(
            batch_capacity=self.config.batch_size, device=self.device
        )
        self._step_index = 0

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.AdamW:
        assert_tier1_fp32_runtime_r11(model)
        return torch.optim.AdamW(
            model.parameters(), lr=float(self.config.lr), weight_decay=float(self.config.weight_decay)
        )

    def resource_stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "static_batch_buffer_data_ptr": self._batch_buffer.storage_data_ptr,
            "static_batch_capacity": self._batch_buffer.batch_capacity,
            "execution_profile": self.execution_profile.as_dict(),
        }
        out.update(self.evaluation_runtime.resource_stats())
        if self.device.type == "cuda" and torch.cuda.is_available():
            out.update(
                {
                    "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                    "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                }
            )
        return out

    def _validate_step_boundary(
        self, model: torch.nn.Module, prepared: PreparedEvidenceR11
    ) -> None:
        assert_tier1_fp32_runtime_r11(model)
        if prepared.packed.device != self.device:
            raise RuntimeError("R11_PREPARED_EVIDENCE_DEVICE_DRIFT")
        if next(model.parameters()).device != self.device:
            raise RuntimeError("R11_MODEL_DEVICE_DRIFT")

    def _train_step(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        prepared: PreparedEvidenceR11,
        ids: torch.Tensor,
        runtime_prevalidated: bool,
        collect_diagnostics: bool,
        materialize_result: bool,
    ) -> TrainingStepResultR11 | None:
        if not runtime_prevalidated:
            self._validate_step_boundary(model, prepared)
        timer = self.telemetry.begin_step(self._step_index, self.device)
        self._step_index += 1

        batch = self._batch_buffer.load(prepared, ids)
        if timer is not None:
            timer.mark("batch_ready")
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward_model(model, batch)
        loss = _student_loss_impl_r11(outputs, batch, validate_weight_denom=False)
        _assert_finite_scalar_r11(loss.loss, "R11_NONFINITE_TRAIN_LOSS")
        if timer is not None:
            timer.mark("forward_done")
        loss.loss.backward()
        if timer is not None:
            timer.mark("backward_done")

        norms: dict[str, float] = {}
        owners = frozenset()
        if collect_diagnostics:
            norms = gradient_group_norms_r11(model)
            owners = frozenset(k for k, v in norms.items() if v > 0.0 and math.isfinite(v))
            if owners != AUTHORIZED_GRADIENT_OWNERS_R11:
                raise RuntimeError(f"R11_GRADIENT_OWNER_SET_DRIFT:{sorted(owners)}")
            if any(not math.isfinite(v) or v <= 0.0 for v in norms.values()):
                raise RuntimeError(f"R11_AUTHORIZED_GRADIENT_DISCONNECT:{norms}")

        pre_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float(self.config.max_grad_norm)
        )
        _assert_finite_scalar_r11(pre_clip, "R11_NONFINITE_PRECLIP_GRADIENT_NORM")
        if timer is not None:
            timer.mark("clip_checks_done")
        optimizer.step()
        if timer is not None:
            timer.mark("optimizer_done")
        self.telemetry.record_step(timer)

        if not materialize_result:
            return None
        if not collect_diagnostics:
            norms = gradient_group_norms_r11(model)
            owners = frozenset(k for k, v in norms.items() if v > 0.0 and math.isfinite(v))
        return TrainingStepResultR11(
            loss=float(loss.loss.detach().cpu()),
            direction_loss=float(loss.direction_loss.detach().cpu()),
            sizing_loss=float(loss.sizing_loss.detach().cpu()),
            gradient_group_norms=norms,
            gradient_owner_set=owners,
            pre_clip_grad_norm=float(pre_clip.detach().cpu()),
        )

    def train_one_step(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        prepared: PreparedEvidenceR11,
        ids: torch.Tensor,
    ) -> TrainingStepResultR11:
        result = self._train_step(
            model=model,
            optimizer=optimizer,
            prepared=prepared,
            ids=ids,
            runtime_prevalidated=False,
            collect_diagnostics=True,
            materialize_result=True,
        )
        if result is None:
            raise AssertionError("R11_INTERNAL_STEP_RESULT_MISSING")
        return result

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
        self._validate_step_boundary(model, campaign.train)

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
        generation_seed = self.config.generation_base_seed + int(generation)
        permutations, permutation_h2d_transfers = prepare_epoch_permutations_r11(
            rows=campaign.train.rows,
            epochs=int(self.config.epochs),
            seed=generation_seed,
            device=self.device,
        )

        batches_per_epoch = math.ceil(campaign.train.rows / int(self.config.batch_size))
        expected_steps = int(self.config.epochs) * batches_per_epoch
        last_step: TrainingStepResultR11 | None = None
        steps = 0
        self.telemetry.sample_device(self.device)
        for epoch in range(int(self.config.epochs)):
            permutation = permutations[epoch]
            for start in range(0, campaign.train.rows, int(self.config.batch_size)):
                ids = permutation[start : start + int(self.config.batch_size)]
                is_first = steps == 0
                is_last = steps == expected_steps - 1
                result = self._train_step(
                    model=model,
                    optimizer=optimizer,
                    prepared=campaign.train,
                    ids=ids,
                    runtime_prevalidated=True,
                    collect_diagnostics=bool(is_first or is_last),
                    materialize_result=bool(is_last),
                )
                if result is not None:
                    last_step = result
                steps += 1
        self.telemetry.sample_device(self.device)

        if last_step is None or steps != expected_steps:
            raise RuntimeError("R11_ZERO_OR_MISMATCHED_OPTIMIZER_STEPS")
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
            "generation_seed": int(generation_seed),
            "permutation_host_to_device_transfers": int(permutation_h2d_transfers),
            "train_evidence_hash": campaign.train.evidence_hash,
            "validation_evidence_hash": campaign.validation.evidence_hash,
            "train_host_to_device_transfers": campaign.train.host_to_device_transfers,
            "validation_host_to_device_transfers": campaign.validation.host_to_device_transfers,
            "train_h2d": {
                "strategy": campaign.train.h2d_strategy,
                "non_blocking": campaign.train.h2d_non_blocking,
                "latency_ms": campaign.train.h2d_latency_ms,
                "benchmark": campaign.train.h2d_benchmark,
            },
            "validation_h2d": {
                "strategy": campaign.validation.h2d_strategy,
                "non_blocking": campaign.validation.h2d_non_blocking,
                "latency_ms": campaign.validation.h2d_latency_ms,
                "benchmark": campaign.validation.h2d_benchmark,
            },
            "static_buffers": {
                "campaign_train_resident": True,
                "campaign_validation_resident": True,
                "reusable_minibatch_buffer": True,
                "minibatch_buffer_data_ptr": self._batch_buffer.storage_data_ptr,
            },
            "gpu_execution_profile": self.execution_profile.as_dict(),
            "parameter_l2_delta": _state_l2_delta(before, after),
            "gradient_group_norms_last_step": last_step.gradient_group_norms,
            "gradient_owner_set_last_step": sorted(last_step.gradient_owner_set),
            "update_group_norms": update_norms,
            "challenger_semantic_sha256": policy_hash_r11(model),
            "validation_before": validation_before,
            "validation_after": validation_after,
            "telemetry": self.telemetry.snapshot(),
            "runtime_resources": self.resource_stats(),
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


__all__ = [
    "AUTHORIZED_GRADIENT_OWNERS_R11",
    "CANONICAL_BATCH_SIZE_R11",
    "CANONICAL_EPOCHS_R11",
    "CUDA_GRAPH_DISABLED_FALLBACK",
    "EvaluationRuntimeR11",
    "PreparedCampaignR11",
    "PreparedEvidenceR11",
    "R11TrainingConfig",
    "SnapshotConsumptionGuardR11",
    "StaticPreparedBatchR11",
    "TrainingRuntimeR11",
    "TrainingStepResultR11",
    "assert_tier1_fp32_runtime_r11",
    "forward_prepared_r11",
    "gradient_group_norms_r11",
    "gradient_owner_set_r11",
    "group_weights_r11",
    "policy_hash_r11",
    "prepare_evidence_campaign_r11",
    "student_loss_from_outputs_r11",
]
