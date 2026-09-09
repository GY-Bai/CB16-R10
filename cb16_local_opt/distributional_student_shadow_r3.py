from __future__ import annotations

"""R11 Science G0 R3: distributional Student belief shadow.

This module is intentionally *not* part of the production Tier-1 decision path.
It consumes already-authorized probabilistic Teacher action laws and trains a
small shadow belief head over a detached Shared Decision Core representation.
The external LONG/FLAT/SHORT + requested-risk contract, Supervisor/Permission,
Physics, canonical generation, and production Student gradient ownership remain
unchanged.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


R11_DISTRIBUTIONAL_STUDENT_SHADOW_R3 = "CB16_R11_DISTRIBUTIONAL_STUDENT_SHADOW_R3_V1"
R3_BELIEF_SEMANTICS = "TEACHER_ACTION_LAW_QUANTILE_DISTILLATION__BELIEF_ONLY"
R3_GRADIENT_OWNER = "Distributional Student Shadow Head"
R3_SHARED_INPUT_GRADIENT_POLICY = "DETACHED_SHARED_REPRESENTATION__NO_PRODUCTION_STUDENT_GRADIENT"
R3_DISTANCE = "TRUNCATED_QUANTILE_W1_ON_TEACHER_QUANTILE_SUPPORT"
R3_HIDDEN_DIM = 64
R3_INIT_GAP_BIAS = -6.0


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256_obj(obj: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(obj)).hexdigest()


def _group_weights(dependence_group_ids: Sequence[str]) -> np.ndarray:
    """Mirror canonical R11 equal-dependence-group mass weighting exactly."""
    if not dependence_group_ids:
        raise RuntimeError("R11_R3_NO_ADMITTED_EVIDENCE")
    counts: dict[str, int] = {}
    for gid in dependence_group_ids:
        counts[str(gid)] = counts.get(str(gid), 0) + 1
    w = np.asarray([1.0 / counts[str(g)] for g in dependence_group_ids], dtype=np.float32)
    return w / max(float(w.mean()), 1e-12)


def _float_tuple(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(x) for x in values)


@dataclass(frozen=True)
class DistributionalEvidenceBatchR3:
    """Immutable admitted Teacher action-law targets plus current Student inputs."""

    parent_ids: tuple[str, ...]
    dependence_group_ids: tuple[str, ...]
    action_grid: tuple[tuple[int, float], ...]
    quantile_levels: tuple[float, ...]
    operator48: torch.Tensor
    medium48: torch.Tensor
    account6: torch.Tensor
    teacher_quantiles: torch.Tensor
    group_weight: torch.Tensor
    evidence_hash: str

    @property
    def rows(self) -> int:
        return len(self.parent_ids)

    @property
    def action_count(self) -> int:
        return len(self.action_grid)

    @property
    def quantile_count(self) -> int:
        return len(self.quantile_levels)

    @property
    def device(self) -> torch.device:
        return self.teacher_quantiles.device

    def validate(self) -> None:
        n, a, q = self.rows, self.action_count, self.quantile_count
        if n <= 0 or a <= 0 or q < 2:
            raise RuntimeError("R11_R3_EMPTY_DISTRIBUTIONAL_BATCH")
        if self.operator48.shape != (n, 48) or self.medium48.shape != (n, 48):
            raise RuntimeError("R11_R3_FROZEN_SENSORY_DIMENSION_DRIFT")
        if self.account6.shape != (n, 6):
            raise RuntimeError("R11_R3_ACCOUNT6_DIMENSION_DRIFT")
        if self.teacher_quantiles.shape != (n, a, q):
            raise RuntimeError("R11_R3_TEACHER_QUANTILE_SHAPE_DRIFT")
        if self.group_weight.shape != (n,):
            raise RuntimeError("R11_R3_GROUP_WEIGHT_SHAPE_DRIFT")
        tensors = (
            self.operator48,
            self.medium48,
            self.account6,
            self.teacher_quantiles,
            self.group_weight,
        )
        if any(t.dtype != torch.float32 for t in tensors):
            raise RuntimeError("R11_R3_NON_FP32_SHADOW_INPUT")
        if any(t.requires_grad for t in tensors):
            raise RuntimeError("R11_R3_TEACHER_OR_FROZEN_INPUT_ENTERED_AUTOGRAD")
        if any(t.device != self.device for t in tensors):
            raise RuntimeError("R11_R3_MIXED_DEVICE_BATCH")
        if not all(bool(torch.isfinite(t).all().item()) for t in tensors):
            raise RuntimeError("R11_R3_NONFINITE_DISTRIBUTIONAL_BATCH")
        levels = np.asarray(self.quantile_levels, dtype=np.float64)
        if np.any(~np.isfinite(levels)) or np.any((levels <= 0.0) | (levels >= 1.0)):
            raise RuntimeError("R11_R3_INVALID_QUANTILE_LEVEL")
        if np.any(np.diff(levels) <= 0.0):
            raise RuntimeError("R11_R3_NONMONOTONE_QUANTILE_LEVEL")
        if torch.any(self.teacher_quantiles[..., 1:] < self.teacher_quantiles[..., :-1]).item():
            raise RuntimeError("R11_R3_TEACHER_QUANTILE_CROSSING")
        if torch.any(self.group_weight <= 0.0).item():
            raise RuntimeError("R11_R3_NONPOSITIVE_DEPENDENCE_GROUP_WEIGHT")
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise RuntimeError("R11_R3_DUPLICATED_ADMITTED_PARENT")
        if len(set(self.action_grid)) != len(self.action_grid):
            raise RuntimeError("R11_R3_DUPLICATED_ACTION_GRID_POINT")

    @classmethod
    def from_evidence(
        cls,
        evidence: Sequence[Any],
        parents: Mapping[str, Any],
        *,
        device: str | torch.device,
    ) -> "DistributionalEvidenceBatchR3":
        admitted = [e for e in evidence if bool(e.admission.admitted)]
        if not admitted:
            raise RuntimeError("R11_R3_NO_ADMITTED_EVIDENCE")

        parent_ids = tuple(str(e.parent_id) for e in admitted)
        group_ids = tuple(str(e.target_dependence_group_id) for e in admitted)
        if len(set(parent_ids)) != len(parent_ids):
            raise RuntimeError("R11_R3_DUPLICATED_ADMITTED_PARENT")

        first_laws = tuple(admitted[0].action_laws)
        if not first_laws:
            raise RuntimeError("R11_R3_TEACHER_ACTION_LAWS_EMPTY")
        action_grid = tuple((int(l.direction), float(l.requested_risk)) for l in first_laws)
        if len(set(action_grid)) != len(action_grid):
            raise RuntimeError("R11_R3_DUPLICATED_ACTION_GRID_POINT")
        quantile_levels = _float_tuple(first_laws[0].quantile_levels)
        if len(quantile_levels) < 2:
            raise RuntimeError("R11_R3_INSUFFICIENT_QUANTILE_LEVELS")

        all_quantiles: list[list[tuple[float, ...]]] = []
        digest_rows: list[dict[str, Any]] = []
        for e in admitted:
            laws = tuple(e.action_laws)
            grid = tuple((int(l.direction), float(l.requested_risk)) for l in laws)
            if grid != action_grid:
                raise RuntimeError(f"R11_R3_ACTION_GRID_DRIFT:{e.parent_id}")
            row: list[tuple[float, ...]] = []
            for law in laws:
                levels = _float_tuple(law.quantile_levels)
                if levels != quantile_levels:
                    raise RuntimeError(f"R11_R3_QUANTILE_LEVEL_DRIFT:{e.parent_id}")
                qs = _float_tuple(law.quantiles)
                if len(qs) != len(quantile_levels):
                    raise RuntimeError(f"R11_R3_QUANTILE_WIDTH_DRIFT:{e.parent_id}")
                if not all(math.isfinite(x) for x in qs):
                    raise RuntimeError(f"R11_R3_NONFINITE_TEACHER_QUANTILE:{e.parent_id}")
                if any(qs[i + 1] < qs[i] for i in range(len(qs) - 1)):
                    raise RuntimeError(f"R11_R3_TEACHER_QUANTILE_CROSSING:{e.parent_id}")
                row.append(qs)
            all_quantiles.append(row)
            digest_rows.append(
                {
                    "parent_id": str(e.parent_id),
                    "dependence_group_id": str(e.target_dependence_group_id),
                    "evidence_id": str(getattr(e, "evidence_id", "")),
                    "teacher_protocol_hash": str(getattr(e, "teacher_protocol_hash", "")),
                    "content_hash": str(getattr(e, "content_hash", "")),
                }
            )

        op = np.asarray([parents[e.parent_id].operator48 for e in admitted], dtype=np.float32)
        med = np.asarray([parents[e.parent_id].medium48 for e in admitted], dtype=np.float32)
        acc = np.asarray([parents[e.parent_id].account6 for e in admitted], dtype=np.float32)
        tq = np.asarray(all_quantiles, dtype=np.float32)
        w = _group_weights(group_ids)
        if op.shape != (len(admitted), 48) or med.shape != (len(admitted), 48):
            raise RuntimeError("R11_R3_FROZEN_SENSORY_DIMENSION_DRIFT")
        if acc.shape != (len(admitted), 6):
            raise RuntimeError("R11_R3_ACCOUNT6_DIMENSION_DRIFT")
        if tq.shape != (len(admitted), len(action_grid), len(quantile_levels)):
            raise RuntimeError("R11_R3_TEACHER_QUANTILE_SHAPE_DRIFT")
        if not np.isfinite(op).all() or not np.isfinite(med).all() or not np.isfinite(acc).all() or not np.isfinite(tq).all():
            raise RuntimeError("R11_R3_NONFINITE_DISTRIBUTIONAL_BATCH")

        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("R11_R3_CUDA_REQUESTED_BUT_UNAVAILABLE")
        tensors = [
            torch.from_numpy(np.ascontiguousarray(x)).to(dev)
            for x in (op, med, acc, tq, w)
        ]
        for t in tensors:
            t.requires_grad_(False)

        evidence_hash = _sha256_obj(
            {
                "schema": "CB16_R11_DISTRIBUTIONAL_EVIDENCE_BATCH_R3_V1",
                "rows": digest_rows,
                "action_grid": action_grid,
                "quantile_levels": quantile_levels,
                "teacher_quantiles_sha256": hashlib.sha256(tq.tobytes(order="C")).hexdigest(),
            }
        )
        out = cls(
            parent_ids=parent_ids,
            dependence_group_ids=group_ids,
            action_grid=action_grid,
            quantile_levels=quantile_levels,
            operator48=tensors[0],
            medium48=tensors[1],
            account6=tensors[2],
            teacher_quantiles=tensors[3],
            group_weight=tensors[4],
            evidence_hash=evidence_hash,
        )
        out.validate()
        return out


class DistributionalStudentShadowHeadR3(nn.Module):
    """Small belief-only head with monotone quantiles by construction."""

    def __init__(
        self,
        *,
        shared_dim: int,
        action_count: int,
        quantile_count: int,
        hidden_dim: int = R3_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if shared_dim <= 0 or action_count <= 0 or quantile_count < 2 or hidden_dim <= 0:
            raise ValueError("R11_R3_INVALID_SHADOW_HEAD_DIMENSION")
        self.shared_dim = int(shared_dim)
        self.action_count = int(action_count)
        self.quantile_count = int(quantile_count)
        self.hidden_dim = int(hidden_dim)
        self.body = nn.Sequential(nn.Linear(self.shared_dim, self.hidden_dim), nn.SiLU())
        self.out = nn.Linear(self.hidden_dim, self.action_count * self.quantile_count)
        self._reset_parameters_r3()

    def _reset_parameters_r3(self) -> None:
        with torch.no_grad():
            bias = self.out.bias.view(self.action_count, self.quantile_count)
            bias[:, 1:].fill_(R3_INIT_GAP_BIAS)

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        if shared.ndim != 2 or shared.shape[1] != self.shared_dim:
            raise RuntimeError("R11_R3_SHARED_REPRESENTATION_SHAPE_DRIFT")
        raw = self.out(self.body(shared.detach())).view(
            shared.shape[0], self.action_count, self.quantile_count
        )
        base = raw[..., :1]
        positive_steps = F.softplus(raw[..., 1:])
        return torch.cat([base, base + torch.cumsum(positive_steps, dim=-1)], dim=-1)


def truncated_quantile_w1_loss_r3(
    predicted_quantiles: torch.Tensor,
    teacher_quantiles: torch.Tensor,
    quantile_levels: Sequence[float],
    group_weight: torch.Tensor,
) -> torch.Tensor:
    """Distill Teacher distributions without treating any realization as a correct label.

    With only Teacher quantiles available, pinball/qCRPS against a realized scalar would
    incorrectly make the future realization a direct Student gradient owner. R3 instead
    matches the two quantile functions over the Teacher's observed quantile support using
    a normalized trapezoidal approximation to truncated W1 distance.
    """
    if predicted_quantiles.shape != teacher_quantiles.shape or predicted_quantiles.ndim != 3:
        raise RuntimeError("R11_R3_PREDICTION_TARGET_SHAPE_DRIFT")
    if teacher_quantiles.requires_grad:
        raise RuntimeError("R11_R3_TEACHER_TARGET_REQUIRES_GRAD")
    if group_weight.ndim != 1 or group_weight.shape[0] != predicted_quantiles.shape[0]:
        raise RuntimeError("R11_R3_GROUP_WEIGHT_SHAPE_DRIFT")
    levels = torch.as_tensor(
        tuple(float(x) for x in quantile_levels),
        dtype=predicted_quantiles.dtype,
        device=predicted_quantiles.device,
    )
    if levels.numel() != predicted_quantiles.shape[-1] or levels.numel() < 2:
        raise RuntimeError("R11_R3_QUANTILE_LEVEL_WIDTH_DRIFT")
    if torch.any(levels[1:] <= levels[:-1]).item():
        raise RuntimeError("R11_R3_NONMONOTONE_QUANTILE_LEVEL")
    width = levels[-1] - levels[0]
    if not bool((width > 0).item()):
        raise RuntimeError("R11_R3_ZERO_QUANTILE_SUPPORT_WIDTH")
    error = torch.abs(predicted_quantiles - teacher_quantiles.detach())
    per_action = torch.trapz(error, levels, dim=-1) / width
    per_row = per_action.mean(dim=-1)
    denom = group_weight.detach().sum().clamp_min(1e-12)
    loss = (per_row * group_weight.detach()).sum() / denom
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("R11_R3_NONFINITE_DISTRIBUTIONAL_LOSS")
    return loss


def quantile_diagnostics_r3(
    predicted_quantiles: torch.Tensor,
    teacher_quantiles: torch.Tensor,
) -> dict[str, float | int | bool]:
    with torch.no_grad():
        pred_cross = predicted_quantiles[..., :-1] - predicted_quantiles[..., 1:]
        teacher_cross = teacher_quantiles[..., :-1] - teacher_quantiles[..., 1:]
        abs_err = torch.abs(predicted_quantiles - teacher_quantiles)
        return {
            "prediction_quantiles_monotone": bool(torch.all(pred_cross <= 0).item()),
            "teacher_quantiles_monotone": bool(torch.all(teacher_cross <= 0).item()),
            "prediction_crossing_count": int(torch.sum(pred_cross > 0).item()),
            "teacher_crossing_count": int(torch.sum(teacher_cross > 0).item()),
            "mean_abs_quantile_error": float(abs_err.mean().item()),
            "max_abs_quantile_error": float(abs_err.max().item()),
        }


def shadow_parameter_report_r3(head: nn.Module) -> dict[str, int | str]:
    params = list(head.named_parameters())
    count = sum(int(p.numel()) for _, p in params)
    trainable = sum(int(p.numel()) for _, p in params if p.requires_grad)
    bytes_fp32 = sum(int(p.numel() * p.element_size()) for _, p in params)
    if any(p.dtype != torch.float32 for _, p in params):
        raise RuntimeError("R11_R3_NON_FP32_SHADOW_PARAMETER")
    if trainable != count:
        raise RuntimeError("R11_R3_FROZEN_SHADOW_PARAMETER")
    return {
        "gradient_owner": R3_GRADIENT_OWNER,
        "parameter_count": count,
        "trainable_parameter_count": trainable,
        "parameter_bytes": bytes_fp32,
        "dtype": "float32",
    }


def assert_shadow_gradient_ownership_r3(
    *,
    production_model: nn.Module,
    shadow_head: nn.Module,
    require_nonzero_shadow_gradient: bool = True,
) -> dict[str, Any]:
    production_grad_names = [name for name, p in production_model.named_parameters() if p.grad is not None]
    if production_grad_names:
        raise RuntimeError(f"R11_R3_PRODUCTION_STUDENT_RECEIVED_SHADOW_GRADIENT:{production_grad_names}")
    nonfinite: list[str] = []
    nonzero: list[str] = []
    missing: list[str] = []
    for name, p in shadow_head.named_parameters():
        if p.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(p.grad).all().item()):
            nonfinite.append(name)
        if bool(torch.any(p.grad != 0).item()):
            nonzero.append(name)
    if nonfinite:
        raise RuntimeError(f"R11_R3_NONFINITE_SHADOW_GRADIENT:{nonfinite}")
    if missing:
        raise RuntimeError(f"R11_R3_SHADOW_PARAMETER_WITHOUT_GRADIENT:{missing}")
    if require_nonzero_shadow_gradient and not nonzero:
        raise RuntimeError("R11_R3_ZERO_SHADOW_GRADIENT_PATH")
    return {
        "authorized_gradient_owner": R3_GRADIENT_OWNER,
        "production_student_gradient_parameter_count": 0,
        "shadow_gradient_parameter_tensors": len(nonzero),
        "teacher_target_autograd": False,
        "shared_representation_detached": True,
    }


def shared_representation_r3(production_model: nn.Module, batch: DistributionalEvidenceBatchR3) -> torch.Tensor:
    """Read production Shared Decision Core as a value, never as an R3 gradient owner."""
    production_model.eval()
    for p in production_model.parameters():
        p.grad = None
    with torch.no_grad():
        out = production_model(batch.operator48, batch.medium48, batch.account6)
        shared = out["shared"]
    if shared.requires_grad:
        raise RuntimeError("R11_R3_SHARED_VALUE_UNEXPECTEDLY_REQUIRES_GRAD")
    if shared.ndim != 2 or shared.shape[0] != batch.rows:
        raise RuntimeError("R11_R3_SHARED_VALUE_SHAPE_DRIFT")
    return shared.detach()
