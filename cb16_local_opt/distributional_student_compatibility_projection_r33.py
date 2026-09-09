from __future__ import annotations

"""R11 Science G0 R3.3 — compatibility-constrained Shared-Core update helpers.

The distributional belief head always receives its full distributional gradient.
Only the disposable Shared Decision Core update direction may be projected, and
only when the *actual Adam direction* is first-order conflicting with the current
canonical Direction/Requested-Risk total gradient.  The canonical gradient is a
compatibility guard, not a new Teacher/truth target and is never added to the
optimization objective.
"""

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch


R33_RUNTIME = "CB16_R11_DISTRIBUTIONAL_STUDENT_COMPATIBILITY_PROJECTION_R3_3_V1"
R33_SEEDS: tuple[int, ...] = (31_337, 271_828, 314_159, 161_803, 424_242)
R33_STEPS = 32
R33_LR = 2e-3
R33_WEIGHT_DECAY = 0.0
R33_GRAD_CLIP = 10.0
R33_BETA1 = 0.9
R33_BETA2 = 0.999
R33_EPS = 1e-8
R33_PROJECTION_REL_TOL = 2e-7


def _validate_tensor_list(xs: Sequence[torch.Tensor], *, code: str) -> None:
    if not xs:
        raise RuntimeError(f"{code}:EMPTY")
    for x in xs:
        if not bool(torch.isfinite(x).all().item()):
            raise RuntimeError(f"{code}:NONFINITE")


def _flatten64(xs: Sequence[torch.Tensor]) -> torch.Tensor:
    _validate_tensor_list(xs, code="R11_R33_FLATTEN")
    return torch.cat([x.detach().reshape(-1).to(dtype=torch.float64) for x in xs])


def _unflatten_like(flat: torch.Tensor, refs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    out: list[torch.Tensor] = []
    pos = 0
    for ref in refs:
        n = int(ref.numel())
        chunk = flat[pos : pos + n].reshape(ref.shape).to(device=ref.device, dtype=ref.dtype)
        out.append(chunk)
        pos += n
    if pos != int(flat.numel()):
        raise RuntimeError("R11_R33_UNFLATTEN_WIDTH_DRIFT")
    return tuple(out)


@dataclass
class ManualAdamDirectionR33:
    """AdamW(weight_decay=0) moment state returning the pre-LR update direction."""

    params: tuple[torch.nn.Parameter, ...]
    beta1: float = R33_BETA1
    beta2: float = R33_BETA2
    eps: float = R33_EPS

    def __post_init__(self) -> None:
        if not self.params:
            raise ValueError("R11_R33_ADAM_EMPTY_PARAMETER_SET")
        if not (0.0 < self.beta1 < 1.0 and 0.0 < self.beta2 < 1.0 and self.eps > 0.0):
            raise ValueError("R11_R33_ADAM_HYPERPARAMETER_DRIFT")
        self.step = 0
        self.exp_avg = [torch.zeros_like(p) for p in self.params]
        self.exp_avg_sq = [torch.zeros_like(p) for p in self.params]

    def directions(self, grads: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(grads) != len(self.params):
            raise RuntimeError("R11_R33_ADAM_GRADIENT_COUNT_DRIFT")
        _validate_tensor_list(grads, code="R11_R33_ADAM_GRADIENT")
        self.step += 1
        bc1 = 1.0 - self.beta1 ** self.step
        bc2 = 1.0 - self.beta2 ** self.step
        out: list[torch.Tensor] = []
        with torch.no_grad():
            for p, g, m, v in zip(self.params, grads, self.exp_avg, self.exp_avg_sq):
                if g.shape != p.shape or g.device != p.device or g.dtype != p.dtype:
                    raise RuntimeError("R11_R33_ADAM_GRADIENT_PARAMETER_DRIFT")
                m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                denom = v.sqrt().div_(math.sqrt(bc2)).add_(self.eps)
                direction = m.div(bc1).div(denom)
                if not bool(torch.isfinite(direction).all().item()):
                    raise RuntimeError("R11_R33_NONFINITE_ADAM_DIRECTION")
                out.append(direction.clone())
        return tuple(out)


def apply_adam_direction_r33(
    params: Sequence[torch.nn.Parameter], directions: Sequence[torch.Tensor], *, lr: float = R33_LR
) -> None:
    if len(params) != len(directions) or lr <= 0.0:
        raise RuntimeError("R11_R33_APPLY_DIRECTION_CONTRACT_DRIFT")
    with torch.no_grad():
        for p, d in zip(params, directions):
            if p.shape != d.shape or p.device != d.device or p.dtype != d.dtype:
                raise RuntimeError("R11_R33_APPLY_DIRECTION_SHAPE_DEVICE_DTYPE_DRIFT")
            p.add_(d, alpha=-float(lr))


def compatibility_project_adam_direction_r33(
    raw_directions: Sequence[torch.Tensor],
    canonical_gradients: Sequence[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Project only a conflicting Adam direction onto canonical gradient's tangent plane.

    Update is ``delta = -lr * direction``.  First-order canonical change is
    ``g_canon dot delta = -lr * dot(g_canon, direction)``.  Thus non-increase to
    first order requires ``dot(g_canon, direction) >= 0``.
    """
    if len(raw_directions) != len(canonical_gradients):
        raise RuntimeError("R11_R33_PROJECTION_TENSOR_COUNT_DRIFT")
    for d, g in zip(raw_directions, canonical_gradients):
        if d.shape != g.shape or d.device != g.device or d.dtype != g.dtype:
            raise RuntimeError("R11_R33_PROJECTION_SHAPE_DEVICE_DTYPE_DRIFT")
    raw64 = _flatten64(raw_directions)
    guard64 = _flatten64(canonical_gradients)
    raw_norm = torch.linalg.vector_norm(raw64)
    guard_norm = torch.linalg.vector_norm(guard64)
    if float(raw_norm.item()) <= 0.0 or float(guard_norm.item()) <= 0.0:
        raise RuntimeError("R11_R33_ZERO_UPDATE_OR_GUARD_NORM")
    dot_before = torch.dot(raw64, guard64)
    projected = bool(float(dot_before.item()) < 0.0)
    if projected:
        denom = torch.dot(guard64, guard64)
        safe64 = raw64 - (dot_before / denom) * guard64
    else:
        safe64 = raw64.clone()

    safe = _unflatten_like(safe64, raw_directions)
    safe64_observed = _flatten64(safe)
    dot_after = torch.dot(safe64_observed, guard64)
    safe_norm = torch.linalg.vector_norm(safe64_observed)
    product = float((safe_norm * guard_norm).item())
    tolerance = R33_PROJECTION_REL_TOL * max(product, 1.0)
    if float(dot_after.item()) < -tolerance:
        raise RuntimeError(
            f"R11_R33_PROJECTED_DIRECTION_STILL_CONFLICTING:{float(dot_after.item())}:{tolerance}"
        )
    cosine_before = float((dot_before / (raw_norm * guard_norm)).item())
    cosine_after = (
        float((dot_after / (safe_norm * guard_norm)).item())
        if float(safe_norm.item()) > 0.0
        else 0.0
    )
    return safe, {
        "projected": projected,
        "dot_before": float(dot_before.item()),
        "dot_after": float(dot_after.item()),
        "cosine_before": cosine_before,
        "cosine_after": cosine_after,
        "raw_adam_direction_l2_norm": float(raw_norm.item()),
        "safe_adam_direction_l2_norm": float(safe_norm.item()),
        "canonical_guard_gradient_l2_norm": float(guard_norm.item()),
        "first_order_canonical_change_raw_per_unit_lr": float(-dot_before.item()),
        "first_order_canonical_change_safe_per_unit_lr": float(-dot_after.item()),
        "projection_relative_tolerance": float(R33_PROJECTION_REL_TOL),
    }


def summarize_projection_steps_r33(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != R33_STEPS:
        raise RuntimeError("R11_R33_PROJECTION_STEP_COUNT_DRIFT")
    projected = [r for r in rows if bool(r["projected"])]
    after = [float(r["dot_after"]) for r in rows]
    before = [float(r["dot_before"]) for r in rows]
    if any(not math.isfinite(x) for x in before + after):
        raise RuntimeError("R11_R33_NONFINITE_PROJECTION_STEP_SUMMARY")
    return {
        "step_count": len(rows),
        "projected_step_count": len(projected),
        "unprojected_step_count": len(rows) - len(projected),
        "minimum_dot_before": float(min(before)),
        "maximum_dot_before": float(max(before)),
        "minimum_dot_after": float(min(after)),
        "maximum_abs_negative_dot_after": float(max([max(0.0, -x) for x in after], default=0.0)),
        "all_safe_directions_nonconflicting_within_tolerance": True,
    }
