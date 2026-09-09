from __future__ import annotations

"""Diagnostics for R11 Science-G0 historical learning connectivity.

This module observes the already-qualified R11 Student objective and training
receipt.  It does not define Teacher, Evidence, Permission, Physics, promotion,
or a new scientific verdict.  The central distinction is intentional:

* connectivity: did admitted historical evidence create legal Student gradients,
  parameter updates, and measurable behavior change?
* scientific quality: did those updates improve a clean evaluation objective?

A connectivity PASS is therefore not an alpha/market-information qualification.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .training_runtime_r11 import (
    AUTHORIZED_GRADIENT_OWNERS_R11,
    PreparedEvidenceR11,
    forward_prepared_r11,
)

SCHEMA = "CB16_R11_SCIENCE_FEEDBACK_DIAGNOSTICS_V1"
SMOOTH_L1_BETA_R11 = 0.05


def _sha256_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _finite_positive_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0.0
        for v in value.values()
    )


def validate_teacher_targets_r11(prepared: PreparedEvidenceR11) -> dict[str, Any]:
    """Independently check the probabilistic targets consumed by the Student."""

    prepared.validate()
    p = prepared.direction_target_probs.detach()
    r = prepared.requested_risk_target.detach()
    w = prepared.group_weight.detach()
    if p.requires_grad or r.requires_grad or w.requires_grad:
        raise RuntimeError("R11_FEEDBACK_TARGET_AUTOGRAD_TAINT")
    if torch.any(p < 0.0).item():
        raise RuntimeError("R11_FEEDBACK_DIRECTION_TARGET_NEGATIVE")
    sums = p.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0.0):
        raise RuntimeError("R11_FEEDBACK_DIRECTION_TARGET_NOT_NORMALIZED")
    if torch.any((r < 0.0) | (r > 1.0)).item():
        raise RuntimeError("R11_FEEDBACK_REQUESTED_RISK_TARGET_OUT_OF_RANGE")
    if torch.any(w <= 0.0).item():
        raise RuntimeError("R11_FEEDBACK_GROUP_WEIGHT_NONPOSITIVE")
    return {
        "rows": prepared.rows,
        "independent_dependence_groups": len(set(prepared.dependence_group_ids)),
        "direction_target_min": float(p.min().detach().cpu()),
        "direction_target_max": float(p.max().detach().cpu()),
        "requested_risk_target_min": float(r.min().detach().cpu()),
        "requested_risk_target_max": float(r.max().detach().cpu()),
        "group_weight_min": float(w.min().detach().cpu()),
        "group_weight_max": float(w.max().detach().cpu()),
        "targets_detached_from_autograd": True,
    }


def independent_loss_breakdown_r11(
    model: torch.nn.Module,
    prepared: PreparedEvidenceR11,
) -> dict[str, float]:
    """Recompute the frozen Student formula without calling the runtime loss helper.

    L_dir = -sum_c q_c log softmax(z)_c
    L_size = SmoothL1(r_hat, r_teacher; beta=.05)
    L = sum_i w_i (L_dir_i + L_size_i) / sum_i w_i
    """

    validate_teacher_targets_r11(prepared)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            out = forward_prepared_r11(model, prepared)
            q = prepared.direction_target_probs.detach()
            risk_target = prepared.requested_risk_target.detach()
            weight = prepared.group_weight.detach()
            logp = F.log_softmax(out["direction_logits"], dim=-1)
            row_direction = -(q * logp).sum(dim=-1)
            row_sizing = F.smooth_l1_loss(
                out["requested_risk_raw"],
                risk_target,
                reduction="none",
                beta=SMOOTH_L1_BETA_R11,
            )
            denom = weight.sum()
            if not torch.isfinite(denom).item() or float(denom.detach().cpu()) <= 0.0:
                raise RuntimeError("R11_FEEDBACK_INVALID_WEIGHT_DENOMINATOR")
            direction = (row_direction * weight).sum() / denom
            sizing = (row_sizing * weight).sum() / denom
            total = ((row_direction + row_sizing) * weight).sum() / denom
            vals = {
                "loss": float(total.detach().cpu()),
                "direction_loss": float(direction.detach().cpu()),
                "sizing_loss": float(sizing.detach().cpu()),
            }
    finally:
        model.train(was_training)
    if not all(math.isfinite(v) for v in vals.values()):
        raise RuntimeError(f"R11_FEEDBACK_NONFINITE_INDEPENDENT_LOSS:{vals}")
    return vals


def capture_behavior_r11(
    model: torch.nn.Module,
    prepared: PreparedEvidenceR11,
) -> dict[str, Any]:
    """Capture Student-owned belief/intent outputs on a fixed probe set."""

    prepared.validate()
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            out = forward_prepared_r11(model, prepared)
            probs = torch.softmax(out["direction_logits"], dim=-1)
            risk = out["requested_risk_raw"]
            action = model.compose_action(out)
            directions = action["direction"]
            p = probs.detach().cpu().numpy().astype(np.float64, copy=True)
            r = risk.detach().cpu().numpy().astype(np.float64, copy=True)
            d = directions.detach().cpu().numpy().astype(np.int64, copy=True)
    finally:
        model.train(was_training)
    if p.shape != (prepared.rows, 3) or r.shape != (prepared.rows,) or d.shape != (prepared.rows,):
        raise RuntimeError(f"R11_FEEDBACK_BEHAVIOR_SHAPE_DRIFT:{p.shape}:{r.shape}:{d.shape}")
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise RuntimeError("R11_FEEDBACK_NONFINITE_BEHAVIOR")
    payload = {
        "direction_probs": p.tolist(),
        "requested_risk": r.tolist(),
        "direction": d.tolist(),
    }
    return {
        **payload,
        "rows": int(prepared.rows),
        "sha256": _sha256_obj(payload),
        "mean_direction_probs": p.mean(axis=0).tolist(),
        "mean_requested_risk": float(r.mean()),
        "long_rate": float(np.mean(d == 1)),
        "flat_rate": float(np.mean(d == 0)),
        "short_rate": float(np.mean(d == -1)),
    }


def behavior_delta_r11(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    if int(before.get("rows", -1)) != int(after.get("rows", -2)):
        raise RuntimeError("R11_FEEDBACK_BEHAVIOR_ROW_MISMATCH")
    pb = np.asarray(before["direction_probs"], dtype=np.float64)
    pa = np.asarray(after["direction_probs"], dtype=np.float64)
    rb = np.asarray(before["requested_risk"], dtype=np.float64)
    ra = np.asarray(after["requested_risk"], dtype=np.float64)
    db = np.asarray(before["direction"], dtype=np.int64)
    da = np.asarray(after["direction"], dtype=np.int64)
    if pb.shape != pa.shape or rb.shape != ra.shape or db.shape != da.shape:
        raise RuntimeError("R11_FEEDBACK_BEHAVIOR_SHAPE_MISMATCH")
    direction_l1 = np.abs(pa - pb).sum(axis=1)
    risk_abs = np.abs(ra - rb)
    any_numeric_change = bool(
        float(direction_l1.max(initial=0.0)) > 0.0 or float(risk_abs.max(initial=0.0)) > 0.0
    )
    return {
        "mean_direction_probability_l1": float(direction_l1.mean()) if len(direction_l1) else 0.0,
        "max_direction_probability_l1": float(direction_l1.max(initial=0.0)),
        "mean_requested_risk_abs_delta": float(risk_abs.mean()) if len(risk_abs) else 0.0,
        "max_requested_risk_abs_delta": float(risk_abs.max(initial=0.0)),
        "discrete_direction_change_rate": float(np.mean(db != da)) if len(db) else 0.0,
        "behavior_fingerprint_changed": str(before.get("sha256")) != str(after.get("sha256")),
        "any_numeric_behavior_change": any_numeric_change,
    }


def _assert_metric_close(label: str, observed: Mapping[str, Any], independent: Mapping[str, Any], atol: float) -> None:
    for key in ("loss", "direction_loss", "sizing_loss"):
        a = float(observed[key])
        b = float(independent[key])
        if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > atol:
            raise RuntimeError(f"R11_FEEDBACK_FORMULA_MISMATCH:{label}:{key}:{a}:{b}:ATOL={atol}")


def audit_training_feedback_r11(
    *,
    training_receipt: Mapping[str, Any],
    independent_before: Mapping[str, Any],
    independent_after: Mapping[str, Any],
    behavior_delta: Mapping[str, Any],
    formula_atol: float = 1e-6,
) -> dict[str, Any]:
    """Fail closed on broken learning connectivity; report quality separately."""

    if training_receipt.get("schema") != "CB16_R11_CHALLENGER_TRAINING_RECEIPT_V1":
        raise RuntimeError("R11_FEEDBACK_TRAINING_RECEIPT_SCHEMA_MISMATCH")
    if training_receipt.get("amp") is not False or training_receipt.get("dtype") != "torch.float32":
        raise RuntimeError("R11_FEEDBACK_TRAINING_NUMERIC_IDENTITY_DRIFT")
    _assert_metric_close(
        "BEFORE", training_receipt["validation_before"], independent_before, float(formula_atol)
    )
    _assert_metric_close(
        "AFTER", training_receipt["validation_after"], independent_after, float(formula_atol)
    )

    owners = frozenset(str(x) for x in training_receipt.get("gradient_owner_set_last_step", ()))
    if owners != AUTHORIZED_GRADIENT_OWNERS_R11:
        raise RuntimeError(
            f"R11_FEEDBACK_GRADIENT_OWNER_SET_DRIFT:{sorted(owners)}:{sorted(AUTHORIZED_GRADIENT_OWNERS_R11)}"
        )
    grad_norms = training_receipt.get("gradient_group_norms_last_step")
    update_norms = training_receipt.get("update_group_norms")
    if not _finite_positive_mapping(grad_norms):
        raise RuntimeError(f"R11_FEEDBACK_GRADIENT_DISCONNECT:{grad_norms}")
    if not _finite_positive_mapping(update_norms):
        raise RuntimeError(f"R11_FEEDBACK_PARAMETER_GROUP_NOT_UPDATED:{update_norms}")
    parameter_delta = float(training_receipt.get("parameter_l2_delta", 0.0))
    if not math.isfinite(parameter_delta) or parameter_delta <= 0.0:
        raise RuntimeError(f"R11_FEEDBACK_PARAMETER_DELTA_NOT_POSITIVE:{parameter_delta}")
    if int(training_receipt.get("optimizer_steps", 0)) <= 0:
        raise RuntimeError("R11_FEEDBACK_ZERO_OPTIMIZER_STEPS")
    if not bool(behavior_delta.get("behavior_fingerprint_changed")) or not bool(
        behavior_delta.get("any_numeric_behavior_change")
    ):
        raise RuntimeError(f"R11_FEEDBACK_PARAMETER_TO_BEHAVIOR_PATH_DEAD:{behavior_delta}")

    before_loss = float(independent_before["loss"])
    after_loss = float(independent_after["loss"])
    delta = after_loss - before_loss
    tolerance = max(1e-9, abs(before_loss) * 1e-9)
    if delta < -tolerance:
        learning_direction = "IMPROVED_ON_FIXED_VALIDATION_TEACHER_OBJECTIVE"
    elif delta > tolerance:
        learning_direction = "WORSENED_ON_FIXED_VALIDATION_TEACHER_OBJECTIVE"
    else:
        learning_direction = "NUMERICALLY_FLAT_ON_FIXED_VALIDATION_TEACHER_OBJECTIVE"

    return {
        "schema": SCHEMA,
        "status": "R11_HISTORICAL_LEARNING_CONNECTIVITY_PASS",
        "connectivity_pass": True,
        "learning_direction": learning_direction,
        "validation_loss_before": before_loss,
        "validation_loss_after": after_loss,
        "validation_loss_delta_after_minus_before": delta,
        "formula_recomputed_independently": True,
        "formula_atol": float(formula_atol),
        "gradient_owner_set": sorted(owners),
        "gradient_group_norms_last_step": dict(grad_norms),
        "update_group_norms": dict(update_norms),
        "parameter_l2_delta": parameter_delta,
        "behavior_delta": dict(behavior_delta),
        "teacher_targets_detached": True,
        "teacher_future_permission_physics_gradient_ownership": "FORBIDDEN__NOT_CLAIMED",
        "scientific_market_information_qualification_claimed": False,
        "profitability_or_alpha_claimed": False,
        "qcrps_claimed": False,
        "qcrps_note": "Current R11 Student head does not emit the conditional utility quantile/CDF object required for a qCRPS qualification claim.",
    }


__all__ = [
    "SCHEMA",
    "SMOOTH_L1_BETA_R11",
    "audit_training_feedback_r11",
    "behavior_delta_r11",
    "capture_behavior_r11",
    "independent_loss_breakdown_r11",
    "validate_teacher_targets_r11",
]
