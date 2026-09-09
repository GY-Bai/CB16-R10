from __future__ import annotations

"""Science diagnostics for the R11 distributional-feedback boundary.

This module does not create a new Teacher, objective, permission rule, or
scientific verdict.  It asks a narrower mechanistic question:

    Which parts of the admitted probabilistic Teacher law are actually on the
    current Student loss/gradient path?

The distinction follows the frozen CB16 V6.1 semantics:
TRUTH != BELIEF != DECISION != PERMISSION.
"""

from dataclasses import replace
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .training_runtime_r11 import (
    PreparedEvidenceR11,
    forward_prepared_r11,
    gradient_group_norms_r11,
    student_loss_from_outputs_r11,
)

SCHEMA = "CB16_R11_DISTRIBUTIONAL_FEEDBACK_BOUNDARY_DIAGNOSTICS_V1"


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _packed_sha256(prepared: PreparedEvidenceR11) -> str:
    x = prepared.packed.detach().cpu().contiguous().numpy()
    return _sha256_bytes(x.tobytes(order="C"))


def audit_teacher_action_laws_r11(evidence: Sequence[Any]) -> dict[str, Any]:
    admitted = [e for e in evidence if bool(e.admission.admitted)]
    if not admitted:
        raise RuntimeError("R11_R2_NO_ADMITTED_TEACHER_EVIDENCE")
    action_counts: list[int] = []
    quantile_levels_seen: set[tuple[float, ...]] = set()
    min_effective_n = float("inf")
    min_unique_groups = 1 << 60
    nonmonotone: list[str] = []
    nonfinite: list[str] = []
    for e in admitted:
        laws = tuple(e.action_laws)
        action_counts.append(len(laws))
        if len(laws) != 9:
            raise RuntimeError(f"R11_R2_ACTION_LAW_GRID_DRIFT:{e.parent_id}:{len(laws)}")
        grid = {(int(l.direction), float(l.requested_risk)) for l in laws}
        if len(grid) != 9 or (0, 0.0) not in grid or {d for d, _ in grid} != {-1, 0, 1}:
            raise RuntimeError(f"R11_R2_ACTION_LAW_GRID_INVALID:{e.parent_id}:{sorted(grid)}")
        for law in laws:
            ql = np.asarray(law.quantile_levels, dtype=np.float64)
            q = np.asarray(law.quantiles, dtype=np.float64)
            quantile_levels_seen.add(tuple(float(x) for x in ql))
            if (
                ql.ndim != 1
                or q.ndim != 1
                or len(ql) != len(q)
                or len(q) < 3
                or np.any(np.diff(ql) <= 0.0)
                or np.any((ql <= 0.0) | (ql >= 1.0))
            ):
                raise RuntimeError(f"R11_R2_QUANTILE_GRID_INVALID:{e.parent_id}")
            if not (
                np.isfinite(q).all()
                and math.isfinite(float(law.mean_utility))
                and math.isfinite(float(law.std_utility))
                and math.isfinite(float(law.effective_dependence_n))
            ):
                nonfinite.append(f"{e.parent_id}:{law.direction}:{law.requested_risk}")
            if np.any(np.diff(q) < -1e-12):
                nonmonotone.append(f"{e.parent_id}:{law.direction}:{law.requested_risk}")
            min_effective_n = min(min_effective_n, float(law.effective_dependence_n))
            min_unique_groups = min(min_unique_groups, int(law.unique_dependence_groups))
    if nonfinite:
        raise RuntimeError(f"R11_R2_NONFINITE_PREDICTIVE_LAW:{nonfinite[:8]}")
    if nonmonotone:
        raise RuntimeError(f"R11_R2_QUANTILE_CROSSING:{nonmonotone[:8]}")
    if len(quantile_levels_seen) != 1:
        raise RuntimeError(f"R11_R2_QUANTILE_LEVEL_DRIFT:{sorted(quantile_levels_seen)}")
    levels = next(iter(quantile_levels_seen))
    return {
        "schema": SCHEMA,
        "admitted_evidence": len(admitted),
        "action_laws_per_evidence": sorted(set(action_counts)),
        "quantile_levels": list(levels),
        "quantiles_monotone": True,
        "all_laws_finite": True,
        "minimum_effective_dependence_n": float(min_effective_n),
        "minimum_unique_dependence_groups_per_law": int(min_unique_groups),
    }


def tail_only_law_perturbation_r11(
    evidence: Sequence[Any], *, scale_multiplier: float = 4.0
) -> list[Any]:
    """Change lower-tail quantiles while preserving mean/std and Student targets."""
    if scale_multiplier <= 0.0:
        raise ValueError("scale_multiplier")
    out: list[Any] = []
    for e in evidence:
        if not bool(e.admission.admitted):
            out.append(e)
            continue
        new_laws = []
        for law in e.action_laws:
            q = np.asarray(law.quantiles, dtype=np.float64).copy()
            if len(q) < 3:
                raise RuntimeError("R11_R2_QUANTILE_GRID_TOO_SHORT_FOR_TAIL_CANARY")
            spread = max(
                1e-5,
                abs(float(law.std_utility)),
                abs(float(q[-1] - q[0])) / 4.0,
            )
            shift = float(scale_multiplier) * spread
            q[0] = min(q[0] - shift, q[1] - 2e-12)
            q[1] = min(q[1] - 0.5 * shift, q[2] - 1e-12)
            if q[0] > q[1]:
                q[0] = q[1] - max(1e-12, 0.5 * shift)
            new_laws.append(replace(law, quantiles=tuple(float(x) for x in q)))
        changed = replace(e, action_laws=tuple(new_laws))
        if changed.direction_target_probs != e.direction_target_probs:
            raise AssertionError("R11_R2_TAIL_CANARY_CHANGED_DIRECTION_TARGET")
        if changed.requested_risk_target != e.requested_risk_target:
            raise AssertionError("R11_R2_TAIL_CANARY_CHANGED_RISK_TARGET")
        out.append(changed)
    return out


def projected_target_perturbation_r11(
    evidence: Sequence[Any], *, probability_shift: float = 0.02, risk_shift: float = 0.03
) -> list[Any]:
    """Positive control: change only fields known to be consumed by Student loss."""
    if probability_shift <= 0.0 or risk_shift <= 0.0:
        raise ValueError("positive shifts required")
    out: list[Any] = []
    for e in evidence:
        if not bool(e.admission.admitted):
            out.append(e)
            continue
        p = np.asarray(e.direction_target_probs, dtype=np.float64).copy()
        lo = int(np.argmin(p))
        hi = int(np.argmax(p))
        if hi == lo:
            hi, lo = 0, 1
        delta = min(float(probability_shift), max(float(p[lo]) * 0.5, 1e-6))
        p[lo] -= delta
        p[hi] += delta
        p /= p.sum()
        risk = min(1.0, max(0.0, float(e.requested_risk_target) + float(risk_shift)))
        if abs(risk - float(e.requested_risk_target)) < 1e-12:
            risk = max(0.0, float(e.requested_risk_target) - float(risk_shift))
        out.append(
            replace(
                e,
                direction_target_probs=tuple(float(x) for x in p),
                requested_risk_target=float(risk),
            )
        )
    return out


def _gradient_probe(
    model: torch.nn.Module, prepared: PreparedEvidenceR11
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    prepared.validate()
    model.zero_grad(set_to_none=True)
    outputs = forward_prepared_r11(model, prepared)
    loss = student_loss_from_outputs_r11(outputs, prepared)
    loss.loss.backward()
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            raise RuntimeError(f"R11_R2_AUTHORIZED_PARAMETER_GRADIENT_MISSING:{name}")
        grads[name] = param.grad.detach().cpu().double().clone()
    group_norms = gradient_group_norms_r11(model)
    model.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.loss.detach().cpu()),
        "direction_loss": float(loss.direction_loss.detach().cpu()),
        "sizing_loss": float(loss.sizing_loss.detach().cpu()),
        "gradient_group_norms": group_norms,
        "prepared_evidence_hash": prepared.evidence_hash,
        "packed_sha256": _packed_sha256(prepared),
    }, grads


def compare_student_projection_paths_r11(
    *,
    model: torch.nn.Module,
    original: PreparedEvidenceR11,
    variant: PreparedEvidenceR11,
    atol: float = 1e-9,
) -> dict[str, Any]:
    a, ga = _gradient_probe(model, original)
    b, gb = _gradient_probe(model, variant)
    if ga.keys() != gb.keys():
        raise RuntimeError("R11_R2_GRADIENT_PARAMETER_SET_DRIFT")
    total = 0.0
    max_abs = 0.0
    for name in ga:
        d = gb[name] - ga[name]
        total += float(torch.sum(d * d))
        max_abs = max(max_abs, float(torch.max(torch.abs(d)).item()))
    grad_l2 = math.sqrt(total)
    loss_delta = float(b["loss"] - a["loss"])
    packed_equal = bool(torch.equal(original.packed.detach().cpu(), variant.packed.detach().cpu()))
    return {
        "original": a,
        "variant": b,
        "packed_tensor_equal": packed_equal,
        "packed_sha256_equal": a["packed_sha256"] == b["packed_sha256"],
        "prepared_evidence_identity_equal": original.evidence_hash == variant.evidence_hash,
        "loss_delta_variant_minus_original": loss_delta,
        "loss_equal_within_atol": abs(loss_delta) <= float(atol),
        "gradient_l2_delta": grad_l2,
        "gradient_max_abs_delta": max_abs,
        "gradient_equal_within_atol": max_abs <= float(atol),
        "atol": float(atol),
    }


def classify_distributional_boundary_r11(
    *,
    teacher_law_audit: Mapping[str, Any],
    tail_projection: Mapping[str, Any],
    target_projection_control: Mapping[str, Any],
) -> dict[str, Any]:
    if not teacher_law_audit.get("quantiles_monotone"):
        raise RuntimeError("R11_R2_TEACHER_LAW_INTEGRITY_NOT_PASS")
    tail_identity_changed = not bool(tail_projection["prepared_evidence_identity_equal"])
    tail_pack_unchanged = bool(tail_projection["packed_tensor_equal"])
    tail_grad_unchanged = bool(tail_projection["gradient_equal_within_atol"])
    target_grad_changed = not bool(target_projection_control["gradient_equal_within_atol"])
    target_pack_changed = not bool(target_projection_control["packed_tensor_equal"])
    if not tail_identity_changed:
        raise RuntimeError("R11_R2_TAIL_CANARY_DID_NOT_CHANGE_EVIDENCE_IDENTITY")
    if not target_pack_changed or not target_grad_changed:
        raise RuntimeError("R11_R2_POSITIVE_CONTROL_STUDENT_TARGET_PATH_DEAD")
    if tail_pack_unchanged and tail_grad_unchanged:
        diagnosis = (
            "DISTRIBUTIONAL_ACTION_LAW_PRESERVED_AS_EVIDENCE_"
            "BUT_NOT_ON_CURRENT_STUDENT_GRADIENT_PATH"
        )
    else:
        diagnosis = "DISTRIBUTIONAL_ACTION_LAW_REACHES_CURRENT_STUDENT_GRADIENT_PATH"
    return {
        "schema": SCHEMA,
        "status": "R11_DISTRIBUTIONAL_FEEDBACK_BOUNDARY_AUDIT_COMPLETE",
        "diagnosis": diagnosis,
        "teacher_distributional_law_integrity": "PASS",
        "tail_only_law_changes_evidence_identity": tail_identity_changed,
        "tail_only_law_changes_student_packed_tensor": not tail_pack_unchanged,
        "tail_only_law_changes_student_gradient": not tail_grad_unchanged,
        "projected_target_positive_control_changes_packed_tensor": target_pack_changed,
        "projected_target_positive_control_changes_gradient": target_grad_changed,
        "architecture_revision_authorized": False,
        "qr_dqn_conversion_authorized": False,
        "new_scientific_verdict_created": False,
    }


__all__ = [
    "SCHEMA",
    "audit_teacher_action_laws_r11",
    "tail_only_law_perturbation_r11",
    "projected_target_perturbation_r11",
    "compare_student_projection_paths_r11",
    "classify_distributional_boundary_r11",
]
