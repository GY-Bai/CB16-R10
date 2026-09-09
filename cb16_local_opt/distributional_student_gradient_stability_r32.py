from __future__ import annotations

"""R11 Science G0 R3.2 — gradient geometry and fixed-seed stability helpers.

R3.2 is shadow-only.  It measures how the full Teacher action-law distributional
objective pushes a disposable Shared Decision Core relative to the already-frozen
canonical Direction/Requested-Risk objective.  It does not combine the objectives,
change production gradient ownership, or authorize a production architecture.
"""

import math
import statistics
from typing import Any, Mapping, Sequence

import torch

from cb16_local_opt.distributional_student_shadow_r3 import truncated_quantile_w1_loss_r3
from cb16_local_opt.training_runtime_r11 import student_loss_from_outputs_r11


R32_RUNTIME = "CB16_R11_DISTRIBUTIONAL_STUDENT_GRADIENT_STABILITY_R3_2_V1"
R32_SEEDS: tuple[int, ...] = (31_337, 271_828, 314_159, 161_803, 424_242)
R32_STEPS = 32
R32_LR = 2e-3
R32_WEIGHT_DECAY = 0.0
R32_GRAD_CLIP = 10.0


def _shared_params(model: torch.nn.Module) -> tuple[torch.nn.Parameter, ...]:
    if not hasattr(model, "shared_core"):
        raise RuntimeError("R11_R32_SHARED_CORE_MISSING")
    params = tuple(model.shared_core.parameters())
    if not params:
        raise RuntimeError("R11_R32_SHARED_CORE_EMPTY")
    return params


def _flatten_grad_tuple(grads: Sequence[torch.Tensor], *, code: str) -> torch.Tensor:
    if not grads:
        raise RuntimeError(f"{code}:EMPTY")
    parts: list[torch.Tensor] = []
    for g in grads:
        if g is None:
            raise RuntimeError(f"{code}:MISSING")
        if not bool(torch.isfinite(g).all().item()):
            raise RuntimeError(f"{code}:NONFINITE")
        parts.append(g.reshape(-1))
    out = torch.cat(parts)
    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError(f"{code}:NONFINITE_FLAT")
    return out


def _geometry(a: torch.Tensor, b: torch.Tensor, *, code: str) -> dict[str, float]:
    if a.shape != b.shape:
        raise RuntimeError(f"{code}:SHAPE_DRIFT")
    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)
    if not bool(torch.isfinite(na).item()) or not bool(torch.isfinite(nb).item()):
        raise RuntimeError(f"{code}:NONFINITE_NORM")
    fa = float(na.item()); fb = float(nb.item())
    if fa <= 0.0 or fb <= 0.0:
        raise RuntimeError(f"{code}:ZERO_GRADIENT_NORM")
    dot = torch.dot(a, b)
    cosine = dot / (na * nb)
    if not bool(torch.isfinite(cosine).item()):
        raise RuntimeError(f"{code}:NONFINITE_COSINE")
    return {
        "distributional_l2_norm": fa,
        "canonical_l2_norm": fb,
        "canonical_over_distributional_norm_ratio": float(fb / fa),
        "dot_product": float(dot.item()),
        "cosine_similarity": float(cosine.item()),
    }


def shared_core_gradient_geometry_r32(
    *,
    disposable_model: torch.nn.Module,
    distributional_head: torch.nn.Module,
    distributional_batch: Any,
    canonical_prepared_batch: Any,
) -> dict[str, Any]:
    """Measure initial Shared-Core gradient geometry without updating any parameter."""
    params = _shared_params(disposable_model)
    for name, p in disposable_model.named_parameters():
        expected = name.startswith("shared_core")
        if bool(p.requires_grad) != expected:
            raise RuntimeError(f"R11_R32_DISPOSABLE_TRAINABILITY_DRIFT:{name}:{p.requires_grad}")
        p.grad = None
    for p in distributional_head.parameters():
        p.grad = None

    outputs = disposable_model(
        distributional_batch.operator48,
        distributional_batch.medium48,
        distributional_batch.account6,
    )
    pred = distributional_head(outputs["shared"])
    dist_loss = truncated_quantile_w1_loss_r3(
        pred,
        distributional_batch.teacher_quantiles,
        distributional_batch.quantile_levels,
        distributional_batch.group_weight,
    )
    dist_grads = torch.autograd.grad(dist_loss, params, retain_graph=False, allow_unused=False)
    dist_flat = _flatten_grad_tuple(dist_grads, code="R11_R32_DISTRIBUTIONAL_GRADIENT")

    def canonical_grad(component: str) -> tuple[torch.Tensor, float]:
        out = disposable_model(
            canonical_prepared_batch.operator48,
            canonical_prepared_batch.medium48,
            canonical_prepared_batch.account6,
        )
        losses = student_loss_from_outputs_r11(out, canonical_prepared_batch)
        loss = {
            "total": losses.loss,
            "direction": losses.direction_loss,
            "sizing": losses.sizing_loss,
        }[component]
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=False)
        return _flatten_grad_tuple(grads, code=f"R11_R32_CANONICAL_{component.upper()}_GRADIENT"), float(loss.detach().item())

    total_grad, total_loss = canonical_grad("total")
    direction_grad, direction_loss = canonical_grad("direction")
    sizing_grad, sizing_loss = canonical_grad("sizing")

    if any(p.grad is not None for p in disposable_model.parameters()):
        raise RuntimeError("R11_R32_AUTOGRAD_MEASUREMENT_LEAKED_INTO_PARAM_GRAD")
    if any(p.grad is not None for p in distributional_head.parameters()):
        raise RuntimeError("R11_R32_AUTOGRAD_MEASUREMENT_LEAKED_INTO_HEAD_GRAD")

    return {
        "measurement_point": "INITIAL_BOOTSTRAP_BEFORE_ANY_R3_2_UPDATE",
        "shared_core_parameter_count": int(sum(p.numel() for p in params)),
        "distributional_loss": float(dist_loss.detach().item()),
        "canonical_loss": {
            "total": total_loss,
            "direction": direction_loss,
            "sizing": sizing_loss,
        },
        "distributional_vs_canonical_total": _geometry(
            dist_flat, total_grad, code="R11_R32_DIST_VS_CANONICAL_TOTAL"
        ),
        "distributional_vs_direction": _geometry(
            dist_flat, direction_grad, code="R11_R32_DIST_VS_DIRECTION"
        ),
        "distributional_vs_sizing": _geometry(
            dist_flat, sizing_grad, code="R11_R32_DIST_VS_SIZING"
        ),
        "parameter_update_performed": False,
    }


def summarize_seed_stability_r32(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(int(x["seed"]) for x in rows) != R32_SEEDS:
        raise RuntimeError("R11_R32_SEED_SET_OR_ORDER_DRIFT")
    if len(rows) != len(R32_SEEDS):
        raise RuntimeError("R11_R32_SEED_COUNT_DRIFT")

    shared_minus_head = [float(x["shared_minus_head_validation_distributional_loss"]) for x in rows]
    decision_delta = [float(x["canonical_decision_total_delta"]) for x in rows]
    direction_delta = [float(x["canonical_direction_loss_delta"]) for x in rows]
    sizing_delta = [float(x["canonical_sizing_loss_delta"]) for x in rows]
    total_cos = [float(x["gradient_geometry"]["distributional_vs_canonical_total"]["cosine_similarity"]) for x in rows]
    direction_cos = [float(x["gradient_geometry"]["distributional_vs_direction"]["cosine_similarity"]) for x in rows]
    sizing_cos = [float(x["gradient_geometry"]["distributional_vs_sizing"]["cosine_similarity"]) for x in rows]
    direction_changes = [int(x["hypothetical_direction_changed_rows"]) for x in rows]
    mean_risk_delta = [float(x["mean_abs_requested_risk_delta"]) for x in rows]

    scalar_sets = [
        shared_minus_head,
        decision_delta,
        direction_delta,
        sizing_delta,
        total_cos,
        direction_cos,
        sizing_cos,
        mean_risk_delta,
    ]
    if any(not math.isfinite(v) for xs in scalar_sets for v in xs):
        raise RuntimeError("R11_R32_NONFINITE_SEED_AGGREGATE_INPUT")

    def stats(xs: Sequence[float]) -> dict[str, float]:
        return {
            "min": float(min(xs)),
            "max": float(max(xs)),
            "mean": float(statistics.fmean(xs)),
            "median": float(statistics.median(xs)),
        }

    shared_better = sum(v < 0.0 for v in shared_minus_head)
    decision_better = sum(v < 0.0 for v in decision_delta)
    direction_better = sum(v < 0.0 for v in direction_delta)
    sizing_better = sum(v < 0.0 for v in sizing_delta)
    total_positive_cos = sum(v > 0.0 for v in total_cos)

    if shared_better == len(rows) and decision_better == len(rows):
        architecture_pattern = "ALL_SEEDS_SHARED_BETTER_DISTRIBUTIONAL_AND_LOWER_CANONICAL_DECISION_LOSS"
    elif shared_better == len(rows):
        architecture_pattern = "ALL_SEEDS_SHARED_BETTER_DISTRIBUTIONAL__CANONICAL_DECISION_MIXED"
    elif shared_better > len(rows) // 2:
        architecture_pattern = "MAJORITY_SEEDS_SHARED_BETTER_DISTRIBUTIONAL"
    else:
        architecture_pattern = "SHARED_CORE_DISTRIBUTIONAL_ADVANTAGE_NOT_STABLE_ACROSS_SEEDS"

    return {
        "seed_count": len(rows),
        "seed_set": list(R32_SEEDS),
        "architecture_pattern": architecture_pattern,
        "shared_core_better_distributional_seed_count": int(shared_better),
        "canonical_total_loss_lower_seed_count": int(decision_better),
        "canonical_direction_loss_lower_seed_count": int(direction_better),
        "canonical_sizing_loss_lower_seed_count": int(sizing_better),
        "positive_distributional_vs_canonical_total_cosine_seed_count": int(total_positive_cos),
        "shared_minus_head_validation_distributional_loss": stats(shared_minus_head),
        "canonical_decision_total_delta": stats(decision_delta),
        "canonical_direction_loss_delta": stats(direction_delta),
        "canonical_sizing_loss_delta": stats(sizing_delta),
        "gradient_cosine_distributional_vs_canonical_total": stats(total_cos),
        "gradient_cosine_distributional_vs_direction": stats(direction_cos),
        "gradient_cosine_distributional_vs_sizing": stats(sizing_cos),
        "hypothetical_direction_changed_rows": {
            "min": int(min(direction_changes)),
            "max": int(max(direction_changes)),
            "median": float(statistics.median(direction_changes)),
            "mean": float(statistics.fmean(direction_changes)),
        },
        "mean_abs_requested_risk_delta": stats(mean_risk_delta),
        "role": "ARCHITECTURE_STABILITY_AND_GRADIENT_GEOMETRY_ONLY__NOT_MARKET_INFORMATION_ADJUDICATION",
    }
