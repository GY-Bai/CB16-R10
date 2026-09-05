from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch

FORBIDDEN_STUDENT_TAINTS = {
    "TEACHER_ONLY",
    "FUTURE_ONLY",
    "PERMISSION_ONLY",
    "TRUTH_ONLY",
    "SHADOW_ONLY",
}

FORBIDDEN_HINDSIGHT_KEYS = {
    "BEST_ACTION",
    "CORRECT_DIRECTION",
    "REALIZED_WINNER",
    "HINDSIGHT_OPTIMAL_DIRECTION",
}


class ConformanceError(RuntimeError):
    pass


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def guard_action_authority(
    *,
    requested_direction: np.ndarray,
    requested_risk: np.ndarray,
    executable_direction: np.ndarray,
    executable_risk: np.ndarray,
    atol: float = 1e-12,
) -> None:
    rd = np.asarray(requested_direction)
    rr = np.asarray(requested_risk, dtype=float)
    ed = np.asarray(executable_direction)
    er = np.asarray(executable_risk, dtype=float)
    if not (rd.shape == rr.shape == ed.shape == er.shape):
        raise ConformanceError("ACTION_ARRAY_SHAPE_MISMATCH")
    if np.any((rr < 0) | (rr > 1) | (er < 0) | (er > 1)):
        raise ConformanceError("RISK_OUT_OF_RANGE")
    if np.any((rd == 0) & (np.abs(rr) > atol)):
        raise ConformanceError("REQUESTED_FLAT_NONZERO_RISK")
    if np.any((ed == 0) & (np.abs(er) > atol)):
        raise ConformanceError("EXECUTABLE_FLAT_NONZERO_RISK")
    if np.any(er - rr > atol):
        raise ConformanceError("SUPERVISOR_INCREASED_REQUESTED_RISK")


def guard_student_payload(payload: Any) -> None:
    """Recursively reject taint tags and hindsight-label keys in Student payloads."""
    if isinstance(payload, Mapping):
        for k, v in payload.items():
            ku = str(k).upper()
            if ku in FORBIDDEN_HINDSIGHT_KEYS:
                raise ConformanceError(f"FORBIDDEN_HINDSIGHT_LABEL:{ku}")
            if ku in {"TAINT", "VISIBILITY", "AUTHORITY_CLASS"} and str(v).upper() in FORBIDDEN_STUDENT_TAINTS:
                raise ConformanceError(f"FORBIDDEN_STUDENT_TAINT:{v}")
            guard_student_payload(v)
    elif isinstance(payload, (list, tuple)):
        for x in payload:
            guard_student_payload(x)


def guard_counterfactual_independence(*, parent_contexts: int, branches: int, claimed_independent: int) -> None:
    if claimed_independent > parent_contexts:
        raise ConformanceError(
            f"COUNTERFACTUAL_INDEPENDENCE_INFLATION parent={parent_contexts} "
            f"branches={branches} claimed={claimed_independent}"
        )


def guard_generation_lineage(
    *,
    trajectory_generation: int,
    trajectory_weight_hash: str,
    generator_generation: int,
    generator_weight_hash: str,
) -> None:
    if trajectory_generation != generator_generation:
        raise ConformanceError("TRAJECTORY_GENERATION_MISMATCH")
    if trajectory_weight_hash != generator_weight_hash:
        raise ConformanceError("TRAJECTORY_WEIGHT_HASH_MISMATCH")


@dataclass(frozen=True)
class GradientRule:
    prefix: str
    allowed: bool


def guard_gradient_ownership(
    model: torch.nn.Module,
    rules: Iterable[GradientRule],
    *,
    allowed_min_norm: float = 0.0,
    forbidden_atol: float = 0.0,
) -> dict[str, float]:
    rules = tuple(rules)
    report: dict[str, float] = {}
    matched = set()
    for name, p in model.named_parameters():
        norm = 0.0 if p.grad is None else float(p.grad.detach().norm().cpu())
        report[name] = norm
        for i, rule in enumerate(rules):
            if name.startswith(rule.prefix):
                matched.add(i)
                if rule.allowed:
                    if norm <= allowed_min_norm:
                        raise ConformanceError(f"AUTHORIZED_GRADIENT_MISSING:{name}:{norm}")
                else:
                    if norm > forbidden_atol:
                        raise ConformanceError(f"FORBIDDEN_GRADIENT_PRESENT:{name}:{norm}")
                break
    if len(matched) != len(rules):
        missing = [rules[i].prefix for i in range(len(rules)) if i not in matched]
        raise ConformanceError(f"GRADIENT_RULE_PREFIX_NOT_FOUND:{missing}")
    return report


def guard_empty_evidence_no_update(
    before: Mapping[str, torch.Tensor],
    after_model: torch.nn.Module,
    *,
    atol: float = 0.0,
) -> None:
    for name, p in after_model.named_parameters():
        if name not in before:
            raise ConformanceError(f"PARAMETER_SET_CHANGED:{name}")
        if not torch.allclose(before[name], p.detach().cpu(), atol=atol, rtol=0):
            raise ConformanceError(f"EMPTY_EVIDENCE_CHANGED_PARAMETER:{name}")


def snapshot_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


class EvidenceConsumptionGuard:
    """In-memory guard useful inside one trainer process.

    Durable exactly-once identity belongs in async_trajectory_pipeline.ExactlyOnceLedger.
    """

    def __init__(self):
        self._seen: dict[str, str] = {}

    def consume(self, evidence_id: str, content_hash: str) -> bool:
        old = self._seen.get(evidence_id)
        if old is None:
            self._seen[evidence_id] = content_hash
            return True
        if old != content_hash:
            raise ConformanceError("EVIDENCE_ID_CONTENT_CONFLICT")
        return False


def run_basic_guard_canaries() -> dict[str, str]:
    out = {}
    try:
        guard_action_authority(
            requested_direction=np.array([1]),
            requested_risk=np.array([0.2]),
            executable_direction=np.array([1]),
            executable_risk=np.array([0.3]),
        )
        out["supervisor_increase"] = "FAIL_NOT_DETECTED"
    except ConformanceError:
        out["supervisor_increase"] = "PASS"

    try:
        guard_student_payload({"x": {"taint": "FUTURE_ONLY", "value": 1}})
        out["future_taint"] = "FAIL_NOT_DETECTED"
    except ConformanceError:
        out["future_taint"] = "PASS"

    try:
        guard_counterfactual_independence(parent_contexts=10, branches=240, claimed_independent=240)
        out["branch_inflation"] = "FAIL_NOT_DETECTED"
    except ConformanceError:
        out["branch_inflation"] = "PASS"
    return out
