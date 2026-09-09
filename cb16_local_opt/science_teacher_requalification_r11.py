from __future__ import annotations

"""R11 shadow requalification utilities for dependence-balanced Teacher geometry.

No production Teacher authority is changed here.  The shadow differs from the
frozen R6 Teacher only in normalization geometry: every independent future
(dependence group) contributes equal total normalization mass, and exact
Student-context replicas inside that future are deduplicated.
"""

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np

from .probabilistic_teacher_r6 import DependenceAwareProbabilisticTeacherR6
from .scientific_controls_r6 import (
    FORMULATIONS,
    DependenceAwareControlSuiteConfigR6,
    DependenceAwareHistoricalControlSuiteR6,
    pinball_loss,
)

SHADOW_GEOMETRY_VERSION = "CB16_R11_DEPENDENCE_BALANCED_UNIQUE_CONTEXT_GEOMETRY_SHADOW_V1"


class DependenceBalancedProbabilisticTeacherShadowR11(DependenceAwareProbabilisticTeacherR6):
    """R6 Teacher semantics with replica-invariant dependence-balanced scaling."""

    def _normalization(self, *, train_deps, index, feature_override=None):
        if not train_deps:
            raise RuntimeError("R11_R2_2_NO_TRAIN_DEPENDENCE_GROUPS_FOR_NORMALIZATION")
        group_arrays = []
        for dep in train_deps:
            seen = set()
            rows = []
            for p in index.parents_by_dependence_group[dep]:
                context_id = index.rows_by_parent[p][0].student_context_object_id
                if context_id in seen:
                    continue
                seen.add(context_id)
                feat = (
                    feature_override[p]
                    if feature_override is not None
                    else index.rows_by_parent[p][0].context_features
                )
                rows.append(np.asarray(feat, dtype=np.float64))
            if not rows:
                raise RuntimeError(f"R11_R2_2_EMPTY_DEPENDENCE_GROUP:{dep}")
            group_arrays.append(np.stack(rows, axis=0))

        # Each independent future has equal total mass; different AccountStates
        # inside that future share its mass. Exact replicas add zero mass.
        group_means = np.stack([x.mean(axis=0) for x in group_arrays], axis=0)
        mean = group_means.mean(axis=0)
        group_second = np.stack(
            [np.mean((x - mean) ** 2, axis=0) for x in group_arrays], axis=0
        )
        std = np.sqrt(np.maximum(group_second.mean(axis=0), 0.0))
        return mean, np.where(std < 1e-8, 1.0, std)


class DependenceBalancedHistoricalControlSuiteShadowR11(DependenceAwareHistoricalControlSuiteR6):
    """Exact R6 F0/F1/F2/F3 control suite with only shadow geometry substituted."""

    def __init__(self, config: DependenceAwareControlSuiteConfigR6):
        super().__init__(config)
        self.teacher = DependenceBalancedProbabilisticTeacherShadowR11(config.teacher)


def receipt_to_dict(receipt) -> dict[str, Any]:
    return asdict(receipt)


def formulation_qcrps(receipt) -> dict[str, float]:
    return {x.formulation: float(x.qcrps) for x in receipt.formulations}


def compare_control_receipts_r11(current, shadow) -> dict[str, Any]:
    c = formulation_qcrps(current)
    s = formulation_qcrps(shadow)
    if set(c) != set(s):
        raise RuntimeError("R11_R2_2_FORMULATION_SET_DRIFT")
    per = {f: float(s[f] - c[f]) for f in sorted(c)}
    return {
        "shadow_minus_current_qcrps": per,
        "maximum_abs_qcrps_delta": float(max(abs(x) for x in per.values())),
        "f2_minus_f0_delta_difference": float(shadow.f2_minus_f0.mean_delta - current.f2_minus_f0.mean_delta),
        "f3_minus_f2_delta_difference": float(shadow.f3_minus_f2.mean_delta - current.f3_minus_f2.mean_delta),
        "f1_minus_f0_delta_difference": float(shadow.f1_minus_f0.mean_delta - current.f1_minus_f0.mean_delta),
        "current_status": current.status,
        "shadow_status": shadow.status,
    }


def coverage_diagnostics_r11(
    suite: DependenceAwareHistoricalControlSuiteR6,
    samples,
    *,
    target_parent_ids: Sequence[str],
    eligible_train_dependence_group_ids: Sequence[str],
) -> dict[str, Any]:
    """Coverage on exactly the same laws used by the R6 qCRPS control suite."""
    idx = suite.teacher.index(samples)
    target_set = set(target_parent_ids)
    targets = [p for p in idx.parents if p in target_set]
    eligible = set(eligible_train_dependence_group_ids)
    out = {}
    for formulation in FORMULATIONS:
        by_group_qcrps: dict[str, list[float]] = {}
        cover50 = []
        cover80 = []
        cover90 = []
        laws_missing = 0
        branches = 0
        for p in targets:
            laws = suite._laws(
                formulation=formulation,
                target_parent=p,
                idx=idx,
                eligible_train_deps=eligible,
            )
            if not laws:
                laws_missing += len(idx.rows_by_parent[p])
                continue
            for row in idx.rows_by_parent[p]:
                law = laws.get((row.direction, float(row.requested_risk)))
                if law is None:
                    laws_missing += 1
                    continue
                if law.effective_dependence_n < suite.config.teacher.min_effective_dependence_n:
                    laws_missing += 1
                    continue
                if law.nearest_distance > suite.config.teacher.max_nearest_distance:
                    laws_missing += 1
                    continue
                losses = [
                    pinball_loss(float(row.realized_utility), float(q), float(pred))
                    for q, pred in zip(law.quantile_levels, law.quantiles)
                ]
                score = float(2.0 * np.mean(losses))
                dep = idx.parent_dependence_group[p]
                by_group_qcrps.setdefault(dep, []).append(score)
                qmap = {float(q): float(v) for q, v in zip(law.quantile_levels, law.quantiles)}
                y = float(row.realized_utility)
                cover50.append(float(qmap[0.25] <= y <= qmap[0.75]))
                cover80.append(float(qmap[0.10] <= y <= qmap[0.90]))
                cover90.append(float(qmap[0.05] <= y <= qmap[0.95]))
                branches += 1
        group_scores = [float(np.mean(v)) for _, v in sorted(by_group_qcrps.items()) if v]
        out[formulation] = {
            "qcrps_group_weighted": float(np.mean(group_scores)) if group_scores else None,
            "independent_validation_groups": len(group_scores),
            "scored_branches": int(branches),
            "missing_or_unscorable_branches": int(laws_missing),
            "coverage50": float(np.mean(cover50)) if cover50 else None,
            "coverage80": float(np.mean(cover80)) if cover80 else None,
            "coverage90": float(np.mean(cover90)) if cover90 else None,
            "coverage50_abs_error": abs(float(np.mean(cover50)) - 0.50) if cover50 else None,
            "coverage80_abs_error": abs(float(np.mean(cover80)) - 0.80) if cover80 else None,
            "coverage90_abs_error": abs(float(np.mean(cover90)) - 0.90) if cover90 else None,
        }
    return {
        "geometry": type(suite.teacher).__name__,
        "formulations": out,
        "metric": "DEPENDENCE_GROUP_WEIGHTED_DISCRETE_QCRPS_LOWER_IS_BETTER",
        "coverage_is_diagnostic_not_OBJECTIVE_REVISION": True,
    }


def compare_coverage_r11(current: Mapping[str, Any], shadow: Mapping[str, Any]) -> dict[str, Any]:
    per = {}
    max_q = 0.0
    max_cov = 0.0
    for f in FORMULATIONS:
        c = current["formulations"][f]
        s = shadow["formulations"][f]
        qd = float(s["qcrps_group_weighted"] - c["qcrps_group_weighted"])
        cov = {
            k: float(s[k] - c[k])
            for k in ("coverage50", "coverage80", "coverage90")
        }
        max_q = max(max_q, abs(qd))
        max_cov = max(max_cov, *(abs(x) for x in cov.values()))
        per[f] = {"qcrps_delta": qd, "coverage_deltas": cov}
    return {
        "per_formulation": per,
        "maximum_abs_qcrps_delta": float(max_q),
        "maximum_abs_coverage_delta": float(max_cov),
    }


__all__ = [
    "SHADOW_GEOMETRY_VERSION",
    "DependenceBalancedProbabilisticTeacherShadowR11",
    "DependenceBalancedHistoricalControlSuiteShadowR11",
    "receipt_to_dict",
    "compare_control_receipts_r11",
    "coverage_diagnostics_r11",
    "compare_coverage_r11",
]
