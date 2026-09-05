from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .probabilistic_teacher_r6 import DependenceAwareProbabilisticTeacherR6
from .r102_evidence_cache import ParentContextR102
from .r102_learning import VAL_TEACHER_CONFIG_R102


def discrete_qcrps(y: float, quantile_levels: Sequence[float], quantiles: Sequence[float]) -> float:
    y = float(y); t = np.asarray(quantile_levels, dtype=np.float64); q = np.asarray(quantiles, dtype=np.float64)
    e = y - q
    pinball = np.maximum(t * e, (t - 1.0) * e)
    return float(2.0 * np.mean(pinball))


def _cyclic_group_market_shuffle(parents: Mapping[str, ParentContextR102]) -> dict[str, tuple[float, ...]]:
    # Shuffle market packets at the whole future-group level, preserving own AccountState6.
    by_split_symbol = defaultdict(list)
    group_market = {}
    group_meta = {}
    for p in parents.values():
        group_market.setdefault(p.dependence_group_id, p.operator48 + p.medium48)
        group_meta[p.dependence_group_id] = (p.split, p.symbol)
    for g, (split, sym) in group_meta.items(): by_split_symbol[(split, sym)].append(g)
    map_group = {}
    for key, gs in by_split_symbol.items():
        gs = sorted(gs)
        if len(gs) < 2:
            for g in gs: map_group[g] = g
        else:
            # deterministic non-identity cyclic permutation, no outcome inspection.
            for i, g in enumerate(gs): map_group[g] = gs[(i + 1) % len(gs)]
    out = {}
    for p in parents.values():
        shuffled_market = group_market[map_group[p.dependence_group_id]]
        out[p.parent_id] = tuple(shuffled_market + p.account6)
    return out


def run_f0_f1_f2_f3_controls(samples, parents: Mapping[str, ParentContextR102]) -> dict[str, Any]:
    teacher = DependenceAwareProbabilisticTeacherR6(VAL_TEACHER_CONFIG_R102)
    idx = teacher.index(samples)
    train_groups = {p.dependence_group_id for p in parents.values() if p.split == "TRAIN"}
    val_parents = [p for p in parents.values() if p.split == "VALIDATION" and p.parent_id in idx.rows_by_parent]
    f1 = {p.parent_id: tuple(p.account6) for p in parents.values()}
    f3 = _cyclic_group_market_shuffle(parents)

    modes = {
        "F0_CLIMATOLOGY": (None, True),
        "F1_ACCOUNT_ONLY": (f1, False),
        "F2_TRUE_MARKET_ACCOUNT": (None, False),
        "F3_SHUFFLED_MARKET_ACCOUNT": (f3, False),
    }
    results = {}
    for name, (override, climatology) in modes.items():
        per_group = defaultdict(list); missing = 0
        cover50 = []; cover80 = []; cover90 = []
        for p in val_parents:
            train_deps = teacher.train_dependence_groups_for_target(
                target_parent=p.parent_id, index=idx, eligible_train_dependence_groups=train_groups,
            )
            target_features = (
                override[p.parent_id] if override is not None else idx.rows_by_parent[p.parent_id][0].context_features
            )
            for branch in idx.rows_by_parent[p.parent_id]:
                law = teacher.predictive_law(
                    target_features=target_features, train_deps=train_deps, index=idx,
                    direction=branch.direction, risk=branch.requested_risk,
                    feature_override=override, equal_weight_climatology=climatology,
                )
                if law is None:
                    missing += 1; continue
                score = discrete_qcrps(branch.realized_utility, law.quantile_levels, law.quantiles)
                per_group[p.dependence_group_id].append(score)
                q = dict(zip(law.quantile_levels, law.quantiles))
                # Quantile grid contains 10/25/75/90; 50/80/90 central interval diagnostics.
                cover50.append(float(q[0.25] <= branch.realized_utility <= q[0.75]))
                cover80.append(float(q[0.10] <= branch.realized_utility <= q[0.90]))
                cover90.append(float(q[0.05] <= branch.realized_utility <= q[0.95]))
        group_means = [float(np.mean(v)) for _, v in sorted(per_group.items()) if v]
        results[name] = {
            "qcrps": float(np.mean(group_means)) if group_means else None,
            "independent_validation_groups": len(group_means), "missing_laws": int(missing),
            "coverage50": float(np.mean(cover50)) if cover50 else None,
            "coverage80": float(np.mean(cover80)) if cover80 else None,
            "coverage90": float(np.mean(cover90)) if cover90 else None,
        }
    f2 = results["F2_TRUE_MARKET_ACCOUNT"]["qcrps"]
    f3q = results["F3_SHUFFLED_MARKET_ACCOUNT"]["qcrps"]
    delta = None if f2 is None or f3q is None else float(f3q - f2)  # positive means true market better (lower qCRPS)
    return {
        "schema": "CB16_R10_2_F0_F1_F2_F3_CONTROL_REPORT_V1",
        "status": "DIAGNOSTIC_ONLY_NOT_PIPELINE_PASS_DRIVER",
        "metric": "group_weighted_discrete_qCRPS_lower_is_better",
        "controls": results,
        "F3_minus_F2": delta,
        "interpretation": (
            "TRUE_MARKET_BETTER_THAN_SHUFFLE" if delta is not None and delta > 0 else
            "TRUE_MARKET_NOT_BETTER_THAN_SHUFFLE" if delta is not None else "UNAVAILABLE"
        ),
        "no_result_rescue": True,
    }
