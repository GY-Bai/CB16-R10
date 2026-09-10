from __future__ import annotations

"""H5.11 common-support rank-geometry mechanism decomposition.

Only the empirical measure over the already-frozen H5.8 aligned
(Operator-distance rank percentile, Medium-distance rank percentile) geometry is
changed.  The 3x3 grid is design-only.  Continuous H5.8 rank order, support,
normalization, distances, utility target, structured Medium null, scenario
weighting, and future-group macro aggregation remain unchanged.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from . import state_utility_geometry_transport_audit_h55 as h55
from .medium48_temporal_halfblock_stability_h510 import H510_EXPECTED, orientation_h510, _future_group_timestamp_h510
from .operator_conditional_medium_geometry_h58 import H58_SHIFTS, partial_spearman_rank_h58
from .teacher_temporal_transport_audit_h5 import (
    H5_SCENARIOS,
    _eval_group_scenario_rows_h5,
    _fold_material_h5,
    require,
)
from .teacher_vectorized_r11 import build_columnar_teacher_index_r11
from .time_local_vs_forward_geometry_contrast_h56 import _same_scenario_leave_group_out_normalization_h56

H511_RUNTIME = "CB16_R11_H5_11_COMMON_SUPPORT_RANK_GEOMETRY_MECHANISM_DECOMPOSITION_R0_V1"
H511_GRID_N = 3
H511_CELL_N = 9
H511_EPS = 1e-12
H511_BASELINE_TOL = 1e-12
H511_MIN_CELL_CLOCKS = 3
H511_PAIRS = ((1, 2), (2, 3), (3, 4), (4, 5))
H511_NATIVE_ORIENTATION = {
    1: "ANTI_ALIGNMENT",
    2: "POSITIVE_ALIGNMENT",
    3: "ANTI_ALIGNMENT",
    4: "POSITIVE_ALIGNMENT",
    5: "POSITIVE_ALIGNMENT",
}
H511_FLIP_PAIRS = ((1, 2), (2, 3), (3, 4))
H511_CONTROL_PAIR = (4, 5)


def _raw_rank_h511(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    require(a.size >= 2 and np.isfinite(a).all(), "H511_BAD_RANK_VECTOR")
    r = np.asarray(h55._rankdata_average_h55(a), dtype=np.float64)
    require(r.shape == a.shape and np.isfinite(r).all(), "H511_BAD_AVERAGE_RANK")
    return r


def rank_percentile_h511(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return average-tie rank and its [0,1] percentile; used only for design cells."""
    r = _raw_rank_h511(x)
    pct = (r - 1.0) / float(len(r) - 1)
    require(np.isfinite(pct).all() and np.all(pct >= -H511_EPS) and np.all(pct <= 1.0 + H511_EPS), "H511_RANK_PERCENTILE_RANGE")
    return r, np.clip(pct, 0.0, 1.0)


def cell_ids_h511(operator_pct: np.ndarray, medium_pct: np.ndarray) -> np.ndarray:
    o = np.asarray(operator_pct, dtype=np.float64).reshape(-1)
    m = np.asarray(medium_pct, dtype=np.float64).reshape(-1)
    require(o.shape == m.shape and o.size >= 1, "H511_CELL_SHAPE")
    require(np.isfinite(o).all() and np.isfinite(m).all(), "H511_CELL_NONFINITE")
    require(np.all(o >= -H511_EPS) and np.all(o <= 1.0 + H511_EPS), "H511_OPERATOR_PCT_RANGE")
    require(np.all(m >= -H511_EPS) and np.all(m <= 1.0 + H511_EPS), "H511_MEDIUM_PCT_RANGE")
    edges = np.asarray([1.0 / 3.0, 2.0 / 3.0], dtype=np.float64)
    ob = np.searchsorted(edges, np.clip(o, 0.0, 1.0), side="right").astype(np.int32)
    mb = np.searchsorted(edges, np.clip(m, 0.0, 1.0), side="right").astype(np.int32)
    cells = ob * H511_GRID_N + mb
    require(np.all(cells >= 0) and np.all(cells < H511_CELL_N), "H511_CELL_ID_RANGE")
    return np.ascontiguousarray(cells)


def _weighted_corr_h511(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, bool]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    require(x.shape == y.shape == w.shape and x.size >= 2, "H511_WEIGHTED_CORR_SHAPE")
    require(np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all(), "H511_WEIGHTED_CORR_NONFINITE")
    require(np.all(w >= 0.0), "H511_NEGATIVE_WEIGHT")
    sw = float(np.sum(w))
    require(sw > H511_EPS, "H511_ZERO_TOTAL_WEIGHT")
    wn = w / sw
    xm = float(np.sum(wn * x)); ym = float(np.sum(wn * y))
    xc = x - xm; yc = y - ym
    vx = float(np.sum(wn * xc * xc)); vy = float(np.sum(wn * yc * yc))
    if vx <= H511_EPS or vy <= H511_EPS:
        return 0.0, True
    cov = float(np.sum(wn * xc * yc))
    raw = cov / float(np.sqrt(vx * vy))
    require(np.isfinite(raw) and -1.0 - 1e-10 <= raw <= 1.0 + 1e-10, "H511_WEIGHTED_CORR_RANGE")
    return float(np.clip(raw, -1.0, 1.0)), False


def weighted_partial_rank_h511(
    operator_rank: np.ndarray,
    medium_rank: np.ndarray,
    utility_rank: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Weighted partial Pearson on frozen rank values; not claimed as canonical weighted Spearman."""
    rho_mo, deg_mo = _weighted_corr_h511(medium_rank, operator_rank, weight)
    rho_uo, deg_uo = _weighted_corr_h511(utility_rank, operator_rank, weight)
    rho_mu, deg_mu = _weighted_corr_h511(medium_rank, utility_rank, weight)
    if deg_mo or deg_uo or deg_mu:
        return 0.0, {
            "rho_medium_utility": float(rho_mu),
            "rho_medium_operator": float(rho_mo),
            "rho_operator_utility": float(rho_uo),
            "zero_information": True,
        }
    d1 = max(0.0, 1.0 - rho_mo * rho_mo)
    d2 = max(0.0, 1.0 - rho_uo * rho_uo)
    denom = float(np.sqrt(d1 * d2))
    if denom <= H511_EPS:
        partial = 0.0
        zero = True
    else:
        raw = float((rho_mu - rho_mo * rho_uo) / denom)
        require(np.isfinite(raw) and -1.0 - 1e-10 <= raw <= 1.0 + 1e-10, "H511_PARTIAL_RANGE")
        partial = float(np.clip(raw, -1.0, 1.0))
        zero = False
    return partial, {
        "rho_medium_utility": float(rho_mu),
        "rho_medium_operator": float(rho_mo),
        "rho_operator_utility": float(rho_uo),
        "zero_information": bool(zero),
    }


def _fold_context_h511(*, fold_spec: Mapping[str, Any], all_train_parents: Mapping[str, Any], all_train_samples: Sequence[Any]):
    train_parents, eval_parents, parents, train_samples, eval_samples, samples = _fold_material_h5(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    index = build_columnar_teacher_index_r11(samples)
    eval_parent_ids = tuple(sorted(eval_parents))
    eval_order, eval_rows_map = _eval_group_scenario_rows_h5(index=index, eval_parents=eval_parents, eval_parent_ids=eval_parent_ids)
    require(len(eval_order) >= 33, "H511_TOO_FEW_LOCAL_GROUPS")
    normalizers = _same_scenario_leave_group_out_normalization_h56(index=index, eval_order=eval_order, eval_rows_map=eval_rows_map)
    return index, eval_order, eval_rows_map, normalizers


def _feature_rank_cells_h511(*, index, gid: str, scenario: str, eval_order: Sequence[str], eval_rows_map: Mapping[str, Mapping[str, int]], mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    support_groups = [g for g in eval_order if g != gid]
    require(len(support_groups) >= 32, "H511_LOCAL_SUPPORT_TOO_SMALL")
    target_row = int(eval_rows_map[gid][scenario])
    support_rows = np.asarray([eval_rows_map[g][scenario] for g in support_groups], dtype=np.int32)
    target_feature = np.asarray(index.features[target_row], dtype=np.float64).reshape(1, -1)
    support_features = np.asarray(index.features[support_rows], dtype=np.float64)
    d_operator = h55._pairwise_state_distance_h55(
        target_features=target_feature,
        support_features=support_features,
        train_mean=mean,
        train_std=std,
        active=h55.active_dimensions_h55("OPERATOR48", int(index.feature_dim)),
    )[0]
    d_medium = h55._pairwise_state_distance_h55(
        target_features=target_feature,
        support_features=support_features,
        train_mean=mean,
        train_std=std,
        active=h55.active_dimensions_h55("MEDIUM48", int(index.feature_dim)),
    )[0]
    orank, opct = rank_percentile_h511(d_operator)
    mrank, mpct = rank_percentile_h511(d_medium)
    cells = cell_ids_h511(opct, mpct)
    return {
        "support_groups": tuple(str(x) for x in support_groups),
        "support_rows": support_rows,
        "target_row": target_row,
        "d_operator": np.asarray(d_operator, dtype=np.float64),
        "d_medium": np.asarray(d_medium, dtype=np.float64),
        "operator_rank": orank,
        "medium_rank": mrank,
        "cell_ids": cells,
    }


def run_fold_design_h511(*, fold_spec: Mapping[str, Any], all_train_parents: Mapping[str, Any], all_train_samples: Sequence[Any]) -> dict[str, Any]:
    """Outcome-blind pass: utilities are never read here."""
    fold = int(fold_spec["fold"])
    index, eval_order, eval_rows_map, normalizers = _fold_context_h511(
        fold_spec=fold_spec, all_train_parents=all_train_parents, all_train_samples=all_train_samples
    )
    mass = np.zeros(H511_CELL_N, dtype=np.float64)
    clocks_by_cell: list[set[int]] = [set() for _ in range(H511_CELL_N)]
    anchor_scenario_count = 0
    support_min = 10**9; support_max = 0
    for gid in eval_order:
        ts = _future_group_timestamp_h510(str(gid))
        for scenario in H5_SCENARIOS:
            mean, std = normalizers[(str(gid), str(scenario))]
            x = _feature_rank_cells_h511(index=index, gid=str(gid), scenario=str(scenario), eval_order=eval_order, eval_rows_map=eval_rows_map, mean=mean, std=std)
            cells = x["cell_ids"]
            n = int(len(cells))
            support_min = min(support_min, n); support_max = max(support_max, n)
            mass += np.bincount(cells, minlength=H511_CELL_N).astype(np.float64) / float(n)
            for c in np.unique(cells):
                clocks_by_cell[int(c)].add(int(ts))
            anchor_scenario_count += 1
    require(anchor_scenario_count == len(eval_order) * len(H5_SCENARIOS), "H511_DESIGN_SCENARIO_COUNT_DRIFT")
    mass /= float(anchor_scenario_count)
    require(abs(float(np.sum(mass)) - 1.0) <= 1e-12, "H511_DESIGN_MASS_NOT_ONE")
    return {
        "fold": fold,
        "cell_mass": mass.tolist(),
        "unique_target_decision_clocks_by_cell": [len(x) for x in clocks_by_cell],
        "eval_dependence_groups": int(len(eval_order)),
        "anchor_scenario_count": int(anchor_scenario_count),
        "support_future_groups_min": int(support_min),
        "support_future_groups_max": int(support_max),
        "utility_accessed": False,
        "teacher_used": False,
        "student_used": False,
    }


def build_pair_overlap_design_h511(design_a: Mapping[str, Any], design_b: Mapping[str, Any]) -> dict[str, Any]:
    fa = int(design_a["fold"]); fb = int(design_b["fold"])
    require((fa, fb) in H511_PAIRS, f"H511_UNREGISTERED_PAIR:{fa}:{fb}")
    p_a = np.asarray(design_a["cell_mass"], dtype=np.float64)
    p_b = np.asarray(design_b["cell_mass"], dtype=np.float64)
    require(p_a.shape == p_b.shape == (H511_CELL_N,), "H511_PAIR_MASS_SHAPE")
    raw = np.zeros(H511_CELL_N, dtype=np.float64)
    mask = (p_a > 0.0) & (p_b > 0.0)
    raw[mask] = p_a[mask] * p_b[mask] / (p_a[mask] + p_b[mask])
    raw_sum = float(np.sum(raw))
    require(np.isfinite(raw_sum) and raw_sum > H511_EPS, "H511_NO_OVERLAP_MASS")
    q = raw / raw_sum
    ma = np.zeros_like(q); mb = np.zeros_like(q)
    ma[mask] = q[mask] / p_a[mask]
    mb[mask] = q[mask] / p_b[mask]
    positive = np.flatnonzero(q > 0.0)
    op_bins = {int(c) // H511_GRID_N for c in positive.tolist()}
    med_bins = {int(c) % H511_GRID_N for c in positive.tolist()}
    clocks_a = np.asarray(design_a["unique_target_decision_clocks_by_cell"], dtype=np.int64)
    clocks_b = np.asarray(design_b["unique_target_decision_clocks_by_cell"], dtype=np.int64)
    clock_ok = bool(all(int(clocks_a[c]) >= H511_MIN_CELL_CLOCKS and int(clocks_b[c]) >= H511_MIN_CELL_CLOCKS for c in positive.tolist()))
    axis_ok = bool(len(op_bins) >= 2 and len(med_bins) >= 2)
    valid = bool(clock_ok and axis_ok)
    return {
        "pair": [fa, fb],
        "p_a": p_a.tolist(),
        "p_b": p_b.tolist(),
        "q": q.tolist(),
        "multiplier_a": ma.tolist(),
        "multiplier_b": mb.tolist(),
        "positive_q_cells": positive.astype(int).tolist(),
        "operator_bins_spanned": sorted(op_bins),
        "medium_bins_spanned": sorted(med_bins),
        "unique_target_decision_clocks_a_by_cell": clocks_a.astype(int).tolist(),
        "unique_target_decision_clocks_b_by_cell": clocks_b.astype(int).tolist(),
        "clock_support_gate_passed": clock_ok,
        "axis_span_gate_passed": axis_ok,
        "common_support_valid": valid,
        "utility_used_to_construct_q": False,
        "propensity_model_fit": False,
        "q_recomputed_for_null": False,
    }


def _scenario_payload_h511(*, index, gid: str, scenario: str, eval_order: Sequence[str], eval_rows_map: Mapping[str, Mapping[str, int]], mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    x = _feature_rank_cells_h511(index=index, gid=gid, scenario=scenario, eval_order=eval_order, eval_rows_map=eval_rows_map, mean=mean, std=std)
    utility_mse = h55._pairwise_profile_mse_h55(
        np.asarray(index.utilities[x["target_row"]], dtype=np.float64).reshape(1, -1),
        np.asarray(index.utilities[x["support_rows"]], dtype=np.float64),
        centered=True,
    )[0]
    urank = _raw_rank_h511(utility_mse)
    ones = np.ones(len(urank), dtype=np.float64)
    aligned, _ = weighted_partial_rank_h511(x["operator_rank"], x["medium_rank"], urank, ones)
    inherited_aligned, _ = partial_spearman_rank_h58(x["d_operator"], x["d_medium"], utility_mse)
    require(abs(float(aligned) - float(inherited_aligned)) <= H511_BASELINE_TOL, "H511_H58_ALIGNED_STAT_DRIFT")
    nulls: dict[int, float] = {}
    null_medium_ranks: dict[int, np.ndarray] = {}
    for shift in H58_SHIFTS:
        k = int(shift) % len(x["d_medium"])
        require(k != 0, f"H511_NULL_IDENTITY_SHIFT:{shift}")
        rotated = np.roll(x["d_medium"], -k)
        mr = _raw_rank_h511(rotated)
        rho, _ = weighted_partial_rank_h511(x["operator_rank"], mr, urank, ones)
        inherited, _ = partial_spearman_rank_h58(x["d_operator"], rotated, utility_mse)
        require(abs(float(rho) - float(inherited)) <= H511_BASELINE_TOL, f"H511_H58_NULL_STAT_DRIFT:{shift}")
        nulls[int(shift)] = float(rho)
        null_medium_ranks[int(shift)] = mr
    return {
        "scenario": str(scenario),
        "operator_rank": x["operator_rank"],
        "medium_rank": x["medium_rank"],
        "utility_rank": urank,
        "cell_ids": x["cell_ids"],
        "null_medium_rank_by_shift": null_medium_ranks,
        "native_aligned_partial_rho": float(aligned),
        "native_null_partial_rho_by_shift": nulls,
    }


def prepare_fold_payload_h511(*, fold_spec: Mapping[str, Any], all_train_parents: Mapping[str, Any], all_train_samples: Sequence[Any]) -> dict[str, Any]:
    fold = int(fold_spec["fold"])
    index, eval_order, eval_rows_map, normalizers = _fold_context_h511(
        fold_spec=fold_spec, all_train_parents=all_train_parents, all_train_samples=all_train_samples
    )
    groups: list[dict[str, Any]] = []
    mass = np.zeros(H511_CELL_N, dtype=np.float64)
    for gid in eval_order:
        scenarios = []
        for scenario in H5_SCENARIOS:
            mean, std = normalizers[(str(gid), str(scenario))]
            p = _scenario_payload_h511(index=index, gid=str(gid), scenario=str(scenario), eval_order=eval_order, eval_rows_map=eval_rows_map, mean=mean, std=std)
            scenarios.append(p)
            cells = p["cell_ids"]
            mass += np.bincount(cells, minlength=H511_CELL_N).astype(np.float64) / float(len(cells))
        groups.append({"future_group_id": str(gid), "timestamp_ms": _future_group_timestamp_h510(str(gid)), "scenarios": scenarios})
    mass /= float(len(eval_order) * len(H5_SCENARIOS))
    require(abs(float(np.sum(mass)) - 1.0) <= 1e-12, "H511_PAYLOAD_MASS_NOT_ONE")
    return {"fold": fold, "groups": groups, "cell_mass": mass.tolist(), "eval_dependence_groups": int(len(eval_order))}


def native_fold_summary_h511(payload: Mapping[str, Any]) -> dict[str, Any]:
    group_rows = []
    for g in payload["groups"]:
        ss = g["scenarios"]
        group_rows.append({
            "aligned": float(np.mean([float(x["native_aligned_partial_rho"]) for x in ss])),
            "nulls": {int(s): float(np.mean([float(x["native_null_partial_rho_by_shift"][int(s)]) for x in ss])) for s in H58_SHIFTS},
        })
    aligned = float(np.mean([x["aligned"] for x in group_rows]))
    nulls = {int(s): float(np.mean([x["nulls"][int(s)] for x in group_rows])) for s in H58_SHIFTS}
    med = float(statistics.median(nulls.values()))
    fold = int(payload["fold"]); expected = H510_EXPECTED[fold]
    aerr = abs(aligned - float(expected["aligned_partial_rho"])); nerr = abs(med - float(expected["null_median"]))
    return {
        "fold": fold,
        "aligned_partial_rho": aligned,
        "null_partial_rho_by_shift": nulls,
        "null_median_partial_rho": med,
        "aligned_minus_null_median": float(aligned - med),
        "orientation": orientation_h510(aligned, med),
        "expected_orientation": H511_NATIVE_ORIENTATION[fold],
        "aligned_abs_error_vs_h5_8": float(aerr),
        "null_median_abs_error_vs_h5_8": float(nerr),
        "reproduced": bool(aerr <= H511_BASELINE_TOL and nerr <= H511_BASELINE_TOL and orientation_h510(aligned, med) == H511_NATIVE_ORIENTATION[fold]),
    }


def _post_weight_cell_mass_h511(p: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    x = np.asarray(p, dtype=np.float64) * np.asarray(multiplier, dtype=np.float64)
    s = float(np.sum(x)); require(s > H511_EPS, "H511_POST_WEIGHT_ZERO_MASS")
    return x / s


def evaluate_fold_overlap_h511(payload: Mapping[str, Any], multiplier: Sequence[float], q: Sequence[float]) -> dict[str, Any]:
    m = np.asarray(multiplier, dtype=np.float64)
    qv = np.asarray(q, dtype=np.float64)
    require(m.shape == qv.shape == (H511_CELL_N,), "H511_EVAL_WEIGHT_SHAPE")
    group_rows = []
    zero_count = 0
    min_weight_sum = float("inf")
    max_weight_sum = 0.0
    for g in payload["groups"]:
        scenario_rows = []
        for srow in g["scenarios"]:
            w = m[np.asarray(srow["cell_ids"], dtype=np.int32)]
            sw = float(np.sum(w)); require(sw > H511_EPS, "H511_TARGET_SCENARIO_NO_OVERLAP_WEIGHT")
            min_weight_sum = min(min_weight_sum, sw); max_weight_sum = max(max_weight_sum, sw)
            aligned, ar = weighted_partial_rank_h511(srow["operator_rank"], srow["medium_rank"], srow["utility_rank"], w)
            nulls = {}
            scenario_zero = bool(ar["zero_information"])
            for shift in H58_SHIFTS:
                rho, rr = weighted_partial_rank_h511(srow["operator_rank"], srow["null_medium_rank_by_shift"][int(shift)], srow["utility_rank"], w)
                nulls[int(shift)] = float(rho)
                scenario_zero = bool(scenario_zero or rr["zero_information"])
            zero_count += int(scenario_zero)
            scenario_rows.append({"aligned": float(aligned), "nulls": nulls})
        group_rows.append({
            "aligned": float(np.mean([x["aligned"] for x in scenario_rows])),
            "nulls": {int(s): float(np.mean([x["nulls"][int(s)] for x in scenario_rows])) for s in H58_SHIFTS},
        })
    aligned = float(np.mean([x["aligned"] for x in group_rows]))
    nulls = {int(s): float(np.mean([x["nulls"][int(s)] for x in group_rows])) for s in H58_SHIFTS}
    med = float(statistics.median(nulls.values()))
    p = np.asarray(payload["cell_mass"], dtype=np.float64)
    post = _post_weight_cell_mass_h511(p, m)
    maxerr = float(np.max(np.abs(post - qv)))
    require(maxerr <= 1e-12, f"H511_POST_WEIGHT_Q_MISMATCH:{maxerr}")
    return {
        "fold": int(payload["fold"]),
        "aligned_partial_rho": aligned,
        "null_partial_rho_by_shift": nulls,
        "null_median_partial_rho": med,
        "aligned_minus_null_median": float(aligned - med),
        "orientation": orientation_h510(aligned, med),
        "zero_information_target_scenario_count": int(zero_count),
        "post_weight_cell_mass": post.tolist(),
        "post_weight_max_abs_error_vs_q": maxerr,
        "target_scenario_weight_sum_min": float(min_weight_sum),
        "target_scenario_weight_sum_max": float(max_weight_sum),
        "same_aligned_candidate_weights_used_for_all_nulls": True,
        "q_recomputed_for_null": False,
    }


def adjudicate_h511(native: Sequence[Mapping[str, Any]], pair_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    native_rows = sorted(native, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in native_rows) == (1, 2, 3, 4, 5), "H511_NATIVE_FOLD_SET_DRIFT")
    baseline_ok = bool(all(bool(x["reproduced"]) for x in native_rows))
    by_pair = {(int(x["pair"][0]), int(x["pair"][1])): x for x in pair_results}
    require(tuple(sorted(by_pair)) == H511_PAIRS, "H511_PAIR_SET_DRIFT")
    support_ok = bool(all(bool(x["common_support_valid"]) for x in pair_results))
    if not baseline_ok:
        classification = "EXECUTION_BLOCKED__H5_8_BASELINE_REPRODUCTION_FAILED"
    elif not support_ok:
        classification = "EXECUTION_BLOCKED__INSUFFICIENT_COMMON_SUPPORT"
    else:
        retained = []
        clean_removed = []
        for pair in H511_FLIP_PAIRS:
            x = by_pair[pair]
            oa = str(x["side_a"]["orientation"]); ob = str(x["side_b"]["orientation"])
            retained.append(oa == H511_NATIVE_ORIENTATION[pair[0]] and ob == H511_NATIVE_ORIENTATION[pair[1]])
            clean_removed.append(oa == ob and oa in {"POSITIVE_ALIGNMENT", "ANTI_ALIGNMENT"})
        ctrl = by_pair[H511_CONTROL_PAIR]
        control_stable = bool(str(ctrl["side_a"]["orientation"]) == "POSITIVE_ALIGNMENT" and str(ctrl["side_b"]["orientation"]) == "POSITIVE_ALIGNMENT")
        if all(retained) and control_stable:
            classification = "COARSENED_COMMON_SUPPORT_CONDITIONAL_MAPPING_INSTABILITY_SUPPORTED"
        elif all(clean_removed) and control_stable:
            classification = "COARSENED_OCCUPANCY_STANDARDIZATION_REMOVES_NATIVE_ORIENTATION_TRANSITIONS"
        else:
            classification = "MIXED_OCCUPANCY_AND_CONDITIONAL_MAPPING_EVIDENCE"
    flip_persistence = 0
    flip_clean_removed = 0
    if baseline_ok and support_ok:
        for pair in H511_FLIP_PAIRS:
            x = by_pair[pair]
            oa = str(x["side_a"]["orientation"]); ob = str(x["side_b"]["orientation"])
            flip_persistence += int(oa == H511_NATIVE_ORIENTATION[pair[0]] and ob == H511_NATIVE_ORIENTATION[pair[1]])
            flip_clean_removed += int(oa == ob and oa in {"POSITIVE_ALIGNMENT", "ANTI_ALIGNMENT"})
    return {
        "classification": classification,
        "conclusion": f"H5_11_{classification}",
        "native_h5_8_baseline_reproduction_passed": baseline_ok,
        "all_pair_common_support_gates_passed": support_ok,
        "native_flip_persistence_count_of_3": int(flip_persistence),
        "native_flip_clean_removal_count_of_3": int(flip_clean_removed),
        "positive_control_pair_stable": bool(
            baseline_ok and support_ok and
            str(by_pair[H511_CONTROL_PAIR]["side_a"]["orientation"]) == "POSITIVE_ALIGNMENT" and
            str(by_pair[H511_CONTROL_PAIR]["side_b"]["orientation"]) == "POSITIVE_ALIGNMENT"
        ),
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "teacher_change_authorized": False,
        "student_change_authorized": False,
        "organ_change_authorized": False,
        "learned_gate_or_router_authorized": False,
        "time_or_regime_gate_authorized": False,
    }
