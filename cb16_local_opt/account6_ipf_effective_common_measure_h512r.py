from __future__ import annotations

"""H5.12R: add frozen Account6 relational occupancy to H5.11.2.

The measured statistic remains the H5.11 weighted Operator-conditional-Medium
rank statistic. Account6 is used only to refine the outcome-blind calibration
cells from 3x3 to 3x3x3. Pair-side cell multipliers are calibrated by IPF to
match the actual locally-normalized effective macro cell margins.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from . import state_utility_geometry_transport_audit_h55 as h55
from .common_support_rank_geometry_decomposition_h511 import (
    H511_EPS,
    H511_NATIVE_ORIENTATION,
    H511_PAIRS,
    _fold_context_h511,
    _scenario_payload_h511,
    rank_percentile_h511,
    weighted_partial_rank_h511,
)
from .medium48_temporal_halfblock_stability_h510 import _future_group_timestamp_h510, orientation_h510
from .operator_conditional_medium_geometry_h58 import H58_SHIFTS
from .teacher_temporal_transport_audit_h5 import H5_SCENARIOS, require

H512R_RUNTIME = "CB16_R11_H5_12R_ACCOUNT6_IPF_EFFECTIVE_COMMON_MEASURE_R0_V1"
H512R_GRID_N = 3
H512R_CELL_N = 27
H512R_MIN_CELL_CLOCKS = 3
H512R_INTERNAL_STOP = 1e-13
H512R_FORMAL_TOL = 1e-12
H512R_SIDE_TOL = 2e-12
H512R_MAX_ITER = 10000
H512R_FLIP_PAIRS = ((1, 2), (2, 3), (3, 4))
H512R_CONTROL_PAIR = (4, 5)


def _pct_from_rank_h512r(rank: np.ndarray) -> np.ndarray:
    r = np.asarray(rank, dtype=np.float64).reshape(-1)
    require(r.size >= 2 and np.isfinite(r).all(), "H512R_BAD_RANK")
    pct = (r - 1.0) / float(r.size - 1)
    require(np.all(pct >= -H511_EPS) and np.all(pct <= 1.0 + H511_EPS), "H512R_RANK_PCT_RANGE")
    return np.clip(pct, 0.0, 1.0)


def cell_ids_27_h512r(operator_pct: np.ndarray, medium_pct: np.ndarray, account_pct: np.ndarray) -> np.ndarray:
    o = np.asarray(operator_pct, dtype=np.float64).reshape(-1)
    m = np.asarray(medium_pct, dtype=np.float64).reshape(-1)
    a = np.asarray(account_pct, dtype=np.float64).reshape(-1)
    require(o.shape == m.shape == a.shape and o.size >= 1, "H512R_CELL_SHAPE")
    require(np.isfinite(o).all() and np.isfinite(m).all() and np.isfinite(a).all(), "H512R_CELL_NONFINITE")
    edges = np.asarray([1.0 / 3.0, 2.0 / 3.0], dtype=np.float64)
    ob = np.searchsorted(edges, np.clip(o, 0.0, 1.0), side="right").astype(np.int32)
    mb = np.searchsorted(edges, np.clip(m, 0.0, 1.0), side="right").astype(np.int32)
    ab = np.searchsorted(edges, np.clip(a, 0.0, 1.0), side="right").astype(np.int32)
    cells = ob * 9 + mb * 3 + ab
    require(np.all(cells >= 0) and np.all(cells < H512R_CELL_N), "H512R_CELL_RANGE")
    return np.ascontiguousarray(cells)


def _scenario_payload_h512r(
    *, index, gid: str, scenario: str, eval_order: Sequence[str], eval_rows_map: Mapping[str, Mapping[str, int]],
    mean: np.ndarray, std: np.ndarray,
) -> dict[str, Any]:
    base = _scenario_payload_h511(
        index=index, gid=gid, scenario=scenario, eval_order=eval_order,
        eval_rows_map=eval_rows_map, mean=mean, std=std,
    )
    support_groups = [g for g in eval_order if g != gid]
    target_row = int(eval_rows_map[gid][scenario])
    support_rows = np.asarray([eval_rows_map[g][scenario] for g in support_groups], dtype=np.int32)
    target_feature = np.asarray(index.features[target_row], dtype=np.float64).reshape(1, -1)
    support_features = np.asarray(index.features[support_rows], dtype=np.float64)
    d_account = h55._pairwise_state_distance_h55(
        target_features=target_feature,
        support_features=support_features,
        train_mean=mean,
        train_std=std,
        active=h55.active_dimensions_h55("ACCOUNT6", int(index.feature_dim)),
    )[0]
    account_rank, account_pct = rank_percentile_h511(d_account)
    operator_pct = _pct_from_rank_h512r(np.asarray(base["operator_rank"], dtype=np.float64))
    medium_pct = _pct_from_rank_h512r(np.asarray(base["medium_rank"], dtype=np.float64))
    cells27 = cell_ids_27_h512r(operator_pct, medium_pct, account_pct)
    account_degenerate = bool(np.all(np.asarray(d_account, dtype=np.float64) == float(d_account[0])))
    if account_degenerate:
        require(np.all(np.asarray(account_pct, dtype=np.float64) == 0.5), "H512R_DEGENERATE_ACCOUNT_NOT_MIDDLE_RANK")
    return {
        "scenario": str(scenario),
        "operator_rank": np.asarray(base["operator_rank"], dtype=np.float64),
        "medium_rank": np.asarray(base["medium_rank"], dtype=np.float64),
        "utility_rank": np.asarray(base["utility_rank"], dtype=np.float64),
        "null_medium_rank_by_shift": {int(k): np.asarray(v, dtype=np.float64) for k, v in base["null_medium_rank_by_shift"].items()},
        "account_rank": np.asarray(account_rank, dtype=np.float64),
        "account_distance": np.asarray(d_account, dtype=np.float64),
        "account_rank_percentile": np.asarray(account_pct, dtype=np.float64),
        "cell_ids": cells27,
        "account_distance_degenerate": account_degenerate,
        "support_count": int(len(support_rows)),
    }


def prepare_fold_payload_h512r(
    *, fold_spec: Mapping[str, Any], all_train_parents: Mapping[str, Any], all_train_samples: Sequence[Any]
) -> dict[str, Any]:
    fold = int(fold_spec["fold"])
    index, eval_order, eval_rows_map, normalizers = _fold_context_h511(
        fold_spec=fold_spec, all_train_parents=all_train_parents, all_train_samples=all_train_samples
    )
    groups: list[dict[str, Any]] = []
    mass = np.zeros(H512R_CELL_N, dtype=np.float64)
    clocks_by_cell: list[set[int]] = [set() for _ in range(H512R_CELL_N)]
    degenerate = 0
    nondegenerate = 0
    support_min = 10**9
    support_max = 0
    for gid in eval_order:
        scenarios: list[dict[str, Any]] = []
        ts = _future_group_timestamp_h510(str(gid))
        for scenario in H5_SCENARIOS:
            mean, std = normalizers[(str(gid), str(scenario))]
            row = _scenario_payload_h512r(
                index=index, gid=str(gid), scenario=str(scenario), eval_order=eval_order,
                eval_rows_map=eval_rows_map, mean=mean, std=std,
            )
            scenarios.append(row)
            cells = np.asarray(row["cell_ids"], dtype=np.int32)
            n = int(len(cells))
            support_min = min(support_min, n)
            support_max = max(support_max, n)
            mass += np.bincount(cells, minlength=H512R_CELL_N).astype(np.float64) / float(n)
            for c in np.unique(cells):
                clocks_by_cell[int(c)].add(int(ts))
            degenerate += int(bool(row["account_distance_degenerate"]))
            nondegenerate += int(not bool(row["account_distance_degenerate"]))
        groups.append({"future_group_id": str(gid), "timestamp_ms": int(ts), "scenarios": scenarios})
    macro_rows = int(len(eval_order) * len(H5_SCENARIOS))
    mass /= float(macro_rows)
    require(abs(float(np.sum(mass)) - 1.0) <= H512R_FORMAL_TOL, "H512R_FOLD_MASS_NOT_ONE")
    return {
        "fold": fold,
        "groups": groups,
        "cell_mass": mass.tolist(),
        "unique_target_decision_clocks_by_cell": [len(x) for x in clocks_by_cell],
        "eval_dependence_groups": int(len(eval_order)),
        "macro_row_count": macro_rows,
        "account_degenerate_target_scenario_count": int(degenerate),
        "account_nondegenerate_target_scenario_count": int(nondegenerate),
        "account_context_eligible": bool(nondegenerate >= 1),
        "support_future_groups_min": int(support_min),
        "support_future_groups_max": int(support_max),
    }


def build_pair_design_h512r(payload_a: Mapping[str, Any], payload_b: Mapping[str, Any]) -> dict[str, Any]:
    fa = int(payload_a["fold"]); fb = int(payload_b["fold"])
    require((fa, fb) in H511_PAIRS, f"H512R_UNREGISTERED_PAIR:{fa}:{fb}")
    p_a = np.asarray(payload_a["cell_mass"], dtype=np.float64)
    p_b = np.asarray(payload_b["cell_mass"], dtype=np.float64)
    require(p_a.shape == p_b.shape == (H512R_CELL_N,), "H512R_PAIR_MASS_SHAPE")
    raw = np.zeros(H512R_CELL_N, dtype=np.float64)
    mask = (p_a > 0.0) & (p_b > 0.0)
    raw[mask] = p_a[mask] * p_b[mask] / (p_a[mask] + p_b[mask])
    require(float(np.sum(raw)) > H511_EPS, "H512R_NO_OVERLAP")
    q = raw / float(np.sum(raw))
    positive = np.flatnonzero(q > 0.0).astype(np.int32)
    op_bins = {int(c) // 9 for c in positive.tolist()}
    med_bins = {(int(c) % 9) // 3 for c in positive.tolist()}
    acc_bins = {int(c) % 3 for c in positive.tolist()}
    clocks_a = np.asarray(payload_a["unique_target_decision_clocks_by_cell"], dtype=np.int64)
    clocks_b = np.asarray(payload_b["unique_target_decision_clocks_by_cell"], dtype=np.int64)
    clock_ok = bool(all(int(clocks_a[c]) >= H512R_MIN_CELL_CLOCKS and int(clocks_b[c]) >= H512R_MIN_CELL_CLOCKS for c in positive.tolist()))
    axis_ok = bool(len(op_bins) >= 2 and len(med_bins) >= 2 and len(acc_bins) >= 2)
    account_ok = bool(payload_a["account_context_eligible"] and payload_b["account_context_eligible"])
    return {
        "pair": [fa, fb],
        "p_a": p_a.tolist(), "p_b": p_b.tolist(), "q": q.tolist(),
        "positive_q_cells": positive.astype(int).tolist(),
        "operator_bins_spanned": sorted(op_bins), "medium_bins_spanned": sorted(med_bins), "account_bins_spanned": sorted(acc_bins),
        "unique_target_decision_clocks_a_by_cell": clocks_a.astype(int).tolist(),
        "unique_target_decision_clocks_b_by_cell": clocks_b.astype(int).tolist(),
        "clock_support_gate_passed": clock_ok,
        "axis_span_gate_passed": axis_ok,
        "account_context_gate_passed": account_ok,
        "common_support_valid": bool(clock_ok and axis_ok and account_ok),
        "utility_used_to_construct_q": False,
    }


def _macro_cell_rows_h512r(payload: Mapping[str, Any]) -> list[np.ndarray]:
    rows = [np.asarray(s["cell_ids"], dtype=np.int32) for g in payload["groups"] for s in g["scenarios"]]
    require(len(rows) == int(payload["macro_row_count"]), "H512R_MACRO_ROW_COUNT_DRIFT")
    return rows


def _effective_mass_h512r(cell_rows: Sequence[np.ndarray], multiplier: np.ndarray) -> dict[str, Any]:
    m = np.asarray(multiplier, dtype=np.float64).reshape(-1)
    require(m.shape == (H512R_CELL_N,), "H512R_MULTIPLIER_SHAPE")
    require(np.isfinite(m).all() and np.all(m >= 0.0), "H512R_BAD_MULTIPLIER")
    mass = np.zeros(H512R_CELL_N, dtype=np.float64)
    raw_sums: list[float] = []
    zero_rows = 0
    for row in cell_rows:
        cells = np.asarray(row, dtype=np.int32).reshape(-1)
        w = m[cells]
        sw = float(np.sum(w))
        if not np.isfinite(sw) or sw <= H511_EPS:
            zero_rows += 1
            continue
        mass += np.bincount(cells, weights=w / sw, minlength=H512R_CELL_N).astype(np.float64)
        raw_sums.append(sw)
    if zero_rows == 0:
        mass /= float(len(cell_rows))
        require(abs(float(np.sum(mass)) - 1.0) <= H512R_FORMAL_TOL, "H512R_EFFECTIVE_MASS_NOT_ONE")
    return {
        "effective_macro_cell_mass": mass.tolist(),
        "zero_positive_support_macro_row_count": int(zero_rows),
        "macro_row_count": int(len(cell_rows)),
        "raw_local_weight_sum_min": float(min(raw_sums)) if raw_sums else 0.0,
        "raw_local_weight_sum_max": float(max(raw_sums)) if raw_sums else 0.0,
    }


def calibrate_ipf_h512r(*, cell_rows: Sequence[np.ndarray], p: Sequence[float], q: Sequence[float]) -> dict[str, Any]:
    pv = np.asarray(p, dtype=np.float64).reshape(-1)
    qv = np.asarray(q, dtype=np.float64).reshape(-1)
    require(pv.shape == qv.shape == (H512R_CELL_N,), "H512R_IPF_VECTOR_SHAPE")
    positive = qv > 0.0
    require(np.any(positive) and np.all(pv[positive] > 0.0), "H512R_IPF_BAD_POSITIVE_SUPPORT")
    m = np.zeros(H512R_CELL_N, dtype=np.float64)
    m[positive] = qv[positive] / pv[positive]

    def gauge(x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64).copy()
        require(np.all(y[positive] > 0.0) and np.isfinite(y[positive]).all(), "H512R_IPF_BAD_POSITIVE_MULTIPLIER")
        gm = float(np.exp(np.mean(np.log(y[positive]))))
        require(np.isfinite(gm) and gm > 0.0, "H512R_IPF_BAD_GAUGE")
        y[positive] /= gm
        y[~positive] = 0.0
        return y

    m = gauge(m)
    initial_receipt = _effective_mass_h512r(cell_rows, m)
    zero_rows = int(initial_receipt["zero_positive_support_macro_row_count"])
    if zero_rows > 0:
        return {
            "converged": False, "blocked_zero_support_rows": True, "zero_positive_support_macro_row_count": zero_rows,
            "iterations": 0, "multiplier": m.tolist(), "q": qv.tolist(),
        }
    cur = np.asarray(initial_receipt["effective_macro_cell_mass"], dtype=np.float64)
    initial_err = float(np.max(np.abs(cur - qv)))
    converged = False
    iterations = 0
    final_err = initial_err
    for it in range(1, H512R_MAX_ITER + 1):
        receipt = _effective_mass_h512r(cell_rows, m)
        require(int(receipt["zero_positive_support_macro_row_count"]) == 0, "H512R_IPF_ROW_SUPPORT_LOST")
        cur = np.asarray(receipt["effective_macro_cell_mass"], dtype=np.float64)
        err = float(np.max(np.abs(cur - qv)))
        if err <= H512R_INTERNAL_STOP:
            converged = True
            iterations = it - 1
            final_err = err
            break
        require(np.all(cur[positive] > 0.0), "H512R_IPF_ZERO_POSITIVE_MARGIN")
        m[positive] *= qv[positive] / cur[positive]
        m = gauge(m)
        iterations = it
    if not converged:
        receipt = _effective_mass_h512r(cell_rows, m)
        require(int(receipt["zero_positive_support_macro_row_count"]) == 0, "H512R_IPF_ROW_SUPPORT_LOST_FINAL")
        cur = np.asarray(receipt["effective_macro_cell_mass"], dtype=np.float64)
        final_err = float(np.max(np.abs(cur - qv)))
        converged = bool(final_err <= H512R_INTERNAL_STOP)
    posm = m[positive]
    return {
        "converged": bool(converged), "blocked_zero_support_rows": False,
        "zero_positive_support_macro_row_count": 0, "iterations": int(iterations),
        "initial_max_abs_effective_margin_minus_q": initial_err,
        "final_max_abs_effective_margin_minus_q": final_err,
        "multiplier": m.tolist(), "q": qv.tolist(),
        "positive_q_cell_count": int(np.sum(positive)),
        "multiplier_positive_min": float(np.min(posm)),
        "multiplier_positive_max": float(np.max(posm)),
        "multiplier_positive_max_to_min_ratio": float(np.max(posm) / np.min(posm)),
        "damping_used": False, "regularization_used": False, "clipping_used": False,
    }


def evaluate_payload_h512r(payload: Mapping[str, Any], multiplier: Sequence[float], q: Sequence[float]) -> dict[str, Any]:
    m = np.asarray(multiplier, dtype=np.float64).reshape(-1)
    qv = np.asarray(q, dtype=np.float64).reshape(-1)
    require(m.shape == qv.shape == (H512R_CELL_N,), "H512R_EVAL_SHAPE")
    group_rows: list[dict[str, Any]] = []
    zero_count = 0
    for group in payload["groups"]:
        scenario_rows: list[dict[str, Any]] = []
        for srow in group["scenarios"]:
            w = m[np.asarray(srow["cell_ids"], dtype=np.int32)]
            require(float(np.sum(w)) > H511_EPS, "H512R_EVAL_ZERO_LOCAL_WEIGHT")
            aligned, ar = weighted_partial_rank_h511(srow["operator_rank"], srow["medium_rank"], srow["utility_rank"], w)
            nulls: dict[int, float] = {}
            scenario_zero = bool(ar["zero_information"])
            for shift in H58_SHIFTS:
                rho, rr = weighted_partial_rank_h511(
                    srow["operator_rank"], srow["null_medium_rank_by_shift"][int(shift)], srow["utility_rank"], w
                )
                nulls[int(shift)] = float(rho)
                scenario_zero = bool(scenario_zero or rr["zero_information"])
            zero_count += int(scenario_zero)
            scenario_rows.append({"aligned": float(aligned), "nulls": nulls})
        group_rows.append({
            "aligned": float(np.mean([x["aligned"] for x in scenario_rows])),
            "nulls": {int(sh): float(np.mean([x["nulls"][int(sh)] for x in scenario_rows])) for sh in H58_SHIFTS},
        })
    aligned = float(np.mean([x["aligned"] for x in group_rows]))
    nulls = {int(sh): float(np.mean([x["nulls"][int(sh)] for x in group_rows])) for sh in H58_SHIFTS}
    med = float(statistics.median(nulls.values()))
    eff = _effective_mass_h512r(_macro_cell_rows_h512r(payload), m)
    require(int(eff["zero_positive_support_macro_row_count"]) == 0, "H512R_EVAL_ZERO_SUPPORT_ROW")
    em = np.asarray(eff["effective_macro_cell_mass"], dtype=np.float64)
    return {
        "fold": int(payload["fold"]), "aligned_partial_rho": aligned,
        "null_partial_rho_by_shift": nulls, "null_median_partial_rho": med,
        "aligned_minus_null_median": float(aligned - med),
        "orientation": orientation_h510(aligned, med),
        "zero_information_target_scenario_count": int(zero_count),
        "effective_macro_cell_mass": em.tolist(),
        "effective_max_abs_error_vs_q": float(np.max(np.abs(em - qv))),
        "same_aligned_candidate_weights_used_for_all_nulls": True,
        "q_recomputed_for_null": False,
    }


def run_pair_h512r(*, design: Mapping[str, Any], payload_a: Mapping[str, Any], payload_b: Mapping[str, Any]) -> dict[str, Any]:
    fa, fb = [int(x) for x in design["pair"]]
    q = np.asarray(design["q"], dtype=np.float64)
    if not bool(design["account_context_gate_passed"]):
        return {"pair": [fa, fb], "blocked": "EXECUTION_BLOCKED__ACCOUNT6_RELATIONAL_CONTEXT_DEGENERATE"}
    if not bool(design["clock_support_gate_passed"] and design["axis_span_gate_passed"]):
        return {"pair": [fa, fb], "blocked": "EXECUTION_BLOCKED__INSUFFICIENT_3D_COMMON_SUPPORT"}
    cal_a = calibrate_ipf_h512r(cell_rows=_macro_cell_rows_h512r(payload_a), p=design["p_a"], q=q)
    cal_b = calibrate_ipf_h512r(cell_rows=_macro_cell_rows_h512r(payload_b), p=design["p_b"], q=q)
    if bool(cal_a.get("blocked_zero_support_rows")) or bool(cal_b.get("blocked_zero_support_rows")):
        return {"pair": [fa, fb], "blocked": "EXECUTION_BLOCKED__TARGET_SCENARIO_HAS_NO_3D_COMMON_SUPPORT", "calibration": {"side_a": cal_a, "side_b": cal_b}}
    if not bool(cal_a["converged"] and cal_b["converged"]):
        return {"pair": [fa, fb], "blocked": "EXECUTION_BLOCKED__IPF_3D_CALIBRATION_INFEASIBLE_OR_NONCONVERGENT", "calibration": {"side_a": cal_a, "side_b": cal_b}}
    ea = evaluate_payload_h512r(payload_a, cal_a["multiplier"], q)
    eb = evaluate_payload_h512r(payload_b, cal_b["multiplier"], q)
    side_diff = float(np.max(np.abs(np.asarray(ea["effective_macro_cell_mass"]) - np.asarray(eb["effective_macro_cell_mass"]))))
    closure = bool(ea["effective_max_abs_error_vs_q"] <= H512R_FORMAL_TOL and eb["effective_max_abs_error_vs_q"] <= H512R_FORMAL_TOL and side_diff <= H512R_SIDE_TOL)
    return {
        "pair": [fa, fb], "blocked": None, "q": q.tolist(),
        "positive_q_cells": list(design["positive_q_cells"]),
        "calibration": {"side_a": cal_a, "side_b": cal_b},
        "side_a": ea, "side_b": eb,
        "effective_side_to_side_max_abs": side_diff,
        "exact_effective_margin_closure": closure,
    }


def classify_h512r(pair_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocked = [str(x["blocked"]) for x in pair_results if x.get("blocked")]
    if blocked:
        classification = blocked[0]
        return {"classification": classification, "conclusion": f"H5_12R_{classification}", "native_flip_persistence_count_of_3": 0, "native_flip_clean_removal_count_of_3": 0, "positive_control_pair_stable": False}
    if not all(bool(x["exact_effective_margin_closure"]) for x in pair_results):
        classification = "EXECUTION_BLOCKED__IPF_3D_EFFECTIVE_MARGIN_CLOSURE_FAILED"
        return {"classification": classification, "conclusion": f"H5_12R_{classification}", "native_flip_persistence_count_of_3": 0, "native_flip_clean_removal_count_of_3": 0, "positive_control_pair_stable": False}
    by_pair = {(int(x["pair"][0]), int(x["pair"][1])): x for x in pair_results}
    retained = []
    removed = []
    for pair in H512R_FLIP_PAIRS:
        row = by_pair[pair]
        oa = str(row["side_a"]["orientation"]); ob = str(row["side_b"]["orientation"])
        retained.append(oa == H511_NATIVE_ORIENTATION[pair[0]] and ob == H511_NATIVE_ORIENTATION[pair[1]])
        removed.append(oa == ob and oa in {"POSITIVE_ALIGNMENT", "ANTI_ALIGNMENT"})
    ctrl = by_pair[H512R_CONTROL_PAIR]
    control = bool(ctrl["side_a"]["orientation"] == "POSITIVE_ALIGNMENT" and ctrl["side_b"]["orientation"] == "POSITIVE_ALIGNMENT")
    if all(retained) and control:
        classification = "ACCOUNT6_EXACT_EFFECTIVE_MARGIN_STANDARDIZATION__NATIVE_TRANSITIONS_PERSIST"
    elif all(removed) and control:
        classification = "ACCOUNT6_EXACT_EFFECTIVE_MARGIN_STANDARDIZATION__NATIVE_TRANSITIONS_REMOVED"
    else:
        classification = "ACCOUNT6_EXACT_EFFECTIVE_MARGIN_STANDARDIZATION__MIXED"
    return {
        "classification": classification, "conclusion": f"H5_12R_{classification}",
        "native_flip_persistence_count_of_3": int(sum(retained)),
        "native_flip_clean_removal_count_of_3": int(sum(removed)),
        "positive_control_pair_stable": control,
    }
