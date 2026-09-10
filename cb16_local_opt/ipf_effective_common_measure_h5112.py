from __future__ import annotations

"""H5.11.2 exact effective-macro-cell-margin calibration.

This module preserves the H5.11 scientific statistic and changes only the
cell-multiplier estimation.  The H5.8/H5.11 weighted rank statistic normalizes
candidate weights separately inside each target-future-group x scenario.  We
therefore calibrate the nine Operator/Medium cell multipliers against the
*effective* macro candidate-cell margin after that local normalization.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .common_support_rank_geometry_decomposition_h511 import (
    H511_CELL_N,
    H511_CONTROL_PAIR,
    H511_EPS,
    H511_FLIP_PAIRS,
    H511_NATIVE_ORIENTATION,
    H511_PAIRS,
    _post_weight_cell_mass_h511,
    weighted_partial_rank_h511,
)
from .medium48_temporal_halfblock_stability_h510 import orientation_h510
from .operator_conditional_medium_geometry_h58 import H58_SHIFTS
from .teacher_temporal_transport_audit_h5 import require

H5112_RUNTIME = "CB16_R11_H5_11_2_IPF_EFFECTIVE_COMMON_MEASURE_R0_V1"
H5112_INTERNAL_STOP = 1e-13
H5112_FORMAL_TOL = 1e-12
H5112_SIDE_TO_SIDE_TOL = 2e-12
H5112_MAX_ITER = 10000


def payload_cell_rows_h5112(payload: Mapping[str, Any]) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for group in payload["groups"]:
        for scenario in group["scenarios"]:
            cells = np.asarray(scenario["cell_ids"], dtype=np.int32).reshape(-1)
            require(cells.size >= 1, "H5112_EMPTY_CELL_ROW")
            require(np.all(cells >= 0) and np.all(cells < H511_CELL_N), "H5112_CELL_RANGE")
            rows.append(cells)
    require(len(rows) >= 1, "H5112_NO_MACRO_ROWS")
    return rows


def effective_macro_cell_mass_h5112(
    cell_rows: Sequence[np.ndarray], multiplier: Sequence[float]
) -> dict[str, Any]:
    m = np.asarray(multiplier, dtype=np.float64).reshape(-1)
    require(m.shape == (H511_CELL_N,), "H5112_MULTIPLIER_SHAPE")
    require(np.isfinite(m).all() and np.all(m > 0.0), "H5112_BAD_MULTIPLIER")
    require(len(cell_rows) >= 1, "H5112_NO_MACRO_ROWS")

    mass = np.zeros(H511_CELL_N, dtype=np.float64)
    row_sums: list[float] = []
    counts: list[int] = []
    for row in cell_rows:
        cells = np.asarray(row, dtype=np.int32).reshape(-1)
        require(cells.size >= 1, "H5112_EMPTY_CELL_ROW")
        require(np.all(cells >= 0) and np.all(cells < H511_CELL_N), "H5112_CELL_RANGE")
        w = m[cells]
        sw = float(np.sum(w))
        require(np.isfinite(sw) and sw > H511_EPS, "H5112_ZERO_LOCAL_WEIGHT")
        mass += np.bincount(cells, weights=w / sw, minlength=H511_CELL_N).astype(np.float64)
        row_sums.append(sw)
        counts.append(int(cells.size))

    mass /= float(len(cell_rows))
    require(np.isfinite(mass).all(), "H5112_EFFECTIVE_MASS_NONFINITE")
    require(abs(float(np.sum(mass)) - 1.0) <= H5112_FORMAL_TOL, "H5112_EFFECTIVE_MASS_NOT_ONE")
    return {
        "effective_macro_cell_mass": mass.tolist(),
        "macro_row_count": int(len(cell_rows)),
        "candidate_count_min": int(min(counts)),
        "candidate_count_max": int(max(counts)),
        "raw_local_weight_sum_min": float(min(row_sums)),
        "raw_local_weight_sum_max": float(max(row_sums)),
        "raw_local_weight_sum_mean": float(np.mean(row_sums)),
        "raw_local_weight_sum_std": float(np.std(row_sums, ddof=0)),
    }


def _remove_global_gauge_h5112(multiplier: np.ndarray) -> np.ndarray:
    m = np.asarray(multiplier, dtype=np.float64).reshape(-1)
    require(m.shape == (H511_CELL_N,), "H5112_GAUGE_SHAPE")
    require(np.isfinite(m).all() and np.all(m > 0.0), "H5112_GAUGE_BAD_MULTIPLIER")
    gm = float(np.exp(np.mean(np.log(m))))
    require(np.isfinite(gm) and gm > 0.0, "H5112_BAD_GEOMETRIC_MEAN")
    out = m / gm
    require(np.isfinite(out).all() and np.all(out > 0.0), "H5112_GAUGE_NONFINITE")
    return out


def calibrate_ipf_h5112(
    *,
    cell_rows: Sequence[np.ndarray],
    q: Sequence[float],
    initial_multiplier: Sequence[float],
) -> dict[str, Any]:
    qv = np.asarray(q, dtype=np.float64).reshape(-1)
    require(qv.shape == (H511_CELL_N,), "H5112_Q_SHAPE")
    require(np.isfinite(qv).all() and np.all(qv > 0.0), "H5112_Q_MUST_BE_STRICTLY_POSITIVE")
    require(abs(float(np.sum(qv)) - 1.0) <= H5112_FORMAL_TOL, "H5112_Q_NOT_ONE")
    m = _remove_global_gauge_h5112(np.asarray(initial_multiplier, dtype=np.float64))

    initial = np.asarray(effective_macro_cell_mass_h5112(cell_rows, m)["effective_macro_cell_mass"], dtype=np.float64)
    initial_err = float(np.max(np.abs(initial - qv)))
    converged = False
    iterations = 0
    final_mass = initial
    final_err = initial_err

    for it in range(1, H5112_MAX_ITER + 1):
        cur = np.asarray(effective_macro_cell_mass_h5112(cell_rows, m)["effective_macro_cell_mass"], dtype=np.float64)
        err = float(np.max(np.abs(cur - qv)))
        if err <= H5112_INTERNAL_STOP:
            converged = True
            iterations = it - 1
            final_mass = cur
            final_err = err
            break
        require(np.all(cur > 0.0), "H5112_ZERO_EFFECTIVE_CELL_DURING_IPF")
        ratio = qv / cur
        require(np.isfinite(ratio).all() and np.all(ratio > 0.0), "H5112_BAD_IPF_RATIO")
        m = _remove_global_gauge_h5112(m * ratio)
        iterations = it
    if not converged:
        final_mass = np.asarray(effective_macro_cell_mass_h5112(cell_rows, m)["effective_macro_cell_mass"], dtype=np.float64)
        final_err = float(np.max(np.abs(final_mass - qv)))
        converged = bool(final_err <= H5112_INTERNAL_STOP)

    return {
        "converged": bool(converged),
        "iterations": int(iterations),
        "internal_stop_tolerance": H5112_INTERNAL_STOP,
        "max_iterations": H5112_MAX_ITER,
        "initial_max_abs_effective_margin_minus_q": float(initial_err),
        "final_max_abs_effective_margin_minus_q": float(final_err),
        "multiplier": m.tolist(),
        "multiplier_min": float(np.min(m)),
        "multiplier_max": float(np.max(m)),
        "multiplier_max_to_min_ratio": float(np.max(m) / np.min(m)),
        "final_effective_macro_cell_mass": final_mass.tolist(),
        "q": qv.tolist(),
        "damping_used": False,
        "regularization_used": False,
        "clipping_used": False,
    }


def evaluate_fold_multiplier_h5112(
    payload: Mapping[str, Any], multiplier: Sequence[float], q: Sequence[float]
) -> dict[str, Any]:
    """Exact H5.11 statistic with only the obsolete pooled-Q receipt removed."""
    m = np.asarray(multiplier, dtype=np.float64).reshape(-1)
    qv = np.asarray(q, dtype=np.float64).reshape(-1)
    require(m.shape == qv.shape == (H511_CELL_N,), "H5112_EVAL_WEIGHT_SHAPE")
    require(np.isfinite(m).all() and np.all(m > 0.0), "H5112_EVAL_BAD_MULTIPLIER")

    group_rows: list[dict[str, Any]] = []
    zero_count = 0
    min_weight_sum = float("inf")
    max_weight_sum = 0.0
    for group in payload["groups"]:
        scenario_rows: list[dict[str, Any]] = []
        for srow in group["scenarios"]:
            cells = np.asarray(srow["cell_ids"], dtype=np.int32)
            w = m[cells]
            sw = float(np.sum(w))
            require(sw > H511_EPS, "H5112_TARGET_SCENARIO_ZERO_WEIGHT")
            min_weight_sum = min(min_weight_sum, sw)
            max_weight_sum = max(max_weight_sum, sw)
            aligned, ar = weighted_partial_rank_h511(
                srow["operator_rank"], srow["medium_rank"], srow["utility_rank"], w
            )
            nulls: dict[int, float] = {}
            scenario_zero = bool(ar["zero_information"])
            for shift in H58_SHIFTS:
                rho, rr = weighted_partial_rank_h511(
                    srow["operator_rank"],
                    srow["null_medium_rank_by_shift"][int(shift)],
                    srow["utility_rank"],
                    w,
                )
                nulls[int(shift)] = float(rho)
                scenario_zero = bool(scenario_zero or rr["zero_information"])
            zero_count += int(scenario_zero)
            scenario_rows.append({"aligned": float(aligned), "nulls": nulls})
        group_rows.append({
            "aligned": float(np.mean([x["aligned"] for x in scenario_rows])),
            "nulls": {
                int(shift): float(np.mean([x["nulls"][int(shift)] for x in scenario_rows]))
                for shift in H58_SHIFTS
            },
        })

    aligned = float(np.mean([x["aligned"] for x in group_rows]))
    nulls = {
        int(shift): float(np.mean([x["nulls"][int(shift)] for x in group_rows]))
        for shift in H58_SHIFTS
    }
    median_null = float(statistics.median(nulls.values()))
    cell_rows = payload_cell_rows_h5112(payload)
    effective = effective_macro_cell_mass_h5112(cell_rows, m)
    effective_mass = np.asarray(effective["effective_macro_cell_mass"], dtype=np.float64)
    effective_err = float(np.max(np.abs(effective_mass - qv)))
    p = np.asarray(payload["cell_mass"], dtype=np.float64)
    nominal = _post_weight_cell_mass_h511(p, m)

    return {
        "fold": int(payload["fold"]),
        "aligned_partial_rho": aligned,
        "null_partial_rho_by_shift": nulls,
        "null_median_partial_rho": median_null,
        "aligned_minus_null_median": float(aligned - median_null),
        "orientation": orientation_h510(aligned, median_null),
        "zero_information_target_scenario_count": int(zero_count),
        "effective_macro_cell_mass": effective_mass.tolist(),
        "effective_max_abs_error_vs_q": effective_err,
        "nominal_pooled_post_weight_cell_mass": nominal.tolist(),
        "nominal_pooled_max_abs_error_vs_q": float(np.max(np.abs(nominal - qv))),
        "target_scenario_weight_sum_min": float(min_weight_sum),
        "target_scenario_weight_sum_max": float(max_weight_sum),
        "same_aligned_candidate_weights_used_for_all_nulls": True,
        "q_recomputed_for_null": False,
    }


def run_pair_h5112(
    *,
    pair_design: Mapping[str, Any],
    payload_a: Mapping[str, Any],
    payload_b: Mapping[str, Any],
) -> dict[str, Any]:
    fa, fb = [int(x) for x in pair_design["pair"]]
    require((fa, fb) in H511_PAIRS, f"H5112_UNREGISTERED_PAIR:{fa}:{fb}")
    require(int(payload_a["fold"]) == fa and int(payload_b["fold"]) == fb, "H5112_PAIR_PAYLOAD_DRIFT")
    require(bool(pair_design["common_support_valid"]), "H5112_PARENT_COMMON_SUPPORT_INVALID")
    q = np.asarray(pair_design["q"], dtype=np.float64)
    require(q.shape == (H511_CELL_N,) and np.all(q > 0.0), "H5112_PAIR_Q_NOT_FULL_SUPPORT")

    old_a = np.asarray(pair_design["multiplier_a"], dtype=np.float64)
    old_b = np.asarray(pair_design["multiplier_b"], dtype=np.float64)
    reproduction_a = evaluate_fold_multiplier_h5112(payload_a, old_a, q)
    reproduction_b = evaluate_fold_multiplier_h5112(payload_b, old_b, q)

    cal_a = calibrate_ipf_h5112(
        cell_rows=payload_cell_rows_h5112(payload_a), q=q, initial_multiplier=old_a
    )
    cal_b = calibrate_ipf_h5112(
        cell_rows=payload_cell_rows_h5112(payload_b), q=q, initial_multiplier=old_b
    )
    eval_a = evaluate_fold_multiplier_h5112(payload_a, cal_a["multiplier"], q)
    eval_b = evaluate_fold_multiplier_h5112(payload_b, cal_b["multiplier"], q)
    side_diff = float(np.max(np.abs(
        np.asarray(eval_a["effective_macro_cell_mass"], dtype=np.float64)
        - np.asarray(eval_b["effective_macro_cell_mass"], dtype=np.float64)
    )))
    closure = bool(
        cal_a["converged"]
        and cal_b["converged"]
        and float(eval_a["effective_max_abs_error_vs_q"]) <= H5112_FORMAL_TOL
        and float(eval_b["effective_max_abs_error_vs_q"]) <= H5112_FORMAL_TOL
        and side_diff <= H5112_SIDE_TO_SIDE_TOL
    )
    return {
        "pair": [fa, fb],
        "q": q.tolist(),
        "old_h5_11_reproduction": {"side_a": reproduction_a, "side_b": reproduction_b},
        "calibration": {"side_a": cal_a, "side_b": cal_b},
        "side_a": eval_a,
        "side_b": eval_b,
        "effective_side_to_side_max_abs": side_diff,
        "exact_effective_margin_closure": closure,
    }


def classify_h5112(pair_results: Sequence[Mapping[str, Any]], reproduction_passed: bool) -> dict[str, Any]:
    by_pair = {(int(x["pair"][0]), int(x["pair"][1])): x for x in pair_results}
    require(tuple(sorted(by_pair)) == H511_PAIRS, "H5112_PAIR_SET_DRIFT")
    if not reproduction_passed:
        classification = "EXECUTION_BLOCKED__H5_11_SIBLING_EVALUATOR_REPRODUCTION_FAILED"
    elif not all(bool(x["calibration"]["side_a"]["converged"]) and bool(x["calibration"]["side_b"]["converged"]) for x in pair_results):
        classification = "EXECUTION_BLOCKED__IPF_CALIBRATION_INFEASIBLE_OR_NONCONVERGENT"
    elif not all(bool(x["exact_effective_margin_closure"]) for x in pair_results):
        classification = "EXECUTION_BLOCKED__IPF_EFFECTIVE_MARGIN_CLOSURE_FAILED"
    else:
        retained = []
        removed = []
        for pair in H511_FLIP_PAIRS:
            row = by_pair[pair]
            oa = str(row["side_a"]["orientation"])
            ob = str(row["side_b"]["orientation"])
            retained.append(oa == H511_NATIVE_ORIENTATION[pair[0]] and ob == H511_NATIVE_ORIENTATION[pair[1]])
            removed.append(oa == ob and oa in {"POSITIVE_ALIGNMENT", "ANTI_ALIGNMENT"})
        ctrl = by_pair[H511_CONTROL_PAIR]
        control_stable = bool(
            str(ctrl["side_a"]["orientation"]) == "POSITIVE_ALIGNMENT"
            and str(ctrl["side_b"]["orientation"]) == "POSITIVE_ALIGNMENT"
        )
        if all(retained) and control_stable:
            classification = "EXACT_MACRO_CELL_OCCUPANCY_STANDARDIZATION__NATIVE_TRANSITIONS_PERSIST"
        elif all(removed) and control_stable:
            classification = "EXACT_MACRO_CELL_OCCUPANCY_STANDARDIZATION__NATIVE_TRANSITIONS_REMOVED"
        else:
            classification = "EXACT_MACRO_CELL_OCCUPANCY_STANDARDIZATION__MIXED"

    persist = 0
    clean_removed = 0
    if classification.startswith("EXACT_MACRO_CELL_OCCUPANCY_STANDARDIZATION__"):
        for pair in H511_FLIP_PAIRS:
            row = by_pair[pair]
            oa = str(row["side_a"]["orientation"])
            ob = str(row["side_b"]["orientation"])
            persist += int(oa == H511_NATIVE_ORIENTATION[pair[0]] and ob == H511_NATIVE_ORIENTATION[pair[1]])
            clean_removed += int(oa == ob and oa in {"POSITIVE_ALIGNMENT", "ANTI_ALIGNMENT"})
    ctrl = by_pair[H511_CONTROL_PAIR]
    return {
        "classification": classification,
        "conclusion": f"H5_11_2_{classification}",
        "native_flip_persistence_count_of_3": int(persist),
        "native_flip_clean_removal_count_of_3": int(clean_removed),
        "positive_control_pair_stable": bool(
            str(ctrl["side_a"]["orientation"]) == "POSITIVE_ALIGNMENT"
            and str(ctrl["side_b"]["orientation"]) == "POSITIVE_ALIGNMENT"
        ),
        "market_information_verdict_changed": False,
        "canonical_change_authorized": False,
        "h5_12_execution_authorized": False,
    }
