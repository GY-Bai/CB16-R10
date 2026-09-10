from __future__ import annotations

"""H5.14 target-side Operator/Medium neighborhood-concordance falsification.

The only scientific manipulation is the target future-group empirical measure
across a preregistered outcome-blind scalar kappa = Spearman(d_Operator,
d_Medium) on the frozen local support.  Candidate-side H5.12R 27-cell margins
are re-calibrated to the exact same pair-specific Q under the new target row
masses so target and candidate occupancy are not confounded.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .account6_ipf_effective_common_measure_h512r import H512R_CELL_N
from .common_support_rank_geometry_decomposition_h511 import H511_EPS, H511_NATIVE_ORIENTATION, H511_PAIRS, weighted_partial_rank_h511
from .medium48_temporal_halfblock_stability_h510 import orientation_h510
from .operator_conditional_medium_geometry_h58 import H58_SHIFTS
from .teacher_temporal_transport_audit_h5 import require

H514_RUNTIME = "CB16_R11_H5_14_TARGET_OM_CONCORDANCE_OCCUPANCY_R0_V1"
H514_TARGET_BINS = 3
H514_MIN_TARGETS_PER_BIN = 10
H514_TOL = 1e-12
H514_INTERNAL_STOP = 1e-13
H514_SIDE_TOL = 2e-12
H514_MAX_ITER = 10000
H514_FLIP_PAIRS = ((1, 2), (2, 3), (3, 4))
H514_CONTROL_PAIR = (4, 5)


def _unweighted_corr_h514(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    require(a.shape == b.shape and a.size >= 2, "H514_CORR_SHAPE")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "H514_CORR_NONFINITE")
    ac = a - float(np.mean(a)); bc = b - float(np.mean(b))
    den = float(np.sqrt(np.sum(ac * ac) * np.sum(bc * bc)))
    require(np.isfinite(den) and den > H511_EPS, "H514_KAPPA_DEGENERATE")
    raw = float(np.sum(ac * bc) / den)
    require(np.isfinite(raw) and -1.0 - 1e-10 <= raw <= 1.0 + 1e-10, "H514_KAPPA_RANGE")
    return float(np.clip(raw, -1.0, 1.0))


def target_kappa_payload_h514(payload: Mapping[str, Any]) -> dict[str, Any]:
    group_ids: list[str] = []
    kappas: list[float] = []
    max_scenario_spread = 0.0
    scenario_values: dict[str, list[float]] = {}
    for group in payload["groups"]:
        gid = str(group["future_group_id"])
        vals = [
            _unweighted_corr_h514(row["operator_rank"], row["medium_rank"])
            for row in group["scenarios"]
        ]
        require(len(vals) == 6, "H514_SCENARIO_COUNT_DRIFT")
        spread = float(max(vals) - min(vals))
        max_scenario_spread = max(max_scenario_spread, spread)
        group_ids.append(gid)
        kappas.append(float(np.mean(vals)))
        scenario_values[gid] = vals
    require(len(group_ids) == int(payload["eval_dependence_groups"]), "H514_TARGET_GROUP_COUNT_DRIFT")
    return {
        "fold": int(payload["fold"]),
        "future_group_ids": group_ids,
        "kappa": kappas,
        "max_scenario_kappa_spread": max_scenario_spread,
        "scenario_consistency_passed": bool(max_scenario_spread <= H514_TOL),
        "utility_used": False,
    }


def _target_bin_ids_h514(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    e = np.asarray(edges, dtype=np.float64).reshape(-1)
    require(e.shape == (2,) and np.isfinite(e).all() and e[0] <= e[1], "H514_BAD_BIN_EDGES")
    return np.searchsorted(e, x, side="right").astype(np.int32)


def build_target_pair_design_h514(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    fa = int(a["fold"]); fb = int(b["fold"])
    require((fa, fb) in H511_PAIRS, f"H514_UNREGISTERED_PAIR:{fa}:{fb}")
    ka = np.asarray(a["kappa"], dtype=np.float64)
    kb = np.asarray(b["kappa"], dtype=np.float64)
    require(ka.size >= 30 and kb.size >= 30 and np.isfinite(ka).all() and np.isfinite(kb).all(), "H514_BAD_KAPPA_VECTOR")
    pooled = np.concatenate([ka, kb])
    edges = np.quantile(pooled, [1.0 / 3.0, 2.0 / 3.0], method="linear")
    ba = _target_bin_ids_h514(ka, edges); bb = _target_bin_ids_h514(kb, edges)
    ca = np.bincount(ba, minlength=H514_TARGET_BINS).astype(np.int64)
    cb = np.bincount(bb, minlength=H514_TARGET_BINS).astype(np.int64)
    pa = ca.astype(np.float64) / float(len(ka)); pb = cb.astype(np.float64) / float(len(kb))
    raw = np.zeros(H514_TARGET_BINS, dtype=np.float64)
    mask = (pa > 0.0) & (pb > 0.0)
    raw[mask] = pa[mask] * pb[mask] / (pa[mask] + pb[mask])
    require(float(np.sum(raw)) > H511_EPS, "H514_NO_TARGET_KAPPA_OVERLAP")
    qk = raw / float(np.sum(raw))
    positive = np.flatnonzero(qk > 0.0)
    support_ok = bool(
        len(positive) == 3
        and all(int(ca[i]) >= H514_MIN_TARGETS_PER_BIN and int(cb[i]) >= H514_MIN_TARGETS_PER_BIN for i in positive)
    )

    ma = np.zeros(H514_TARGET_BINS, dtype=np.float64); mb = np.zeros(H514_TARGET_BINS, dtype=np.float64)
    ma[positive] = qk[positive] / pa[positive]
    mb[positive] = qk[positive] / pb[positive]
    wa = ma[ba]; wb = mb[bb]
    require(float(np.sum(wa)) > H511_EPS and float(np.sum(wb)) > H511_EPS, "H514_ZERO_TARGET_WEIGHT")
    wa = wa / float(np.sum(wa)); wb = wb / float(np.sum(wb))
    eff_a = np.bincount(ba, weights=wa, minlength=3).astype(np.float64)
    eff_b = np.bincount(bb, weights=wb, minlength=3).astype(np.float64)
    require(float(np.max(np.abs(eff_a - qk))) <= H514_TOL, "H514_TARGET_Q_A_MISMATCH")
    require(float(np.max(np.abs(eff_b - qk))) <= H514_TOL, "H514_TARGET_Q_B_MISMATCH")
    return {
        "pair": [fa, fb],
        "edges": edges.tolist(),
        "counts_a": ca.astype(int).tolist(), "counts_b": cb.astype(int).tolist(),
        "p_a": pa.tolist(), "p_b": pb.tolist(), "qk": qk.tolist(),
        "bin_ids_a": ba.astype(int).tolist(), "bin_ids_b": bb.astype(int).tolist(),
        "target_weights_a": wa.tolist(), "target_weights_b": wb.tolist(),
        "effective_target_bin_mass_a": eff_a.tolist(), "effective_target_bin_mass_b": eff_b.tolist(),
        "common_support_valid": support_ok,
        "scenario_consistency_passed": bool(a["scenario_consistency_passed"] and b["scenario_consistency_passed"]),
        "utility_used_to_construct_target_measure": False,
    }


def macro_row_masses_h514(payload: Mapping[str, Any], group_weights: Sequence[float]) -> np.ndarray:
    gw = np.asarray(group_weights, dtype=np.float64).reshape(-1)
    require(gw.shape == (len(payload["groups"]),), "H514_GROUP_WEIGHT_SHAPE")
    require(np.isfinite(gw).all() and np.all(gw >= 0.0), "H514_BAD_GROUP_WEIGHT")
    require(abs(float(np.sum(gw)) - 1.0) <= H514_TOL, "H514_GROUP_WEIGHT_NOT_ONE")
    rows: list[float] = []
    for w, group in zip(gw.tolist(), payload["groups"]):
        require(len(group["scenarios"]) == 6, "H514_SCENARIO_COUNT_DRIFT")
        rows.extend([float(w) / 6.0] * 6)
    out = np.asarray(rows, dtype=np.float64)
    require(abs(float(np.sum(out)) - 1.0) <= H514_TOL, "H514_ROW_MASS_NOT_ONE")
    return out


def macro_cell_rows_h514(payload: Mapping[str, Any]) -> list[np.ndarray]:
    rows = [np.asarray(s["cell_ids"], dtype=np.int32) for g in payload["groups"] for s in g["scenarios"]]
    require(len(rows) == int(payload["macro_row_count"]), "H514_MACRO_ROW_COUNT_DRIFT")
    return rows


def weighted_effective_cell_mass_h514(
    cell_rows: Sequence[np.ndarray], row_masses: Sequence[float], multiplier: Sequence[float]
) -> dict[str, Any]:
    rm = np.asarray(row_masses, dtype=np.float64).reshape(-1)
    m = np.asarray(multiplier, dtype=np.float64).reshape(-1)
    require(len(cell_rows) == len(rm) and len(cell_rows) >= 1, "H514_EFFECTIVE_ROW_SHAPE")
    require(m.shape == (H512R_CELL_N,), "H514_EFFECTIVE_MULTIPLIER_SHAPE")
    require(np.isfinite(rm).all() and np.all(rm >= 0.0) and abs(float(np.sum(rm)) - 1.0) <= H514_TOL, "H514_BAD_ROW_MASS")
    require(np.isfinite(m).all() and np.all(m > 0.0), "H514_BAD_CANDIDATE_MULTIPLIER")
    mass = np.zeros(H512R_CELL_N, dtype=np.float64)
    zero_rows = 0
    raw_sums: list[float] = []
    for cells, row_mass in zip(cell_rows, rm.tolist()):
        c = np.asarray(cells, dtype=np.int32).reshape(-1)
        w = m[c]
        sw = float(np.sum(w))
        if not np.isfinite(sw) or sw <= H511_EPS:
            zero_rows += 1
            continue
        mass += float(row_mass) * np.bincount(c, weights=w / sw, minlength=H512R_CELL_N).astype(np.float64)
        raw_sums.append(sw)
    if zero_rows == 0:
        require(abs(float(np.sum(mass)) - 1.0) <= H514_TOL, "H514_EFFECTIVE_CELL_MASS_NOT_ONE")
    return {
        "effective_cell_mass": mass.tolist(),
        "zero_positive_support_macro_row_count": int(zero_rows),
        "raw_local_weight_sum_min": float(min(raw_sums)) if raw_sums else 0.0,
        "raw_local_weight_sum_max": float(max(raw_sums)) if raw_sums else 0.0,
    }


def _positive_gauge_h514(m: np.ndarray) -> np.ndarray:
    x = np.asarray(m, dtype=np.float64).reshape(-1).copy()
    require(x.shape == (H512R_CELL_N,) and np.isfinite(x).all() and np.all(x > 0.0), "H514_GAUGE_BAD_MULTIPLIER")
    gm = float(np.exp(np.mean(np.log(x))))
    require(np.isfinite(gm) and gm > 0.0, "H514_GAUGE_BAD_MEAN")
    return x / gm


def calibrate_candidate_margin_h514(
    *, cell_rows: Sequence[np.ndarray], row_masses: Sequence[float], q27: Sequence[float], initial_multiplier: Sequence[float]
) -> dict[str, Any]:
    q = np.asarray(q27, dtype=np.float64).reshape(-1)
    require(q.shape == (H512R_CELL_N,) and np.isfinite(q).all() and np.all(q > 0.0), "H514_Q27_BAD")
    require(abs(float(np.sum(q)) - 1.0) <= H514_TOL, "H514_Q27_NOT_ONE")
    m = _positive_gauge_h514(np.asarray(initial_multiplier, dtype=np.float64))
    first = weighted_effective_cell_mass_h514(cell_rows, row_masses, m)
    if int(first["zero_positive_support_macro_row_count"]) > 0:
        return {"converged": False, "blocked_zero_support_rows": True, "iterations": 0, "multiplier": m.tolist()}
    cur = np.asarray(first["effective_cell_mass"], dtype=np.float64)
    initial_err = float(np.max(np.abs(cur - q)))
    converged = False; iterations = 0; final_err = initial_err
    for it in range(1, H514_MAX_ITER + 1):
        receipt = weighted_effective_cell_mass_h514(cell_rows, row_masses, m)
        require(int(receipt["zero_positive_support_macro_row_count"]) == 0, "H514_CANDIDATE_SUPPORT_LOST")
        cur = np.asarray(receipt["effective_cell_mass"], dtype=np.float64)
        err = float(np.max(np.abs(cur - q)))
        if err <= H514_INTERNAL_STOP:
            converged = True; iterations = it - 1; final_err = err; break
        require(np.all(cur > 0.0), "H514_ZERO_CANDIDATE_MARGIN")
        m = _positive_gauge_h514(m * (q / cur))
        iterations = it
    if not converged:
        receipt = weighted_effective_cell_mass_h514(cell_rows, row_masses, m)
        cur = np.asarray(receipt["effective_cell_mass"], dtype=np.float64)
        final_err = float(np.max(np.abs(cur - q)))
        converged = bool(final_err <= H514_INTERNAL_STOP)
    return {
        "converged": bool(converged), "blocked_zero_support_rows": False,
        "iterations": int(iterations), "initial_max_abs_error_vs_q27": initial_err,
        "final_max_abs_error_vs_q27": final_err, "multiplier": m.tolist(),
        "multiplier_min": float(np.min(m)), "multiplier_max": float(np.max(m)),
        "multiplier_max_to_min_ratio": float(np.max(m) / np.min(m)),
    }


def evaluate_payload_h514(
    payload: Mapping[str, Any], group_weights: Sequence[float], candidate_multiplier: Sequence[float], q27: Sequence[float]
) -> dict[str, Any]:
    gw = np.asarray(group_weights, dtype=np.float64).reshape(-1)
    cm = np.asarray(candidate_multiplier, dtype=np.float64).reshape(-1)
    q = np.asarray(q27, dtype=np.float64).reshape(-1)
    require(gw.shape == (len(payload["groups"]),) and abs(float(np.sum(gw)) - 1.0) <= H514_TOL, "H514_EVAL_GROUP_WEIGHT")
    require(cm.shape == q.shape == (H512R_CELL_N,), "H514_EVAL_CANDIDATE_SHAPE")
    group_aligned: list[float] = []
    group_nulls: dict[int, list[float]] = {int(s): [] for s in H58_SHIFTS}
    zero_count = 0
    for group in payload["groups"]:
        scenario_aligned: list[float] = []
        scenario_nulls: dict[int, list[float]] = {int(s): [] for s in H58_SHIFTS}
        for row in group["scenarios"]:
            cells = np.asarray(row["cell_ids"], dtype=np.int32)
            w = cm[cells]
            aligned, ar = weighted_partial_rank_h511(row["operator_rank"], row["medium_rank"], row["utility_rank"], w)
            scenario_aligned.append(float(aligned)); zero_count += int(bool(ar["zero_information"]))
            for shift in H58_SHIFTS:
                nr, _ = weighted_partial_rank_h511(row["operator_rank"], row["null_medium_rank_by_shift"][int(shift)], row["utility_rank"], w)
                scenario_nulls[int(shift)].append(float(nr))
        group_aligned.append(float(np.mean(scenario_aligned)))
        for shift in H58_SHIFTS:
            group_nulls[int(shift)].append(float(np.mean(scenario_nulls[int(shift)])))
    aligned = float(np.sum(gw * np.asarray(group_aligned, dtype=np.float64)))
    nulls = {int(s): float(np.sum(gw * np.asarray(group_nulls[int(s)], dtype=np.float64))) for s in H58_SHIFTS}
    med = float(statistics.median(nulls.values()))
    row_masses = macro_row_masses_h514(payload, gw)
    eff = weighted_effective_cell_mass_h514(macro_cell_rows_h514(payload), row_masses, cm)
    require(int(eff["zero_positive_support_macro_row_count"]) == 0, "H514_EVAL_CANDIDATE_SUPPORT_LOST")
    em = np.asarray(eff["effective_cell_mass"], dtype=np.float64)
    return {
        "fold": int(payload["fold"]), "aligned_partial_rho": aligned,
        "null_partial_rho_by_shift": nulls, "null_median_partial_rho": med,
        "aligned_minus_null_median": float(aligned - med),
        "orientation": orientation_h510(aligned, med),
        "zero_information_target_scenario_count": int(zero_count),
        "effective_candidate_cell_mass": em.tolist(),
        "effective_candidate_max_abs_error_vs_q27": float(np.max(np.abs(em - q))),
    }


def run_pair_h514(
    *, target_design: Mapping[str, Any], h512r_pair_result: Mapping[str, Any], payload_a: Mapping[str, Any], payload_b: Mapping[str, Any]
) -> dict[str, Any]:
    fa, fb = [int(x) for x in target_design["pair"]]
    if not bool(target_design["scenario_consistency_passed"]):
        return {"pair": [fa, fb], "blocked": "EXECUTION_BLOCKED__TARGET_KAPPA_SCENARIO_DRIFT"}
    if not bool(target_design["common_support_valid"]):
        return {"pair": [fa, fb], "blocked": "EXECUTION_BLOCKED__INSUFFICIENT_TARGET_KAPPA_COMMON_SUPPORT"}
    q27 = np.asarray(h512r_pair_result["q"], dtype=np.float64)
    wa = target_design["target_weights_a"]; wb = target_design["target_weights_b"]
    rma = macro_row_masses_h514(payload_a, wa); rmb = macro_row_masses_h514(payload_b, wb)
    cra = calibrate_candidate_margin_h514(
        cell_rows=macro_cell_rows_h514(payload_a), row_masses=rma, q27=q27,
        initial_multiplier=h512r_pair_result["calibration"]["side_a"]["multiplier"],
    )
    crb = calibrate_candidate_margin_h514(
        cell_rows=macro_cell_rows_h514(payload_b), row_masses=rmb, q27=q27,
        initial_multiplier=h512r_pair_result["calibration"]["side_b"]["multiplier"],
    )
    if bool(cra.get("blocked_zero_support_rows")) or bool(crb.get("blocked_zero_support_rows")):
        return {"pair": [fa, fb], "blocked": "EXECUTION_BLOCKED__TARGET_WEIGHTED_CANDIDATE_COMMON_SUPPORT_LOST", "candidate_calibration": {"side_a": cra, "side_b": crb}}
    if not bool(cra["converged"] and crb["converged"]):
        return {"pair": [fa, fb], "blocked": "EXECUTION_BLOCKED__TARGET_WEIGHTED_CANDIDATE_IPF_NONCONVERGENT", "candidate_calibration": {"side_a": cra, "side_b": crb}}
    ea = evaluate_payload_h514(payload_a, wa, cra["multiplier"], q27)
    eb = evaluate_payload_h514(payload_b, wb, crb["multiplier"], q27)
    side_diff = float(np.max(np.abs(np.asarray(ea["effective_candidate_cell_mass"]) - np.asarray(eb["effective_candidate_cell_mass"]))))
    closure = bool(ea["effective_candidate_max_abs_error_vs_q27"] <= H514_TOL and eb["effective_candidate_max_abs_error_vs_q27"] <= H514_TOL and side_diff <= H514_SIDE_TOL)
    if not closure:
        return {"pair": [fa, fb], "blocked": "EXECUTION_BLOCKED__TARGET_WEIGHTED_CANDIDATE_MARGIN_CLOSURE_FAILED", "side_a": ea, "side_b": eb, "candidate_calibration": {"side_a": cra, "side_b": crb}, "candidate_side_to_side_max_abs": side_diff}
    parent_delta = float(h512r_pair_result["side_b"]["aligned_partial_rho"] - h512r_pair_result["side_a"]["aligned_partial_rho"])
    new_delta = float(eb["aligned_partial_rho"] - ea["aligned_partial_rho"])
    ratio = float(abs(new_delta) / abs(parent_delta)) if abs(parent_delta) > H511_EPS else None
    return {
        "pair": [fa, fb], "blocked": None,
        "target_design": dict(target_design),
        "candidate_q27": q27.tolist(),
        "candidate_calibration": {"side_a": cra, "side_b": crb},
        "side_a": ea, "side_b": eb,
        "candidate_side_to_side_max_abs": side_diff,
        "candidate_margin_closure": True,
        "parent_delta_rho": parent_delta,
        "target_standardized_delta_rho": new_delta,
        "absolute_delta_ratio": ratio,
    }


def classify_h514(pair_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocked = [str(x["blocked"]) for x in pair_results if x.get("blocked")]
    if blocked:
        return {"classification": blocked[0], "native_flip_persistence_count_of_3": 0, "native_flip_clean_removal_count_of_3": 0, "positive_control_pair_stable": False}
    by = {(int(x["pair"][0]), int(x["pair"][1])): x for x in pair_results}
    require(set(by) == set(H514_FLIP_PAIRS + (H514_CONTROL_PAIR,)), "H514_PAIR_SET_DRIFT")
    retained: list[bool] = []; removed: list[bool] = []
    for pair in H514_FLIP_PAIRS:
        x = by[pair]
        oa = str(x["side_a"]["orientation"]); ob = str(x["side_b"]["orientation"])
        retained.append(oa == H511_NATIVE_ORIENTATION[pair[0]] and ob == H511_NATIVE_ORIENTATION[pair[1]])
        removed.append(oa == ob and oa in {"POSITIVE_ALIGNMENT", "ANTI_ALIGNMENT"})
    ctrl = by[H514_CONTROL_PAIR]
    control = bool(ctrl["side_a"]["orientation"] == "POSITIVE_ALIGNMENT" and ctrl["side_b"]["orientation"] == "POSITIVE_ALIGNMENT")
    if all(retained) and control:
        cls = "TARGET_OM_CONCORDANCE_OCCUPANCY_INSUFFICIENT__GLOBAL_OFFSET_TRANSITIONS_PERSIST"
    elif all(removed) and control:
        cls = "TARGET_OM_CONCORDANCE_OCCUPANCY_EXPLAINS_NATIVE_TRANSITIONS"
    else:
        cls = "TARGET_OM_CONCORDANCE_OCCUPANCY_PARTIAL_OR_MIXED"
    return {
        "classification": cls,
        "conclusion": f"H5_14_{cls}",
        "native_flip_persistence_count_of_3": int(sum(retained)),
        "native_flip_clean_removal_count_of_3": int(sum(removed)),
        "positive_control_pair_stable": control,
        "market_information_verdict_changed": False,
        "canonical_change_authorized": False,
        "runtime_gate_authorized": False,
    }
