from __future__ import annotations

"""H5.13 exact 27-cell decomposition of the frozen H5.12R statistic.

No support, representation, distance, utility, null, or calibration object is
changed.  The H5.12R weighted partial-rank statistic is decomposed into signed
cell contributions via the weighted residual identity for partial correlation.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .account6_ipf_effective_common_measure_h512r import H512R_CELL_N
from .common_support_rank_geometry_decomposition_h511 import H511_EPS, weighted_partial_rank_h511
from .operator_conditional_medium_geometry_h58 import H58_SHIFTS
from .teacher_temporal_transport_audit_h5 import require

H513_RUNTIME = "CB16_R11_H5_13_WITHIN_CELL_PARTIAL_MAP_TRANSPORT_R0_V1"
H513_TOL = 1e-12
H513_FLIP_PAIRS = ((1, 2), (2, 3), (3, 4))
H513_CONTROL_PAIR = (4, 5)


def _weighted_standardize_h513(x: np.ndarray, wn: np.ndarray) -> tuple[np.ndarray, bool]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    w = np.asarray(wn, dtype=np.float64).reshape(-1)
    require(a.shape == w.shape and a.size >= 2, "H513_STANDARDIZE_SHAPE")
    require(np.isfinite(a).all() and np.isfinite(w).all(), "H513_STANDARDIZE_NONFINITE")
    mu = float(np.sum(w * a))
    c = a - mu
    var = float(np.sum(w * c * c))
    if var <= H511_EPS:
        return np.zeros_like(a), True
    return c / float(np.sqrt(var)), False


def partial_cell_contributions_h513(
    operator_rank: np.ndarray,
    medium_rank: np.ndarray,
    utility_rank: np.ndarray,
    cell_ids: np.ndarray,
    weight: np.ndarray,
) -> dict[str, Any]:
    o = np.asarray(operator_rank, dtype=np.float64).reshape(-1)
    m = np.asarray(medium_rank, dtype=np.float64).reshape(-1)
    u = np.asarray(utility_rank, dtype=np.float64).reshape(-1)
    cells = np.asarray(cell_ids, dtype=np.int32).reshape(-1)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    require(o.shape == m.shape == u.shape == cells.shape == w.shape and o.size >= 2, "H513_ROW_SHAPE")
    require(np.isfinite(o).all() and np.isfinite(m).all() and np.isfinite(u).all() and np.isfinite(w).all(), "H513_ROW_NONFINITE")
    require(np.all(w >= 0.0), "H513_NEGATIVE_WEIGHT")
    require(np.all(cells >= 0) and np.all(cells < H512R_CELL_N), "H513_CELL_RANGE")
    sw = float(np.sum(w))
    require(sw > H511_EPS, "H513_ZERO_WEIGHT")
    wn = w / sw

    zo, do = _weighted_standardize_h513(o, wn)
    zm, dm = _weighted_standardize_h513(m, wn)
    zu, du = _weighted_standardize_h513(u, wn)
    frozen_partial, frozen_receipt = weighted_partial_rank_h511(o, m, u, w)
    if do or dm or du:
        contrib = np.zeros(H512R_CELL_N, dtype=np.float64)
        require(bool(frozen_receipt["zero_information"]), "H513_DEGENERACY_DRIFT")
        require(abs(float(frozen_partial)) <= H513_TOL, "H513_ZERO_INFO_PARTIAL_NONZERO")
        return {
            "partial_rho": 0.0,
            "cell_contribution": contrib.tolist(),
            "zero_information": True,
            "reconstruction_error": 0.0,
        }

    rho_mo = float(np.sum(wn * zm * zo))
    rho_uo = float(np.sum(wn * zu * zo))
    denom = float(np.sqrt(max(0.0, 1.0 - rho_mo * rho_mo) * max(0.0, 1.0 - rho_uo * rho_uo)))
    if denom <= H511_EPS:
        contrib = np.zeros(H512R_CELL_N, dtype=np.float64)
        require(bool(frozen_receipt["zero_information"]), "H513_DENOM_DEGENERACY_DRIFT")
        require(abs(float(frozen_partial)) <= H513_TOL, "H513_DENOM_ZERO_PARTIAL_NONZERO")
        return {
            "partial_rho": 0.0,
            "cell_contribution": contrib.tolist(),
            "zero_information": True,
            "reconstruction_error": 0.0,
        }

    rm = zm - rho_mo * zo
    ru = zu - rho_uo * zo
    candidate = wn * rm * ru / denom
    contrib = np.bincount(cells, weights=candidate, minlength=H512R_CELL_N).astype(np.float64)
    reconstructed = float(np.sum(contrib))
    err = abs(reconstructed - float(frozen_partial))
    require(err <= H513_TOL, f"H513_ROW_RECONSTRUCTION:{err}")
    return {
        "partial_rho": float(frozen_partial),
        "cell_contribution": contrib.tolist(),
        "zero_information": bool(frozen_receipt["zero_information"]),
        "reconstruction_error": float(err),
    }


def _map_from_contribution_h513(contribution: np.ndarray, rho: float, q: np.ndarray) -> dict[str, Any]:
    c = np.asarray(contribution, dtype=np.float64).reshape(-1)
    qv = np.asarray(q, dtype=np.float64).reshape(-1)
    require(c.shape == qv.shape == (H512R_CELL_N,), "H513_MAP_SHAPE")
    require(np.isfinite(c).all() and np.isfinite(qv).all() and np.all(qv > 0.0), "H513_MAP_BAD_Q")
    require(abs(float(np.sum(qv)) - 1.0) <= H513_TOL, "H513_Q_NOT_ONE")
    require(abs(float(np.sum(c)) - float(rho)) <= H513_TOL, "H513_FOLD_CONTRIBUTION_RECONSTRUCTION")
    g = c / qv
    qmean = float(np.sum(qv * g))
    require(abs(qmean - float(rho)) <= H513_TOL, "H513_G_QMEAN_DRIFT")
    h = g - float(rho)
    hmean = float(np.sum(qv * h))
    require(abs(hmean) <= H513_TOL, "H513_H_NOT_CENTERED")
    var = float(np.sum(qv * h * h))
    return {
        "g": g.tolist(),
        "h": h.tolist(),
        "q_weighted_mean_g": qmean,
        "q_weighted_mean_h": hmean,
        "q_weighted_variance_h": var,
    }


def evaluate_payload_maps_h513(payload: Mapping[str, Any], multiplier: Sequence[float], q: Sequence[float]) -> dict[str, Any]:
    mcell = np.asarray(multiplier, dtype=np.float64).reshape(-1)
    qv = np.asarray(q, dtype=np.float64).reshape(-1)
    require(mcell.shape == qv.shape == (H512R_CELL_N,), "H513_EVAL_SHAPE")
    require(np.all(qv > 0.0), "H513_REQUIRES_FULL_27_CELL_Q")

    group_partials: list[float] = []
    group_contribs: list[np.ndarray] = []
    group_null_partials: dict[int, list[float]] = {int(s): [] for s in H58_SHIFTS}
    group_null_contribs: dict[int, list[np.ndarray]] = {int(s): [] for s in H58_SHIFTS}
    max_row_err = 0.0
    zero_count = 0

    for group in payload["groups"]:
        scenario_partials: list[float] = []
        scenario_contribs: list[np.ndarray] = []
        scenario_null_partials: dict[int, list[float]] = {int(s): [] for s in H58_SHIFTS}
        scenario_null_contribs: dict[int, list[np.ndarray]] = {int(s): [] for s in H58_SHIFTS}
        for row in group["scenarios"]:
            cells = np.asarray(row["cell_ids"], dtype=np.int32)
            w = mcell[cells]
            aligned = partial_cell_contributions_h513(row["operator_rank"], row["medium_rank"], row["utility_rank"], cells, w)
            scenario_partials.append(float(aligned["partial_rho"]))
            scenario_contribs.append(np.asarray(aligned["cell_contribution"], dtype=np.float64))
            max_row_err = max(max_row_err, float(aligned["reconstruction_error"]))
            zero_count += int(bool(aligned["zero_information"]))
            for shift in H58_SHIFTS:
                null = partial_cell_contributions_h513(
                    row["operator_rank"], row["null_medium_rank_by_shift"][int(shift)], row["utility_rank"], cells, w
                )
                scenario_null_partials[int(shift)].append(float(null["partial_rho"]))
                scenario_null_contribs[int(shift)].append(np.asarray(null["cell_contribution"], dtype=np.float64))
                max_row_err = max(max_row_err, float(null["reconstruction_error"]))
        group_partials.append(float(np.mean(scenario_partials)))
        group_contribs.append(np.mean(np.stack(scenario_contribs, axis=0), axis=0))
        for shift in H58_SHIFTS:
            group_null_partials[int(shift)].append(float(np.mean(scenario_null_partials[int(shift)])))
            group_null_contribs[int(shift)].append(np.mean(np.stack(scenario_null_contribs[int(shift)], axis=0), axis=0))

    rho = float(np.mean(group_partials))
    contribution = np.mean(np.stack(group_contribs, axis=0), axis=0)
    fold_err = abs(float(np.sum(contribution)) - rho)
    require(fold_err <= H513_TOL, f"H513_FOLD_RECONSTRUCTION:{fold_err}")
    aligned_map = _map_from_contribution_h513(contribution, rho, qv)

    null_rho: dict[int, float] = {}
    null_maps: dict[int, dict[str, Any]] = {}
    null_fold_err: dict[int, float] = {}
    for shift in H58_SHIFTS:
        nr = float(np.mean(group_null_partials[int(shift)]))
        nc = np.mean(np.stack(group_null_contribs[int(shift)], axis=0), axis=0)
        ne = abs(float(np.sum(nc)) - nr)
        require(ne <= H513_TOL, f"H513_NULL_FOLD_RECONSTRUCTION:{shift}:{ne}")
        null_rho[int(shift)] = nr
        null_fold_err[int(shift)] = ne
        null_maps[int(shift)] = _map_from_contribution_h513(nc, nr, qv)

    return {
        "fold": int(payload["fold"]),
        "aligned_partial_rho": rho,
        "aligned_cell_contribution": contribution.tolist(),
        "aligned_map": aligned_map,
        "null_partial_rho_by_shift": null_rho,
        "null_maps_by_shift": null_maps,
        "max_row_reconstruction_error": float(max_row_err),
        "fold_reconstruction_error": float(fold_err),
        "null_fold_reconstruction_error_by_shift": null_fold_err,
        "zero_information_row_count_aligned": int(zero_count),
    }


def q_weighted_corr_h513(a: Sequence[float], b: Sequence[float], q: Sequence[float]) -> tuple[float, bool]:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    w = np.asarray(q, dtype=np.float64).reshape(-1)
    require(x.shape == y.shape == w.shape == (H512R_CELL_N,), "H513_CORR_SHAPE")
    require(np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all(), "H513_CORR_NONFINITE")
    require(np.all(w > 0.0) and abs(float(np.sum(w)) - 1.0) <= H513_TOL, "H513_CORR_BAD_Q")
    xm = float(np.sum(w * x)); ym = float(np.sum(w * y))
    xc = x - xm; yc = y - ym
    vx = float(np.sum(w * xc * xc)); vy = float(np.sum(w * yc * yc))
    if vx <= H511_EPS or vy <= H511_EPS:
        return 0.0, True
    corr = float(np.sum(w * xc * yc) / np.sqrt(vx * vy))
    require(np.isfinite(corr) and -1.0 - 1e-10 <= corr <= 1.0 + 1e-10, "H513_CORR_RANGE")
    return float(np.clip(corr, -1.0, 1.0)), False


def evaluate_pair_transport_h513(side_a: Mapping[str, Any], side_b: Mapping[str, Any], q: Sequence[float]) -> dict[str, Any]:
    qv = np.asarray(q, dtype=np.float64).reshape(-1)
    ha = np.asarray(side_a["aligned_map"]["h"], dtype=np.float64)
    hb = np.asarray(side_b["aligned_map"]["h"], dtype=np.float64)
    aligned_corr, aligned_deg = q_weighted_corr_h513(ha, hb, qv)
    require(not aligned_deg, "H513_ALIGNED_CELL_RATE_MAP_DEGENERATE")

    null_corrs: list[float] = []
    null_degenerate = 0
    for sa in H58_SHIFTS:
        hna = side_a["null_maps_by_shift"][int(sa)]["h"]
        for sb in H58_SHIFTS:
            hnb = side_b["null_maps_by_shift"][int(sb)]["h"]
            c, deg = q_weighted_corr_h513(hna, hnb, qv)
            null_degenerate += int(deg)
            if not deg:
                null_corrs.append(float(c))
    require(null_degenerate == 0 and len(null_corrs) == 25, "H513_NULL_CELL_RATE_MAP_DEGENERATE")
    null_median = float(statistics.median(null_corrs))
    pattern_pass = bool(aligned_corr > 0.0 and aligned_corr > null_median)

    rho_a = float(side_a["aligned_partial_rho"]); rho_b = float(side_b["aligned_partial_rho"])
    delta = rho_b - rho_a
    pattern_sq = float(np.sum(qv * (hb - ha) ** 2))
    ga = np.asarray(side_a["aligned_map"]["g"], dtype=np.float64)
    gb = np.asarray(side_b["aligned_map"]["g"], dtype=np.float64)
    total_sq = float(np.sum(qv * (gb - ga) ** 2))
    rhs = float(delta * delta + pattern_sq)
    identity_err = abs(total_sq - rhs)
    require(identity_err <= H513_TOL, f"H513_ORTHOGONAL_IDENTITY:{identity_err}")
    level_fraction = 0.0 if total_sq <= H511_EPS else float((delta * delta) / total_sq)
    return {
        "aligned_centered_pattern_similarity": aligned_corr,
        "null_null_similarity_values": null_corrs,
        "null_null_similarity_median": null_median,
        "pattern_transport_pass": pattern_pass,
        "delta_rho": delta,
        "pattern_difference_squared": pattern_sq,
        "total_g_difference_squared": total_sq,
        "orthogonal_identity_error": identity_err,
        "level_fraction": level_fraction,
        "level_dominant": bool(level_fraction > 0.5),
        "pattern_dominant": bool(level_fraction < 0.5),
    }


def classify_h513(pair_results: Sequence[Mapping[str, Any]], parent_reproduction_passed: bool) -> dict[str, Any]:
    if not parent_reproduction_passed:
        return {"classification": "EXECUTION_BLOCKED__H5_12R_PARENT_REPRODUCTION_FAILED"}
    by_pair = {(int(x["pair"][0]), int(x["pair"][1])): x for x in pair_results}
    require(set(by_pair) == set(H513_FLIP_PAIRS + (H513_CONTROL_PAIR,)), "H513_PAIR_SET_DRIFT")
    flips = [by_pair[p] for p in H513_FLIP_PAIRS]
    ctrl = by_pair[H513_CONTROL_PAIR]
    if all(bool(x["transport"]["pattern_transport_pass"] and x["transport"]["level_dominant"]) for x in flips) and bool(ctrl["transport"]["pattern_transport_pass"]):
        cls = "CENTERED_CELL_PATTERN_TRANSPORTS__GLOBAL_LEVEL_SHIFT_DOMINANT"
    elif all(bool((not x["transport"]["pattern_transport_pass"]) and x["transport"]["pattern_dominant"]) for x in flips) and bool(ctrl["transport"]["pattern_transport_pass"]):
        cls = "CENTERED_CELL_PATTERN_NONTRANSPORT__WITHIN_CELL_REORGANIZATION_DOMINANT"
    else:
        cls = "MIXED_LEVEL_AND_WITHIN_CELL_MAPPING_DRIFT"
    return {
        "classification": cls,
        "conclusion": f"H5_13_{cls}",
        "flip_pattern_transport_pass_count_of_3": int(sum(bool(x["transport"]["pattern_transport_pass"]) for x in flips)),
        "flip_level_dominant_count_of_3": int(sum(bool(x["transport"]["level_dominant"]) for x in flips)),
        "flip_pattern_dominant_count_of_3": int(sum(bool(x["transport"]["pattern_dominant"]) for x in flips)),
        "positive_control_pattern_transport_pass": bool(ctrl["transport"]["pattern_transport_pass"]),
        "market_information_verdict_changed": False,
        "canonical_change_authorized": False,
        "runtime_gate_authorized": False,
    }
