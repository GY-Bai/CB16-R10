from __future__ import annotations

"""H5.11.1 outcome-unused audit of the effective measure induced by H5.11.

H5.11 records normalize(P_E * cell_multiplier) == Q.  Its actual weighted
partial-correlation statistic, however, normalizes candidate weights separately
inside each target-future-group x scenario before macro-averaging those local
statistics.  This module reconstructs the candidate weight mass after that local
normalization.  Utility values are not referenced by any audit computation.
"""

from typing import Any, Mapping, Sequence

import numpy as np

from .common_support_rank_geometry_decomposition_h511 import (
    H511_CELL_N,
    H511_EPS,
    H511_PAIRS,
    _feature_rank_cells_h511,
    _fold_context_h511,
    _post_weight_cell_mass_h511,
    build_pair_overlap_design_h511,
    run_fold_design_h511,
)
from .teacher_temporal_transport_audit_h5 import H5_SCENARIOS, require

H5111_RUNTIME = "CB16_R11_H5_11_1_EFFECTIVE_MEASURE_CLOSURE_R0_V1"
H5111_TOL = 1e-12


def effective_macro_mass_from_cell_rows_h5111(
    cell_rows: Sequence[np.ndarray], multiplier: Sequence[float]
) -> dict[str, Any]:
    """Apply H5.11 local candidate-weight normalization then equal macro weighting."""
    m = np.asarray(multiplier, dtype=np.float64).reshape(-1)
    require(m.shape == (H511_CELL_N,), "H5111_MULTIPLIER_SHAPE")
    require(np.isfinite(m).all() and np.all(m >= 0.0), "H5111_BAD_MULTIPLIER")
    require(len(cell_rows) >= 1, "H5111_NO_MACRO_ROWS")

    mass = np.zeros(H511_CELL_N, dtype=np.float64)
    raw_sums: list[float] = []
    candidate_counts: list[int] = []
    for row in cell_rows:
        cells = np.asarray(row, dtype=np.int32).reshape(-1)
        require(cells.size >= 1, "H5111_EMPTY_CELL_ROW")
        require(np.all(cells >= 0) and np.all(cells < H511_CELL_N), "H5111_CELL_RANGE")
        w = m[cells]
        sw = float(np.sum(w))
        require(np.isfinite(sw) and sw > H511_EPS, "H5111_LOCAL_ZERO_WEIGHT")
        mass += np.bincount(cells, weights=w / sw, minlength=H511_CELL_N).astype(np.float64)
        raw_sums.append(sw)
        candidate_counts.append(int(cells.size))

    mass /= float(len(cell_rows))
    require(np.isfinite(mass).all(), "H5111_EFFECTIVE_MASS_NONFINITE")
    require(abs(float(np.sum(mass)) - 1.0) <= H5111_TOL, "H5111_EFFECTIVE_MASS_NOT_ONE")
    return {
        "effective_macro_candidate_mass": mass.tolist(),
        "macro_row_count": int(len(cell_rows)),
        "candidate_count_min": int(min(candidate_counts)),
        "candidate_count_max": int(max(candidate_counts)),
        "target_scenario_raw_multiplier_sum_min": float(min(raw_sums)),
        "target_scenario_raw_multiplier_sum_max": float(max(raw_sums)),
        "target_scenario_raw_multiplier_sum_mean": float(np.mean(raw_sums)),
        "target_scenario_raw_multiplier_sum_std": float(np.std(raw_sums, ddof=0)),
    }


def fold_cell_rows_h5111(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: Sequence[Any],
) -> dict[str, Any]:
    """Rebuild exact H5.11 feature-only cell rows; no utility array is referenced."""
    fold = int(fold_spec["fold"])
    index, eval_order, eval_rows_map, normalizers = _fold_context_h511(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    rows: list[np.ndarray] = []
    for gid in eval_order:
        for scenario in H5_SCENARIOS:
            mean, std = normalizers[(str(gid), str(scenario))]
            x = _feature_rank_cells_h511(
                index=index,
                gid=str(gid),
                scenario=str(scenario),
                eval_order=eval_order,
                eval_rows_map=eval_rows_map,
                mean=mean,
                std=std,
            )
            rows.append(np.asarray(x["cell_ids"], dtype=np.int32))
    require(len(rows) == len(eval_order) * len(H5_SCENARIOS), "H5111_MACRO_ROW_COUNT_DRIFT")
    return {
        "fold": fold,
        "cell_rows": rows,
        "eval_dependence_groups": int(len(eval_order)),
        "macro_row_count": int(len(rows)),
        "utility_values_referenced": False,
    }


def _distance_receipt_h5111(effective: np.ndarray, q: np.ndarray) -> dict[str, float]:
    d = np.asarray(effective, dtype=np.float64) - np.asarray(q, dtype=np.float64)
    require(d.shape == (H511_CELL_N,) and np.isfinite(d).all(), "H5111_DISTANCE_SHAPE")
    l1 = float(np.sum(np.abs(d)))
    return {
        "max_abs": float(np.max(np.abs(d))),
        "l1": l1,
        "total_variation": float(0.5 * l1),
    }


def audit_pair_h5111(
    *,
    pair_design: Mapping[str, Any],
    fold_rows_a: Mapping[str, Any],
    fold_rows_b: Mapping[str, Any],
) -> dict[str, Any]:
    fa, fb = [int(x) for x in pair_design["pair"]]
    require((fa, fb) in H511_PAIRS, f"H5111_UNREGISTERED_PAIR:{fa}:{fb}")
    require(int(fold_rows_a["fold"]) == fa and int(fold_rows_b["fold"]) == fb, "H5111_PAIR_FOLD_DRIFT")

    p_a = np.asarray(pair_design["p_a"], dtype=np.float64)
    p_b = np.asarray(pair_design["p_b"], dtype=np.float64)
    q = np.asarray(pair_design["q"], dtype=np.float64)
    m_a = np.asarray(pair_design["multiplier_a"], dtype=np.float64)
    m_b = np.asarray(pair_design["multiplier_b"], dtype=np.float64)
    require(p_a.shape == p_b.shape == q.shape == m_a.shape == m_b.shape == (H511_CELL_N,), "H5111_PAIR_VECTOR_SHAPE")

    nominal_a = _post_weight_cell_mass_h511(p_a, m_a)
    nominal_b = _post_weight_cell_mass_h511(p_b, m_b)
    nominal_a_err = float(np.max(np.abs(nominal_a - q)))
    nominal_b_err = float(np.max(np.abs(nominal_b - q)))
    require(nominal_a_err <= H5111_TOL and nominal_b_err <= H5111_TOL, "H5111_NOMINAL_Q_REPRODUCTION_FAIL")

    ea = effective_macro_mass_from_cell_rows_h5111(fold_rows_a["cell_rows"], m_a)
    eb = effective_macro_mass_from_cell_rows_h5111(fold_rows_b["cell_rows"], m_b)
    va = np.asarray(ea["effective_macro_candidate_mass"], dtype=np.float64)
    vb = np.asarray(eb["effective_macro_candidate_mass"], dtype=np.float64)
    da = _distance_receipt_h5111(va, q)
    db = _distance_receipt_h5111(vb, q)
    dab = _distance_receipt_h5111(va, vb)
    exact = bool(da["max_abs"] <= H5111_TOL and db["max_abs"] <= H5111_TOL and dab["max_abs"] <= H5111_TOL)

    return {
        "pair": [fa, fb],
        "q": q.tolist(),
        "nominal_post_weight_mass_a": nominal_a.tolist(),
        "nominal_post_weight_mass_b": nominal_b.tolist(),
        "nominal_a_max_abs_error_vs_q": nominal_a_err,
        "nominal_b_max_abs_error_vs_q": nominal_b_err,
        "side_a": {
            **ea,
            "distance_vs_q": da,
        },
        "side_b": {
            **eb,
            "distance_vs_q": db,
        },
        "effective_side_to_side_distance": dab,
        "exact_effective_common_measure_closure": exact,
        "utility_values_referenced_by_audit": False,
    }


def run_h5111(
    *,
    fold_specs: Sequence[Mapping[str, Any]],
    all_train_parents: Mapping[str, Any],
    all_train_samples: Sequence[Any],
) -> dict[str, Any]:
    specs = sorted(fold_specs, key=lambda x: int(x["fold"]))
    require([int(x["fold"]) for x in specs] == [1, 2, 3, 4, 5], "H5111_FOLD_SET_DRIFT")

    designs = [
        run_fold_design_h511(
            fold_spec=s,
            all_train_parents=all_train_parents,
            all_train_samples=all_train_samples,
        )
        for s in specs
    ]
    by_design = {int(x["fold"]): x for x in designs}
    pair_designs = [build_pair_overlap_design_h511(by_design[a], by_design[b]) for a, b in H511_PAIRS]
    require(all(bool(x["common_support_valid"]) for x in pair_designs), "H5111_FROZEN_H511_COMMON_SUPPORT_DRIFT")

    fold_rows = [
        fold_cell_rows_h5111(
            fold_spec=s,
            all_train_parents=all_train_parents,
            all_train_samples=all_train_samples,
        )
        for s in specs
    ]
    by_rows = {int(x["fold"]): x for x in fold_rows}
    pair_audits = [
        audit_pair_h5111(
            pair_design=d,
            fold_rows_a=by_rows[int(d["pair"][0])],
            fold_rows_b=by_rows[int(d["pair"][1])],
        )
        for d in pair_designs
    ]
    exact = bool(all(bool(x["exact_effective_common_measure_closure"]) for x in pair_audits))
    classification = (
        "H5_11_EXACT_EFFECTIVE_COMMON_MEASURE_CLAIM_CONFIRMED"
        if exact
        else "H5_11_EXACT_EFFECTIVE_COMMON_MEASURE_CLAIM_FALSE__SHARED_CELL_MULTIPLIER_ROBUSTNESS_ONLY"
    )
    return {
        "classification": classification,
        "conclusion": f"H5_11_1_{classification}",
        "exact_closure_all_pairs": exact,
        "fold_designs": designs,
        "pair_designs": pair_designs,
        "pair_audits": pair_audits,
        "utility_values_referenced_by_audit": False,
        "market_information_verdict_changed": False,
        "canonical_change_authorized": False,
        "h5_12_execution_authorized": False,
    }
