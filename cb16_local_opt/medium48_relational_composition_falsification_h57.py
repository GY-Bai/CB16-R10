from __future__ import annotations

"""H5.7 Teacher-free/Student-free Medium48 relational composition falsification.

The experiment stays on the exact H5.6 time-local support.  It decomposes the
frozen Market96 standardized RMS-Euclidean distance into Operator48 and Medium48
squared-distance contributions, then breaks only the mapping from Medium48
squared-distance contribution to support-future-group identity.  No synthetic
Operator/Medium feature vector is constructed.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from . import state_utility_geometry_transport_audit_h55 as h55
from .teacher_temporal_transport_audit_h5 import (
    H5_SCENARIOS,
    H5_SHIFTS,
    _eval_group_scenario_rows_h5,
    _fold_material_h5,
    require,
)
from .teacher_vectorized_r11 import build_columnar_teacher_index_r11
from .time_local_vs_forward_geometry_contrast_h56 import (
    _rho_and_ratio_h56,
    _same_scenario_leave_group_out_normalization_h56,
)

H57_RUNTIME = "CB16_R11_H5_7_MEDIUM48_RELATIONAL_COMPOSITION_FALSIFICATION_R0_V1"
H57_SHIFTS = tuple(int(x) for x in H5_SHIFTS)
H57_MARKET_IDENTITY_ATOL = 1e-10
H57_TAUGAP_BOUNDARY_ATOL = 1e-12


def tau_gap_h57(reference_loss: np.ndarray, predicted_distance: np.ndarray) -> tuple[float, dict[str, int]]:
    """Gao-Oard-style tau_GAP with preregistered neutral handling of exact ties.

    Lower reference_loss is better and lower predicted_distance is better.
    Reference-score gaps provide the magnitude weights.  A prediction tie receives
    half credit.  If one rank has zero total reference gap against all better/equal
    ranks, that rank receives neutral half credit so it contributes zero information
    after mapping the average from [0,1] to [-1,1].
    """

    ref = np.asarray(reference_loss, dtype=np.float64).reshape(-1)
    pred = np.asarray(predicted_distance, dtype=np.float64).reshape(-1)
    require(ref.shape == pred.shape and len(ref) >= 2, "H57_TAUGAP_SHAPE")
    require(np.isfinite(ref).all() and np.isfinite(pred).all(), "H57_TAUGAP_NONFINITE")

    order = np.argsort(ref, kind="stable")
    fractions: list[float] = []
    degenerate_rank_count = 0
    reference_zero_gap_pair_count = 0
    prediction_tie_pair_count = 0

    for pos in range(1, len(order)):
        cur = int(order[pos])
        prior = order[:pos]
        gaps = ref[cur] - ref[prior]
        require(np.all(gaps >= 0.0), "H57_REFERENCE_ORDER_DRIFT")
        reference_zero_gap_pair_count += int(np.sum(gaps == 0.0))
        denom = float(np.sum(gaps))
        if denom == 0.0:
            fractions.append(0.5)
            degenerate_rank_count += 1
            continue

        pprev = pred[prior]
        pcur = float(pred[cur])
        credit = np.where(pprev < pcur, 1.0, np.where(pprev > pcur, 0.0, 0.5))
        prediction_tie_pair_count += int(np.sum((gaps > 0.0) & (pprev == pcur)))
        q_raw = float(np.dot(gaps, credit) / denom)
        require(np.isfinite(q_raw), "H57_TAUGAP_FRACTION_NONFINITE")
        require(
            -H57_TAUGAP_BOUNDARY_ATOL <= q_raw <= 1.0 + H57_TAUGAP_BOUNDARY_ATOL,
            f"H57_TAUGAP_FRACTION_RANGE:{q_raw}",
        )
        q = float(np.clip(q_raw, 0.0, 1.0))
        fractions.append(q)

    require(len(fractions) == len(ref) - 1, "H57_TAUGAP_RANK_COUNT")
    tau_raw = float(2.0 * np.mean(np.asarray(fractions, dtype=np.float64)) - 1.0)
    require(np.isfinite(tau_raw), "H57_TAUGAP_NONFINITE_RESULT")
    require(
        -1.0 - H57_TAUGAP_BOUNDARY_ATOL <= tau_raw <= 1.0 + H57_TAUGAP_BOUNDARY_ATOL,
        f"H57_TAUGAP_RANGE:{tau_raw}",
    )
    tau = float(np.clip(tau_raw, -1.0, 1.0))
    return tau, {
        "degenerate_reference_gap_rank_count": int(degenerate_rank_count),
        "reference_zero_gap_pair_count": int(reference_zero_gap_pair_count),
        "prediction_tie_pair_count": int(prediction_tie_pair_count),
    }


def _scenario_target_h57(
    *,
    index,
    gid: str,
    scenario: str,
    eval_order: Sequence[str],
    eval_rows_map: Mapping[str, Mapping[str, int]],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    support_groups = [g for g in eval_order if g != gid]
    require(len(support_groups) >= 32, "H57_LOCAL_SUPPORT_TOO_SMALL")
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
    d_market_direct = h55._pairwise_state_distance_h55(
        target_features=target_feature,
        support_features=support_features,
        train_mean=mean,
        train_std=std,
        active=h55.active_dimensions_h55("MARKET96", int(index.feature_dim)),
    )[0]
    d_market = np.sqrt(0.5 * (d_operator * d_operator + d_medium * d_medium))
    identity_error = float(np.max(np.abs(d_market_direct - d_market)))
    require(identity_error <= H57_MARKET_IDENTITY_ATOL, f"H57_MARKET_DECOMPOSITION_DRIFT:{identity_error}")

    d_operator_duplicate_96 = np.sqrt(0.5 * (d_operator * d_operator + d_operator * d_operator))
    duplicate_error = float(np.max(np.abs(d_operator_duplicate_96 - d_operator)))
    require(duplicate_error <= H57_MARKET_IDENTITY_ATOL, f"H57_OPERATOR_DUPLICATE_IDENTITY_DRIFT:{duplicate_error}")

    utility_mse = h55._pairwise_profile_mse_h55(
        np.asarray(index.utilities[target_row], dtype=np.float64).reshape(1, -1),
        np.asarray(index.utilities[support_rows], dtype=np.float64),
        centered=True,
    )[0]

    tau_operator, tie_operator = tau_gap_h57(utility_mse, d_operator)
    tau_true_market, tie_market = tau_gap_h57(utility_mse, d_market)
    rho_operator, _, _, _ = _rho_and_ratio_h56(d_operator, utility_mse)
    rho_true_market, _, _, _ = _rho_and_ratio_h56(d_market, utility_mse)

    medium_sq = np.ascontiguousarray(d_medium * d_medium)
    null_tau: dict[int, float] = {}
    null_rho: dict[int, float] = {}
    null_multiset_identity: dict[int, bool] = {}
    for shift in H57_SHIFTS:
        k = int(shift) % len(medium_sq)
        require(k != 0, f"H57_NULL_IDENTITY_SHIFT:{shift}:{len(medium_sq)}")
        rotated_medium_sq = np.roll(medium_sq, -k)
        same_multiset = bool(np.array_equal(np.sort(rotated_medium_sq), np.sort(medium_sq)))
        require(same_multiset, f"H57_MEDIUM_DISTANCE_MULTISET_DRIFT:{shift}")
        null_distance = np.sqrt(0.5 * (d_operator * d_operator + rotated_medium_sq))
        t, _ = tau_gap_h57(utility_mse, null_distance)
        r, _, _, _ = _rho_and_ratio_h56(null_distance, utility_mse)
        null_tau[int(shift)] = float(t)
        null_rho[int(shift)] = float(r)
        null_multiset_identity[int(shift)] = same_multiset

    return {
        "tau_operator": float(tau_operator),
        "tau_true_market": float(tau_true_market),
        "delta_true_market_minus_operator": float(tau_true_market - tau_operator),
        "tau_null_by_shift": null_tau,
        "rho_operator_secondary": float(rho_operator),
        "rho_true_market_secondary": float(rho_true_market),
        "rho_null_by_shift_secondary": null_rho,
        "support_future_group_count": int(len(support_groups)),
        "market_distance_decomposition_max_abs_error": identity_error,
        "operator_duplicate_96d_max_abs_error": duplicate_error,
        "null_medium_squared_distance_multiset_identity": null_multiset_identity,
        "operator_tie_receipt": tie_operator,
        "true_market_tie_receipt": tie_market,
        "utility_used_to_select_support": False,
        "synthetic_feature_vector_created": False,
    }


def run_fold_h57(
    *, fold_spec: Mapping[str, Any], all_train_parents: Mapping[str, Any], all_train_samples: Sequence[Any]
) -> dict[str, Any]:
    train_parents, eval_parents, parents, train_samples, eval_samples, samples = _fold_material_h5(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    index = build_columnar_teacher_index_r11(samples)
    eval_parent_ids = tuple(sorted(eval_parents))
    eval_order, eval_rows_map = _eval_group_scenario_rows_h5(
        index=index, eval_parents=eval_parents, eval_parent_ids=eval_parent_ids
    )
    require(len(eval_order) >= 33, "H57_TOO_FEW_LOCAL_GROUPS")
    normalizers = _same_scenario_leave_group_out_normalization_h56(
        index=index, eval_order=eval_order, eval_rows_map=eval_rows_map
    )

    group_rows: list[dict[str, Any]] = []
    max_market_error = 0.0
    max_duplicate_error = 0.0
    all_multisets = True
    min_support = 10**9
    max_support = 0

    for gid in eval_order:
        scenario_rows = []
        for scenario in H5_SCENARIOS:
            mean, std = normalizers[(str(gid), str(scenario))]
            x = _scenario_target_h57(
                index=index,
                gid=str(gid),
                scenario=str(scenario),
                eval_order=eval_order,
                eval_rows_map=eval_rows_map,
                mean=mean,
                std=std,
            )
            scenario_rows.append(x)
            max_market_error = max(max_market_error, float(x["market_distance_decomposition_max_abs_error"]))
            max_duplicate_error = max(max_duplicate_error, float(x["operator_duplicate_96d_max_abs_error"]))
            all_multisets = bool(all_multisets and all(x["null_medium_squared_distance_multiset_identity"].values()))
            min_support = min(min_support, int(x["support_future_group_count"]))
            max_support = max(max_support, int(x["support_future_group_count"]))

        group_rows.append({
            "future_group_id": str(gid),
            "tau_operator": float(np.mean([x["tau_operator"] for x in scenario_rows])),
            "tau_true_market": float(np.mean([x["tau_true_market"] for x in scenario_rows])),
            "delta_true_market_minus_operator": float(np.mean([x["delta_true_market_minus_operator"] for x in scenario_rows])),
            "tau_null_by_shift": {
                int(s): float(np.mean([x["tau_null_by_shift"][int(s)] for x in scenario_rows])) for s in H57_SHIFTS
            },
            "rho_operator_secondary": float(np.mean([x["rho_operator_secondary"] for x in scenario_rows])),
            "rho_true_market_secondary": float(np.mean([x["rho_true_market_secondary"] for x in scenario_rows])),
            "rho_null_by_shift_secondary": {
                int(s): float(np.mean([x["rho_null_by_shift_secondary"][int(s)] for x in scenario_rows])) for s in H57_SHIFTS
            },
        })

    tau_operator = float(np.mean([x["tau_operator"] for x in group_rows]))
    tau_true_market = float(np.mean([x["tau_true_market"] for x in group_rows]))
    tau_null = {int(s): float(np.mean([x["tau_null_by_shift"][int(s)] for x in group_rows])) for s in H57_SHIFTS}
    median_null = float(statistics.median(tau_null.values()))
    return {
        "fold": int(fold_spec["fold"]),
        "tau_operator": tau_operator,
        "tau_true_market": tau_true_market,
        "delta_true_market_minus_operator": float(tau_true_market - tau_operator),
        "tau_null_by_shift": tau_null,
        "median_tau_null": median_null,
        "aligned_minus_null_median": float(tau_true_market - median_null),
        "aligned_gt_each_null_count": int(sum(tau_true_market > tau_null[int(s)] for s in H57_SHIFTS)),
        "aligned_lt_each_null_count": int(sum(tau_true_market < tau_null[int(s)] for s in H57_SHIFTS)),
        "rho_operator_secondary": float(np.mean([x["rho_operator_secondary"] for x in group_rows])),
        "rho_true_market_secondary": float(np.mean([x["rho_true_market_secondary"] for x in group_rows])),
        "rho_null_by_shift_secondary": {
            int(s): float(np.mean([x["rho_null_by_shift_secondary"][int(s)] for x in group_rows])) for s in H57_SHIFTS
        },
        "eval_dependence_groups": int(len(eval_order)),
        "local_support_future_groups_min": int(min_support),
        "local_support_future_groups_max": int(max_support),
        "market_distance_decomposition_max_abs_error": float(max_market_error),
        "operator_duplicate_96d_max_abs_error": float(max_duplicate_error),
        "all_null_medium_distance_multisets_preserved": bool(all_multisets),
        "future_group_macro_aggregation": True,
        "scenario_equal_weight_aggregation": True,
        "teacher_compilation_used": False,
        "teacher_kernel_used": False,
        "teacher_support_selection_used": False,
        "student_training_used": False,
        "student_inference_used": False,
        "synthetic_feature_vectors_created": False,
        "group_rows": group_rows,
    }


def _direction_gate_h57(rows: Sequence[Mapping[str, Any]], *, field: str, positive: bool) -> dict[str, Any]:
    vals = [float(x[field]) for x in rows]
    good = [(v > 0.0) if positive else (v < 0.0) for v in vals]
    count = int(sum(good))
    late = bool(good[3] and good[4])
    return {
        "passed": bool(count >= 4 and late),
        "fold_count": count,
        "both_late_folds": late,
        "per_fold": {str(i + 1): float(vals[i]) for i in range(5)},
    }


def _null_gate_h57(rows: Sequence[Mapping[str, Any]], *, beneficial: bool) -> dict[str, Any]:
    if beneficial:
        fold_good = [float(x["tau_true_market"]) > float(x["median_tau_null"]) for x in rows]
        pairs = int(sum(int(x["aligned_gt_each_null_count"]) for x in rows))
    else:
        fold_good = [float(x["tau_true_market"]) < float(x["median_tau_null"]) for x in rows]
        pairs = int(sum(int(x["aligned_lt_each_null_count"]) for x in rows))
    count = int(sum(fold_good))
    late = bool(fold_good[3] and fold_good[4])
    return {
        "passed": bool(count >= 4 and late and pairs >= 20),
        "fold_count": count,
        "both_late_folds": late,
        "pairwise_count_of_25": pairs,
        "per_fold_aligned_minus_null_median": {
            str(i + 1): float(rows[i]["aligned_minus_null_median"]) for i in range(5)
        },
    }


def adjudicate_h57(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in rows) == (1, 2, 3, 4, 5), "H57_FOLD_SET_DRIFT")
    require(all(bool(x["all_null_medium_distance_multisets_preserved"]) for x in rows), "H57_NULL_MULTISET_GUARD_FAIL")
    require(all(float(x["market_distance_decomposition_max_abs_error"]) <= H57_MARKET_IDENTITY_ATOL for x in rows), "H57_MARKET_IDENTITY_GUARD_FAIL")
    require(all(float(x["operator_duplicate_96d_max_abs_error"]) <= H57_MARKET_IDENTITY_ATOL for x in rows), "H57_DUPLICATE_IDENTITY_GUARD_FAIL")

    beat_operator = _direction_gate_h57(rows, field="delta_true_market_minus_operator", positive=True)
    worse_operator = _direction_gate_h57(rows, field="delta_true_market_minus_operator", positive=False)
    aligned_benefit = _null_gate_h57(rows, beneficial=True)
    aligned_harm = _null_gate_h57(rows, beneficial=False)

    if aligned_benefit["passed"] and beat_operator["passed"]:
        classification = "ALIGNED_MEDIUM_BENEFITS_AND_MARKET_BEATS_OPERATOR"
    elif aligned_benefit["passed"] and worse_operator["passed"]:
        classification = "ALIGNED_MEDIUM_BENEFICIAL_BUT_INSUFFICIENT_TO_BEAT_OPERATOR"
    elif aligned_harm["passed"] and worse_operator["passed"]:
        classification = "ALIGNED_MEDIUM_RELATIONAL_STRUCTURE_SPECIFICALLY_HARMS_COMPOSITE_GEOMETRY"
    elif (not aligned_benefit["passed"]) and (not aligned_harm["passed"]) and worse_operator["passed"]:
        classification = "NO_STABLE_ALIGNMENT_SPECIFIC_EFFECT__GENERIC_AUGMENTATION_COMPATIBLE"
    else:
        classification = "MEDIUM48_RELATIONAL_COMPOSITION_MIXED_UNRESOLVED"

    return {
        "classification": classification,
        "conclusion": f"H5_7_{classification}",
        "true_market_beats_operator_gate": beat_operator,
        "true_market_worse_than_operator_gate": worse_operator,
        "aligned_medium_beats_null_gate": aligned_benefit,
        "aligned_medium_worse_than_null_gate": aligned_harm,
        "all_geometry_identity_guards_pass": True,
        "null_is_structured_negative_control_not_formal_permutation_test": True,
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "teacher_change_authorized": False,
        "student_change_authorized": False,
        "organ_change_authorized": False,
        "promotion_authorized": False,
        "r7_evaluation_authorized": False,
        "final_opening_authorized": False,
    }
