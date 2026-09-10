from __future__ import annotations

"""H5.8 Teacher-free/Student-free Operator-conditional Medium48 geometry audit.

The experiment stays on the exact H5.6/H5.7 time-local support. For each anchor,
Operator48, Medium48, and centered 9-action utility-profile distances are ranked.
Medium and utility rank vectors are projected orthogonally off the Operator rank
vector; their residual cosine is the preregistered partial-Spearman statistic.
No fusion weight, Teacher, Student, or synthetic feature vector is used.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from . import state_utility_geometry_transport_audit_h55 as h55
from .state_utility_geometry_transport_audit_h55_clarified import normalized_rank_rows_clarified_h55
from .teacher_temporal_transport_audit_h5 import (
    H5_SCENARIOS,
    H5_SHIFTS,
    _eval_group_scenario_rows_h5,
    _fold_material_h5,
    require,
)
from .teacher_vectorized_r11 import build_columnar_teacher_index_r11
from .time_local_vs_forward_geometry_contrast_h56 import _same_scenario_leave_group_out_normalization_h56

H58_RUNTIME = "CB16_R11_H5_8_OPERATOR_CONDITIONAL_MEDIUM48_GEOMETRY_R0_V1"
H58_SHIFTS = tuple(int(x) for x in H5_SHIFTS)
H58_RESIDUAL_EPS = 1e-12
H58_H57_FAILURE_FOLDS = (1, 3)


def partial_spearman_rank_h58(
    operator_distance: np.ndarray,
    medium_distance: np.ndarray,
    utility_distance: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Partial Pearson correlation of Spearman ranks, conditioning on Operator rank."""
    o = np.asarray(operator_distance, dtype=np.float64).reshape(1, -1)
    m = np.asarray(medium_distance, dtype=np.float64).reshape(1, -1)
    u = np.asarray(utility_distance, dtype=np.float64).reshape(1, -1)
    require(o.shape == m.shape == u.shape and o.shape[1] >= 2, "H58_PARTIAL_SHAPE")
    require(np.isfinite(o).all() and np.isfinite(m).all() and np.isfinite(u).all(), "H58_PARTIAL_NONFINITE")

    orank, odeg = normalized_rank_rows_clarified_h55(o)
    mrank, mdeg = normalized_rank_rows_clarified_h55(m)
    urank, udeg = normalized_rank_rows_clarified_h55(u)
    ov, mv, uv = orank[0], mrank[0], urank[0]

    rho_mo = float(np.dot(mv, ov))
    rho_uo = float(np.dot(uv, ov))
    rho_mu = float(np.dot(mv, uv))

    if bool(odeg[0]):
        return 0.0, {
            "rho_medium_utility": rho_mu,
            "rho_medium_operator": rho_mo,
            "rho_operator_utility": rho_uo,
            "operator_rank_degenerate": True,
            "medium_rank_degenerate": bool(mdeg[0]),
            "utility_rank_degenerate": bool(udeg[0]),
            "medium_residual_norm": 0.0,
            "utility_residual_norm": 0.0,
            "zero_information": True,
        }

    mres = mv - rho_mo * ov
    ures = uv - rho_uo * ov
    mn = float(np.linalg.norm(mres))
    un = float(np.linalg.norm(ures))
    if (not np.isfinite(mn)) or (not np.isfinite(un)):
        raise RuntimeError("H58_PARTIAL_RESIDUAL_NONFINITE")
    zero = bool(mn <= H58_RESIDUAL_EPS or un <= H58_RESIDUAL_EPS)
    if zero:
        partial = 0.0
    else:
        raw = float(np.dot(mres, ures) / (mn * un))
        require(np.isfinite(raw) and -1.0 - 1e-12 <= raw <= 1.0 + 1e-12, "H58_PARTIAL_RANGE")
        partial = float(np.clip(raw, -1.0, 1.0))

    return partial, {
        "rho_medium_utility": rho_mu,
        "rho_medium_operator": rho_mo,
        "rho_operator_utility": rho_uo,
        "operator_rank_degenerate": bool(odeg[0]),
        "medium_rank_degenerate": bool(mdeg[0]),
        "utility_rank_degenerate": bool(udeg[0]),
        "medium_residual_norm": mn,
        "utility_residual_norm": un,
        "zero_information": zero,
    }


def _scenario_target_h58(
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
    require(len(support_groups) >= 32, "H58_LOCAL_SUPPORT_TOO_SMALL")
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
    utility_mse = h55._pairwise_profile_mse_h55(
        np.asarray(index.utilities[target_row], dtype=np.float64).reshape(1, -1),
        np.asarray(index.utilities[support_rows], dtype=np.float64),
        centered=True,
    )[0]

    aligned, receipt = partial_spearman_rank_h58(d_operator, d_medium, utility_mse)
    nulls: dict[int, float] = {}
    multiset_identity: dict[int, bool] = {}
    for shift in H58_SHIFTS:
        k = int(shift) % len(d_medium)
        require(k != 0, f"H58_NULL_IDENTITY_SHIFT:{shift}:{len(d_medium)}")
        rotated = np.roll(d_medium, -k)
        same = bool(np.array_equal(np.sort(rotated), np.sort(d_medium)))
        require(same, f"H58_MEDIUM_DISTANCE_MULTISET_DRIFT:{shift}")
        rho, _ = partial_spearman_rank_h58(d_operator, rotated, utility_mse)
        nulls[int(shift)] = float(rho)
        multiset_identity[int(shift)] = same

    return {
        "aligned_partial_rho": float(aligned),
        "null_partial_rho_by_shift": nulls,
        "rho_medium_utility": float(receipt["rho_medium_utility"]),
        "rho_medium_operator": float(receipt["rho_medium_operator"]),
        "rho_operator_utility": float(receipt["rho_operator_utility"]),
        "zero_information": bool(receipt["zero_information"]),
        "operator_rank_degenerate": bool(receipt["operator_rank_degenerate"]),
        "medium_rank_degenerate": bool(receipt["medium_rank_degenerate"]),
        "utility_rank_degenerate": bool(receipt["utility_rank_degenerate"]),
        "support_future_group_count": int(len(support_groups)),
        "null_medium_distance_multiset_identity": multiset_identity,
        "utility_used_to_select_support": False,
        "synthetic_feature_vector_created": False,
    }


def run_fold_h58(
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
    require(len(eval_order) >= 33, "H58_TOO_FEW_LOCAL_GROUPS")
    normalizers = _same_scenario_leave_group_out_normalization_h56(
        index=index, eval_order=eval_order, eval_rows_map=eval_rows_map
    )

    group_rows: list[dict[str, Any]] = []
    min_support = 10**9
    max_support = 0
    all_multisets = True
    zero_count = 0
    target_count = 0

    for gid in eval_order:
        scenario_rows = []
        for scenario in H5_SCENARIOS:
            mean, std = normalizers[(str(gid), str(scenario))]
            x = _scenario_target_h58(
                index=index,
                gid=str(gid),
                scenario=str(scenario),
                eval_order=eval_order,
                eval_rows_map=eval_rows_map,
                mean=mean,
                std=std,
            )
            scenario_rows.append(x)
            min_support = min(min_support, int(x["support_future_group_count"]))
            max_support = max(max_support, int(x["support_future_group_count"]))
            all_multisets = bool(all_multisets and all(x["null_medium_distance_multiset_identity"].values()))
            zero_count += int(x["zero_information"])
            target_count += 1

        group_rows.append({
            "future_group_id": str(gid),
            "aligned_partial_rho": float(np.mean([x["aligned_partial_rho"] for x in scenario_rows])),
            "null_partial_rho_by_shift": {
                int(s): float(np.mean([x["null_partial_rho_by_shift"][int(s)] for x in scenario_rows]))
                for s in H58_SHIFTS
            },
            "rho_medium_utility": float(np.mean([x["rho_medium_utility"] for x in scenario_rows])),
            "rho_medium_operator": float(np.mean([x["rho_medium_operator"] for x in scenario_rows])),
            "rho_operator_utility": float(np.mean([x["rho_operator_utility"] for x in scenario_rows])),
        })

    aligned = float(np.mean([x["aligned_partial_rho"] for x in group_rows]))
    nulls = {
        int(s): float(np.mean([x["null_partial_rho_by_shift"][int(s)] for x in group_rows]))
        for s in H58_SHIFTS
    }
    median_null = float(statistics.median(nulls.values()))
    return {
        "fold": int(fold_spec["fold"]),
        "aligned_partial_rho": aligned,
        "null_partial_rho_by_shift": nulls,
        "median_null_partial_rho": median_null,
        "aligned_minus_null_median": float(aligned - median_null),
        "aligned_positive": bool(aligned > 0.0),
        "aligned_gt_null_median": bool(aligned > median_null),
        "aligned_gt_each_null_count": int(sum(aligned > nulls[int(s)] for s in H58_SHIFTS)),
        "rho_medium_utility": float(np.mean([x["rho_medium_utility"] for x in group_rows])),
        "rho_medium_operator": float(np.mean([x["rho_medium_operator"] for x in group_rows])),
        "rho_operator_utility": float(np.mean([x["rho_operator_utility"] for x in group_rows])),
        "eval_dependence_groups": int(len(eval_order)),
        "local_support_future_groups_min": int(min_support),
        "local_support_future_groups_max": int(max_support),
        "zero_information_target_count": int(zero_count),
        "target_row_count": int(target_count),
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


def adjudicate_h58(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in rows) == (1, 2, 3, 4, 5), "H58_FOLD_SET_DRIFT")
    require(all(bool(x["all_null_medium_distance_multisets_preserved"]) for x in rows), "H58_NULL_MULTISET_GUARD_FAIL")

    positive = [bool(x["aligned_positive"]) for x in rows]
    gt_med = [bool(x["aligned_gt_null_median"]) for x in rows]
    pair_count = int(sum(int(x["aligned_gt_each_null_count"]) for x in rows))
    positive_count = int(sum(positive))
    median_count = int(sum(gt_med))
    positive_late = bool(positive[3] and positive[4])
    median_late = bool(gt_med[3] and gt_med[4])
    global_pass = bool(
        positive_count >= 4
        and positive_late
        and median_count >= 4
        and median_late
        and pair_count >= 20
    )

    f1, f3 = rows[0], rows[2]
    failure_survives = bool(
        float(f1["aligned_partial_rho"]) > 0.0
        and float(f3["aligned_partial_rho"]) > 0.0
        and float(f1["aligned_partial_rho"]) > float(f1["median_null_partial_rho"])
        and float(f3["aligned_partial_rho"]) > float(f3["median_null_partial_rho"])
    )
    failure_anti = bool(
        float(f1["aligned_partial_rho"]) < 0.0
        and float(f3["aligned_partial_rho"]) < 0.0
        and float(f1["aligned_partial_rho"]) < float(f1["median_null_partial_rho"])
        and float(f3["aligned_partial_rho"]) < float(f3["median_null_partial_rho"])
    )

    if global_pass and failure_survives:
        classification = "OPERATOR_CONDITIONAL_MEDIUM_VALUE_SUPPORTED__COMPOSITION_EXTRACTION_BOTTLENECK"
    elif global_pass and not failure_survives:
        classification = "OPERATOR_CONDITIONAL_MEDIUM_VALUE_SUPPORTED__H5_7_FAILURE_FOLDS_MIXED"
    elif failure_anti:
        classification = "H5_7_FAILURE_FOLDS_SHOW_MEDIUM_CONDITIONAL_ANTI_ALIGNMENT"
    elif not global_pass and not failure_anti:
        classification = "OPERATOR_CONDITIONAL_MEDIUM_GEOMETRY_NOT_SUPPORTED"
    else:
        classification = "OPERATOR_CONDITIONAL_MEDIUM_GEOMETRY_MIXED_UNRESOLVED"

    return {
        "classification": classification,
        "conclusion": f"H5_8_{classification}",
        "global_gate": {
            "passed": global_pass,
            "positive_fold_count": positive_count,
            "positive_both_late_folds": positive_late,
            "beats_null_median_fold_count": median_count,
            "beats_null_median_both_late_folds": median_late,
            "pairwise_null_count_of_25": pair_count,
            "per_fold_aligned_partial_rho": {str(i + 1): float(rows[i]["aligned_partial_rho"]) for i in range(5)},
            "per_fold_aligned_minus_null_median": {str(i + 1): float(rows[i]["aligned_minus_null_median"]) for i in range(5)},
        },
        "h5_7_failure_fold_gate": {
            "folds": [1, 3],
            "conditional_value_survives": failure_survives,
            "conditional_anti_alignment": failure_anti,
            "fold_1_partial_rho": float(f1["aligned_partial_rho"]),
            "fold_1_null_median": float(f1["median_null_partial_rho"]),
            "fold_3_partial_rho": float(f3["aligned_partial_rho"]),
            "fold_3_null_median": float(f3["median_null_partial_rho"]),
        },
        "partial_spearman_is_general_conditional_independence_test": False,
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
