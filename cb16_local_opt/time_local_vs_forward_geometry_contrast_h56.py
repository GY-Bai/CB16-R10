from __future__ import annotations

"""H5.6 Teacher-free time-local versus time-forward state/utility geometry contrast.

The TIME_FORWARD arm is the clarified H5.5 computation.  TIME_LOCAL compares every
eval target only against other future groups in the same eval block and same account
scenario.  The target future group is excluded from both local normalization and local
support.  Utilities are read only after state distances are constructed.
"""

import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from . import state_utility_geometry_transport_audit_h55 as h55
from .state_utility_geometry_transport_audit_h55_clarified import (
    nearest_decile_ratio_clarified_h55,
    normalized_rank_rows_clarified_h55,
    run_fold_h55_clarified,
)
from .teacher_temporal_transport_audit_h5 import (
    H5_SCENARIOS,
    H5_SHIFTS,
    _eval_group_scenario_rows_h5,
    _fold_material_h5,
    require,
    shuffled_target_feature_index_h5,
)
from .teacher_vectorized_r11 import build_columnar_teacher_index_r11

H56_RUNTIME = "CB16_R11_H5_6_TIME_LOCAL_VS_FORWARD_STATE_UTILITY_GEOMETRY_CONTRAST_R0_V1"
H56_METRICS = h55.H55_METRICS


def _leave_group_out_normalization_h56(*, index, eval_order: Sequence[str], eval_rows_map: Mapping[str, Mapping[str, int]]):
    all_rows = np.asarray(
        [eval_rows_map[g][s] for g in eval_order for s in H5_SCENARIOS], dtype=np.int32
    )
    x = np.asarray(index.features[all_rows], dtype=np.float64)
    total_sum = np.sum(x, axis=0)
    total_sumsq = np.sum(x * x, axis=0)
    total_n = int(len(all_rows))
    require(total_n >= 6 * 33, "H56_LOCAL_BLOCK_TOO_SMALL")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for gid in eval_order:
        excluded = np.asarray([eval_rows_map[gid][s] for s in H5_SCENARIOS], dtype=np.int32)
        ex = np.asarray(index.features[excluded], dtype=np.float64)
        n = total_n - len(excluded)
        require(n > 0, "H56_LOCAL_NORMALIZATION_EMPTY")
        mean = (total_sum - np.sum(ex, axis=0)) / float(n)
        second = (total_sumsq - np.sum(ex * ex, axis=0)) / float(n)
        var = np.maximum(second - mean * mean, 0.0)
        std = np.sqrt(var)
        std = np.where(std < 1e-8, 1.0, std)
        require(np.isfinite(mean).all() and np.isfinite(std).all(), "H56_LOCAL_NORMALIZATION_NONFINITE")
        out[str(gid)] = (np.ascontiguousarray(mean), np.ascontiguousarray(std))
    return out


def _rho_and_ratio_h56(state_distance: np.ndarray, centered_mse: np.ndarray) -> tuple[float, float, bool, bool]:
    d = np.asarray(state_distance, dtype=np.float64).reshape(1, -1)
    u = np.asarray(centered_mse, dtype=np.float64).reshape(1, -1)
    state_rank, state_deg = normalized_rank_rows_clarified_h55(d)
    util_rank, util_deg = normalized_rank_rows_clarified_h55(u)
    rho = float(np.einsum("ij,ij->i", state_rank, util_rank, optimize=True)[0])
    ratio = float(nearest_decile_ratio_clarified_h55(d, u, state_deg)[0])
    require(np.isfinite(rho) and np.isfinite(ratio), "H56_LOCAL_NONFINITE_METRIC")
    return rho, ratio, bool(state_deg[0]), bool(util_deg[0])


def _local_metric_h56(
    *,
    index,
    eval_order: Sequence[str],
    eval_rows_map: Mapping[str, Mapping[str, int]],
    normalizers: Mapping[str, tuple[np.ndarray, np.ndarray]],
    metric: str,
) -> dict[str, Any]:
    active = h55.active_dimensions_h55(metric, int(index.feature_dim))
    aligned_rows: list[float] = []
    ratio_rows: list[float] = []
    state_deg_count = 0
    util_deg_count = 0
    support_sizes: list[int] = []
    shuffle_rows: dict[int, list[float]] = {int(s): [] for s in H5_SHIFTS}

    n_groups = len(eval_order)
    require(n_groups - 1 >= 32, f"H56_LOCAL_SUPPORT_TOO_SMALL:{n_groups-1}")
    source_pos = {int(s): (np.arange(n_groups, dtype=np.int32) + int(s)) % n_groups for s in H5_SHIFTS}
    for shift, src in source_pos.items():
        require(np.all(src != np.arange(n_groups, dtype=np.int32)), f"H56_LOCAL_ROTATION_FIXED_POINT:{shift}")

    for i, gid in enumerate(eval_order):
        mean, std = normalizers[str(gid)]
        support_groups = [g for j, g in enumerate(eval_order) if j != i]
        support_sizes.append(len(support_groups))
        for scenario in H5_SCENARIOS:
            target_row = int(eval_rows_map[gid][scenario])
            support_rows = np.asarray([eval_rows_map[g][scenario] for g in support_groups], dtype=np.int32)
            target_feature = np.asarray(index.features[target_row], dtype=np.float64).reshape(1, -1)
            support_features = np.asarray(index.features[support_rows], dtype=np.float64)
            d = h55._pairwise_state_distance_h55(
                target_features=target_feature,
                support_features=support_features,
                train_mean=mean,
                train_std=std,
                active=active,
            )[0]
            centered_mse = h55._pairwise_profile_mse_h55(
                np.asarray(index.utilities[target_row], dtype=np.float64).reshape(1, -1),
                np.asarray(index.utilities[support_rows], dtype=np.float64),
                centered=True,
            )[0]
            rho, ratio, state_deg, util_deg = _rho_and_ratio_h56(d, centered_mse)
            aligned_rows.append(rho)
            ratio_rows.append(ratio)
            state_deg_count += int(state_deg)
            util_deg_count += int(util_deg)

            for shift in H5_SHIFTS:
                src_gid = eval_order[int(source_pos[int(shift)][i])]
                src_row = int(eval_rows_map[src_gid][scenario])
                rotated_feature = np.asarray(index.features[src_row], dtype=np.float64).reshape(1, -1)
                rd = h55._pairwise_state_distance_h55(
                    target_features=rotated_feature,
                    support_features=support_features,
                    train_mean=mean,
                    train_std=std,
                    active=active,
                )[0]
                rrho, _, _, _ = _rho_and_ratio_h56(rd, centered_mse)
                shuffle_rows[int(shift)].append(rrho)

    expected_targets = len(eval_order) * len(H5_SCENARIOS)
    require(len(aligned_rows) == expected_targets, "H56_LOCAL_TARGET_COUNT_DRIFT")
    require(all(len(v) == expected_targets for v in shuffle_rows.values()), "H56_LOCAL_SHUFFLE_COUNT_DRIFT")
    aligned = float(np.mean(np.asarray(aligned_rows, dtype=np.float64)))
    shuffles = {int(s): float(np.mean(np.asarray(v, dtype=np.float64))) for s, v in shuffle_rows.items()}
    med = float(statistics.median(shuffles.values()))
    return {
        "aligned_centered_profile_spearman_rho": aligned,
        "aligned_nearest_decile_centered_mse_ratio": float(np.mean(np.asarray(ratio_rows, dtype=np.float64))),
        "shuffle_centered_profile_spearman_rho": shuffles,
        "median_shuffle_centered_profile_spearman_rho": med,
        "aligned_minus_median_shuffle_rho": float(aligned - med),
        "aligned_positive": bool(aligned > 0.0),
        "aligned_gt_shuffle_median": bool(aligned > med),
        "aligned_gt_each_shuffle_count": int(sum(aligned > float(shuffles[int(s)]) for s in H5_SHIFTS)),
        "degenerate_state_rank_row_count": int(state_deg_count),
        "degenerate_centered_utility_rank_row_count": int(util_deg_count),
        "target_row_count": int(expected_targets),
        "local_support_future_groups_min": int(min(support_sizes)),
        "local_support_future_groups_max": int(max(support_sizes)),
        "target_future_group_excluded": True,
        "utility_used_to_select_support": False,
    }


def run_fold_h56(
    *, fold_spec: Mapping[str, Any], all_train_parents: Mapping[str, Any], all_train_samples: Sequence[Any]
) -> dict[str, Any]:
    forward = run_fold_h55_clarified(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
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
    require(len(eval_order) >= 33, "H56_TOO_FEW_LOCAL_GROUPS")
    normalizers = _leave_group_out_normalization_h56(index=index, eval_order=eval_order, eval_rows_map=eval_rows_map)

    rotation_receipts = {}
    for shift in H5_SHIFTS:
        shuffled, receipt = shuffled_target_feature_index_h5(
            index=index,
            eval_parents=eval_parents,
            eval_parent_ids=eval_parent_ids,
            shift=int(shift),
        )
        k = int(shift) % len(eval_order)
        for scenario in H5_SCENARIOS:
            dst = np.asarray([eval_rows_map[g][scenario] for g in eval_order], dtype=np.int32)
            src = dst[(np.arange(len(eval_order), dtype=np.int32) + k) % len(eval_order)]
            require(np.array_equal(shuffled.features[dst], index.features[src]), f"H56_ROTATION_MAPPING:{shift}:{scenario}")
            require(np.array_equal(shuffled.utilities[dst], index.utilities[dst]), f"H56_ROTATED_UTILITY:{shift}:{scenario}")
        rotation_receipts[int(shift)] = receipt

    local_metrics = {
        metric: _local_metric_h56(
            index=index,
            eval_order=eval_order,
            eval_rows_map=eval_rows_map,
            normalizers=normalizers,
            metric=metric,
        )
        for metric in H56_METRICS
    }
    return {
        "fold": int(fold_spec["fold"]),
        "forward": forward,
        "local_state_metrics": local_metrics,
        "eval_dependence_groups": len(eval_order),
        "rotation_receipts": rotation_receipts,
        "teacher_compilation_used": False,
        "teacher_kernel_used": False,
        "teacher_support_selection_used": False,
        "local_support_rule": "SAME_EVAL_BLOCK_OTHER_FUTURE_GROUPS_SAME_SCENARIO_TARGET_GROUP_EXCLUDED",
        "local_normalization_rule": "ALL_OTHER_EVAL_PARENT_ROWS_TARGET_FUTURE_GROUP_SIX_ROWS_EXCLUDED",
        "utility_read_after_state_distance_only": True,
    }


def _local_gate_h56(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    vals = [x["local_state_metrics"][metric] for x in rows]
    positive = sum(bool(x["aligned_positive"]) for x in vals)
    shuffle = sum(bool(x["aligned_gt_shuffle_median"]) for x in vals)
    pairs = sum(int(x["aligned_gt_each_shuffle_count"]) for x in vals)
    positive_late = all(bool(vals[f-1]["aligned_positive"]) for f in (4,5))
    shuffle_late = all(bool(vals[f-1]["aligned_gt_shuffle_median"]) for f in (4,5))
    supported = bool(positive >= 4 and positive_late and shuffle >= 4 and shuffle_late and pairs >= 20)
    return {
        "local_geometry_supported": supported,
        "positive_fold_count": int(positive),
        "positive_both_late_folds": bool(positive_late),
        "beats_shuffle_median_fold_count": int(shuffle),
        "beats_shuffle_median_both_late_folds": bool(shuffle_late),
        "pairwise_shuffle_count_of_25": int(pairs),
        "per_fold": [
            {
                "fold": int(rows[i]["fold"]),
                "local_rho": float(vals[i]["aligned_centered_profile_spearman_rho"]),
                "median_shuffle_rho": float(vals[i]["median_shuffle_centered_profile_spearman_rho"]),
                "local_minus_shuffle_median_rho": float(vals[i]["aligned_minus_median_shuffle_rho"]),
                "local_positive": bool(vals[i]["aligned_positive"]),
                "local_gt_shuffle_median": bool(vals[i]["aligned_gt_shuffle_median"]),
                "local_gt_each_shuffle_count": int(vals[i]["aligned_gt_each_shuffle_count"]),
                "nearest_decile_ratio": float(vals[i]["aligned_nearest_decile_centered_mse_ratio"]),
            }
            for i in range(5)
        ],
    }


def _contrast_gate_h56(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    local = [float(x["local_state_metrics"][metric]["aligned_centered_profile_spearman_rho"]) for x in rows]
    forward = [float(x["forward"]["state_metrics"][metric]["aligned_centered_profile_spearman_rho"]) for x in rows]
    delta = [l-f for l,f in zip(local, forward)]
    count = sum(d > 0.0 for d in delta)
    both_late = bool(delta[3] > 0.0 and delta[4] > 0.0)
    late_mean = float(np.mean(np.asarray(delta[3:5], dtype=np.float64)))
    passed = bool(count >= 4 and both_late and late_mean > 0.0)
    return {
        "local_gt_forward_fold_count": int(count),
        "local_gt_forward_both_late_folds": both_late,
        "late_mean_local_minus_forward_rho": late_mean,
        "contrast_supported": passed,
        "per_fold_local_minus_forward_rho": {str(i+1): float(delta[i]) for i in range(5)},
    }


def adjudicate_h56(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in rows) == (1,2,3,4,5), "H56_FOLD_SET_DRIFT")
    forward_summary = h55.adjudicate_h55([x["forward"] for x in rows])
    require(forward_summary["classification"] == "MARKET_STATE_UTILITY_GEOMETRY_TEMPORAL_NONTRANSPORT", "H56_FORWARD_H55_CLASS_DRIFT")
    local_gates = {m: _local_gate_h56(rows, m) for m in H56_METRICS}
    contrast = {m: _contrast_gate_h56(rows, m) for m in H56_METRICS}
    market_local = bool(local_gates["MARKET96"]["local_geometry_supported"])
    operator_local = bool(local_gates["OPERATOR48"]["local_geometry_supported"])
    full_local = bool(local_gates["FULL102"]["local_geometry_supported"])
    medium_local = bool(local_gates["MEDIUM48"]["local_geometry_supported"])
    if market_local and not forward_summary["metric_gates"]["MARKET96"]["metric_transport_supported"] and contrast["MARKET96"]["contrast_supported"]:
        classification = "TIME_LOCAL_MARKET_GEOMETRY_EXISTS__CROSS_TIME_NONTRANSFER_SUPPORTED"
    elif operator_local and not forward_summary["metric_gates"]["OPERATOR48"]["metric_transport_supported"] and not market_local:
        classification = "TIME_LOCAL_OPERATOR_GEOMETRY_EXISTS__MEDIUM48_DEGRADES_CROSS_TIME_COMPOSITION"
    elif (not full_local) and (not market_local) and (not operator_local) and (not medium_local):
        classification = "STATE_UTILITY_GEOMETRY_WEAK_EVEN_TIME_LOCAL"
    else:
        classification = "LOCAL_VS_FORWARD_GEOMETRY_MIXED_UNRESOLVED"
    identities = all(
        all(r["target_feature_multiset_preserved"] is True and r["train_features_byte_identical"] is True and r["scenario_identity_preserved"] is True for r in x["rotation_receipts"].values())
        for x in rows
    )
    return {
        "schema": "CB16_R11_M_SERIES_H5_6_TIME_LOCAL_VS_FORWARD_STATE_UTILITY_GEOMETRY_CONTRAST_R0_ADJUDICATION_V1",
        "classification": classification,
        "conclusion": f"H5_6_{classification}",
        "forward_h5_5_reproduction": forward_summary,
        "local_metric_gates": local_gates,
        "local_vs_forward_contrast": contrast,
        "all_rotation_identity_guards_pass": bool(identities),
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "teacher_change_authorized": False,
        "student_change_authorized": False,
        "r7_evaluation_authorized": False,
        "final_opening_authorized": False,
    }
