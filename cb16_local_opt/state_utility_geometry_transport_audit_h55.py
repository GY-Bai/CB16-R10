from __future__ import annotations

"""H5.5 direct state-to-counterfactual-utility geometry transport audit.

No Teacher law, support selector, kernel, quantile compiler, Student, optimizer, or
policy is used.  State distances are computed first from frozen representations;
realized 9-action counterfactual utilities are then used only as post-distance
geometry truth.  Whole-future-group rotations are exactly the H5 transform.
"""

import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .teacher_temporal_transport_audit_h5 import (
    H5_SCENARIOS,
    H5_SHIFTS,
    _eval_group_scenario_rows_h5,
    _fold_material_h5,
    require,
    shuffled_target_feature_index_h5,
)
from .teacher_vectorized_r11 import build_columnar_teacher_index_r11

H55_RUNTIME = "CB16_R11_H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_R0_V1"
H55_METRICS = ("FULL102", "MARKET96", "OPERATOR48", "MEDIUM48", "ACCOUNT6")


def active_dimensions_h55(metric: str, feature_dim: int) -> np.ndarray:
    require(int(feature_dim) == 102, f"H55_FEATURE_DIM:{feature_dim}")
    m = str(metric)
    if m == "FULL102":
        return np.arange(0, 102, dtype=np.int32)
    if m == "MARKET96":
        return np.arange(0, 96, dtype=np.int32)
    if m == "OPERATOR48":
        return np.arange(0, 48, dtype=np.int32)
    if m == "MEDIUM48":
        return np.arange(48, 96, dtype=np.int32)
    if m == "ACCOUNT6":
        return np.arange(96, 102, dtype=np.int32)
    raise RuntimeError(f"H55_UNKNOWN_METRIC:{m}")


def _rankdata_average_h55(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    require(a.ndim == 1 and len(a) >= 2 and np.isfinite(a).all(), "H55_BAD_RANK_INPUT")
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        value = a[order[i]]
        while j < len(a) and a[order[j]] == value:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _normalized_rank_rows_h55(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    require(a.ndim == 2 and a.shape[1] >= 2 and np.isfinite(a).all(), "H55_BAD_RANK_MATRIX")
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        r = _rankdata_average_h55(a[i])
        r -= float(np.mean(r))
        norm = float(np.linalg.norm(r))
        require(norm > 0.0 and np.isfinite(norm), "H55_DEGENERATE_RANK_ROW")
        out[i] = r / norm
    return np.ascontiguousarray(out)


def _pairwise_state_distance_h55(
    *, target_features: np.ndarray, support_features: np.ndarray, train_mean: np.ndarray, train_std: np.ndarray, active: np.ndarray
) -> np.ndarray:
    t = (np.asarray(target_features, dtype=np.float64)[:, active] - train_mean[active]) / train_std[active]
    s = (np.asarray(support_features, dtype=np.float64)[:, active] - train_mean[active]) / train_std[active]
    tn = np.einsum("ij,ij->i", t, t, optimize=True)
    sn = np.einsum("ij,ij->i", s, s, optimize=True)
    sq = (tn[:, None] + sn[None, :] - 2.0 * (t @ s.T)) / float(len(active))
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq, out=sq)


def _pairwise_profile_mse_h55(target_util: np.ndarray, support_util: np.ndarray, *, centered: bool) -> np.ndarray:
    t = np.asarray(target_util, dtype=np.float64)
    s = np.asarray(support_util, dtype=np.float64)
    require(t.ndim == 2 and s.ndim == 2 and t.shape[1] == s.shape[1] == 9, "H55_UTILITY_PROFILE_SHAPE")
    if centered:
        t = t - np.mean(t, axis=1, keepdims=True)
        s = s - np.mean(s, axis=1, keepdims=True)
    tn = np.einsum("ij,ij->i", t, t, optimize=True)
    sn = np.einsum("ij,ij->i", s, s, optimize=True)
    sq = (tn[:, None] + sn[None, :] - 2.0 * (t @ s.T)) / 9.0
    np.maximum(sq, 0.0, out=sq)
    require(np.isfinite(sq).all(), "H55_NONFINITE_PROFILE_DISTANCE")
    return np.ascontiguousarray(sq)


def _train_group_scenario_rows_h55(*, index, train_parents: Mapping[str, Any]) -> tuple[list[str], dict[str, dict[str, int]]]:
    rows: dict[str, dict[str, int]] = {}
    dep_ts: dict[str, int] = {}
    for pid, p in train_parents.items():
        gid = str(p.dependence_group_id)
        scenario = str(p.scenario)
        require(scenario in H5_SCENARIOS, f"H55_UNKNOWN_TRAIN_SCENARIO:{scenario}")
        rows.setdefault(gid, {})
        require(scenario not in rows[gid], f"H55_DUPLICATE_TRAIN_SCENARIO:{gid}:{scenario}")
        rows[gid][scenario] = int(index.parent_row_by_id[pid])
        dep_ts[gid] = int(p.decision_time_ms)
    expected = set(H5_SCENARIOS)
    for gid, mapping in rows.items():
        require(set(mapping) == expected, f"H55_TRAIN_SCENARIO_SET_DRIFT:{gid}")
    order = sorted(rows, key=lambda g: (dep_ts[g], g))
    return order, rows


def _aggregate_scenario_group_h55(values_by_scenario: Mapping[str, np.ndarray]) -> float:
    arrays = [np.asarray(values_by_scenario[s], dtype=np.float64) for s in H5_SCENARIOS]
    require(arrays and all(a.ndim == 1 for a in arrays), "H55_BAD_SCENARIO_AGG")
    n = len(arrays[0])
    require(n > 0 and all(len(a) == n for a in arrays), "H55_SCENARIO_LENGTH_DRIFT")
    matrix = np.stack(arrays, axis=0)
    require(np.isfinite(matrix).all(), "H55_NONFINITE_SCENARIO_AGG")
    return float(np.mean(np.mean(matrix, axis=0)))


def _scenario_geometry_h55(
    *, index, train_mean: np.ndarray, train_std: np.ndarray, train_order: Sequence[str], train_rows_map: Mapping[str, Mapping[str, int]],
    eval_order: Sequence[str], eval_rows_map: Mapping[str, Mapping[str, int]], metric: str,
) -> dict[str, Any]:
    active = active_dimensions_h55(metric, int(index.feature_dim))
    aligned_rho_by_scenario: dict[str, np.ndarray] = {}
    raw_rho_by_scenario: dict[str, np.ndarray] = {}
    nearest_ratio_by_scenario: dict[str, np.ndarray] = {}
    state_rank_norm_by_scenario: dict[str, np.ndarray] = {}
    centered_rank_norm_by_scenario: dict[str, np.ndarray] = {}
    raw_rank_norm_by_scenario: dict[str, np.ndarray] = {}
    centered_mse_by_scenario: dict[str, np.ndarray] = {}
    state_distance_by_scenario: dict[str, np.ndarray] = {}

    for scenario in H5_SCENARIOS:
        target_rows = np.asarray([eval_rows_map[g][scenario] for g in eval_order], dtype=np.int32)
        support_rows = np.asarray([train_rows_map[g][scenario] for g in train_order], dtype=np.int32)
        d = _pairwise_state_distance_h55(
            target_features=index.features[target_rows],
            support_features=index.features[support_rows],
            train_mean=train_mean,
            train_std=train_std,
            active=active,
        )
        centered_mse = _pairwise_profile_mse_h55(index.utilities[target_rows], index.utilities[support_rows], centered=True)
        raw_mse = _pairwise_profile_mse_h55(index.utilities[target_rows], index.utilities[support_rows], centered=False)
        state_rank = _normalized_rank_rows_h55(d)
        centered_rank = _normalized_rank_rows_h55(centered_mse)
        raw_rank = _normalized_rank_rows_h55(raw_mse)
        aligned = np.einsum("ij,ij->i", state_rank, centered_rank, optimize=True)
        raw = np.einsum("ij,ij->i", state_rank, raw_rank, optimize=True)
        nearest_n = max(1, int(math.ceil(0.10 * d.shape[1])))
        order = np.argsort(d, axis=1, kind="stable")[:, :nearest_n]
        near_values = np.take_along_axis(centered_mse, order, axis=1)
        ratio = np.mean(near_values, axis=1) / np.maximum(np.mean(centered_mse, axis=1), 1e-30)
        aligned_rho_by_scenario[scenario] = aligned
        raw_rho_by_scenario[scenario] = raw
        nearest_ratio_by_scenario[scenario] = ratio
        state_rank_norm_by_scenario[scenario] = state_rank
        centered_rank_norm_by_scenario[scenario] = centered_rank
        raw_rank_norm_by_scenario[scenario] = raw_rank
        centered_mse_by_scenario[scenario] = centered_mse
        state_distance_by_scenario[scenario] = d

    return {
        "aligned_rho": _aggregate_scenario_group_h55(aligned_rho_by_scenario),
        "aligned_raw_rho": _aggregate_scenario_group_h55(raw_rho_by_scenario),
        "aligned_nearest_decile_centered_mse_ratio": _aggregate_scenario_group_h55(nearest_ratio_by_scenario),
        "state_rank_norm_by_scenario": state_rank_norm_by_scenario,
        "centered_rank_norm_by_scenario": centered_rank_norm_by_scenario,
        "raw_rank_norm_by_scenario": raw_rank_norm_by_scenario,
        "centered_mse_by_scenario": centered_mse_by_scenario,
        "state_distance_by_scenario": state_distance_by_scenario,
    }


def run_fold_h55(
    *, fold_spec: Mapping[str, Any], all_train_parents: Mapping[str, Any], all_train_samples: Sequence[Any]
) -> dict[str, Any]:
    fold = int(fold_spec["fold"])
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
    train_order, train_rows_map = _train_group_scenario_rows_h55(index=index, train_parents=train_parents)
    require(len(train_order) >= 10 and len(eval_order) >= 2, "H55_INSUFFICIENT_GROUPS")

    train_rows = np.asarray([index.parent_row_by_id[pid] for pid in sorted(train_parents)], dtype=np.int32)
    train_x = np.asarray(index.features[train_rows], dtype=np.float64)
    train_mean = np.mean(train_x, axis=0)
    train_std = np.std(train_x, axis=0, ddof=0)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)
    require(np.isfinite(train_mean).all() and np.isfinite(train_std).all(), "H55_BAD_TRAIN_NORMALIZATION")

    aligned_detail = {
        metric: _scenario_geometry_h55(
            index=index,
            train_mean=train_mean,
            train_std=train_std,
            train_order=train_order,
            train_rows_map=train_rows_map,
            eval_order=eval_order,
            eval_rows_map=eval_rows_map,
            metric=metric,
        )
        for metric in H55_METRICS
    }

    rotation_receipts: dict[int, dict[str, Any]] = {}
    source_pos_by_shift: dict[int, np.ndarray] = {}
    for shift in H5_SHIFTS:
        shuffled_index, receipt = shuffled_target_feature_index_h5(
            index=index,
            eval_parents=eval_parents,
            eval_parent_ids=eval_parent_ids,
            shift=int(shift),
        )
        k = int(shift) % len(eval_order)
        require(k != 0, f"H55_IDENTITY_ROTATION:{shift}")
        src = (np.arange(len(eval_order), dtype=np.int32) + k) % len(eval_order)
        for scenario in H5_SCENARIOS:
            dst_rows = np.asarray([eval_rows_map[g][scenario] for g in eval_order], dtype=np.int32)
            src_rows = dst_rows[src]
            require(
                np.array_equal(shuffled_index.features[dst_rows], index.features[src_rows]),
                f"H55_ROTATION_MAPPING_DRIFT:{shift}:{scenario}",
            )
            require(
                np.array_equal(shuffled_index.utilities[dst_rows], index.utilities[dst_rows]),
                f"H55_ROTATED_UTILITY_DRIFT:{shift}:{scenario}",
            )
        rotation_receipts[int(shift)] = receipt
        source_pos_by_shift[int(shift)] = src

    metric_results: dict[str, Any] = {}
    for metric in H55_METRICS:
        detail = aligned_detail[metric]
        shuffle_rho: dict[int, float] = {}
        shuffle_raw_rho: dict[int, float] = {}
        shuffle_nearest_ratio: dict[int, float] = {}
        for shift in H5_SHIFTS:
            src = source_pos_by_shift[int(shift)]
            rho_s: dict[str, np.ndarray] = {}
            raw_s: dict[str, np.ndarray] = {}
            ratio_s: dict[str, np.ndarray] = {}
            for scenario in H5_SCENARIOS:
                state_rank = detail["state_rank_norm_by_scenario"][scenario]
                centered_rank = detail["centered_rank_norm_by_scenario"][scenario]
                raw_rank = detail["raw_rank_norm_by_scenario"][scenario]
                rho_s[scenario] = np.einsum("ij,ij->i", state_rank[src], centered_rank, optimize=True)
                raw_s[scenario] = np.einsum("ij,ij->i", state_rank[src], raw_rank, optimize=True)
                d = detail["state_distance_by_scenario"][scenario]
                centered_mse = detail["centered_mse_by_scenario"][scenario]
                nearest_n = max(1, int(math.ceil(0.10 * d.shape[1])))
                nearest = np.argsort(d[src], axis=1, kind="stable")[:, :nearest_n]
                near_values = np.take_along_axis(centered_mse, nearest, axis=1)
                ratio_s[scenario] = np.mean(near_values, axis=1) / np.maximum(np.mean(centered_mse, axis=1), 1e-30)
            shuffle_rho[int(shift)] = _aggregate_scenario_group_h55(rho_s)
            shuffle_raw_rho[int(shift)] = _aggregate_scenario_group_h55(raw_s)
            shuffle_nearest_ratio[int(shift)] = _aggregate_scenario_group_h55(ratio_s)

        med = float(statistics.median(shuffle_rho.values()))
        metric_results[metric] = {
            "aligned_centered_profile_spearman_rho": float(detail["aligned_rho"]),
            "aligned_raw_profile_spearman_rho": float(detail["aligned_raw_rho"]),
            "aligned_nearest_decile_centered_mse_ratio": float(detail["aligned_nearest_decile_centered_mse_ratio"]),
            "shuffle_centered_profile_spearman_rho": shuffle_rho,
            "shuffle_raw_profile_spearman_rho": shuffle_raw_rho,
            "shuffle_nearest_decile_centered_mse_ratio": shuffle_nearest_ratio,
            "median_shuffle_centered_profile_spearman_rho": med,
            "aligned_minus_median_shuffle_rho": float(detail["aligned_rho"] - med),
            "aligned_positive": bool(float(detail["aligned_rho"]) > 0.0),
            "aligned_gt_shuffle_median": bool(float(detail["aligned_rho"]) > med),
            "aligned_gt_each_shuffle_count": int(sum(float(detail["aligned_rho"]) > float(shuffle_rho[int(s)]) for s in H5_SHIFTS)),
        }

    return {
        "fold": fold,
        "train_parent_contexts": len(train_parents),
        "eval_parent_contexts": len(eval_parents),
        "train_dependence_groups": len(train_order),
        "eval_dependence_groups": len(eval_order),
        "train_to_eval_gap_hours": float(fold_spec["train_to_eval_gap_hours"]),
        "state_metrics": metric_results,
        "rotation_receipts": rotation_receipts,
        "teacher_compilation_used": False,
        "teacher_kernel_used": False,
        "teacher_support_selection_used": False,
        "support_parent_rule": "SAME_SCENARIO_UNIQUE_PARENT_PER_TRAIN_FUTURE_GROUP",
        "utility_read_after_state_distance_only": True,
    }


def _metric_gate_h55(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    metric_rows = [x["state_metrics"][metric] for x in rows]
    positive_count = sum(bool(x["aligned_positive"]) for x in metric_rows)
    shuffle_count = sum(bool(x["aligned_gt_shuffle_median"]) for x in metric_rows)
    pair_count = sum(int(x["aligned_gt_each_shuffle_count"]) for x in metric_rows)
    positive_late = all(bool(metric_rows[f - 1]["aligned_positive"]) for f in (4, 5))
    shuffle_late = all(bool(metric_rows[f - 1]["aligned_gt_shuffle_median"]) for f in (4, 5))
    supported = bool(positive_count >= 4 and positive_late and shuffle_count >= 4 and shuffle_late and pair_count >= 20)
    return {
        "metric_transport_supported": supported,
        "positive_fold_count": int(positive_count),
        "positive_both_late_folds": bool(positive_late),
        "beats_shuffle_median_fold_count": int(shuffle_count),
        "beats_shuffle_median_both_late_folds": bool(shuffle_late),
        "pairwise_shuffle_count_of_25": int(pair_count),
        "per_fold": [
            {
                "fold": int(rows[i]["fold"]),
                "aligned_rho": float(metric_rows[i]["aligned_centered_profile_spearman_rho"]),
                "median_shuffle_rho": float(metric_rows[i]["median_shuffle_centered_profile_spearman_rho"]),
                "aligned_minus_median_shuffle_rho": float(metric_rows[i]["aligned_minus_median_shuffle_rho"]),
                "aligned_positive": bool(metric_rows[i]["aligned_positive"]),
                "aligned_gt_shuffle_median": bool(metric_rows[i]["aligned_gt_shuffle_median"]),
                "aligned_gt_each_shuffle_count": int(metric_rows[i]["aligned_gt_each_shuffle_count"]),
                "aligned_raw_rho": float(metric_rows[i]["aligned_raw_profile_spearman_rho"]),
                "aligned_nearest_decile_centered_mse_ratio": float(metric_rows[i]["aligned_nearest_decile_centered_mse_ratio"]),
            }
            for i in range(5)
        ],
    }


def adjudicate_h55(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in rows) == (1, 2, 3, 4, 5), "H55_FOLD_SET_DRIFT")
    gates = {metric: _metric_gate_h55(rows, metric) for metric in H55_METRICS}
    full = gates["FULL102"]["metric_transport_supported"]
    market = gates["MARKET96"]["metric_transport_supported"]
    operator = gates["OPERATOR48"]["metric_transport_supported"]
    medium = gates["MEDIUM48"]["metric_transport_supported"]
    if full:
        classification = "FULL_STATE_GEOMETRY_TRANSPORT_SUPPORTED"
    elif operator and not medium:
        classification = "OPERATOR48_TRANSPORT_SURVIVES_MEDIUM48_SUSPECT"
    elif medium and not operator:
        classification = "MEDIUM48_TRANSPORT_SURVIVES_OPERATOR48_SUSPECT"
    elif (not full) and (not market) and (not operator) and (not medium):
        classification = "MARKET_STATE_UTILITY_GEOMETRY_TEMPORAL_NONTRANSPORT"
    else:
        classification = "MIXED_STATE_UTILITY_GEOMETRY_TRANSPORT_UNRESOLVED"

    rotations_ok = all(
        all(
            r["target_feature_multiset_preserved"] is True
            and r["train_features_byte_identical"] is True
            and r["scenario_identity_preserved"] is True
            for r in fold["rotation_receipts"].values()
        )
        for fold in rows
    )
    return {
        "schema": "CB16_R11_M_SERIES_H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_R0_ADJUDICATION_V1",
        "classification": classification,
        "metric_gates": gates,
        "all_rotation_identity_guards_pass": bool(rotations_ok),
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "teacher_change_authorized": False,
        "student_change_authorized": False,
        "promotion_authorized": False,
        "r7_evaluation_authorized": False,
        "final_opening_authorized": False,
        "conclusion": f"H5_5_{classification}",
    }
