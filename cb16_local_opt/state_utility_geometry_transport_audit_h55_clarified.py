from __future__ import annotations

"""Pre-valid-result clarification adapter for H5.5.

The original H5.5 implementation fail-closed before any fold result because some
state-distance rows have zero rank variance.  This adapter changes only the
pre-registered boundary convention authorized by the clarified gate:

* zero-variance state- or utility-rank rows contribute Spearman rho = 0.0;
* zero-variance state-distance rows contribute nearest-decile ratio = 1.0;
* all such rows are counted and reported per fold/metric.

All metrics, H5 rotations, outer folds, utility profiles, aggregation and final
classification remain the original H5.5 definitions.
"""

import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from . import state_utility_geometry_transport_audit_h55 as base
from .teacher_temporal_transport_audit_h5 import (
    H5_SCENARIOS,
    H5_SHIFTS,
    _eval_group_scenario_rows_h5,
    _fold_material_h5,
    require,
    shuffled_target_feature_index_h5,
)
from .teacher_vectorized_r11 import build_columnar_teacher_index_r11

H55_CLARIFIED_RUNTIME = "CB16_R11_H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_R0_CLARIFIED_V1"


def normalized_rank_rows_clarified_h55(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=np.float64)
    require(a.ndim == 2 and a.shape[1] >= 2 and np.isfinite(a).all(), "H55C_BAD_RANK_MATRIX")
    out = np.empty_like(a)
    degenerate = np.zeros(a.shape[0], dtype=np.bool_)
    for i in range(a.shape[0]):
        r = base._rankdata_average_h55(a[i])
        r -= float(np.mean(r))
        norm = float(np.linalg.norm(r))
        if not np.isfinite(norm):
            raise RuntimeError("H55C_NONFINITE_RANK_NORM")
        if norm == 0.0:
            out[i] = 0.0
            degenerate[i] = True
        else:
            out[i] = r / norm
    return np.ascontiguousarray(out), np.ascontiguousarray(degenerate)


def nearest_decile_ratio_clarified_h55(
    state_distance: np.ndarray,
    centered_profile_mse: np.ndarray,
    state_degenerate: np.ndarray,
) -> np.ndarray:
    d = np.asarray(state_distance, dtype=np.float64)
    u = np.asarray(centered_profile_mse, dtype=np.float64)
    deg = np.asarray(state_degenerate, dtype=np.bool_)
    require(d.ndim == 2 and u.shape == d.shape and deg.shape == (d.shape[0],), "H55C_NEAREST_SHAPE")
    nearest_n = max(1, int(math.ceil(0.10 * d.shape[1])))
    order = np.argsort(d, axis=1, kind="stable")[:, :nearest_n]
    near_values = np.take_along_axis(u, order, axis=1)
    ratio = np.mean(near_values, axis=1) / np.maximum(np.mean(u, axis=1), 1e-30)
    ratio = np.asarray(ratio, dtype=np.float64)
    ratio[deg] = 1.0
    require(np.isfinite(ratio).all(), "H55C_NONFINITE_NEAREST_RATIO")
    return ratio


def _scenario_geometry_clarified_h55(
    *,
    index,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    train_order: Sequence[str],
    train_rows_map: Mapping[str, Mapping[str, int]],
    eval_order: Sequence[str],
    eval_rows_map: Mapping[str, Mapping[str, int]],
    metric: str,
) -> dict[str, Any]:
    active = base.active_dimensions_h55(metric, int(index.feature_dim))
    aligned_rho_by_scenario: dict[str, np.ndarray] = {}
    raw_rho_by_scenario: dict[str, np.ndarray] = {}
    nearest_ratio_by_scenario: dict[str, np.ndarray] = {}
    state_rank_norm_by_scenario: dict[str, np.ndarray] = {}
    centered_rank_norm_by_scenario: dict[str, np.ndarray] = {}
    raw_rank_norm_by_scenario: dict[str, np.ndarray] = {}
    state_degenerate_by_scenario: dict[str, np.ndarray] = {}
    centered_degenerate_by_scenario: dict[str, np.ndarray] = {}
    raw_degenerate_by_scenario: dict[str, np.ndarray] = {}
    centered_mse_by_scenario: dict[str, np.ndarray] = {}
    state_distance_by_scenario: dict[str, np.ndarray] = {}

    for scenario in H5_SCENARIOS:
        target_rows = np.asarray([eval_rows_map[g][scenario] for g in eval_order], dtype=np.int32)
        support_rows = np.asarray([train_rows_map[g][scenario] for g in train_order], dtype=np.int32)
        d = base._pairwise_state_distance_h55(
            target_features=index.features[target_rows],
            support_features=index.features[support_rows],
            train_mean=train_mean,
            train_std=train_std,
            active=active,
        )
        centered_mse = base._pairwise_profile_mse_h55(
            index.utilities[target_rows], index.utilities[support_rows], centered=True
        )
        raw_mse = base._pairwise_profile_mse_h55(
            index.utilities[target_rows], index.utilities[support_rows], centered=False
        )
        state_rank, state_deg = normalized_rank_rows_clarified_h55(d)
        centered_rank, centered_deg = normalized_rank_rows_clarified_h55(centered_mse)
        raw_rank, raw_deg = normalized_rank_rows_clarified_h55(raw_mse)
        aligned = np.einsum("ij,ij->i", state_rank, centered_rank, optimize=True)
        raw = np.einsum("ij,ij->i", state_rank, raw_rank, optimize=True)
        ratio = nearest_decile_ratio_clarified_h55(d, centered_mse, state_deg)

        aligned_rho_by_scenario[scenario] = aligned
        raw_rho_by_scenario[scenario] = raw
        nearest_ratio_by_scenario[scenario] = ratio
        state_rank_norm_by_scenario[scenario] = state_rank
        centered_rank_norm_by_scenario[scenario] = centered_rank
        raw_rank_norm_by_scenario[scenario] = raw_rank
        state_degenerate_by_scenario[scenario] = state_deg
        centered_degenerate_by_scenario[scenario] = centered_deg
        raw_degenerate_by_scenario[scenario] = raw_deg
        centered_mse_by_scenario[scenario] = centered_mse
        state_distance_by_scenario[scenario] = d

    return {
        "aligned_rho": base._aggregate_scenario_group_h55(aligned_rho_by_scenario),
        "aligned_raw_rho": base._aggregate_scenario_group_h55(raw_rho_by_scenario),
        "aligned_nearest_decile_centered_mse_ratio": base._aggregate_scenario_group_h55(nearest_ratio_by_scenario),
        "state_rank_norm_by_scenario": state_rank_norm_by_scenario,
        "centered_rank_norm_by_scenario": centered_rank_norm_by_scenario,
        "raw_rank_norm_by_scenario": raw_rank_norm_by_scenario,
        "state_degenerate_by_scenario": state_degenerate_by_scenario,
        "centered_degenerate_by_scenario": centered_degenerate_by_scenario,
        "raw_degenerate_by_scenario": raw_degenerate_by_scenario,
        "centered_mse_by_scenario": centered_mse_by_scenario,
        "state_distance_by_scenario": state_distance_by_scenario,
        "degenerate_state_rank_row_count": int(sum(int(np.sum(x)) for x in state_degenerate_by_scenario.values())),
        "degenerate_centered_utility_rank_row_count": int(sum(int(np.sum(x)) for x in centered_degenerate_by_scenario.values())),
        "degenerate_raw_utility_rank_row_count": int(sum(int(np.sum(x)) for x in raw_degenerate_by_scenario.values())),
        "target_row_count": int(len(eval_order) * len(H5_SCENARIOS)),
    }


def run_fold_h55_clarified(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: Sequence[Any],
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
    train_order, train_rows_map = base._train_group_scenario_rows_h55(index=index, train_parents=train_parents)
    require(len(train_order) >= 10 and len(eval_order) >= 2, "H55C_INSUFFICIENT_GROUPS")

    train_rows = np.asarray([index.parent_row_by_id[pid] for pid in sorted(train_parents)], dtype=np.int32)
    train_x = np.asarray(index.features[train_rows], dtype=np.float64)
    train_mean = np.mean(train_x, axis=0)
    train_std = np.std(train_x, axis=0, ddof=0)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)
    require(np.isfinite(train_mean).all() and np.isfinite(train_std).all(), "H55C_BAD_TRAIN_NORMALIZATION")

    aligned_detail = {
        metric: _scenario_geometry_clarified_h55(
            index=index,
            train_mean=train_mean,
            train_std=train_std,
            train_order=train_order,
            train_rows_map=train_rows_map,
            eval_order=eval_order,
            eval_rows_map=eval_rows_map,
            metric=metric,
        )
        for metric in base.H55_METRICS
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
        require(k != 0, f"H55C_IDENTITY_ROTATION:{shift}")
        src = (np.arange(len(eval_order), dtype=np.int32) + k) % len(eval_order)
        for scenario in H5_SCENARIOS:
            dst_rows = np.asarray([eval_rows_map[g][scenario] for g in eval_order], dtype=np.int32)
            src_rows = dst_rows[src]
            require(
                np.array_equal(shuffled_index.features[dst_rows], index.features[src_rows]),
                f"H55C_ROTATION_MAPPING_DRIFT:{shift}:{scenario}",
            )
            require(
                np.array_equal(shuffled_index.utilities[dst_rows], index.utilities[dst_rows]),
                f"H55C_ROTATED_UTILITY_DRIFT:{shift}:{scenario}",
            )
        rotation_receipts[int(shift)] = receipt
        source_pos_by_shift[int(shift)] = src

    metric_results: dict[str, Any] = {}
    for metric in base.H55_METRICS:
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
                source_degenerate = detail["state_degenerate_by_scenario"][scenario][src]
                ratio_s[scenario] = nearest_decile_ratio_clarified_h55(
                    d[src], centered_mse, source_degenerate
                )
            shuffle_rho[int(shift)] = base._aggregate_scenario_group_h55(rho_s)
            shuffle_raw_rho[int(shift)] = base._aggregate_scenario_group_h55(raw_s)
            shuffle_nearest_ratio[int(shift)] = base._aggregate_scenario_group_h55(ratio_s)

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
            "aligned_gt_each_shuffle_count": int(
                sum(float(detail["aligned_rho"]) > float(shuffle_rho[int(s)]) for s in H5_SHIFTS)
            ),
            "degenerate_state_rank_row_count": int(detail["degenerate_state_rank_row_count"]),
            "degenerate_centered_utility_rank_row_count": int(detail["degenerate_centered_utility_rank_row_count"]),
            "degenerate_raw_utility_rank_row_count": int(detail["degenerate_raw_utility_rank_row_count"]),
            "target_row_count": int(detail["target_row_count"]),
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
        "degenerate_rank_convention": "ZERO_INFORMATION_RHO_ZERO__STATE_NEAREST_RATIO_ONE",
    }
