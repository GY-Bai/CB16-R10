from __future__ import annotations

"""H5.15 bounded full-current-state linear forward transport probe.

This module does not change H5.12R support, distances, utility, candidate
weights, or structured nulls.  It exposes the frozen target-row 102D state and
uses a fixed alpha=1 linear ridge trained only on the earlier side of each
adjacent environment pair to predict row-level local alignment statistics on
the later side.  Calendar/fold id and realized future are never input features.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .common_support_rank_geometry_decomposition_h511 import (
    H511_EPS,
    H511_NATIVE_ORIENTATION,
    H511_PAIRS,
    _fold_context_h511,
    weighted_partial_rank_h511,
)
from .medium48_temporal_halfblock_stability_h510 import orientation_h510
from .operator_conditional_medium_geometry_h58 import H58_SHIFTS
from .teacher_temporal_transport_audit_h5 import H5_SCENARIOS, require

H515_RUNTIME = "CB16_R11_H5_15_FULL_STATE_LINEAR_OFFSET_TRANSPORT_R0_V1"
H515_DIM = 102
H515_ALPHA = 1.0
H515_TOL = 1e-12
H515_SHIFT_CONTROLS = (1, 7, 13, 23, 31)
H515_FLIP_PAIRS = ((1, 2), (2, 3), (3, 4))
H515_CONTROL_PAIR = (4, 5)


def _as_full_state_h515(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    require(a.shape == (H515_DIM,), f"H515_FULL_STATE_DIM:{a.shape}")
    require(np.isfinite(a).all(), "H515_FULL_STATE_NONFINITE")
    return np.ascontiguousarray(a)


def build_side_dataset_h515(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: Sequence[Any],
    payload: Mapping[str, Any],
    candidate_multiplier: Sequence[float],
) -> dict[str, Any]:
    """Expose frozen target state and recompute exact H5.12R row labels."""
    index, eval_order, eval_rows_map, _ = _fold_context_h511(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    require(int(index.feature_dim) == H515_DIM, f"H515_INDEX_DIM:{index.feature_dim}")
    payload_ids = [str(g["future_group_id"]) for g in payload["groups"]]
    require(payload_ids == [str(g) for g in eval_order], "H515_PAYLOAD_ORDER_DRIFT")
    require(len(set(payload_ids)) == len(payload_ids), "H515_DUPLICATE_FUTURE_GROUP")

    cm = np.asarray(candidate_multiplier, dtype=np.float64).reshape(-1)
    require(cm.shape == (27,) and np.isfinite(cm).all() and np.all(cm > 0.0), "H515_BAD_CANDIDATE_MULTIPLIER")

    x_groups: list[np.ndarray] = []
    y_aligned_groups: list[np.ndarray] = []
    y_null_groups: dict[int, list[np.ndarray]] = {int(s): [] for s in H58_SHIFTS}
    zero_aligned = 0
    zero_null = 0

    for pgroup in payload["groups"]:
        gid = str(pgroup["future_group_id"])
        scenarios = list(pgroup["scenarios"])
        require(len(scenarios) == len(H5_SCENARIOS) == 6, "H515_SCENARIO_COUNT_DRIFT")
        by_scenario = {str(r["scenario"]): r for r in scenarios}
        require(set(by_scenario) == set(str(s) for s in H5_SCENARIOS), "H515_SCENARIO_SET_DRIFT")
        x_rows: list[np.ndarray] = []
        ya: list[float] = []
        yn: dict[int, list[float]] = {int(s): [] for s in H58_SHIFTS}
        for scenario in H5_SCENARIOS:
            s = str(scenario)
            target_row = int(eval_rows_map[gid][s])
            x_rows.append(_as_full_state_h515(index.features[target_row]))
            row = by_scenario[s]
            cells = np.asarray(row["cell_ids"], dtype=np.int32).reshape(-1)
            w = cm[cells]
            aligned, ar = weighted_partial_rank_h511(
                row["operator_rank"], row["medium_rank"], row["utility_rank"], w
            )
            ya.append(float(aligned))
            zero_aligned += int(bool(ar["zero_information"]))
            for shift in H58_SHIFTS:
                rho, rr = weighted_partial_rank_h511(
                    row["operator_rank"],
                    row["null_medium_rank_by_shift"][int(shift)],
                    row["utility_rank"],
                    w,
                )
                yn[int(shift)].append(float(rho))
                zero_null += int(bool(rr["zero_information"]))
        x_groups.append(np.stack(x_rows, axis=0))
        y_aligned_groups.append(np.asarray(ya, dtype=np.float64))
        for shift in H58_SHIFTS:
            y_null_groups[int(shift)].append(np.asarray(yn[int(shift)], dtype=np.float64))

    x = np.stack(x_groups, axis=0)
    ya = np.stack(y_aligned_groups, axis=0)
    yn = {int(s): np.stack(v, axis=0) for s, v in y_null_groups.items()}
    require(x.shape == (len(payload_ids), 6, H515_DIM), f"H515_X_SHAPE:{x.shape}")
    require(ya.shape == (len(payload_ids), 6), f"H515_Y_SHAPE:{ya.shape}")
    require(all(v.shape == ya.shape for v in yn.values()), "H515_NULL_Y_SHAPE")
    require(np.isfinite(x).all() and np.isfinite(ya).all() and all(np.isfinite(v).all() for v in yn.values()), "H515_DATA_NONFINITE")
    return {
        "fold": int(payload["fold"]),
        "future_group_ids": payload_ids,
        "x": x,
        "y_aligned": ya,
        "y_null_by_shift": yn,
        "aligned_mean": float(np.mean(ya)),
        "null_mean_by_shift": {int(s): float(np.mean(v)) for s, v in yn.items()},
        "null_median": float(statistics.median(float(np.mean(v)) for v in yn.values())),
        "zero_information_aligned_row_count": int(zero_aligned),
        "zero_information_null_row_count": int(zero_null),
        "calendar_or_fold_id_used_as_input": False,
        "realized_utility_used_as_input": False,
    }


def shift_state_bundles_h515(x: np.ndarray, shift: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    require(a.ndim == 3 and a.shape[1:] == (6, H515_DIM), "H515_SHIFT_X_SHAPE")
    require(a.shape[0] > abs(int(shift)), "H515_SHIFT_TOO_LARGE")
    out = np.roll(a, int(shift), axis=0)
    require(out.shape == a.shape and np.isfinite(out).all(), "H515_SHIFT_OUTPUT")
    return np.ascontiguousarray(out)


def _train_standardize_h515(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xa = np.asarray(train_x, dtype=np.float64).reshape(-1, H515_DIM)
    xb = np.asarray(eval_x, dtype=np.float64).reshape(-1, H515_DIM)
    require(np.isfinite(xa).all() and np.isfinite(xb).all(), "H515_STANDARDIZE_NONFINITE")
    mean = np.mean(xa, axis=0)
    std = np.std(xa, axis=0, ddof=0)
    constant = std <= H515_TOL
    safe = std.copy(); safe[constant] = 1.0
    za = (xa - mean) / safe
    zb = (xb - mean) / safe
    require(np.isfinite(za).all() and np.isfinite(zb).all(), "H515_STANDARDIZED_NONFINITE")
    return za, zb, {
        "constant_coordinate_count": int(np.sum(constant)),
        "train_standardized_abs_mean_max": float(np.max(np.abs(np.mean(za, axis=0)))),
        "train_standardized_std_nonconstant_min": float(np.min(np.std(za[:, ~constant], axis=0))) if np.any(~constant) else None,
        "train_standardized_std_nonconstant_max": float(np.max(np.std(za[:, ~constant], axis=0))) if np.any(~constant) else None,
    }


def fit_predict_ridge_h515(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    alpha: float = H515_ALPHA,
) -> dict[str, Any]:
    require(abs(float(alpha) - H515_ALPHA) <= 0.0, "H515_ALPHA_DRIFT")
    za, zb, norm = _train_standardize_h515(train_x, eval_x)
    y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    require(y.shape == (za.shape[0],) and np.isfinite(y).all(), "H515_RIDGE_Y_SHAPE")
    ymean = float(np.mean(y))
    yc = y - ymean
    gram = za.T @ za + float(alpha) * np.eye(H515_DIM, dtype=np.float64)
    rhs = za.T @ yc
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("H515_RIDGE_SOLVE") from exc
    pred_train = ymean + za @ beta
    pred_eval = ymean + zb @ beta
    require(np.isfinite(beta).all() and np.isfinite(pred_train).all() and np.isfinite(pred_eval).all(), "H515_RIDGE_NONFINITE")
    return {
        "train_label_mean": ymean,
        "beta": beta,
        "pred_train": pred_train,
        "pred_eval": pred_eval,
        "beta_l2": float(np.linalg.norm(beta)),
        "normalization_receipt": norm,
    }


def _group_means_h515(row_values: Sequence[float], group_count: int) -> np.ndarray:
    a = np.asarray(row_values, dtype=np.float64).reshape(group_count, 6)
    require(np.isfinite(a).all(), "H515_GROUP_MEAN_NONFINITE")
    return np.mean(a, axis=1)


def _qmse_h515(pred_rows: Sequence[float], true_rows: np.ndarray) -> float:
    true = np.asarray(true_rows, dtype=np.float64)
    require(true.ndim == 2 and true.shape[1] == 6, "H515_QMSE_TRUE_SHAPE")
    pg = _group_means_h515(pred_rows, true.shape[0])
    tg = np.mean(true, axis=1)
    return float(np.mean((pg - tg) ** 2))


def _fit_target_family_h515(train_x: np.ndarray, eval_x: np.ndarray, side_a: Mapping[str, Any], side_b: Mapping[str, Any]) -> dict[str, Any]:
    aligned = fit_predict_ridge_h515(train_x, side_a["y_aligned"], eval_x)
    null_models: dict[int, dict[str, Any]] = {}
    for shift in H58_SHIFTS:
        null_models[int(shift)] = fit_predict_ridge_h515(train_x, side_a["y_null_by_shift"][int(shift)], eval_x)
    g_b = int(len(side_b["future_group_ids"]))
    pred_aligned_mean = float(np.mean(aligned["pred_eval"]))
    pred_null_mean = {int(s): float(np.mean(m["pred_eval"])) for s, m in null_models.items()}
    pred_null_median = float(statistics.median(pred_null_mean.values()))
    pred_orientation = orientation_h510(pred_aligned_mean, pred_null_median)
    aligned_qmse = _qmse_h515(aligned["pred_eval"], side_b["y_aligned"])
    baseline_rows = np.full(g_b * 6, float(side_a["aligned_mean"]), dtype=np.float64)
    baseline_qmse = _qmse_h515(baseline_rows, side_b["y_aligned"])
    return {
        "predicted_aligned_mean_b": pred_aligned_mean,
        "predicted_null_mean_by_shift_b": pred_null_mean,
        "predicted_null_median_b": pred_null_median,
        "predicted_orientation_b": pred_orientation,
        "aligned_group_qmse_b": aligned_qmse,
        "baseline_group_qmse_b": baseline_qmse,
        "aligned_qmse_gain": float(baseline_qmse - aligned_qmse),
        "aligned_beta_l2": float(aligned["beta_l2"]),
        "aligned_normalization_receipt": aligned["normalization_receipt"],
        "null_beta_l2_by_shift": {int(s): float(m["beta_l2"]) for s, m in null_models.items()},
    }


def evaluate_forward_pair_h515(
    *, pair: Sequence[int], side_a: Mapping[str, Any], side_b: Mapping[str, Any]
) -> dict[str, Any]:
    fa, fb = [int(x) for x in pair]
    require((fa, fb) in H511_PAIRS, f"H515_PAIR:{fa}:{fb}")
    require(int(side_a["fold"]) == fa and int(side_b["fold"]) == fb, "H515_SIDE_FOLD_DRIFT")
    xa = np.asarray(side_a["x"], dtype=np.float64)
    xb = np.asarray(side_b["x"], dtype=np.float64)
    actual_a_orientation = orientation_h510(float(side_a["aligned_mean"]), float(side_a["null_median"]))
    actual_b_orientation = orientation_h510(float(side_b["aligned_mean"]), float(side_b["null_median"]))

    aligned_model = _fit_target_family_h515(xa, xb, side_a, side_b)
    controls: dict[int, dict[str, Any]] = {}
    for shift in H515_SHIFT_CONTROLS:
        shifted = shift_state_bundles_h515(xa, int(shift))
        controls[int(shift)] = _fit_target_family_h515(shifted, xb, side_a, side_b)

    observed_delta = float(side_b["aligned_mean"] - side_a["aligned_mean"])
    predicted_delta = float(aligned_model["predicted_aligned_mean_b"] - side_a["aligned_mean"])
    flip = (fa, fb) in H515_FLIP_PAIRS
    if flip:
        require(abs(observed_delta) > H511_EPS, "H515_ZERO_FLIP_DELTA")
        level_recovery = 1.0 - abs(predicted_delta - observed_delta) / abs(observed_delta)
        control_recovery = [
            1.0 - abs(float(c["predicted_aligned_mean_b"] - side_a["aligned_mean"]) - observed_delta) / abs(observed_delta)
            for c in controls.values()
        ]
        control_recovery_median = float(statistics.median(control_recovery))
        level_gate = bool(level_recovery > 0.0 and level_recovery > control_recovery_median)
    else:
        level_recovery = None
        control_recovery = []
        control_recovery_median = None
        level_gate = None

    control_qmse_gain = [float(c["aligned_qmse_gain"]) for c in controls.values()]
    control_qmse_gain_median = float(statistics.median(control_qmse_gain))
    qmse_gate = bool(
        float(aligned_model["aligned_qmse_gain"]) > 0.0
        and float(aligned_model["aligned_qmse_gain"]) > control_qmse_gain_median
    )
    orientation_gate = bool(
        str(aligned_model["predicted_orientation_b"]) == actual_b_orientation
        and ((not flip) or actual_b_orientation != actual_a_orientation)
    )
    pair_pass = bool(flip and level_gate and qmse_gate and orientation_gate)
    return {
        "pair": [fa, fb],
        "actual_side_a_aligned_mean": float(side_a["aligned_mean"]),
        "actual_side_a_null_median": float(side_a["null_median"]),
        "actual_side_a_orientation": actual_a_orientation,
        "actual_side_b_aligned_mean": float(side_b["aligned_mean"]),
        "actual_side_b_null_median": float(side_b["null_median"]),
        "actual_side_b_orientation": actual_b_orientation,
        "aligned_model": aligned_model,
        "shift_controls": controls,
        "observed_delta_rho": observed_delta,
        "predicted_delta_rho": predicted_delta,
        "level_recovery": level_recovery,
        "shift_control_level_recovery_values": control_recovery,
        "shift_control_level_recovery_median": control_recovery_median,
        "level_gate": level_gate,
        "shift_control_qmse_gain_values": control_qmse_gain,
        "shift_control_qmse_gain_median": control_qmse_gain_median,
        "qmse_gate": qmse_gate,
        "orientation_gate": orientation_gate,
        "native_flip_pair_pass": pair_pass,
    }


def classify_h515(pair_results: Sequence[Mapping[str, Any]], parent_ok: bool) -> dict[str, Any]:
    if not parent_ok:
        return {"classification": "EXECUTION_BLOCKED__H5_14_H5_13_H5_12R_PARENT_REPRODUCTION_FAILED"}
    by_pair = {(int(p["pair"][0]), int(p["pair"][1])): p for p in pair_results}
    require(set(by_pair) == set(H515_FLIP_PAIRS + (H515_CONTROL_PAIR,)), "H515_PAIR_SET_DRIFT")
    flip_passes = [bool(by_pair[p]["native_flip_pair_pass"]) for p in H515_FLIP_PAIRS]
    ctrl = by_pair[H515_CONTROL_PAIR]
    ctrl_ok = bool(
        ctrl["actual_side_a_orientation"] == "POSITIVE_ALIGNMENT"
        and ctrl["actual_side_b_orientation"] == "POSITIVE_ALIGNMENT"
        and ctrl["aligned_model"]["predicted_orientation_b"] == "POSITIVE_ALIGNMENT"
    )
    if all(flip_passes) and ctrl_ok:
        classification = "FULL_102D_LINEAR_CURRENT_STATE_OFFSET_TRANSPORT_SUPPORTED"
    elif not any(flip_passes) and ctrl_ok:
        classification = "FULL_102D_LINEAR_CURRENT_STATE_OFFSET_TRANSPORT_NOT_SUPPORTED"
    else:
        classification = "FULL_102D_LINEAR_CURRENT_STATE_OFFSET_TRANSPORT_PARTIAL_OR_MIXED"
    return {
        "classification": classification,
        "native_flip_pair_pass_count_of_3": int(sum(flip_passes)),
        "positive_control_predicted_orientation_pass": ctrl_ok,
    }
