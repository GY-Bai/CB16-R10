from __future__ import annotations

"""H5.16 fixed-RF full-state nonlinear utility-invariance probe.

Science-only diagnostic.  This module consumes the frozen H5.5 target-row
Full102 state and the same-row centered 9-action realized-utility profile.  It
never uses H5.12R pair/side q or IPF labels and never creates a runtime regime
feature, Teacher change, Student change, or canonical architecture change.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .teacher_temporal_transport_audit_h5 import (
    H5_SCENARIOS,
    _eval_group_scenario_rows_h5,
    _fold_material_h5,
    require,
)
from .teacher_vectorized_r11 import build_columnar_teacher_index_r11

H516_RUNTIME = "CB16_R11_H5_16_FULL_STATE_NONLINEAR_UTILITY_INVARIANCE_R0_V1"
H516_DIM = 102
H516_UTILITY_DIM = 9
H516_SHIFTS = (1, 7, 13, 23, 31)
H516_PAIRS = ((1, 2), (2, 3), (3, 4), (4, 5))
H516_NATIVE_PAIRS = ((1, 2), (2, 3), (3, 4))
H516_CROSSFIT_BLOCKS = 5
H516_SEED = 51616


def _regressor_h516() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        criterion="squared_error",
        max_features=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=5,
        bootstrap=True,
        max_samples=None,
        ccp_alpha=0.0,
        random_state=H516_SEED,
        n_jobs=1,
    )


def _classifier_h516() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_features=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=5,
        bootstrap=True,
        max_samples=None,
        ccp_alpha=0.0,
        class_weight=None,
        random_state=H516_SEED,
        n_jobs=1,
    )


def build_fold_dataset_h516(
    *, fold_spec: Mapping[str, Any], all_train_parents: Mapping[str, Any], all_train_samples: Sequence[Any]
) -> dict[str, Any]:
    """Build only the frozen H5.5 eval-row state and common utility target."""
    fold = int(fold_spec["fold"])
    _, eval_parents, _, _, _, samples = _fold_material_h5(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    index = build_columnar_teacher_index_r11(samples)
    require(int(index.feature_dim) == H516_DIM, f"H516_FEATURE_DIM:{index.feature_dim}")
    eval_ids = tuple(sorted(eval_parents))
    order, rows = _eval_group_scenario_rows_h5(index=index, eval_parents=eval_parents, eval_parent_ids=eval_ids)
    require(bool(order), f"H516_EMPTY_EVAL:{fold}")

    ts_by_gid: dict[str, int] = {}
    scenario_count: dict[str, int] = {}
    for p in eval_parents.values():
        gid = str(p.dependence_group_id)
        ts = int(p.decision_time_ms)
        if gid in ts_by_gid:
            require(ts_by_gid[gid] == ts, f"H516_GROUP_CLOCK_DRIFT:{fold}:{gid}")
        ts_by_gid[gid] = ts
        scenario_count[gid] = scenario_count.get(gid, 0) + 1
    require(all(scenario_count.get(g, 0) == len(H5_SCENARIOS) == 6 for g in order), f"H516_SCENARIO_COUNT_DRIFT:{fold}")
    require(order == sorted(order, key=lambda g: (ts_by_gid[g], g)), f"H516_GROUP_ORDER_DRIFT:{fold}")

    x_groups: list[np.ndarray] = []
    y_groups: list[np.ndarray] = []
    for gid in order:
        xr: list[np.ndarray] = []
        yr: list[np.ndarray] = []
        for scenario in H5_SCENARIOS:
            row = int(rows[gid][scenario])
            x = np.asarray(index.features[row], dtype=np.float64).reshape(-1)
            u = np.asarray(index.utilities[row], dtype=np.float64).reshape(-1)
            require(x.shape == (H516_DIM,) and np.isfinite(x).all(), f"H516_STATE_ROW_DRIFT:{fold}:{gid}:{scenario}")
            require(u.shape == (H516_UTILITY_DIM,) and np.isfinite(u).all(), f"H516_UTILITY_ROW_DRIFT:{fold}:{gid}:{scenario}")
            uc = u - float(np.mean(u))
            require(np.isfinite(uc).all() and abs(float(np.mean(uc))) <= 1e-12, f"H516_CENTERING_DRIFT:{fold}:{gid}:{scenario}")
            xr.append(x)
            yr.append(uc)
        x_groups.append(np.stack(xr, axis=0))
        y_groups.append(np.stack(yr, axis=0))

    x = np.stack(x_groups, axis=0)
    y = np.stack(y_groups, axis=0)
    timestamps = np.asarray([ts_by_gid[g] for g in order], dtype=np.int64)
    require(x.shape == (len(order), 6, H516_DIM), f"H516_X_SHAPE:{x.shape}")
    require(y.shape == (len(order), 6, H516_UTILITY_DIM), f"H516_Y_SHAPE:{y.shape}")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "H516_DATA_NONFINITE")
    return {
        "fold": fold,
        "future_group_ids": tuple(str(g) for g in order),
        "timestamps_ms": timestamps,
        "x": np.ascontiguousarray(x),
        "y": np.ascontiguousarray(y),
        "unique_decision_clock_count": int(len(np.unique(timestamps))),
        "future_group_count": int(len(order)),
        "scenario_count_per_group": 6,
        "state_dimension": H516_DIM,
        "utility_dimension": H516_UTILITY_DIM,
    }


def clock_blocks_h516(timestamps: Sequence[int]) -> tuple[np.ndarray, ...]:
    clocks = np.asarray(sorted({int(x) for x in timestamps}), dtype=np.int64)
    require(len(clocks) >= H516_CROSSFIT_BLOCKS, f"H516_TOO_FEW_CLOCKS:{len(clocks)}")
    q, r = divmod(len(clocks), H516_CROSSFIT_BLOCKS)
    out: list[np.ndarray] = []
    start = 0
    for i in range(H516_CROSSFIT_BLOCKS):
        n = q + (1 if i < r else 0)
        block = clocks[start : start + n]
        require(len(block) > 0, f"H516_EMPTY_CLOCK_BLOCK:{i}")
        out.append(np.asarray(block, dtype=np.int64))
        start += n
    require(start == len(clocks), "H516_CLOCK_PARTITION_DRIFT")
    require(np.array_equal(np.concatenate(out), clocks), "H516_CLOCK_PARTITION_ORDER_DRIFT")
    return tuple(out)


def row_weights_h516(timestamps: Sequence[int]) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    clocks, counts = np.unique(ts, return_counts=True)
    require(len(clocks) > 0 and len(ts) > 0, "H516_EMPTY_WEIGHT_INPUT")
    count_by_clock = {int(c): int(n) for c, n in zip(clocks.tolist(), counts.tolist())}
    w = np.asarray([1.0 / (len(clocks) * count_by_clock[int(t)] * 6.0) for t in ts for _ in range(6)], dtype=np.float64)
    require(np.isfinite(w).all() and np.all(w > 0.0) and abs(float(np.sum(w)) - 1.0) <= 1e-12, "H516_WEIGHT_DRIFT")
    return w


def clock_equal_target_mean_h516(y: np.ndarray, timestamps: Sequence[int]) -> np.ndarray:
    a = np.asarray(y, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    require(a.shape == (len(ts), 6, H516_UTILITY_DIM), f"H516_MEAN_SHAPE:{a.shape}")
    group_mean = np.mean(a, axis=1)
    clock_mean = [np.mean(group_mean[ts == c], axis=0) for c in sorted(set(int(x) for x in ts.tolist()))]
    out = np.mean(np.stack(clock_mean, axis=0), axis=0)
    require(out.shape == (H516_UTILITY_DIM,) and np.isfinite(out).all(), "H516_TARGET_MEAN_NONFINITE")
    return out


def clock_equal_profile_mse_h516(pred: np.ndarray, y: np.ndarray, timestamps: Sequence[int]) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    require(p.shape == t.shape == (len(ts), 6, H516_UTILITY_DIM), f"H516_LOSS_SHAPE:{p.shape}:{t.shape}")
    require(np.isfinite(p).all() and np.isfinite(t).all(), "H516_LOSS_NONFINITE_INPUT")
    group_loss = np.mean(np.mean((p - t) ** 2, axis=2), axis=1)
    clock_loss = [float(np.mean(group_loss[ts == c])) for c in sorted(set(int(x) for x in ts.tolist()))]
    out = float(np.mean(clock_loss))
    require(np.isfinite(out) and out >= 0.0, "H516_LOSS_NONFINITE")
    return out


def _clock_equal_scalar_h516(values: np.ndarray, timestamps: Sequence[int]) -> float:
    a = np.asarray(values, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    require(a.shape == (len(ts), 6), f"H516_SCALAR_SHAPE:{a.shape}")
    group = np.mean(a, axis=1)
    out = float(np.mean([np.mean(group[ts == c]) for c in sorted(set(int(x) for x in ts.tolist()))]))
    require(np.isfinite(out), "H516_SCALAR_NONFINITE")
    return out


def shift_group_state_h516(x: np.ndarray, timestamps: Sequence[int], group_ids: Sequence[str], shift: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    gids = tuple(str(g) for g in group_ids)
    require(a.shape == (len(ts), 6, H516_DIM) and len(gids) == len(ts), "H516_SHIFT_SHAPE")
    order = np.asarray(sorted(range(len(ts)), key=lambda i: (int(ts[i]), gids[i])), dtype=np.int64)
    n = len(order)
    k = int(shift) % n
    require(k != 0, f"H516_NEGATIVE_CONTROL_IDENTITY:{shift}:{n}")
    src_pos = (np.arange(n, dtype=np.int64) + k) % n
    out = np.array(a, copy=True)
    out[order] = a[order[src_pos]]
    require(not np.array_equal(out, a), f"H516_NEGATIVE_CONTROL_STATE_IDENTITY:{shift}:{n}")
    return np.ascontiguousarray(out)


def _fit_predict_regressor_h516(train_x: np.ndarray, train_y: np.ndarray, train_ts: Sequence[int], eval_x: np.ndarray) -> np.ndarray:
    tx = np.asarray(train_x, dtype=np.float64)
    ty = np.asarray(train_y, dtype=np.float64)
    ex = np.asarray(eval_x, dtype=np.float64)
    require(tx.ndim == 3 and tx.shape[1] == 6 and tx.shape[2] in (102, 103), "H516_RF_TRAIN_X_SHAPE")
    require(ty.shape == (tx.shape[0], 6, H516_UTILITY_DIM), "H516_RF_TRAIN_Y_SHAPE")
    require(ex.ndim == 3 and ex.shape[1] == 6 and ex.shape[2] == tx.shape[2], "H516_RF_EVAL_X_SHAPE")
    model = _regressor_h516()
    model.fit(tx.reshape(-1, tx.shape[2]), ty.reshape(-1, H516_UTILITY_DIM), sample_weight=row_weights_h516(train_ts))
    pred = np.asarray(model.predict(ex.reshape(-1, ex.shape[2])), dtype=np.float64).reshape(ex.shape[0], 6, H516_UTILITY_DIM)
    require(np.isfinite(pred).all(), "H516_RF_PRED_NONFINITE")
    return pred


def _augment_environment_h516(x: np.ndarray, env: Sequence[int]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    e = np.asarray(env, dtype=np.float64).reshape(-1)
    require(a.shape == (len(e), 6, H516_DIM), "H516_ENV_AUGMENT_SHAPE")
    ee = np.repeat(e[:, None, None], 6, axis=1)
    out = np.concatenate([a, ee], axis=2)
    require(out.shape == (len(e), 6, H516_DIM + 1) and np.isfinite(out).all(), "H516_ENV_AUGMENT_OUTPUT")
    return np.ascontiguousarray(out)


def run_local_arm_h516(dataset: Mapping[str, Any]) -> dict[str, Any]:
    fold = int(dataset["fold"])
    x = np.asarray(dataset["x"], dtype=np.float64)
    y = np.asarray(dataset["y"], dtype=np.float64)
    ts = np.asarray(dataset["timestamps_ms"], dtype=np.int64)
    gids = tuple(dataset["future_group_ids"])
    blocks = clock_blocks_h516(ts)
    pred = np.empty_like(y)
    base = np.empty_like(y)
    control_pred = {s: np.empty_like(y) for s in H516_SHIFTS}
    block_receipts: list[dict[str, Any]] = []
    for i, block in enumerate(blocks):
        emask = np.isin(ts, block)
        tmask = ~emask
        require(np.any(emask) and np.any(tmask), f"H516_LOCAL_EMPTY_SPLIT:{fold}:{i}")
        pred[emask] = _fit_predict_regressor_h516(x[tmask], y[tmask], ts[tmask], x[emask])
        mu = clock_equal_target_mean_h516(y[tmask], ts[tmask])
        base[emask] = np.broadcast_to(mu, base[emask].shape)
        train_gids = tuple(gids[j] for j in np.flatnonzero(tmask))
        for s in H516_SHIFTS:
            sx = shift_group_state_h516(x[tmask], ts[tmask], train_gids, int(s))
            control_pred[int(s)][emask] = _fit_predict_regressor_h516(sx, y[tmask], ts[tmask], x[emask])
        block_receipts.append({
            "block": i + 1,
            "heldout_clock_count": int(len(block)),
            "train_clock_count": int(len(np.unique(ts[tmask]))),
            "heldout_group_count": int(np.sum(emask)),
            "train_group_count": int(np.sum(tmask)),
        })
    baseline_loss = clock_equal_profile_mse_h516(base, y, ts)
    model_loss = clock_equal_profile_mse_h516(pred, y, ts)
    gain = float(baseline_loss - model_loss)
    control_losses = {int(s): clock_equal_profile_mse_h516(p, y, ts) for s, p in control_pred.items()}
    control_gains = {int(s): float(baseline_loss - v) for s, v in control_losses.items()}
    med = float(statistics.median(control_gains.values()))
    return {
        "fold": fold,
        "baseline_loss": baseline_loss,
        "true_binding_loss": model_loss,
        "true_binding_gain": gain,
        "shift_control_loss": control_losses,
        "shift_control_gain": control_gains,
        "median_shift_control_gain": med,
        "positive_gain": bool(gain > 0.0),
        "beats_shift_median": bool(gain > med),
        "beats_each_shift_count": int(sum(gain > float(v) for v in control_gains.values())),
        "clock_blocks": block_receipts,
    }


def run_forward_arm_h516(side_a: Mapping[str, Any], side_b: Mapping[str, Any]) -> dict[str, Any]:
    pair = (int(side_a["fold"]), int(side_b["fold"]))
    require(pair in H516_PAIRS, f"H516_FORWARD_PAIR:{pair}")
    xa, ya = np.asarray(side_a["x"]), np.asarray(side_a["y"])
    xb, yb = np.asarray(side_b["x"]), np.asarray(side_b["y"])
    tsa, tsb = np.asarray(side_a["timestamps_ms"]), np.asarray(side_b["timestamps_ms"])
    pred = _fit_predict_regressor_h516(xa, ya, tsa, xb)
    mu = clock_equal_target_mean_h516(ya, tsa)
    base = np.broadcast_to(mu, yb.shape)
    baseline_loss = clock_equal_profile_mse_h516(base, yb, tsb)
    model_loss = clock_equal_profile_mse_h516(pred, yb, tsb)
    gain = float(baseline_loss - model_loss)
    controls: dict[int, float] = {}
    for s in H516_SHIFTS:
        sx = shift_group_state_h516(xa, tsa, side_a["future_group_ids"], int(s))
        cp = _fit_predict_regressor_h516(sx, ya, tsa, xb)
        controls[int(s)] = float(baseline_loss - clock_equal_profile_mse_h516(cp, yb, tsb))
    med = float(statistics.median(controls.values()))
    return {
        "pair": list(pair),
        "baseline_loss_b": baseline_loss,
        "true_binding_loss_b": model_loss,
        "true_binding_gain": gain,
        "shift_control_gain": controls,
        "median_shift_control_gain": med,
        "pair_pass": bool(gain > 0.0 and gain > med),
        "beats_each_shift_count": int(sum(gain > float(v) for v in controls.values())),
    }


def _pooled_pair_h516(side_a: Mapping[str, Any], side_b: Mapping[str, Any]) -> dict[str, Any]:
    pair = (int(side_a["fold"]), int(side_b["fold"]))
    require(pair in H516_PAIRS, f"H516_POOL_PAIR:{pair}")
    tsa = np.asarray(side_a["timestamps_ms"], dtype=np.int64)
    tsb = np.asarray(side_b["timestamps_ms"], dtype=np.int64)
    overlap = set(int(x) for x in tsa.tolist()) & set(int(x) for x in tsb.tolist())
    require(not overlap, f"H516_POOLED_CLOCK_MULTI_ENV:{pair}:{len(overlap)}")
    x = np.concatenate([np.asarray(side_a["x"]), np.asarray(side_b["x"])], axis=0)
    y = np.concatenate([np.asarray(side_a["y"]), np.asarray(side_b["y"])], axis=0)
    ts = np.concatenate([tsa, tsb])
    env = np.concatenate([np.zeros(len(tsa), dtype=np.int8), np.ones(len(tsb), dtype=np.int8)])
    gids = tuple(side_a["future_group_ids"]) + tuple(side_b["future_group_ids"])
    clock_env: dict[int, int] = {}
    for t, e in zip(ts.tolist(), env.tolist()):
        if int(t) in clock_env:
            require(clock_env[int(t)] == int(e), f"H516_POOLED_CLOCK_MULTI_ENV:{pair}:{t}")
        clock_env[int(t)] = int(e)
    return {"pair": pair, "x": x, "y": y, "ts": ts, "env": env, "gids": gids, "clock_env": clock_env}


def _shift_clock_environment_h516(ts: Sequence[int], env: Sequence[int], shift: int) -> np.ndarray:
    t = np.asarray(ts, dtype=np.int64).reshape(-1)
    e = np.asarray(env, dtype=np.int8).reshape(-1)
    clocks = np.asarray(sorted(set(int(x) for x in t.tolist())), dtype=np.int64)
    base = np.asarray([int(e[np.flatnonzero(t == c)[0]]) for c in clocks], dtype=np.int8)
    for c, v in zip(clocks.tolist(), base.tolist()):
        require(np.all(e[t == c] == v), f"H516_CLOCK_ENV_DRIFT:{c}")
    k = int(shift) % len(clocks)
    require(k != 0, f"H516_FAKE_ENV_SHIFT_IDENTITY_INDEX:{shift}:{len(clocks)}")
    shifted = base[(np.arange(len(base), dtype=np.int64) + k) % len(base)]
    require(not np.array_equal(base, shifted), f"H516_FAKE_ENV_LABEL_IDENTITY:{shift}:{len(clocks)}")
    mapping = {int(c): int(v) for c, v in zip(clocks.tolist(), shifted.tolist())}
    out = np.asarray([mapping[int(v)] for v in t.tolist()], dtype=np.int8)
    require(int(np.sum(out)) == int(np.sum(e)), "H516_FAKE_ENV_MARGINAL_DRIFT")
    return out


def _support_classifier_oof_h516(x: np.ndarray, env: np.ndarray, ts: np.ndarray, blocks: Sequence[np.ndarray]) -> dict[str, float]:
    pred_class = np.empty((len(ts), 6), dtype=np.float64)
    pred_prob = np.empty((len(ts), 6), dtype=np.float64)
    for block in blocks:
        emask = np.isin(ts, block)
        tmask = ~emask
        tx = x[tmask].reshape(-1, H516_DIM)
        te = np.repeat(env[tmask], 6)
        require(set(int(v) for v in np.unique(te).tolist()) == {0, 1}, "H516_SUPPORT_CLASSIFIER_TRAIN_CLASS_DRIFT")
        clf = _classifier_h516()
        clf.fit(tx, te, sample_weight=row_weights_h516(ts[tmask]))
        require(tuple(int(v) for v in clf.classes_.tolist()) == (0, 1), "H516_SUPPORT_CLASSIFIER_CLASS_ORDER")
        ex = x[emask].reshape(-1, H516_DIM)
        proba = np.asarray(clf.predict_proba(ex), dtype=np.float64)
        pc = np.asarray(clf.predict(ex), dtype=np.int8)
        true = np.repeat(env[emask], 6)
        idx = np.arange(len(true), dtype=np.int64)
        ptrue = proba[idx, true]
        pred_class[emask] = (pc == true).astype(np.float64).reshape(-1, 6)
        eps = np.finfo(np.float64).eps
        pred_prob[emask] = (-np.log(np.clip(ptrue, eps, 1.0))).reshape(-1, 6)
    return {
        "oof_accuracy_clock_equal": _clock_equal_scalar_h516(pred_class, ts),
        "oof_log_loss_clock_equal": _clock_equal_scalar_h516(pred_prob, ts),
    }


def run_environment_arm_h516(side_a: Mapping[str, Any], side_b: Mapping[str, Any]) -> dict[str, Any]:
    pooled = _pooled_pair_h516(side_a, side_b)
    pair = pooled["pair"]
    x, y, ts, env = pooled["x"], pooled["y"], pooled["ts"], pooled["env"]
    blocks = clock_blocks_h516(ts)
    px = np.empty_like(y)
    pxe = np.empty_like(y)
    fake_pred = {s: np.empty_like(y) for s in H516_SHIFTS}
    fake_env = {s: _shift_clock_environment_h516(ts, env, int(s)) for s in H516_SHIFTS}
    for block in blocks:
        emask = np.isin(ts, block)
        tmask = ~emask
        px[emask] = _fit_predict_regressor_h516(x[tmask], y[tmask], ts[tmask], x[emask])
        pxe[emask] = _fit_predict_regressor_h516(
            _augment_environment_h516(x[tmask], env[tmask]), y[tmask], ts[tmask], _augment_environment_h516(x[emask], env[emask])
        )
        for s in H516_SHIFTS:
            fe = fake_env[int(s)]
            fake_pred[int(s)][emask] = _fit_predict_regressor_h516(
                _augment_environment_h516(x[tmask], fe[tmask]), y[tmask], ts[tmask], _augment_environment_h516(x[emask], fe[emask])
            )
    loss_x = clock_equal_profile_mse_h516(px, y, ts)
    loss_true = clock_equal_profile_mse_h516(pxe, y, ts)
    increment = float(loss_x - loss_true)
    fake_losses = {int(s): clock_equal_profile_mse_h516(p, y, ts) for s, p in fake_pred.items()}
    fake_inc = {int(s): float(loss_x - v) for s, v in fake_losses.items()}
    med = float(statistics.median(fake_inc.values()))
    support = _support_classifier_oof_h516(x, env, ts, blocks)
    return {
        "pair": list(pair),
        "x_only_oof_loss": loss_x,
        "x_plus_true_environment_oof_loss": loss_true,
        "true_environment_increment": increment,
        "fake_environment_oof_loss": fake_losses,
        "fake_environment_increment": fake_inc,
        "median_fake_environment_increment": med,
        "pair_pass": bool(increment > 0.0 and increment > med),
        "beats_each_fake_environment_count": int(sum(increment > float(v) for v in fake_inc.values())),
        "support_separability_secondary": support,
        "pooled_unique_clock_count": int(len(np.unique(ts))),
        "pooled_group_count": int(len(ts)),
        "environment_index_runtime_feature": False,
    }


def classify_h516(
    local_results: Sequence[Mapping[str, Any]],
    forward_results: Sequence[Mapping[str, Any]],
    environment_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    local = sorted(local_results, key=lambda r: int(r["fold"]))
    require(tuple(int(r["fold"]) for r in local) == (1, 2, 3, 4, 5), "H516_LOCAL_FOLD_SET_DRIFT")
    positive_count = int(sum(bool(r["positive_gain"]) for r in local))
    median_count = int(sum(bool(r["beats_shift_median"]) for r in local))
    pairwise = int(sum(int(r["beats_each_shift_count"]) for r in local))
    late_positive = all(bool(local[i - 1]["positive_gain"]) for i in (4, 5))
    late_median = all(bool(local[i - 1]["beats_shift_median"]) for i in (4, 5))
    local_supported = bool(positive_count >= 4 and late_positive and median_count >= 4 and late_median and pairwise >= 20)

    fwd = {tuple(int(x) for x in r["pair"]): r for r in forward_results}
    env = {tuple(int(x) for x in r["pair"]): r for r in environment_results}
    require(set(fwd) == set(H516_PAIRS) and set(env) == set(H516_PAIRS), "H516_PAIR_SET_DRIFT")
    native_forward = int(sum(bool(fwd[p]["pair_pass"]) for p in H516_NATIVE_PAIRS))
    native_environment = int(sum(bool(env[p]["pair_pass"]) for p in H516_NATIVE_PAIRS))

    if local_supported and native_forward == 3 and native_environment == 0:
        classification = "FULL102_NONLINEAR_LOCAL_MAP_AND_NATIVE_FORWARD_TRANSPORT_SUPPORTED__NO_ENVIRONMENT_INCREMENT"
    elif local_supported and native_forward == 0 and native_environment == 3:
        classification = "FULL102_NONLINEAR_LOCAL_MAP_EXISTS__FORWARD_NONTRANSPORT__ENVIRONMENT_INDEX_INCREMENT_WITHIN_FIXED_RF"
    elif not local_supported:
        classification = "FULL102_NONLINEAR_UTILITY_MAP_WEAK_EVEN_LOCAL"
    else:
        classification = "FULL102_NONLINEAR_LOCAL_FORWARD_OR_ENVIRONMENT_INCREMENT_MIXED"
    return {
        "classification": classification,
        "local_map_supported": local_supported,
        "local_positive_gain_fold_count": positive_count,
        "local_positive_gain_both_late": bool(late_positive),
        "local_beats_shift_median_fold_count": median_count,
        "local_beats_shift_median_both_late": bool(late_median),
        "local_pairwise_shift_wins_of_25": pairwise,
        "native_forward_pair_pass_count": native_forward,
        "native_environment_increment_pair_pass_count": native_environment,
        "stability_forward_pair_pass": bool(fwd[(4, 5)]["pair_pass"]),
        "stability_environment_increment_pair_pass": bool(env[(4, 5)]["pair_pass"]),
    }
