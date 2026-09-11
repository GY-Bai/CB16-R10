from __future__ import annotations

"""H5.17 exploratory three-arm forward representation probe.

This module is deliberately non-authoritative. It reuses the frozen H5.16
centered Utility9 target, clock-equal weighting, fixed RandomForestRegressor
configuration, and whole-future-group shift controls. It does not touch FINAL,
download data, or change canonical Teacher/Student/Physics/Supervisor semantics.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from cb16_local_opt.full_state_nonlinear_utility_invariance_h516 import (
    H516_DIM,
    H516_SEED,
    H516_SHIFTS,
    H516_UTILITY_DIM,
    _regressor_h516,
    clock_equal_profile_mse_h516,
    clock_equal_target_mean_h516,
    row_weights_h516,
    shift_group_state_h516,
)
from cb16_local_opt.teacher_temporal_transport_audit_h5 import (
    H5_SCENARIOS,
    _eval_group_scenario_rows_h5,
    _fold_material_h5,
)
from cb16_local_opt.teacher_vectorized_r11 import build_columnar_teacher_index_r11

MARKET_DIM = 96
ACCOUNT_DIM = 6
BOTTLENECK_DIM = 8
RANDOM_PROJECTION_SEED = 51708
RIDGE_ALPHA = 1.0
PURGE_HOURS = 256
SCENARIO_COUNT = 6


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def _group_weights(ts: Sequence[int]) -> np.ndarray:
    rw = row_weights_h516(ts).reshape(-1, SCENARIO_COUNT)
    w = rw.sum(axis=1)
    require(abs(float(w.sum()) - 1.0) <= 1e-12, "H517_GROUP_WEIGHT_DRIFT")
    return w


def _weighted_mean_std(x: np.ndarray, w: np.ndarray, floor: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=np.float64)
    ww = np.asarray(w, dtype=np.float64).reshape(-1)
    require(a.ndim == 2 and len(a) == len(ww), "H517_SCALE_SHAPE")
    ww = ww / ww.sum()
    mu = np.sum(a * ww[:, None], axis=0)
    var = np.sum(((a - mu) ** 2) * ww[:, None], axis=0)
    sd = np.sqrt(np.maximum(var, 0.0))
    sd = np.where(sd < floor, 1.0, sd)
    return mu, sd


def _side_dataset(index: Any, side_parents: Mapping[str, Any], fold: int, side: str) -> dict[str, Any]:
    ids = tuple(sorted(side_parents))
    order, rows = _eval_group_scenario_rows_h5(
        index=index, eval_parents=side_parents, eval_parent_ids=ids
    )
    ts_by_gid: dict[str, int] = {}
    for p in side_parents.values():
        gid = str(p.dependence_group_id)
        t = int(p.decision_time_ms)
        if gid in ts_by_gid:
            require(ts_by_gid[gid] == t, f"H517_GROUP_CLOCK_DRIFT:{fold}:{side}:{gid}")
        ts_by_gid[gid] = t

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for gid in order:
        xr: list[np.ndarray] = []
        yr: list[np.ndarray] = []
        for scenario in H5_SCENARIOS:
            row = int(rows[gid][scenario])
            x = np.asarray(index.features[row], dtype=np.float64).reshape(-1)
            u = np.asarray(index.utilities[row], dtype=np.float64).reshape(-1)
            require(x.shape == (H516_DIM,) and np.isfinite(x).all(), f"H517_X_DRIFT:{fold}:{side}")
            require(u.shape == (H516_UTILITY_DIM,) and np.isfinite(u).all(), f"H517_Y_DRIFT:{fold}:{side}")
            uc = u - float(np.mean(u))
            require(abs(float(np.mean(uc))) <= 1e-12, f"H517_CENTER_DRIFT:{fold}:{side}")
            xr.append(x)
            yr.append(uc)
        xs.append(np.stack(xr, axis=0))
        ys.append(np.stack(yr, axis=0))
    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    ts = np.asarray([ts_by_gid[g] for g in order], dtype=np.int64)
    return {
        "fold": int(fold),
        "side": side,
        "future_group_ids": tuple(str(g) for g in order),
        "timestamps_ms": ts,
        "x": np.ascontiguousarray(x),
        "y": np.ascontiguousarray(y),
        "future_group_count": int(len(order)),
        "unique_decision_clock_count": int(len(np.unique(ts))),
    }


def build_fold_pair_h517(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: Sequence[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fold = int(fold_spec["fold"])
    train_parents, eval_parents, _, _, _, samples = _fold_material_h5(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    index = build_columnar_teacher_index_r11(samples)
    train = _side_dataset(index, train_parents, fold, "train")
    evald = _side_dataset(index, eval_parents, fold, "eval")
    gap_ms = int(evald["timestamps_ms"].min()) - int(train["timestamps_ms"].max())
    require(
        gap_ms >= PURGE_HOURS * 3_600_000,
        f"H517_LABEL_MATURITY_PURGE_DRIFT:{fold}:{gap_ms}",
    )
    train["train_to_eval_gap_hours"] = float(gap_ms / 3_600_000)
    evald["train_to_eval_gap_hours"] = float(gap_ms / 3_600_000)
    return train, evald


def block_profile_decomposition(
    pred: np.ndarray, y: np.ndarray, ts: Sequence[int]
) -> dict[str, float]:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    require(p.shape == t.shape and p.ndim == 3 and p.shape[1:] == (6, 9), "H517_DECOMP_SHAPE")
    w = row_weights_h516(ts)
    pf = p.reshape(-1, H516_UTILITY_DIM)
    tf = t.reshape(-1, H516_UTILITY_DIM)
    mu_p = np.sum(pf * w[:, None], axis=0)
    mu_t = np.sum(tf * w[:, None], axis=0)
    level = float(np.mean((mu_p - mu_t) ** 2))
    shape = float(np.sum(w * np.mean(((pf - mu_p) - (tf - mu_t)) ** 2, axis=1)))
    total = float(np.sum(w * np.mean((pf - tf) ** 2, axis=1)))
    require(abs(total - level - shape) <= 1e-12 * max(1.0, abs(total)), "H517_DECOMP_IDENTITY")
    return {
        "total": total,
        "level": level,
        "shape": shape,
        "level_share": float(level / total) if total > 0.0 else 0.0,
    }


def _market_scaler(train_x: np.ndarray, train_ts: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(train_x, dtype=np.float64)
    spread = float(np.max(np.std(x[:, :, :MARKET_DIM], axis=1)))
    require(spread <= 1e-12, f"H517_MARKET_SIBLING_DRIFT:{spread}")
    market = x[:, 0, :MARKET_DIM]
    return _weighted_mean_std(market, _group_weights(train_ts))


def _account_aux_scaler(train_x: np.ndarray, train_ts: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    account = np.asarray(train_x, dtype=np.float64)[:, :, MARKET_DIM:].reshape(-1, ACCOUNT_DIM)
    return _weighted_mean_std(account, row_weights_h516(train_ts))


def _orthonormal_random_projection() -> np.ndarray:
    rng = np.random.default_rng(RANDOM_PROJECTION_SEED)
    q, _ = np.linalg.qr(rng.normal(size=(MARKET_DIM, BOTTLENECK_DIM)))
    return np.ascontiguousarray(q[:, :BOTTLENECK_DIM], dtype=np.float64)


def transform_random_arm(
    train_x: np.ndarray, eval_x: np.ndarray, train_ts: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mu, sd = _market_scaler(train_x, train_ts)
    q = _orthonormal_random_projection()

    def tx(x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64)
        z = ((a[:, :, :MARKET_DIM] - mu) / sd) @ q
        return np.ascontiguousarray(np.concatenate([z, a[:, :, MARKET_DIM:]], axis=2))

    return tx(train_x), tx(eval_x), {
        "kind": "FIXED_ORTHONORMAL_GAUSSIAN_PROJECTION",
        "seed": RANDOM_PROJECTION_SEED,
        "dimension": BOTTLENECK_DIM,
    }


def fit_supervised_encoder(
    train_x: np.ndarray, train_y: np.ndarray, train_ts: Sequence[int]
) -> dict[str, Any]:
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    mu_m, sd_m = _market_scaler(x, train_ts)
    mu_a, sd_a = _account_aux_scaler(x, train_ts)
    m = ((x[:, :, :MARKET_DIM] - mu_m) / sd_m).reshape(-1, MARKET_DIM)
    a = ((x[:, :, MARKET_DIM:] - mu_a) / sd_a).reshape(-1, ACCOUNT_DIM)
    target = y.reshape(-1, H516_UTILITY_DIM)
    sw = row_weights_h516(train_ts)

    joint = np.concatenate([m, a], axis=1)
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(joint, target, sample_weight=sw)
    market_coef = np.asarray(ridge.coef_, dtype=np.float64).T[:MARKET_DIM, :]
    u, s, _ = np.linalg.svd(market_coef, full_matrices=False)
    enc = np.asarray(u[:, :BOTTLENECK_DIM], dtype=np.float64)
    for j in range(enc.shape[1]):
        k = int(np.argmax(np.abs(enc[:, j])))
        if enc[k, j] < 0.0:
            enc[:, j] *= -1.0
    z = m @ enc
    aux_x = np.concatenate([z, a], axis=1)
    aux = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    aux.fit(aux_x, target, sample_weight=sw)
    aux_pred = np.asarray(aux.predict(aux_x), dtype=np.float64)
    aux_mse = float(np.sum(sw * np.mean((aux_pred - target) ** 2, axis=1)))
    return {
        "market_mu": mu_m,
        "market_sd": sd_m,
        "account_aux_mu": mu_a,
        "account_aux_sd": sd_a,
        "encoder": np.ascontiguousarray(enc),
        "singular_values": np.asarray(s, dtype=np.float64),
        "auxiliary_train_clock_equal_mse": aux_mse,
        "ridge_alpha": RIDGE_ALPHA,
        "dimension": BOTTLENECK_DIM,
    }


def transform_supervised_arm(x: np.ndarray, encoder: Mapping[str, Any]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    z = ((a[:, :, :MARKET_DIM] - encoder["market_mu"]) / encoder["market_sd"]) @ encoder["encoder"]
    return np.ascontiguousarray(np.concatenate([z, a[:, :, MARKET_DIM:]], axis=2))


def fit_predict_rf(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_ts: Sequence[int],
    eval_x: np.ndarray,
) -> tuple[np.ndarray, float]:
    tx = np.asarray(train_x, dtype=np.float64)
    ty = np.asarray(train_y, dtype=np.float64)
    ex = np.asarray(eval_x, dtype=np.float64)
    require(tx.ndim == 3 and ex.ndim == 3 and tx.shape[1] == ex.shape[1] == 6, "H517_RF_X_SHAPE")
    require(ty.shape == (tx.shape[0], 6, 9), "H517_RF_Y_SHAPE")
    model = _regressor_h516()
    model.fit(tx.reshape(-1, tx.shape[2]), ty.reshape(-1, 9), sample_weight=row_weights_h516(train_ts))
    train_pred = np.asarray(model.predict(tx.reshape(-1, tx.shape[2])), dtype=np.float64).reshape(ty.shape)
    eval_pred = np.asarray(model.predict(ex.reshape(-1, ex.shape[2])), dtype=np.float64).reshape(ex.shape[0], 6, 9)
    train_loss = clock_equal_profile_mse_h516(train_pred, ty, train_ts)
    return eval_pred, train_loss


def _baseline_prediction(train_y: np.ndarray, train_ts: Sequence[int], eval_shape: tuple[int, ...]) -> np.ndarray:
    mu = clock_equal_target_mean_h516(train_y, train_ts)
    return np.broadcast_to(mu, eval_shape)


def _arm_losses(train: Mapping[str, Any], evald: Mapping[str, Any]) -> dict[str, Any]:
    tx, ty, tts = train["x"], train["y"], train["timestamps_ms"]
    ex, ey, ets = evald["x"], evald["y"], evald["timestamps_ms"]

    pred_a, train_loss_a = fit_predict_rf(tx, ty, tts, ex)

    btx, bex, bmeta = transform_random_arm(tx, ex, tts)
    pred_b, train_loss_b = fit_predict_rf(btx, ty, tts, bex)

    enc = fit_supervised_encoder(tx, ty, tts)
    ctx = transform_supervised_arm(tx, enc)
    cex = transform_supervised_arm(ex, enc)
    pred_c, train_loss_c = fit_predict_rf(ctx, ty, tts, cex)

    base = _baseline_prediction(ty, tts, ey.shape)
    losses = {
        "constant": clock_equal_profile_mse_h516(base, ey, ets),
        "A": clock_equal_profile_mse_h516(pred_a, ey, ets),
        "B": clock_equal_profile_mse_h516(pred_b, ey, ets),
        "C": clock_equal_profile_mse_h516(pred_c, ey, ets),
    }

    controls: dict[int, dict[str, float]] = {}
    control_preds: dict[int, np.ndarray] = {}
    for shift in H516_SHIFTS:
        sx = shift_group_state_h516(tx, tts, train["future_group_ids"], int(shift))
        senc = fit_supervised_encoder(sx, ty, tts)
        sctx = transform_supervised_arm(sx, senc)
        scex = transform_supervised_arm(ex, senc)
        sp, _ = fit_predict_rf(sctx, ty, tts, scex)
        sloss = clock_equal_profile_mse_h516(sp, ey, ets)
        controls[int(shift)] = {"loss": float(sloss), "true_minus_control_gain": float(sloss - losses["C"])}
        control_preds[int(shift)] = sp

    return {
        "fold": int(train["fold"]),
        "train_to_eval_gap_hours": float(train["train_to_eval_gap_hours"]),
        "n_train_groups": int(train["future_group_count"]),
        "n_eval_groups": int(evald["future_group_count"]),
        "n_train_clocks": int(train["unique_decision_clock_count"]),
        "n_eval_clocks": int(evald["unique_decision_clock_count"]),
        "loss": losses,
        "delta": {
            "B_vs_A": float(losses["A"] - losses["B"]),
            "C_vs_A": float(losses["A"] - losses["C"]),
            "C_vs_B": float(losses["B"] - losses["C"]),
            "C_vs_constant": float(losses["constant"] - losses["C"]),
            "C_vs_shift_median": float(statistics.median(v["loss"] for v in controls.values()) - losses["C"]),
        },
        "C_shift_controls": controls,
        "C_shift_wins": int(sum(losses["C"] < v["loss"] for v in controls.values())),
        "train_loss": {"A": train_loss_a, "B": train_loss_b, "C": train_loss_c},
        "C_auxiliary_train_mse": float(enc["auxiliary_train_clock_equal_mse"]),
        "C_singular_values": [float(v) for v in enc["singular_values"][:BOTTLENECK_DIM]],
        "random_projection": bmeta,
        "decomposition": {
            "A": block_profile_decomposition(pred_a, ey, ets),
            "B": block_profile_decomposition(pred_b, ey, ets),
            "C": block_profile_decomposition(pred_c, ey, ets),
        },
        "_predictions": {"A": pred_a, "B": pred_b, "C": pred_c, "constant": base, "C_controls": control_preds},
        "_eval_y": ey,
        "_eval_ts": ets,
    }


def _paired_rule(values: list[float], pooled_delta: float) -> dict[str, Any]:
    require(len(values) == 5, "H517_PAIR_RULE_COUNT")
    return {
        "positive_count": int(sum(v > 0.0 for v in values)),
        "both_late_positive": bool(values[3] > 0.0 and values[4] > 0.0),
        "pooled_delta": float(pooled_delta),
        "pass": bool(sum(v > 0.0 for v in values) >= 4 and values[3] > 0.0 and values[4] > 0.0 and pooled_delta > 0.0),
    }


def _pooled_loss(results: Sequence[Mapping[str, Any]], key: str) -> float:
    preds = np.concatenate([np.asarray(r["_predictions"][key]) for r in results], axis=0)
    ys = np.concatenate([np.asarray(r["_eval_y"]) for r in results], axis=0)
    ts = np.concatenate([np.asarray(r["_eval_ts"], dtype=np.int64) for r in results], axis=0)
    require(len(np.unique(ts)) == sum(len(np.unique(np.asarray(r["_eval_ts"]))) for r in results), "H517_EVAL_CLOCK_OVERLAP")
    return clock_equal_profile_mse_h516(preds, ys, ts)


def summarize_three_arm(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rr = sorted(results, key=lambda x: int(x["fold"]))
    require([int(r["fold"]) for r in rr] == [1, 2, 3, 4, 5], "H517_FOLD_SET")
    pooled = {k: _pooled_loss(rr, k) for k in ("constant", "A", "B", "C")}
    bva = _paired_rule([float(r["delta"]["B_vs_A"]) for r in rr], pooled["A"] - pooled["B"])
    cva = _paired_rule([float(r["delta"]["C_vs_A"]) for r in rr], pooled["A"] - pooled["C"])
    cvb = _paired_rule([float(r["delta"]["C_vs_B"]) for r in rr], pooled["B"] - pooled["C"])
    cvc = _paired_rule([float(r["delta"]["C_vs_constant"]) for r in rr], pooled["constant"] - pooled["C"])

    shift_pooled: dict[int, float] = {}
    for shift in H516_SHIFTS:
        preds = np.concatenate([np.asarray(r["_predictions"]["C_controls"][int(shift)]) for r in rr], axis=0)
        ys = np.concatenate([np.asarray(r["_eval_y"]) for r in rr], axis=0)
        ts = np.concatenate([np.asarray(r["_eval_ts"], dtype=np.int64) for r in rr], axis=0)
        shift_pooled[int(shift)] = clock_equal_profile_mse_h516(preds, ys, ts)
    pooled_shift_median = float(statistics.median(shift_pooled.values()))
    shift_values = [float(r["delta"]["C_vs_shift_median"]) for r in rr]
    shift_rule = _paired_rule(shift_values, pooled_shift_median - pooled["C"])
    shift_wins = int(sum(int(r["C_shift_wins"]) for r in rr))
    shift_rule["wins_of_25"] = shift_wins
    shift_rule["pass"] = bool(shift_rule["pass"] and shift_wins >= 20)

    success = bool(cva["pass"] and cvb["pass"] and cvc["pass"] and shift_rule["pass"])
    return {
        "pooled_loss": pooled,
        "B_vs_A": bva,
        "C_vs_A": cva,
        "C_vs_B": cvb,
        "C_vs_constant": cvc,
        "C_vs_from_scratch_shift": shift_rule,
        "blind_compression_improves_pipeline": bool(bva["pass"]),
        "task_supervised_representation_improves_pipeline": success,
        "interpretation": (
            "THIS_PREDEFINED_LINEAR_SUPERVISED_8D_REPRESENTATION_PLUS_FIXED_RF_IMPROVES_THE_TESTED_PIPELINE"
            if success
            else "THIS_PREDEFINED_LINEAR_SUPERVISED_8D_REPRESENTATION_PLUS_FIXED_RF_DID_NOT_MEET_THE_EXPLORATORY_IMPROVEMENT_GATE"
        ),
    }


def _clock_split_masks(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clocks = np.asarray(sorted(set(int(v) for v in np.asarray(ts).tolist())), dtype=np.int64)
    require(len(clocks) >= 6, "H517_SUPPORT_TOO_FEW_CLOCKS")
    ref_clocks = set(int(v) for v in clocks[::2].tolist())
    ref = np.asarray([int(v) in ref_clocks for v in ts], dtype=bool)
    cal = ~ref
    require(ref.any() and cal.any(), "H517_SUPPORT_EMPTY_SPLIT")
    return ref, cal


def _nn_calibrated_coverage(ref: np.ndarray, cal: np.ndarray, query: np.ndarray) -> dict[str, float]:
    require(len(ref) >= 3 and len(cal) >= 3 and len(query) >= 1, "H517_SUPPORT_POINT_COUNT")
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    zr, zc, zq = (ref - mu) / sd, (cal - mu) / sd, (query - mu) / sd
    nn = NearestNeighbors(n_neighbors=1).fit(zr)
    dc = nn.kneighbors(zc, return_distance=True)[0].reshape(-1)
    dq = nn.kneighbors(zq, return_distance=True)[0].reshape(-1)
    thr = float(np.quantile(dc, 0.95))
    return {
        "threshold_from_clock_isolated_calibration": thr,
        "calibration_coverage": float((dc <= thr).mean()),
        "evaluation_coverage": float((dq <= thr).mean()),
        "median_calibration_distance": float(np.median(dc)),
        "median_evaluation_distance": float(np.median(dq)),
        "distance_ratio": float(np.median(dq) / max(np.median(dc), 1e-300)),
    }


def support_diagnostic(train: Mapping[str, Any], evald: Mapping[str, Any]) -> dict[str, Any]:
    tx = np.asarray(train["x"], dtype=np.float64)
    ex = np.asarray(evald["x"], dtype=np.float64)
    tts = np.asarray(train["timestamps_ms"], dtype=np.int64)
    ref_g, cal_g = _clock_split_masks(tts)

    market_tr = tx[:, 0, :MARKET_DIM]
    market_ev = ex[:, 0, :MARKET_DIM]
    market = _nn_calibrated_coverage(market_tr[ref_g], market_tr[cal_g], market_ev)

    acc_tr = tx[:, :, MARKET_DIM:].reshape(-1, ACCOUNT_DIM)
    acc_ev = ex[:, :, MARKET_DIM:].reshape(-1, ACCOUNT_DIM)
    ref_r = np.repeat(ref_g, SCENARIO_COUNT)
    cal_r = np.repeat(cal_g, SCENARIO_COUNT)
    account = _nn_calibrated_coverage(acc_tr[ref_r], acc_tr[cal_r], acc_ev)

    full_tr = tx.reshape(-1, H516_DIM)
    full_ev = ex.reshape(-1, H516_DIM)
    joint = _nn_calibrated_coverage(full_tr[ref_r], full_tr[cal_r], full_ev)
    return {
        "fold": int(train["fold"]),
        "market96_group_level": market,
        "account6_actual_scenario_rows": account,
        "joint102_actual_scenario_rows": joint,
        "reference_and_calibration_clock_disjoint": True,
        "same_group_cannot_cross_reference_calibration_due_to_clock_bundle_split": True,
        "is_scientific_success_gate": False,
    }


def _synthetic_pair(rng: np.random.Generator, fold: int, W: np.ndarray | None, null: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    def make(n_clocks: int, groups_per_clock: int, start_clock: int) -> dict[str, Any]:
        n = n_clocks * groups_per_clock
        latent = rng.normal(size=(n, BOTTLENECK_DIM))
        market = rng.normal(scale=1.0, size=(n, MARKET_DIM))
        market[:, :BOTTLENECK_DIM] += 2.0 * latent
        market3 = np.repeat(market[:, None, :], SCENARIO_COUNT, axis=1)
        account = rng.normal(size=(n, SCENARIO_COUNT, ACCOUNT_DIM))
        x = np.concatenate([market3, account], axis=2)
        if null:
            y = rng.normal(scale=1.0, size=(n, SCENARIO_COUNT, H516_UTILITY_DIM))
        else:
            sig = latent @ W
            y = np.repeat(sig[:, None, :], SCENARIO_COUNT, axis=1)
            y += 0.25 * account[:, :, :1] * np.linspace(-1.0, 1.0, H516_UTILITY_DIM)[None, None, :]
            y += rng.normal(scale=1.5, size=y.shape)
        y = y - y.mean(axis=2, keepdims=True)
        ts = np.repeat((np.arange(n_clocks, dtype=np.int64) + start_clock) * 3_600_000 + 1_600_000_000_000, groups_per_clock)
        gids = tuple(f"s{fold}_{start_clock}_{i:05d}" for i in range(n))
        return {
            "fold": fold, "future_group_ids": gids, "timestamps_ms": ts,
            "x": np.ascontiguousarray(x), "y": np.ascontiguousarray(y),
            "future_group_count": n, "unique_decision_clock_count": n_clocks,
            "train_to_eval_gap_hours": float(PURGE_HOURS),
        }
    train = make(12, 8, fold * 1000)
    evald = make(6, 8, fold * 1000 + 400)
    return train, evald


def synthetic_pathway_check(seed: int = 20260911) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(BOTTLENECK_DIM, H516_UTILITY_DIM))
    W = W - W.mean(axis=1, keepdims=True)
    pos = [_arm_losses(*_synthetic_pair(rng, f, W, False)) for f in range(1, 6)]
    pos_summary = summarize_three_arm(pos)

    rng0 = np.random.default_rng(seed + 1)
    null = [_arm_losses(*_synthetic_pair(rng0, f, None, True)) for f in range(1, 6)]
    null_summary = summarize_three_arm(null)
    return {
        "fixed_relation_forward_positive": pos_summary,
        "no_binding_null": null_summary,
        "positive_pathway_detected": bool(pos_summary["task_supervised_representation_improves_pipeline"]),
        "null_rejected": bool(not null_summary["task_supervised_representation_improves_pipeline"]),
    }


def strip_internal(result: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(result)
    for k in ("_predictions", "_eval_y", "_eval_ts"):
        out.pop(k, None)
    return out


def known_answer_diagnostic_check(seed: int = 517001) -> dict[str, Any]:
    # Exact block-level profile decomposition.
    ts = np.asarray([1, 1, 2, 2], dtype=np.int64)
    y = np.zeros((4, 6, 9), dtype=np.float64)
    perfect = block_profile_decomposition(y.copy(), y, ts)
    offset = np.asarray([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_level = np.broadcast_to(offset, y.shape).copy()
    pure_level = block_profile_decomposition(p_level, y, ts)
    shape_vec = np.asarray([1.0, -0.7, 0.5, -0.3, 0.2, -0.1, -0.2, -0.1, -0.3])
    shape_vec = shape_vec - shape_vec.mean()
    signs = np.asarray([1.0, -1.0, 1.0, -1.0])[:, None, None]
    p_shape = signs * np.broadcast_to(shape_vec, y.shape)
    pure_shape = block_profile_decomposition(p_shape, y, ts)
    require(perfect["total"] == perfect["level"] == perfect["shape"] == 0.0, "H517_KNOWN_PERFECT")
    require(abs(pure_level["total"] - 2.0 / 9.0) < 1e-12, "H517_KNOWN_LEVEL_TOTAL")
    require(abs(pure_level["level"] - 2.0 / 9.0) < 1e-12 and pure_level["shape"] < 1e-12, "H517_KNOWN_LEVEL_SPLIT")
    require(pure_shape["level"] < 1e-12 and abs(pure_shape["shape"] - pure_shape["total"]) < 1e-12, "H517_KNOWN_SHAPE_SPLIT")

    # Clock-isolated nearest-neighbour calibration on IID vs visibly shifted queries.
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(240, 96))
    cal = rng.normal(size=(240, 96))
    iid = rng.normal(size=(240, 96))
    shifted = rng.normal(size=(240, 96)) + 2.0
    iid_d = _nn_calibrated_coverage(ref, cal, iid)
    shift_d = _nn_calibrated_coverage(ref, cal, shifted)
    require(0.90 <= iid_d["calibration_coverage"] <= 0.97, "H517_KNOWN_CALIBRATION")
    require(iid_d["evaluation_coverage"] > shift_d["evaluation_coverage"], "H517_KNOWN_SUPPORT_SENSITIVITY")
    return {
        "decomposition": {
            "perfect": perfect,
            "pure_profile_level": pure_level,
            "pure_shape": pure_shape,
        },
        "support_tool": {
            "iid": iid_d,
            "shifted": shift_d,
            "iid_coverage_exceeds_shifted": True,
        },
    }
