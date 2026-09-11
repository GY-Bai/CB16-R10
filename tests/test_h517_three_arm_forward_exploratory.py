from __future__ import annotations

import numpy as np

from cb16_local_opt.h517_three_arm_forward_exploratory import (
    BOTTLENECK_DIM,
    MARKET_DIM,
    _nn_calibrated_coverage,
    block_profile_decomposition,
    fit_supervised_encoder,
    transform_supervised_arm,
)


def _ts(n_groups: int) -> np.ndarray:
    return np.repeat(np.arange(n_groups // 2, dtype=np.int64), 2)


def test_block_profile_decomposition_known_answers() -> None:
    ts = _ts(4)
    y = np.zeros((4, 6, 9), dtype=np.float64)

    z = block_profile_decomposition(y.copy(), y, ts)
    assert z["total"] == z["level"] == z["shape"] == 0.0

    offset = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p = np.broadcast_to(offset, y.shape).copy()
    z = block_profile_decomposition(p, y, ts)
    assert abs(z["total"] - 2.0 / 9.0) < 1e-12
    assert abs(z["level"] - 2.0 / 9.0) < 1e-12
    assert z["shape"] < 1e-12

    v = np.array([1.0, -0.7, 0.5, -0.3, 0.2, -0.1, -0.2, -0.1, -0.3])
    v -= v.mean()
    p = np.array([1.0, -1.0, 1.0, -1.0])[:, None, None] * np.broadcast_to(v, y.shape)
    z = block_profile_decomposition(p, y, ts)
    assert z["level"] < 1e-12
    assert abs(z["shape"] - z["total"]) < 1e-12


def test_support_calibration_separates_iid_from_shift() -> None:
    rng = np.random.default_rng(517001)
    ref = rng.normal(size=(240, 96))
    cal = rng.normal(size=(240, 96))
    iid = rng.normal(size=(240, 96))
    shifted = rng.normal(size=(240, 96)) + 2.0
    a = _nn_calibrated_coverage(ref, cal, iid)
    b = _nn_calibrated_coverage(ref, cal, shifted)
    assert 0.90 <= a["calibration_coverage"] <= 0.97
    assert a["evaluation_coverage"] > b["evaluation_coverage"]


def test_supervised_encoder_is_train_only_and_eight_dimensional() -> None:
    rng = np.random.default_rng(51708)
    n = 12
    ts = np.repeat(np.arange(6, dtype=np.int64), 2)
    market = rng.normal(size=(n, MARKET_DIM))
    x = np.concatenate(
        [
            np.repeat(market[:, None, :], 6, axis=1),
            rng.normal(size=(n, 6, 6)),
        ],
        axis=2,
    )
    y = rng.normal(size=(n, 6, 9))
    y -= y.mean(axis=2, keepdims=True)
    enc = fit_supervised_encoder(x, y, ts)
    assert enc["encoder"].shape == (MARKET_DIM, BOTTLENECK_DIM)
    out = transform_supervised_arm(x, enc)
    assert out.shape == (n, 6, BOTTLENECK_DIM + 6)
    assert np.isfinite(out).all()
