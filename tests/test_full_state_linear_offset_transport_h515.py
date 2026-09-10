from __future__ import annotations

import numpy as np

from cb16_local_opt.full_state_linear_offset_transport_h515 import (
    H515_ALPHA,
    H515_DIM,
    classify_h515,
    evaluate_forward_pair_h515,
    fit_predict_ridge_h515,
    shift_state_bundles_h515,
)


def test_bundle_shift_preserves_whole_six_scenario_bundle() -> None:
    x = np.arange(40 * 6 * H515_DIM, dtype=np.float64).reshape(40, 6, H515_DIM)
    out = shift_state_bundles_h515(x, 7)
    np.testing.assert_array_equal(out[7], x[0])
    np.testing.assert_array_equal(out[0], x[-7])
    for g in range(40):
        source = (g - 7) % 40
        np.testing.assert_array_equal(out[g, :, 96:102], x[source, :, 96:102])


def test_fixed_ridge_recovers_forward_linear_signal_without_hyperparameter_search() -> None:
    rng = np.random.default_rng(515)
    ga, gb = 80, 70
    xa = rng.normal(size=(ga, 6, H515_DIM))
    xb = rng.normal(size=(gb, 6, H515_DIM))
    xa[:, :, 0] -= 0.8
    xb[:, :, 0] += 0.8
    beta = np.zeros(H515_DIM); beta[0] = 0.12; beta[7] = -0.04
    ya = xa.reshape(-1, H515_DIM) @ beta + 0.01 * rng.normal(size=ga * 6)
    yb = xb.reshape(-1, H515_DIM) @ beta + 0.01 * rng.normal(size=gb * 6)
    fit = fit_predict_ridge_h515(xa, ya.reshape(ga, 6), xb, alpha=H515_ALPHA)
    pred = fit["pred_eval"]
    mse = float(np.mean((pred - yb) ** 2))
    baseline = float(np.mean((float(np.mean(ya)) - yb) ** 2))
    assert mse < baseline
    assert float(np.mean(pred)) > float(np.mean(ya))


def _synthetic_side(fold: int, x: np.ndarray, slope: float = 0.12):
    y = slope * x[:, :, 0]
    nulls = {s: np.zeros_like(y) for s in (1, 7, 13, 23, 31)}
    return {
        "fold": fold,
        "future_group_ids": [f"g{fold}_{i:03d}" for i in range(x.shape[0])],
        "x": x,
        "y_aligned": y,
        "y_null_by_shift": nulls,
        "aligned_mean": float(np.mean(y)),
        "null_mean_by_shift": {s: 0.0 for s in nulls},
        "null_median": 0.0,
    }


def test_forward_pair_can_predict_synthetic_orientation_flip_and_beat_bundle_shifts() -> None:
    rng = np.random.default_rng(1515)
    g = 96
    xa = rng.normal(size=(g, 6, H515_DIM))
    xb = rng.normal(size=(g, 6, H515_DIM))
    xa[:, :, 0] -= 1.0
    xb[:, :, 0] += 1.0
    side_a = _synthetic_side(1, xa)
    side_b = _synthetic_side(2, xb)
    out = evaluate_forward_pair_h515(pair=(1, 2), side_a=side_a, side_b=side_b)
    assert out["actual_side_a_orientation"] == "ANTI_ALIGNMENT"
    assert out["actual_side_b_orientation"] == "POSITIVE_ALIGNMENT"
    assert out["aligned_model"]["predicted_orientation_b"] == "POSITIVE_ALIGNMENT"
    assert out["level_gate"] is True
    assert out["qmse_gate"] is True
    assert out["orientation_gate"] is True
    assert out["native_flip_pair_pass"] is True


def test_classification_requires_all_three_flip_pairs_and_positive_control() -> None:
    def row(pair, passed, pred="POSITIVE_ALIGNMENT", a="ANTI_ALIGNMENT", b="POSITIVE_ALIGNMENT"):
        return {
            "pair": list(pair),
            "native_flip_pair_pass": passed,
            "actual_side_a_orientation": a,
            "actual_side_b_orientation": b,
            "aligned_model": {"predicted_orientation_b": pred},
        }
    supported = [
        row((1,2), True),
        row((2,3), True, pred="ANTI_ALIGNMENT", a="POSITIVE_ALIGNMENT", b="ANTI_ALIGNMENT"),
        row((3,4), True),
        row((4,5), False, pred="POSITIVE_ALIGNMENT", a="POSITIVE_ALIGNMENT", b="POSITIVE_ALIGNMENT"),
    ]
    assert classify_h515(supported, True)["classification"] == "FULL_102D_LINEAR_CURRENT_STATE_OFFSET_TRANSPORT_SUPPORTED"
    none = [
        row((1,2), False),
        row((2,3), False, pred="ANTI_ALIGNMENT", a="POSITIVE_ALIGNMENT", b="ANTI_ALIGNMENT"),
        row((3,4), False),
        row((4,5), False, pred="POSITIVE_ALIGNMENT", a="POSITIVE_ALIGNMENT", b="POSITIVE_ALIGNMENT"),
    ]
    assert classify_h515(none, True)["classification"] == "FULL_102D_LINEAR_CURRENT_STATE_OFFSET_TRANSPORT_NOT_SUPPORTED"
    mixed = list(supported)
    mixed[1] = row((2,3), False, pred="ANTI_ALIGNMENT", a="POSITIVE_ALIGNMENT", b="ANTI_ALIGNMENT")
    assert classify_h515(mixed, True)["classification"] == "FULL_102D_LINEAR_CURRENT_STATE_OFFSET_TRANSPORT_PARTIAL_OR_MIXED"
    assert classify_h515(supported, False)["classification"] == "EXECUTION_BLOCKED__H5_14_H5_13_H5_12R_PARENT_REPRODUCTION_FAILED"
