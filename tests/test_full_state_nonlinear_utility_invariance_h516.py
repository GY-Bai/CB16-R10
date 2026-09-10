from __future__ import annotations

import inspect

import numpy as np
import pytest

import cb16_local_opt.full_state_nonlinear_utility_invariance_h516 as h516
from cb16_local_opt.full_state_nonlinear_utility_invariance_h516 import (
    H516_DIM,
    H516_SEED,
    H516_SHIFTS,
    H516_UTILITY_DIM,
    _classifier_h516,
    _pooled_pair_h516,
    _regressor_h516,
    _shift_clock_environment_h516,
    classify_h516,
    clock_blocks_h516,
    clock_equal_profile_mse_h516,
    clock_equal_target_mean_h516,
    row_weights_h516,
    shift_group_state_h516,
)


def test_fixed_model_contract_matches_preregistration_and_clarification():
    assert H516_DIM == 102
    assert H516_UTILITY_DIM == 9
    assert H516_SHIFTS == (1, 7, 13, 23, 31)
    assert H516_SEED == 51616
    r = _regressor_h516().get_params()
    assert r["n_estimators"] == 100
    assert r["criterion"] == "squared_error"
    assert r["max_features"] == 10
    assert r["max_depth"] is None
    assert r["min_samples_split"] == 2
    assert r["min_samples_leaf"] == 5
    assert r["bootstrap"] is True
    assert r["max_samples"] is None
    assert r["ccp_alpha"] == 0.0
    assert r["random_state"] == 51616
    assert r["n_jobs"] == 1
    c = _classifier_h516().get_params()
    assert c["n_estimators"] == 100
    assert c["criterion"] == "gini"
    assert c["max_features"] == 10
    assert c["min_samples_leaf"] == 5
    assert c["class_weight"] is None
    assert c["random_state"] == 51616
    assert c["n_jobs"] == 1


def test_clock_crossfit_uses_contiguous_quotient_remainder_blocks_and_keeps_ties():
    ts = [10, 10, 20, 30, 40, 50, 60, 70]
    blocks = clock_blocks_h516(ts)
    assert [b.tolist() for b in blocks] == [[10, 20], [30, 40], [50], [60], [70]]
    assert sum(10 in set(b.tolist()) for b in blocks) == 1


def test_row_weights_equalize_clock_then_groups_then_scenarios():
    ts = [10, 10, 20]
    w = row_weights_h516(ts).reshape(3, 6)
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w[0].sum(), 0.25)
    assert np.isclose(w[1].sum(), 0.25)
    assert np.isclose(w[2].sum(), 0.50)
    assert np.allclose(w[0], np.full(6, 1.0 / 24.0))
    assert np.allclose(w[2], np.full(6, 1.0 / 12.0))


def test_target_mean_and_loss_are_clock_equal_not_row_equal():
    ts = np.asarray([10, 10, 20], dtype=np.int64)
    y = np.zeros((3, 6, 9), dtype=np.float64)
    y[2, :, :] = 2.0
    mean = clock_equal_target_mean_h516(y, ts)
    assert np.allclose(mean, np.ones(9))
    pred = np.zeros_like(y)
    # Clock 10 loss is zero; clock 20 loss is four => equal-clock mean two.
    assert np.isclose(clock_equal_profile_mse_h516(pred, y, ts), 2.0)


def test_group_state_shift_identity_is_defined_by_permutation_not_feature_values():
    # Even byte-identical state bundles are legal so long as the group permutation is nonidentity.
    x = np.zeros((6, 6, 102), dtype=np.float64)
    ts = np.arange(6, dtype=np.int64)
    gids = [f"g{i}" for i in range(6)]
    shifted = shift_group_state_h516(x, ts, gids, 1)
    assert np.array_equal(shifted, x)
    with pytest.raises(RuntimeError, match="H516_NEGATIVE_CONTROL_IDENTITY"):
        shift_group_state_h516(x, ts, gids, 6)


def test_group_state_shift_respects_timestamp_then_gid_order():
    x = np.zeros((4, 6, 102), dtype=np.float64)
    for i in range(4):
        x[i, :, 0] = float(i)
    ts = np.asarray([20, 10, 10, 30], dtype=np.int64)
    gids = ["z", "b", "a", "x"]
    shifted = shift_group_state_h516(x, ts, gids, 1)
    order = [2, 1, 0, 3]  # (10,a),(10,b),(20,z),(30,x)
    expected_source = [1, 0, 3, 2]  # destination row -> source row for +1 source-position shift
    assert shifted[:, 0, 0].tolist() == [float(i) for i in expected_source]
    assert order == sorted(range(4), key=lambda i: (int(ts[i]), gids[i]))


def test_fake_environment_shift_preserves_clock_labels_marginal_and_rejects_identity():
    ts = np.asarray([10, 10, 20, 30, 40, 50, 60], dtype=np.int64)
    env = np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    shifted = _shift_clock_environment_h516(ts, env, 1)
    assert shifted.shape == env.shape
    assert int(shifted.sum()) == int(env.sum())
    assert not np.array_equal(shifted, env)
    assert shifted[0] == shifted[1]
    with pytest.raises(RuntimeError, match="H516_FAKE_ENV_SHIFT_IDENTITY_INDEX"):
        _shift_clock_environment_h516(ts, env, 6)


def _dataset(fold: int, clocks: list[int]) -> dict:
    n = len(clocks)
    return {
        "fold": fold,
        "future_group_ids": tuple(f"f{fold}_g{i}" for i in range(n)),
        "timestamps_ms": np.asarray(clocks, dtype=np.int64),
        "x": np.zeros((n, 6, 102), dtype=np.float64),
        "y": np.zeros((n, 6, 9), dtype=np.float64),
    }


def test_pooled_pair_requires_one_environment_per_decision_clock():
    a = _dataset(1, [10, 20, 30, 40, 50])
    b = _dataset(2, [50, 60, 70, 80, 90])
    with pytest.raises(RuntimeError, match="H516_POOLED_CLOCK_MULTI_ENV"):
        _pooled_pair_h516(a, b)


def _local(fold: int, *, positive: bool = True, median: bool = True, wins: int = 5):
    return {
        "fold": fold,
        "positive_gain": positive,
        "beats_shift_median": median,
        "beats_each_shift_count": wins,
    }


def _pairs(pass_native: tuple[bool, bool, bool], stability: bool = False):
    flags = list(pass_native) + [stability]
    return [{"pair": list(pair), "pair_pass": flag} for pair, flag in zip(((1, 2), (2, 3), (3, 4), (4, 5)), flags)]


def test_classification_stable_nonlinear_strong_class():
    local = [_local(i) for i in range(1, 6)]
    out = classify_h516(local, _pairs((True, True, True)), _pairs((False, False, False)))
    assert out["local_map_supported"] is True
    assert out["classification"] == "FULL102_NONLINEAR_LOCAL_MAP_AND_NATIVE_FORWARD_TRANSPORT_SUPPORTED__NO_ENVIRONMENT_INCREMENT"


def test_classification_nontransport_environment_increment_strong_class():
    local = [_local(i) for i in range(1, 6)]
    out = classify_h516(local, _pairs((False, False, False)), _pairs((True, True, True)))
    assert out["classification"] == "FULL102_NONLINEAR_LOCAL_MAP_EXISTS__FORWARD_NONTRANSPORT__ENVIRONMENT_INDEX_INCREMENT_WITHIN_FIXED_RF"


def test_classification_weak_even_local_and_mixed_are_distinct():
    weak = [_local(i) for i in range(1, 6)]
    weak[0] = _local(1, positive=False, median=False, wins=0)
    weak[1] = _local(2, positive=False, median=False, wins=0)
    out = classify_h516(weak, _pairs((True, True, True)), _pairs((False, False, False)))
    assert out["classification"] == "FULL102_NONLINEAR_UTILITY_MAP_WEAK_EVEN_LOCAL"

    strong_local = [_local(i) for i in range(1, 6)]
    mixed = classify_h516(strong_local, _pairs((True, False, False)), _pairs((True, False, False)))
    assert mixed["classification"] == "FULL102_NONLINEAR_LOCAL_FORWARD_OR_ENVIRONMENT_INCREMENT_MIXED"


def test_local_gate_requires_both_late_folds_and_20_of_25_shift_wins():
    local = [_local(i) for i in range(1, 6)]
    local[4] = _local(5, positive=False, median=False, wins=5)
    out = classify_h516(local, _pairs((False, False, False)), _pairs((True, True, True)))
    assert out["local_map_supported"] is False
    assert out["classification"] == "FULL102_NONLINEAR_UTILITY_MAP_WEAK_EVEN_LOCAL"

    local = [_local(i, wins=4) for i in range(1, 6)]
    out = classify_h516(local, _pairs((False, False, False)), _pairs((True, True, True)))
    assert out["local_pairwise_shift_wins_of_25"] == 20
    assert out["local_map_supported"] is True


def test_helper_has_no_h512r_ipf_teacher_student_or_runtime_gate_path():
    src = inspect.getsource(h516)
    forbidden = (
        "account6_ipf_effective_common_measure_h512r",
        "candidate_multiplier",
        "weighted_partial_rank",
        "compile_validation_targets_only_r5(",
        "_compile_block_r11(",
        "train_student",
        "optimizer.step(",
        "runtime_gate",
    )
    for token in forbidden:
        assert token not in src
    assert "index.utilities" in src
    assert "u - float(np.mean(u))" in src
