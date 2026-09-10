from __future__ import annotations

import inspect

import numpy as np

import cb16_local_opt.state_utility_geometry_transport_audit_h55 as h55
from cb16_local_opt.state_utility_geometry_transport_audit_h55 import (
    H55_METRICS,
    _rankdata_average_h55,
    active_dimensions_h55,
    adjudicate_h55,
)


def test_metric_dimensions_are_frozen_and_disjoint_as_preregistered():
    assert H55_METRICS == ("FULL102", "MARKET96", "OPERATOR48", "MEDIUM48", "ACCOUNT6")
    assert active_dimensions_h55("FULL102", 102).tolist() == list(range(102))
    assert active_dimensions_h55("MARKET96", 102).tolist() == list(range(96))
    assert active_dimensions_h55("OPERATOR48", 102).tolist() == list(range(48))
    assert active_dimensions_h55("MEDIUM48", 102).tolist() == list(range(48, 96))
    assert active_dimensions_h55("ACCOUNT6", 102).tolist() == list(range(96, 102))


def test_rankdata_uses_average_ranks_for_ties():
    x = np.asarray([3.0, 1.0, 1.0, 4.0], dtype=np.float64)
    r = _rankdata_average_h55(x)
    assert np.allclose(r, np.asarray([3.0, 1.5, 1.5, 4.0]))


def test_scientific_helper_contains_no_teacher_compiler_or_student_optimizer_calls():
    src = inspect.getsource(h55)
    forbidden = (
        "_compile_block_r11(",
        "compile_validation_targets_only_r5(",
        "train_student",
        "AdamW(",
        "optimizer.step(",
    )
    for token in forbidden:
        assert token not in src


def _metric(aligned: float, shuffle: float, *, pair_count: int = 5):
    return {
        "aligned_centered_profile_spearman_rho": aligned,
        "aligned_raw_profile_spearman_rho": aligned,
        "aligned_nearest_decile_centered_mse_ratio": 0.8,
        "shuffle_centered_profile_spearman_rho": {s: shuffle for s in (1, 7, 13, 23, 31)},
        "shuffle_raw_profile_spearman_rho": {s: shuffle for s in (1, 7, 13, 23, 31)},
        "shuffle_nearest_decile_centered_mse_ratio": {s: 1.0 for s in (1, 7, 13, 23, 31)},
        "median_shuffle_centered_profile_spearman_rho": shuffle,
        "aligned_minus_median_shuffle_rho": aligned - shuffle,
        "aligned_positive": aligned > 0.0,
        "aligned_gt_shuffle_median": aligned > shuffle,
        "aligned_gt_each_shuffle_count": pair_count,
    }


def _receipt(ok: bool = True):
    return {
        "target_feature_multiset_preserved": ok,
        "train_features_byte_identical": ok,
        "scenario_identity_preserved": ok,
    }


def _fold(fold: int, metric_values: dict[str, tuple[float, float]]):
    return {
        "fold": fold,
        "state_metrics": {m: _metric(*metric_values[m]) for m in H55_METRICS},
        "rotation_receipts": {s: _receipt(True) for s in (1, 7, 13, 23, 31)},
    }


def _all_values(default=(0.2, 0.0)):
    return {m: default for m in H55_METRICS}


def test_classification_full_state_supported_when_full102_passes_gate():
    rows = [_fold(i, _all_values()) for i in range(1, 6)]
    out = adjudicate_h55(rows)
    assert out["metric_gates"]["FULL102"]["metric_transport_supported"] is True
    assert out["classification"] == "FULL_STATE_GEOMETRY_TRANSPORT_SUPPORTED"


def test_classification_operator_survives_when_full_and_medium_fail():
    rows = []
    for i in range(1, 6):
        v = _all_values(default=(-0.1, 0.0))
        v["OPERATOR48"] = (0.2, 0.0)
        rows.append(_fold(i, v))
    out = adjudicate_h55(rows)
    assert out["metric_gates"]["FULL102"]["metric_transport_supported"] is False
    assert out["metric_gates"]["OPERATOR48"]["metric_transport_supported"] is True
    assert out["metric_gates"]["MEDIUM48"]["metric_transport_supported"] is False
    assert out["classification"] == "OPERATOR48_TRANSPORT_SURVIVES_MEDIUM48_SUSPECT"


def test_classification_market_nontransport_when_all_market_metrics_fail():
    rows = [_fold(i, _all_values(default=(-0.1, 0.0))) for i in range(1, 6)]
    out = adjudicate_h55(rows)
    assert out["classification"] == "MARKET_STATE_UTILITY_GEOMETRY_TEMPORAL_NONTRANSPORT"


def test_metric_gate_requires_late_fold_replication_not_just_four_early_wins():
    rows = []
    for i in range(1, 6):
        v = _all_values()
        if i == 5:
            v["FULL102"] = (-0.1, 0.0)
        rows.append(_fold(i, v))
    out = adjudicate_h55(rows)
    gate = out["metric_gates"]["FULL102"]
    assert gate["positive_fold_count"] == 4
    assert gate["positive_both_late_folds"] is False
    assert gate["metric_transport_supported"] is False


def test_rotation_identity_fail_closed_is_reported():
    rows = [_fold(i, _all_values()) for i in range(1, 6)]
    rows[0]["rotation_receipts"][1] = _receipt(False)
    out = adjudicate_h55(rows)
    assert out["all_rotation_identity_guards_pass"] is False
