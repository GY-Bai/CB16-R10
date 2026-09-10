from __future__ import annotations

import inspect

import numpy as np

from cb16_local_opt.time_local_vs_forward_geometry_contrast_h56 import (
    H56_METRICS,
    _rho_and_ratio_h56,
    adjudicate_h56,
    run_fold_h56,
)


def test_h56_metrics_match_frozen_h55_surface():
    assert H56_METRICS == ("FULL102", "MARKET96", "OPERATOR48", "MEDIUM48", "ACCOUNT6")


def test_run_fold_signature_does_not_accept_teacher_or_future_selection_inputs():
    params = set(inspect.signature(run_fold_h56).parameters)
    forbidden = {"teacher", "teacher_config", "kernel", "future_utility", "utility_selector"}
    assert not (params & forbidden)


def test_rho_zero_information_convention_matches_h55_clarification():
    rho, ratio, state_deg, util_deg = _rho_and_ratio_h56(
        np.asarray([1.0, 1.0, 1.0]), np.asarray([0.1, 0.2, 0.3])
    )
    assert rho == 0.0
    assert ratio == 1.0
    assert state_deg is True
    assert util_deg is False


def _metric(local_rho: float, shuf: tuple[float, float, float, float, float]):
    med = sorted(shuf)[2]
    return {
        "aligned_centered_profile_spearman_rho": local_rho,
        "aligned_nearest_decile_centered_mse_ratio": 0.9,
        "median_shuffle_centered_profile_spearman_rho": med,
        "aligned_minus_median_shuffle_rho": local_rho-med,
        "aligned_positive": local_rho > 0,
        "aligned_gt_shuffle_median": local_rho > med,
        "aligned_gt_each_shuffle_count": sum(local_rho > x for x in shuf),
    }


def _forward_metric(rho: float):
    return {
        "aligned_centered_profile_spearman_rho": rho,
        "aligned_raw_profile_spearman_rho": rho,
        "aligned_nearest_decile_centered_mse_ratio": 1.0,
        "median_shuffle_centered_profile_spearman_rho": rho + 0.01,
        "aligned_minus_median_shuffle_rho": -0.01,
        "aligned_positive": rho > 0,
        "aligned_gt_shuffle_median": False,
        "aligned_gt_each_shuffle_count": 2,
    }


def _row(fold: int, market_local: float, operator_local: float = 0.03):
    local = {}
    forward = {}
    for m in H56_METRICS:
        lr = market_local if m in {"FULL102", "MARKET96"} else operator_local if m == "OPERATOR48" else -0.01
        local[m] = _metric(lr, (-0.02, -0.01, 0.0, 0.005, 0.01))
        fr = -0.02 if m in {"FULL102", "MARKET96", "MEDIUM48"} else 0.005 if m == "OPERATOR48" else 0.0
        forward[m] = _forward_metric(fr)
    return {
        "fold": fold,
        "local_state_metrics": local,
        "forward": {"fold":fold,"state_metrics":forward,"rotation_receipts":{}},
        "rotation_receipts": {s:{"target_feature_multiset_preserved":True,"train_features_byte_identical":True,"scenario_identity_preserved":True} for s in (1,7,13,23,31)},
    }


def test_adjudication_can_identify_local_market_geometry_with_forward_failure(monkeypatch):
    # Freeze the forward H5.5 summary shape; this test isolates H5.6 classification logic.
    import cb16_local_opt.time_local_vs_forward_geometry_contrast_h56 as h56
    monkeypatch.setattr(h56.h55, "adjudicate_h55", lambda rows: {
        "classification":"MARKET_STATE_UTILITY_GEOMETRY_TEMPORAL_NONTRANSPORT",
        "metric_gates":{m:{"metric_transport_supported":False} for m in H56_METRICS},
    })
    rows = [_row(i, 0.04) for i in range(1,6)]
    out = adjudicate_h56(rows)
    assert out["classification"] == "TIME_LOCAL_MARKET_GEOMETRY_EXISTS__CROSS_TIME_NONTRANSFER_SUPPORTED"
    assert out["local_metric_gates"]["MARKET96"]["local_geometry_supported"] is True
    assert out["local_vs_forward_contrast"]["MARKET96"]["contrast_supported"] is True
    assert out["all_rotation_identity_guards_pass"] is True


def test_adjudication_fails_to_local_weak_when_all_market_metrics_fail(monkeypatch):
    import cb16_local_opt.time_local_vs_forward_geometry_contrast_h56 as h56
    monkeypatch.setattr(h56.h55, "adjudicate_h55", lambda rows: {
        "classification":"MARKET_STATE_UTILITY_GEOMETRY_TEMPORAL_NONTRANSPORT",
        "metric_gates":{m:{"metric_transport_supported":False} for m in H56_METRICS},
    })
    rows = [_row(i, -0.02, operator_local=-0.02) for i in range(1,6)]
    out = adjudicate_h56(rows)
    assert out["classification"] == "STATE_UTILITY_GEOMETRY_WEAK_EVEN_TIME_LOCAL"
