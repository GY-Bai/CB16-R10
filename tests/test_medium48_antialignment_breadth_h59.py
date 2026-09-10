from __future__ import annotations

import inspect

from cb16_local_opt.medium48_antialignment_breadth_h59 import (
    H59_BASELINE,
    H59_FOLDS,
    H59_SYMBOLS,
    _aggregate_records_h59,
    adjudicate_h59,
    run_fold_h59,
)
from cb16_local_opt.teacher_temporal_transport_audit_h5 import H5_SCENARIOS


def test_h59_frozen_axes_and_failure_folds():
    assert H59_FOLDS == (1, 3)
    assert len(H59_SYMBOLS) == 10
    assert len(set(H59_SYMBOLS)) == 10
    assert len(H5_SCENARIOS) == 6


def test_run_fold_signature_has_no_teacher_student_or_weight_inputs():
    params = set(inspect.signature(run_fold_h59).parameters)
    forbidden = {
        "teacher",
        "teacher_config",
        "kernel",
        "student",
        "organ_weight",
        "fusion_weight",
        "distance_weight",
        "medium_dimensions",
        "regime_selector",
    }
    assert not (params & forbidden)


def _record(gid: str, symbol: str, scenario: str, aligned: float, null: float) -> dict:
    return {
        "future_group_id": gid,
        "symbol": symbol,
        "scenario": scenario,
        "aligned_partial_rho": aligned,
        "null_partial_rho_by_shift": {1: null, 7: null, 13: null, 23: null, 31: null},
    }


def test_aggregate_records_is_future_group_macro_and_scenario_equal_weighted():
    records = []
    for scenario in H5_SCENARIOS:
        records.append(_record("g1", "BTCUSDT", scenario, -0.2, 0.1))
        records.append(_record("g2", "ETHUSDT", scenario, 0.0, 0.1))
    out = _aggregate_records_h59(records)
    assert abs(out["aligned_partial_rho"] + 0.1) <= 1e-15
    assert abs(out["null_median_partial_rho"] - 0.1) <= 1e-15
    assert out["anti_alignment"] is True
    assert out["included_future_groups"] == 2
    assert out["included_scenarios_per_future_group"] == 6


def test_symbol_omission_changes_only_final_target_aggregation():
    records = []
    for scenario in H5_SCENARIOS:
        records.append(_record("g1", "BTCUSDT", scenario, -0.2, 0.1))
        records.append(_record("g2", "ETHUSDT", scenario, -0.4, 0.1))
    out = _aggregate_records_h59(records, omit_symbol="BTCUSDT")
    assert abs(out["aligned_partial_rho"] + 0.4) <= 1e-15
    assert out["included_future_groups"] == 1
    assert out["included_target_rows"] == 6


def test_scenario_omission_keeps_each_future_group_and_five_scenarios():
    records = []
    omitted = H5_SCENARIOS[0]
    for scenario in H5_SCENARIOS:
        records.append(_record("g1", "BTCUSDT", scenario, -0.2, 0.1))
        records.append(_record("g2", "ETHUSDT", scenario, -0.4, 0.1))
    out = _aggregate_records_h59(records, omit_scenario=omitted)
    assert out["included_future_groups"] == 2
    assert out["included_target_rows"] == 10
    assert out["included_scenarios_per_future_group"] == 5


def _fold_result(fold: int, *, symbol_robust: bool, scenario_robust: bool) -> dict:
    base = H59_BASELINE[fold]
    return {
        "fold": fold,
        "baseline": {
            "aligned_partial_rho": base["aligned_partial_rho"],
            "null_median_partial_rho": base["null_median"],
            "anti_alignment": True,
        },
        "symbol_loo_robust": symbol_robust,
        "scenario_loo_robust": scenario_robust,
        "symbol_sensitive_omissions": [] if symbol_robust else ["BTCUSDT"],
        "scenario_sensitive_omissions": [] if scenario_robust else [H5_SCENARIOS[0]],
        "symbol_loo_min_null_minus_aligned_margin": 0.01 if symbol_robust else -0.01,
        "scenario_loo_min_null_minus_aligned_margin": 0.01 if scenario_robust else -0.01,
        "symbol_loo_max_aligned_partial_rho": -0.001 if symbol_robust else 0.001,
        "scenario_loo_max_aligned_partial_rho": -0.001 if scenario_robust else 0.001,
    }


def test_adjudication_can_classify_broad_antialignment():
    out = adjudicate_h59([
        _fold_result(1, symbol_robust=True, scenario_robust=True),
        _fold_result(3, symbol_robust=True, scenario_robust=True),
    ])
    assert out["baseline_reproduction_passed"] is True
    assert out["classification"] == "MEDIUM48_CONDITIONAL_ANTI_ALIGNMENT_BROAD_ACROSS_SYMBOLS_AND_SCENARIOS"


def test_adjudication_can_classify_symbol_sensitive_only():
    out = adjudicate_h59([
        _fold_result(1, symbol_robust=False, scenario_robust=True),
        _fold_result(3, symbol_robust=True, scenario_robust=True),
    ])
    assert out["classification"] == "MEDIUM48_CONDITIONAL_ANTI_ALIGNMENT_SYMBOL_SENSITIVE__SCENARIO_ROBUST"


def test_adjudication_can_classify_scenario_sensitive_only():
    out = adjudicate_h59([
        _fold_result(1, symbol_robust=True, scenario_robust=False),
        _fold_result(3, symbol_robust=True, scenario_robust=True),
    ])
    assert out["classification"] == "MEDIUM48_CONDITIONAL_ANTI_ALIGNMENT_SYMBOL_ROBUST__SCENARIO_SENSITIVE"


def test_adjudication_can_classify_both_sensitive():
    out = adjudicate_h59([
        _fold_result(1, symbol_robust=False, scenario_robust=True),
        _fold_result(3, symbol_robust=True, scenario_robust=False),
    ])
    assert out["classification"] == "MEDIUM48_CONDITIONAL_ANTI_ALIGNMENT_SYMBOL_AND_SCENARIO_SENSITIVE_OR_MIXED"


def test_adjudication_fails_closed_on_baseline_drift():
    a = _fold_result(1, symbol_robust=True, scenario_robust=True)
    b = _fold_result(3, symbol_robust=True, scenario_robust=True)
    a["baseline"] = dict(a["baseline"])
    a["baseline"]["aligned_partial_rho"] += 1e-6
    out = adjudicate_h59([a, b])
    assert out["baseline_reproduction_passed"] is False
    assert out["classification"] == "H5_9_BASELINE_REPRODUCTION_FAILED"
