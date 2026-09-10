from __future__ import annotations

import numpy as np

from cb16_local_opt.target_om_concordance_occupancy_h514 import (
    H514_TOL,
    _unweighted_corr_h514,
    build_target_pair_design_h514,
    calibrate_candidate_margin_h514,
    classify_h514,
    macro_row_masses_h514,
    target_kappa_payload_h514,
    weighted_effective_cell_mass_h514,
)


def _scenario(o, m, cells=None):
    if cells is None:
        cells = np.arange(len(o), dtype=np.int32) % 27
    return {
        "operator_rank": np.asarray(o, dtype=np.float64),
        "medium_rank": np.asarray(m, dtype=np.float64),
        "cell_ids": np.asarray(cells, dtype=np.int32),
    }


def _payload(fold: int, kappas: list[float]) -> dict:
    groups = []
    n = 40
    o = np.arange(1, n + 1, dtype=np.float64)
    for i, k in enumerate(kappas):
        # Construct deterministic rank-like continuous vector with varying concordance.
        base = np.linspace(-1.0, 1.0, n)
        orth = np.sin(np.linspace(0.0, 4.0 * np.pi, n) + i * 0.07)
        m = k * base + np.sqrt(max(0.0, 1.0 - min(k * k, 0.999999))) * orth
        scenarios = [_scenario(o, m) for _ in range(6)]
        groups.append({"future_group_id": f"F{fold}_G{i}", "scenarios": scenarios})
    return {"fold": fold, "groups": groups, "eval_dependence_groups": len(groups), "macro_row_count": len(groups) * 6}


def test_unweighted_corr_identity_and_reverse() -> None:
    x = np.arange(20, dtype=np.float64)
    assert abs(_unweighted_corr_h514(x, x) - 1.0) <= 1e-12
    assert abs(_unweighted_corr_h514(x, x[::-1]) + 1.0) <= 1e-12


def test_target_kappa_is_scenario_consistent_for_market_only_rows() -> None:
    p = _payload(1, [-0.8, -0.2, 0.3, 0.8] * 8)
    out = target_kappa_payload_h514(p)
    assert out["scenario_consistency_passed"] is True
    assert out["max_scenario_kappa_spread"] <= H514_TOL
    assert out["utility_used"] is False


def test_target_pair_design_matches_same_three_bin_measure() -> None:
    # Use explicit kappa payloads so support is safely >10 per bin.
    a = {"fold": 1, "kappa": np.linspace(-0.95, 0.85, 90).tolist(), "scenario_consistency_passed": True}
    b = {"fold": 2, "kappa": np.linspace(-0.75, 0.99, 90).tolist(), "scenario_consistency_passed": True}
    out = build_target_pair_design_h514(a, b)
    assert out["common_support_valid"] is True
    np.testing.assert_allclose(out["effective_target_bin_mass_a"], out["qk"], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(out["effective_target_bin_mass_b"], out["qk"], atol=1e-12, rtol=0.0)
    assert out["utility_used_to_construct_target_measure"] is False


def test_macro_row_masses_preserve_group_authority_and_scenario_equal_weight() -> None:
    p = {"groups": [{"scenarios": [object()] * 6}, {"scenarios": [object()] * 6}]}
    r = macro_row_masses_h514(p, [0.25, 0.75])
    assert abs(float(r.sum()) - 1.0) <= 1e-12
    np.testing.assert_allclose(r[:6], np.full(6, 0.25 / 6.0), atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(r[6:], np.full(6, 0.75 / 6.0), atol=1e-15, rtol=0.0)


def test_generalized_ipf_closes_q27_under_nonuniform_row_masses() -> None:
    # Every row has all cells but with different counts; target row masses are nonuniform.
    rows = []
    for i in range(12):
        counts = 1 + ((np.arange(27) + i) % 4)
        rows.append(np.repeat(np.arange(27, dtype=np.int32), counts))
    rm = np.arange(1, 13, dtype=np.float64); rm /= rm.sum()
    q = np.arange(1, 28, dtype=np.float64); q /= q.sum()
    init = np.ones(27, dtype=np.float64)
    cal = calibrate_candidate_margin_h514(cell_rows=rows, row_masses=rm, q27=q, initial_multiplier=init)
    assert cal["converged"] is True
    assert cal["final_max_abs_error_vs_q27"] <= 1e-12
    eff = weighted_effective_cell_mass_h514(rows, rm, cal["multiplier"])
    np.testing.assert_allclose(eff["effective_cell_mass"], q, atol=1e-12, rtol=0.0)


def test_h514_classification_persist_explain_and_mixed() -> None:
    native = {1: "ANTI_ALIGNMENT", 2: "POSITIVE_ALIGNMENT", 3: "ANTI_ALIGNMENT", 4: "POSITIVE_ALIGNMENT", 5: "POSITIVE_ALIGNMENT"}
    def row(pair, oa, ob):
        return {"pair": list(pair), "blocked": None, "side_a": {"orientation": oa}, "side_b": {"orientation": ob}}
    persist = [row((1,2),native[1],native[2]),row((2,3),native[2],native[3]),row((3,4),native[3],native[4]),row((4,5),native[4],native[5])]
    assert classify_h514(persist)["classification"] == "TARGET_OM_CONCORDANCE_OCCUPANCY_INSUFFICIENT__GLOBAL_OFFSET_TRANSITIONS_PERSIST"
    explain = [row((1,2),"POSITIVE_ALIGNMENT","POSITIVE_ALIGNMENT"),row((2,3),"POSITIVE_ALIGNMENT","POSITIVE_ALIGNMENT"),row((3,4),"POSITIVE_ALIGNMENT","POSITIVE_ALIGNMENT"),row((4,5),"POSITIVE_ALIGNMENT","POSITIVE_ALIGNMENT")]
    assert classify_h514(explain)["classification"] == "TARGET_OM_CONCORDANCE_OCCUPANCY_EXPLAINS_NATIVE_TRANSITIONS"
    mixed = [row((1,2),native[1],native[2]),row((2,3),"POSITIVE_ALIGNMENT","POSITIVE_ALIGNMENT"),row((3,4),native[3],native[4]),row((4,5),native[4],native[5])]
    assert classify_h514(mixed)["classification"] == "TARGET_OM_CONCORDANCE_OCCUPANCY_PARTIAL_OR_MIXED"
