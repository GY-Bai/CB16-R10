from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from cb16_local_opt.medium48_support_selection_causal_audit_h52 import (
    active_dimensions_h52,
    adjudicate_h52,
    score_selected_support_h52,
    selection_signature_guard_h52,
)


@dataclass(frozen=True)
class Parent:
    dependence_group_id: str
    decision_time_ms: int
    symbol: str
    scenario: str


@dataclass(frozen=True)
class Index:
    parent_ids: tuple[str, ...]
    parent_row_by_id: dict[str, int]
    utilities: np.ndarray
    action_count: int = 9


def test_active_dimensions_are_exact_and_disjoint_as_intended():
    full = active_dimensions_h52("FULL102", 102)
    dm = active_dimensions_h52("DROP_MEDIUM48", 102)
    do = active_dimensions_h52("DROP_OPERATOR48", 102)
    assert full.tolist() == list(range(102))
    assert dm.tolist() == [*range(48), *range(96, 102)]
    assert do.tolist() == list(range(48, 102))
    assert not any(48 <= x < 96 for x in dm)
    assert not any(0 <= x < 48 for x in do)


def test_selection_signature_has_no_future_utility_input():
    assert selection_signature_guard_h52() is True


def test_post_selection_profile_metrics_are_group_equal():
    parent_ids = ("E0", "E1", "S0", "S1")
    index = Index(
        parent_ids=parent_ids,
        parent_row_by_id={p: i for i, p in enumerate(parent_ids)},
        utilities=np.asarray([
            np.arange(9, dtype=float),
            np.arange(9, dtype=float) + 1.0,
            np.arange(9, dtype=float) + 0.5,
            np.arange(9, dtype=float) + 2.0,
        ]),
    )
    parents = {
        "E0": Parent("G0", 1000, "BTCUSDT", "A"),
        "E1": Parent("G1", 2000, "ETHUSDT", "B"),
        "S0": Parent("T0", 0, "BTCUSDT", "A"),
        "S1": Parent("T1", 0, "ETHUSDT", "B"),
    }
    selection = {
        "E0": {"selected_parent_rows": (2,), "selected_dep_ids": ("T0",), "nearest_distance": 1.0},
        "E1": {"selected_parent_rows": (3,), "selected_dep_ids": ("T1",), "nearest_distance": 1.0},
    }
    out = score_selected_support_h52(
        index=index,
        eval_parents={"E0": parents["E0"], "E1": parents["E1"]},
        parents=parents,
        selection=selection,
        full_selection=selection,
    )
    assert abs(out["profile_mse"] - 0.625) < 1e-12
    assert abs(out["profile_mae"] - 0.75) < 1e-12
    assert out["selected_support_jaccard_vs_full102_median_group_equal_mean"] == 1.0
    assert out["same_symbol_fraction"] == 1.0
    assert out["same_account_scenario_fraction"] == 1.0


def _fold(fold: int, full: float, medium: float, operator: float, jaccard: float = 0.5):
    return {
        "fold": fold,
        "full102_exact_support_identity_guard": "PASS",
        "metrics": {
            "FULL102": {"profile_mse": full, "selected_support_jaccard_vs_full102_median_group_equal_mean": 1.0},
            "DROP_MEDIUM48": {"profile_mse": medium, "selected_support_jaccard_vs_full102_median_group_equal_mean": jaccard},
            "DROP_OPERATOR48": {"profile_mse": operator, "selected_support_jaccard_vs_full102_median_group_equal_mean": 0.6},
        },
    }


def test_adjudication_requires_late_medium_specificity_and_nontrivial_selection():
    rows = [
        _fold(1, 1.0, 1.1, 1.1),
        _fold(2, 1.0, 1.1, 1.1),
        _fold(3, 1.0, 1.1, 1.1),
        _fold(4, 1.0, 0.9, 1.1, 0.4),
        _fold(5, 1.0, 0.8, 0.9, 0.3),
    ]
    out = adjudicate_h52(rows)
    assert out["overall_support_selection_mechanism_supported"] is True
    assert out["drop_medium48_profile_mse_better_than_full102_both_late_folds"] is True
    assert out["drop_operator48_profile_mse_better_than_full102_both_late_folds"] is False
    assert out["drop_medium48_selection_nontrivial_both_late_folds"] is True


def test_adjudication_fails_if_operator_has_same_late_pattern():
    rows = [
        _fold(1, 1.0, 1.1, 1.1),
        _fold(2, 1.0, 1.1, 1.1),
        _fold(3, 1.0, 1.1, 1.1),
        _fold(4, 1.0, 0.9, 0.8, 0.4),
        _fold(5, 1.0, 0.8, 0.7, 0.3),
    ]
    out = adjudicate_h52(rows)
    assert out["overall_support_selection_mechanism_supported"] is False
    assert out["drop_operator48_profile_mse_better_than_full102_both_late_folds"] is True
