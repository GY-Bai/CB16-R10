from __future__ import annotations

from types import SimpleNamespace

import torch

from cb16_local_opt.reduced_teacher_target_information_audit_r6 import (
    R6_FOLDS,
    R6_SHIFTS,
    R6_SYMBOLS,
    analytic_global_marginal_baseline_r6,
    build_outer_folds_r6,
    summarize_r6,
)
from cb16_local_opt.training_runtime_r11 import PreparedEvidenceR11


def _prepared(rows):
    pack = torch.zeros((len(rows), 107), dtype=torch.float32)
    parent_ids = []
    gids = []
    for i, (p, r) in enumerate(rows):
        pack[i, 102:105] = torch.tensor(p, dtype=torch.float32)
        pack[i, 105] = float(r)
        pack[i, 106] = 1.0
        parent_ids.append(f"P:{i}")
        gids.append(f"G:{i}")
    out = PreparedEvidenceR11(
        parent_ids=tuple(parent_ids),
        dependence_group_ids=tuple(gids),
        packed=pack,
        evidence_hash="x",
        host_to_device_transfers=0,
    )
    out.validate()
    return out


def test_outer_folds_keep_same_clock_and_all_symbols_together():
    start = 1_600_000_000_000
    parents = {}
    for ci in range(60):
        t = start + ci * 256 * 3_600_000
        for symbol in R6_SYMBOLS:
            pid = f"P:{symbol}:{t}"
            parents[pid] = SimpleNamespace(
                parent_id=pid,
                split="TRAIN",
                symbol=symbol,
                decision_time_ms=t,
                dependence_group_id=f"FUT:{symbol}:{t}",
            )
    folds = build_outer_folds_r6(parents)
    assert tuple(x["fold"] for x in folds) == R6_FOLDS
    assert len(folds) == 5
    for row in folds:
        assert set(row["train_clocks"]).isdisjoint(row["eval_clocks"])
        assert row["train_to_eval_gap_hours"] == 256.0
        assert row["eval_clock_count"] == 10
        assert row["eval_group_count"] == 100
        assert set(row["train_symbols"]) == set(R6_SYMBOLS)
        assert set(row["eval_symbols"]) == set(R6_SYMBOLS)


def test_analytic_global_marginal_uses_train_target_means_only():
    train = _prepared([
        ((0.8, 0.1, 0.1), 0.2),
        ((0.2, 0.3, 0.5), 0.6),
    ])
    val = _prepared([
        ((0.5, 0.25, 0.25), 0.4),
        ((0.1, 0.2, 0.7), 0.5),
    ])
    out = analytic_global_marginal_baseline_r6(train, val)
    assert out["no_state_features"] is True
    assert all(abs(a - b) < 1e-7 for a, b in zip(out["direction_distribution"], (0.5, 0.2, 0.3)))
    assert abs(out["requested_risk_mean"] - 0.4) < 1e-7
    assert out["loss"] == out["direction_loss"] + out["sizing_loss"]
    assert out["loss"] > 0.0


def _fold_result(fold: int, *, supported: bool):
    if supported:
        aligned = {"loss": 0.9, "direction_loss": 0.7, "sizing_loss": 0.2}
        sh = {"loss": 1.0, "direction_loss": 0.8, "sizing_loss": 0.2}
        marginal = {"loss": 1.1, "direction_loss": 0.85, "sizing_loss": 0.25}
    else:
        aligned = {"loss": 1.1, "direction_loss": 0.9, "sizing_loss": 0.2}
        sh = {"loss": 1.0, "direction_loss": 0.8, "sizing_loss": 0.2}
        marginal = {"loss": 1.0, "direction_loss": 0.85, "sizing_loss": 0.2}
    arms = {"ALIGNED": {"validation_after": aligned}}
    for shift in R6_SHIFTS:
        arms[f"SHUFFLE_{shift}"] = {"validation_after": dict(sh)}
    return {"fold": fold, "arms": arms, "marginal_baseline": marginal}


def test_summary_primary_rule_is_four_of_five_and_reports_all_25_pairs():
    rows = [_fold_result(i, supported=(i != 5)) for i in R6_FOLDS]
    out = summarize_r6(rows)
    assert out["conditional_information_supported"] is True
    assert out["total_lower_than_shuffle_median_fold_count"] == 4
    assert out["direction_lower_than_shuffle_median_fold_count"] == 4
    assert out["direction_lower_than_marginal_fold_count"] == 4
    assert len(out["all_25_pairwise"]) == 25

    rows = [_fold_result(i, supported=(i <= 3)) for i in R6_FOLDS]
    out = summarize_r6(rows)
    assert out["conditional_information_supported"] is False
    assert out["conclusion"] == "TRAIN_ONLY_CONDITIONAL_INFORMATION_NOT_SUPPORTED"
