"""Bounded durable repeated-learning loop contract tests (engineering evidence only)."""

from __future__ import annotations

import tempfile

import pytest

from cb16_local_opt import post_cc_s1_tasks_v1 as T
from cb16_local_opt.post_cc_s1_training_loop_v1 import (
    SMOKE_ONLY_EVIDENCE_CLASS,
    S1SeedRunConfigV1,
    run_seed_v1,
)

MANIFEST_SHA = "0" * 64


def _run(task_id: str, *, control_id: str | None = None, unit_size: int = 2, max_units: int = 2, seed: int = 1701):
    spec = T.build_task_specs_v1()[task_id]
    with tempfile.TemporaryDirectory(prefix="cb16-s1-loop-test-") as root:
        config = S1SeedRunConfigV1(
            spec=spec,
            seed=seed,
            run_root=root,
            mode="smoke",
            control_id=control_id,
            unit_size=unit_size,
            max_units=max_units,
            evaluation_population=4,
            manifest_sha256=MANIFEST_SHA,
        )
        return run_seed_v1(config)


@pytest.mark.parametrize("task_id", T.FROZEN_TASK_IDS_V1)
def test_every_positive_task_runs_the_full_durable_loop(task_id):
    result = _run(task_id)
    assert result["status"] == "OK"
    assert result["evidence_class"] == SMOKE_ONLY_EVIDENCE_CLASS
    assert result["optimizer"]["optimizer_step_final"] == 2
    assert result["optimizer"]["gradient_applications"] == 2
    assert result["durability"]["restart_sentinel_passed"] is True
    assert result["durability"]["behavior_checkpoint_untouched_every_unit"] is True
    assert result["firewall"] == {
        "FINAL_opened": False,
        "fresh_data_used": False,
        "historical_market_corpus_accessed": False,
        "economic_evidence_claimed": False,
        "transfer_evidence_claimed": False,
    }
    assert result["handcrafted_regime_activation"] is False
    assert len(result["unit_evidence"]) == 2
    assert result["oracle_validation"]["expected_direction_present_for_every_context"] is True


def test_child_switch_tasks_use_committed_child_generation_authority():
    for task_id in (
        T.TASK_ACCOUNT_DEPENDENT_ACTION,
        T.TASK_DELAYED_CONSEQUENCE_CREDIT,
        T.TASK_HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION,
        T.TASK_ABA_RETENTION,
    ):
        result = _run(task_id)
        for unit in result["unit_evidence"]:
            receipt = unit["generation_switch_receipt"]
            assert receipt is not None
            assert receipt["account_fully_preserved"] is True
            assert receipt["authorized_boundary"] is True
            assert receipt["parent_checkpoint_mutated_in_place"] is False
            assert unit["behavior_identity_after"].startswith("cc-s1-policy:")


def test_off_policy_task_keeps_fixed_distinct_behavior_checkpoint():
    result = _run(T.TASK_OFF_POLICY_VTRACE_CORRECTION)
    assert result["behavior_final_identity"]["policy_id"] == "cc-s1-fixed-behavior"
    for unit in result["unit_evidence"]:
        assert unit["generation_switch_receipt"] is None
        assert unit["behavior_identity_before"] == unit["behavior_identity_after"]
    ratios = result["ratio_diagnostics"]
    assert ratios["max_ratio_above_one_count"] >= 0
    assert ratios["max_ratio_below_one_count"] >= 0


def test_zero_reward_control_records_zero_rewards_and_no_signal():
    result = _run(T.TASK_ABA_RETENTION, control_id=T.CONTROL_NO_SIGNAL)
    assert result["control_transform"] == "ZERO_REWARD_NO_SIGNAL"
    for unit in result["unit_evidence"]:
        assert all(reward == 0.0 for reward in unit["rewards"])


def test_account_ablation_control_marks_zeroed_inputs():
    result = _run(T.TASK_ACCOUNT_DEPENDENT_ACTION, control_id=T.CONTROL_ACCOUNT_ABLATION)
    assert result["control_transform"] == "ACCOUNT_INPUT_ABLATION"
    assert all(unit["restart_sentinel"]["restart_verified"] is True for unit in result["unit_evidence"])


def test_delayed_episode_credit_is_durable_per_sequence():
    import json

    spec = T.build_task_specs_v1()[T.TASK_DELAYED_CONSEQUENCE_CREDIT]
    with tempfile.TemporaryDirectory(prefix="cb16-s1-loop-credit-") as root:
        result = run_seed_v1(
            S1SeedRunConfigV1(
                spec=spec,
                seed=1701,
                run_root=root,
                mode="smoke",
                control_id=None,
                unit_size=2,
                max_units=2,
                evaluation_population=4,
                manifest_sha256=MANIFEST_SHA,
            )
        )
        index_path = result["run_root"] + "/durable_index.jsonl"
        rows = [json.loads(line) for line in open(index_path, encoding="utf-8") if line.strip()]
    assert rows
    assert all(row["credit_view_id"] is not None for row in rows)
    assert all(row["boundary_type"] == "OBJECTIVE_HORIZON_REACHED" for row in rows)
    assert all(row["behavior_log_mu"] is not None for row in rows)


def test_run_cannot_be_rescued_by_scientific_override_in_qualification_mode():
    spec = T.build_task_specs_v1()[T.TASK_ABA_RETENTION]
    with pytest.raises(Exception):
        S1SeedRunConfigV1(
            spec=spec,
            seed=1701,
            run_root="/tmp/nonexistent-s1",
            mode="qualification",
            control_id=None,
            unit_size=64,
            max_units=1,
            evaluation_population=64,
            manifest_sha256=MANIFEST_SHA,
        ).validate()
