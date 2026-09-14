"""S1 task environment, oracle and evaluation-population contract tests."""

from __future__ import annotations

import math
import tempfile

import pytest
import torch

from cb16_local_opt import post_cc_s1_tasks_v1 as T


@pytest.fixture(scope="module")
def specs():
    return T.build_task_specs_v1()


def _account_for(spec, context, root):
    account, provenance = T.establish_account_context_v1(
        spec=spec, context=context, lineage="cc-s1-test-lineage", setup_root=root
    )
    return account, provenance


def _row(oracle, direction, risk):
    return next(
        row for row in oracle["rows"] if row["direction"] == direction and row["risk"] == pytest.approx(risk)
    )


def test_frozen_task_ids_and_budgets(specs):
    assert tuple(specs) == T.FROZEN_TASK_IDS_V1
    assert specs[T.TASK_ACCOUNT_DEPENDENT_ACTION].training_decisions == 16384
    assert specs[T.TASK_DELAYED_CONSEQUENCE_CREDIT].training_decisions == 16384
    assert specs[T.TASK_HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION].training_decisions == 16384
    assert specs[T.TASK_OFF_POLICY_VTRACE_CORRECTION].training_decisions == 16384
    assert specs[T.TASK_ABA_RETENTION].training_decisions == 12288
    assert specs[T.TASK_ABA_RETENTION].phase_context_ids == (("A", 4096), ("B", 4096), ("A", 4096))
    for spec in specs.values():
        assert spec.handcrafted_regime_activation is False
        assert spec.objective_orientation == "COMPLETE_SAMPLE_ARITHMETIC_EQUITY_DELTA"


def test_zero_fee_hand_arithmetic_matches_oracle_enumeration(specs):
    """Hand numbers: PnL = (risk * leverage * capital / decision_mark) * price move."""
    with tempfile.TemporaryDirectory() as root:
        cases = [
            (T.TASK_ABA_RETENTION, "A", "LONG", 18.0 / 101.0),
            (T.TASK_ABA_RETENTION, "B", "SHORT", 18.0 / 99.0),
            (T.TASK_DELAYED_CONSEQUENCE_CREDIT, "UP_SIGNAL", "LONG", 18.0 / 101.0),
            (T.TASK_DELAYED_CONSEQUENCE_CREDIT, "DOWN_SIGNAL", "SHORT", 18.0 / 99.0),
            (T.TASK_OFF_POLICY_VTRACE_CORRECTION, "OFF_POLICY_UP", "LONG", 18.0 / 102.0),
            (T.TASK_OFF_POLICY_VTRACE_CORRECTION, "OFF_POLICY_DOWN", "SHORT", 18.0 / 98.0),
        ]
        for task_id, context_id, direction, expected_return in cases:
            spec = specs[task_id]
            context = next(item for item in spec.contexts if item.context_id == context_id)
            account, _provenance = _account_for(spec, context, root)
            oracle = T.enumerate_oracle_v1(spec=spec, context=context, account=account)
            best = _row(oracle, direction, 0.9)
            assert best["expected_arithmetic_return"] == pytest.approx(expected_return, abs=1e-9)
            assert oracle["best_directions"] == (direction,)
            assert oracle["flat_return"] == pytest.approx(0.0, abs=1e-12)


def test_high_bankruptcy_hand_table_and_preflight_relation(specs):
    spec = specs[T.TASK_HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION]
    context = spec.contexts[0]
    with tempfile.TemporaryDirectory() as root:
        account, _provenance = _account_for(spec, context, root)
        oracle = T.enumerate_oracle_v1(spec=spec, context=context, account=account)
        assert _row(oracle, "LONG", 0.9)["expected_arithmetic_return"] == pytest.approx(0.072, abs=1e-12)
        assert _row(oracle, "LONG", 0.9)["expected_bankruptcy_frequency"] == pytest.approx(0.20, abs=1e-12)
        assert _row(oracle, "SHORT", 0.9)["expected_arithmetic_return"] == pytest.approx(-0.072, abs=1e-12)
        assert _row(oracle, "FLAT", 0.0)["expected_arithmetic_return"] == pytest.approx(0.0, abs=1e-12)
        assert _row(oracle, "FLAT", 0.0)["expected_bankruptcy_frequency"] == pytest.approx(0.0, abs=1e-12)
        assert oracle["best_directions"] == ("LONG",)
        assert oracle["non_flat_higher_return_count"] >= 1
        assert oracle["non_flat_higher_return_and_bankruptcy_count"] >= 1
        # the preferred action must not be rejected for its higher bankruptcy frequency
        assert _row(oracle, "LONG", 0.9)["expected_bankruptcy_frequency"] > _row(oracle, "FLAT", 0.0)[
            "expected_bankruptcy_frequency"
        ]


def test_account_dependent_reachability_and_identity(specs):
    spec = specs[T.TASK_ACCOUNT_DEPENDENT_ACTION]
    with tempfile.TemporaryDirectory() as root:
        accounts = {}
        oracles = {}
        for context in spec.contexts:
            account, provenance = _account_for(spec, context, root)
            assert provenance is not None
            assert provenance.setup_direction in ("LONG", "SHORT")
            assert provenance.achieved_risk_fraction == pytest.approx(0.5, abs=1e-9)
            accounts[context.context_id] = account
            oracles[context.context_id] = T.enumerate_oracle_v1(spec=spec, context=context, account=account)
        market_long, account_long, execution_long = T.canonical_brain_vectors_from_account_v1(
            accounts["HELD_LONG"]
        )
        market_short, account_short, execution_short = T.canonical_brain_vectors_from_account_v1(
            accounts["HELD_SHORT"]
        )
        assert market_long == market_short == (0.0, 1.0)
        assert execution_long == execution_short
        assert account_long != account_short
        assert account_long[1] > 0.0 > account_short[1]
        assert oracles["HELD_LONG"]["best_directions"] == ("LONG",)
        assert oracles["HELD_SHORT"]["best_directions"] == ("SHORT",)
        # hand arithmetic for both scored oracle actions (999 equity, 0.001 fee rounding)
        assert _row(oracles["HELD_LONG"], "LONG", 0.5)["expected_arithmetic_return"] == pytest.approx(
            -0.0010009999999999763, abs=1e-12
        )
        assert _row(oracles["HELD_SHORT"], "SHORT", 0.5)["expected_arithmetic_return"] == pytest.approx(
            -0.0010009999999999763, abs=1e-12
        )


def test_delayed_stage_structure_is_explicit(specs):
    spec = specs[T.TASK_DELAYED_CONSEQUENCE_CREDIT]
    assert spec.credit_stage2_no_decision is True
    for context in spec.contexts:
        assert context.dynamics.stage1_mark == context.dynamics.decision_mark
        assert context.dynamics.stage2_is_no_decision_advance is True
        assert len(context.dynamics.stage2_branches) == 1


def test_evaluation_population_uses_dedicated_deterministic_stream(specs):
    spec = specs[T.TASK_ABA_RETENTION]
    context = spec.contexts[0]
    with tempfile.TemporaryDirectory() as root:
        account, _provenance = _account_for(spec, context, root)
        actor = T.make_s1_brain_v1()
        before = {name: tensor.detach().clone() for name, tensor in actor.state_dict().items()}
        first = T.evaluate_context_v1(
            spec=spec, context=context, account=account, actor=actor, seed=1701,
            checkpoint_id="EVAL_A", population=64, zero_account_inputs=False,
        )
        second = T.evaluate_context_v1(
            spec=spec, context=context, account=account, actor=actor, seed=1701,
            checkpoint_id="EVAL_A", population=64, zero_account_inputs=False,
        )
        other = T.evaluate_context_v1(
            spec=spec, context=context, account=account, actor=actor, seed=1701,
            checkpoint_id="EVAL_B", population=64, zero_account_inputs=False,
        )
        after = {name: tensor.detach().clone() for name, tensor in actor.state_dict().items()}
        assert first == second
        assert first != other
        assert all(torch.equal(before[name], after[name]) for name in before)
        assert first["population"] == 64


def test_observation_vectors_contain_no_task_or_phase_identity(specs):
    for spec in specs.values():
        for context in spec.contexts:
            with tempfile.TemporaryDirectory() as root:
                account, _provenance = _account_for(spec, context, root)
                market, account_values, execution_values = T.canonical_brain_vectors_from_account_v1(account)
                assert len(market) == 2 and len(account_values) == 3 and len(execution_values) == 2
                assert all(isinstance(value, float) for value in market + account_values + execution_values)
                assert all(math.isfinite(value) for value in market + account_values + execution_values)


def test_random_control_frozen_realization_is_symmetric(specs):
    spec = specs[T.TASK_ABA_RETENTION]
    contexts = T.frozen_random_control_contexts_v1(spec)
    for context in contexts:
        marks = [branch.terminal_mark for branch in context.dynamics.stage2_branches]
        assert marks == pytest.approx([context.dynamics.decision_mark + 10.0, context.dynamics.decision_mark - 10.0])
        assert [branch.probability for branch in context.dynamics.stage2_branches] == pytest.approx([0.5, 0.5])
    high = T.frozen_random_control_contexts_v1(
        specs[T.TASK_HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION]
    )
    assert [branch.terminal_mark for branch in high[0].dynamics.stage2_branches] == pytest.approx([120.0, 80.0])
