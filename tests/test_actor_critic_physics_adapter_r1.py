from __future__ import annotations

import json
from pathlib import Path

import pytest

from cb16_local_opt.account_economics_r0 import make_account_economics_state_r0
from cb16_local_opt.action_contract_r1 import FLAT, LONG, SHORT, make_target_position_action_r1
from cb16_local_opt.actor_critic_environment_profile_r1 import policy_neutral_environment_profile_r1
from cb16_local_opt.actor_critic_physics_adapter_r1 import (
    EXECUTED,
    NO_POSITION_CHANGE,
    REVERSAL_OPEN_REJECTED,
    Round2MechanicalExecutionConfigR1,
    _state_sha,
    execute_target_position_r1,
)
from cb16_local_opt.actor_critic_supervisor_r1 import make_supervisor_authority_r1

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = json.loads((ROOT / "authority/rearchitecture_r11/CB16_R11_BC_EXECUTION_RULE_TAXONOMY_V1.json").read_text())


def _account(**overrides):
    kwargs = dict(
        account_id="acct",
        cash=100.0,
        position_quantity=0.0,
        position_cost_basis=0.0,
        mark_price=10.0,
        realized_pnl_cumulative=0.0,
        fees_cumulative=0.0,
        funding_cumulative=0.0,
        margin_collateral=0.0,
        liabilities=0.0,
        external_capital_flows_cumulative=0.0,
        economic_responsibility_open=True,
    )
    kwargs.update(overrides)
    return make_account_economics_state_r0(**kwargs)


def _action(direction=LONG, risk=0.5):
    return make_target_position_action_r1(
        action_id="a",
        policy_id="p",
        policy_version="v1",
        target_direction=direction,
        requested_target_risk=(0.0 if direction == FLAT else risk),
    )


def _config(**overrides):
    kwargs = dict(
        fee_rate=0.001,
        slippage_bps=0.0,
        initial_margin_rate=0.5,
        maintenance_margin_rate=0.1,
        max_gross_leverage=2.0,
    )
    kwargs.update(overrides)
    return Round2MechanicalExecutionConfigR1(**kwargs)


def _supervisor(account):
    return make_supervisor_authority_r1(
        authority_id="sup",
        account_id=account.account_id,
        account_state_sha256=_state_sha(account),
        terminated=False,
        truncated=False,
        legal_target_directions=(SHORT, FLAT, LONG),
        max_permitted_target_risk=1.0,
    )


def _execute(account=None, action=None, config=None, **overrides):
    account = account or _account()
    action = action or _action()
    config = config or _config()
    kwargs = dict(
        account=account,
        action=action,
        supervisor_authority=_supervisor(account),
        profile=policy_neutral_environment_profile_r1(),
        taxonomy=TAXONOMY,
        mark_price=10.0,
        legal_envelope_id="env",
        config=config,
    )
    kwargs.update(overrides)
    return execute_target_position_r1(**kwargs)


def test_open_has_fee_margin_and_account_consequence() -> None:
    result = _execute()
    assert result.status == EXECUTED
    assert result.transition_kind == "OPEN_OR_INCREASE"
    assert result.target_quantity == 10.0
    assert result.account_after.position_quantity == 10.0
    assert result.account_after.margin_collateral == 50.0
    assert result.fee_paid == 0.1
    assert result.account_after.cash == 49.9
    assert result.account_after.equity == 99.9


def test_same_nominal_risk_rebalances_when_equity_changes() -> None:
    current = _account(cash=20.0, position_quantity=10.0, position_cost_basis=10.0, margin_collateral=50.0)
    result = _execute(account=current, action=_action(LONG, 0.5))
    assert result.status == EXECUTED
    assert result.target_quantity == 7.0
    assert result.transition_kind == "REDUCE"


def test_resize_increase_and_reduce_are_distinct_account_updates() -> None:
    current = _account(cash=50.0, position_quantity=5.0, position_cost_basis=10.0, margin_collateral=25.0)
    increase = _execute(account=current, action=_action(LONG, 0.5))
    assert increase.account_after.position_quantity > 5.0
    reduced = _execute(account=current, action=_action(LONG, 0.1))
    assert reduced.account_after.position_quantity < 5.0
    assert reduced.account_after.margin_collateral < 25.0


def test_close_is_reachable_and_releases_margin() -> None:
    current = _account(cash=50.0, position_quantity=5.0, position_cost_basis=10.0, margin_collateral=25.0)
    result = _execute(account=current, action=_action(FLAT, 0.0))
    assert result.status == EXECUTED
    assert result.transition_kind == "CLOSE"
    assert result.account_after.position_quantity == 0.0
    assert result.account_after.position_cost_basis == 0.0
    assert result.account_after.margin_collateral == 0.0


def test_true_quantity_noop_has_no_fee_or_account_mutation() -> None:
    current = _account(cash=50.0, position_quantity=5.0, position_cost_basis=10.0, margin_collateral=25.0)
    result = _execute(account=current, action=_action(LONG, 1.0 / 3.0))
    assert result.status == NO_POSITION_CHANGE
    assert result.delta_quantity == 0.0
    assert result.fee_paid == 0.0
    assert result.account_before_sha256 == result.account_after_sha256


def test_reversal_second_leg_failure_leaves_flat_with_close_costs() -> None:
    current = _account(cash=-5.0, position_quantity=5.0, position_cost_basis=10.0, margin_collateral=25.0)
    result = _execute(
        account=current,
        action=_action(SHORT, 1.0),
        config=_config(maintenance_margin_rate=0.9),
    )
    assert result.status == REVERSAL_OPEN_REJECTED
    assert result.account_after.position_quantity == 0.0
    assert result.account_after.margin_collateral == 0.0
    assert result.fee_paid > 0.0
    assert result.second_leg_status == "REJECT_MAINTENANCE"


def test_negative_equity_is_not_clamped_by_execution_adapter() -> None:
    current = _account(
        cash=-30.0,
        position_quantity=1.0,
        position_cost_basis=100.0,
        mark_price=10.0,
        margin_collateral=5.0,
        liabilities=10.0,
    )
    assert current.equity < 0.0
    result = _execute(account=current, action=_action(FLAT, 0.0))
    assert result.account_after.equity < 0.0
    assert result.account_after.cash < 0.0


def test_stale_supervisor_account_binding_fails_closed() -> None:
    current = _account()
    stale = _account(cash=99.0)
    with pytest.raises(RuntimeError, match="SUPERVISOR_ACCOUNT_STATE_STALE"):
        _execute(account=current, supervisor_authority=_supervisor(stale))


def test_policy_neutral_path_has_no_inherited_strategy_exit_rule() -> None:
    profile = policy_neutral_environment_profile_r1()
    for rule in ("stop_loss", "take_profit", "max_hold", "cooldown", "finalize_behavior"):
        assert profile.is_enabled(rule) is False


def test_fee_conservation_on_close() -> None:
    current = _account(cash=50.0, position_quantity=5.0, position_cost_basis=8.0, margin_collateral=25.0)
    before_equity = current.equity
    result = _execute(account=current, action=_action(FLAT, 0.0))
    assert result.realized_pnl == 10.0
    assert abs(result.account_after.equity - (before_equity - result.fee_paid)) < 1e-12
