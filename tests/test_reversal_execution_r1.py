from __future__ import annotations

from cb16_local_opt.action_contract_r1 import FLAT, LONG, SHORT, make_target_position_action_r1
from cb16_local_opt.execution_feasibility_r0 import REJECT_MARGIN
from cb16_local_opt.reversal_execution_r1 import (
    REVERSAL_CLOSED_OPEN_REJECTED,
    REVERSAL_EXECUTED,
    IntermediateFlatAccountR1,
    execute_two_phase_reversal_r1,
)


def _flat(**overrides):
    kwargs = dict(
        account_id="acct",
        account_state_sha256="a" * 64,
        source_direction=LONG,
        close_quantity=10.0,
        close_fee=2.0,
        close_realized_pnl=-7.0,
        equity_after_close=91.0,
        margin_capacity_after_close=91.0,
        execution_price=10.0,
    )
    kwargs.update(overrides)
    return IntermediateFlatAccountR1(**kwargs)


def _short(risk=0.5):
    return make_target_position_action_r1(
        action_id="reverse-open",
        policy_id="p",
        policy_version="v1",
        target_direction=SHORT,
        requested_target_risk=risk,
    )


def _execute(**overrides):
    kwargs = dict(
        intermediate_flat=_flat(),
        opposite_action=_short(),
        legal_envelope_id="env-after-close",
        max_gross_leverage=2.0,
        initial_margin_rate=0.5,
        available_margin_for_new_exposure=100.0,
        maintenance_margin_rate=0.1,
        maintenance_collateral=100.0,
        declared_max_legal_notional=None,
    )
    kwargs.update(overrides)
    return execute_two_phase_reversal_r1(**kwargs)


def test_successful_reversal_sizes_from_intermediate_flat_account() -> None:
    result = _execute()
    assert result.status == REVERSAL_EXECUTED
    assert result.final_direction == SHORT
    # equity_after_close=91 -> leverage cap=182 -> 0.5 target notional=91 -> qty=9.1
    assert result.final_target_quantity == -9.1
    assert result.open_target.account_state_sha256 == "a" * 64
    assert result.open_target.legal_envelope_id == "env-after-close"


def test_second_leg_rejection_leaves_auditable_flat_state() -> None:
    result = _execute(available_margin_for_new_exposure=0.0)
    assert result.status == REVERSAL_CLOSED_OPEN_REJECTED
    assert result.open_feasibility_status == REJECT_MARGIN
    assert result.final_direction == FLAT
    assert result.final_target_quantity == 0.0
    assert result.close_costs_preserved is True
    assert result.close_leg.close_fee == 2.0
    assert result.close_leg.close_realized_pnl == -7.0
    assert result.close_leg.equity_after_close == 91.0


def test_second_leg_uses_post_close_equity_not_pre_close_equity() -> None:
    lower = _execute(intermediate_flat=_flat(equity_after_close=50.0, margin_capacity_after_close=50.0))
    higher = _execute(intermediate_flat=_flat(equity_after_close=100.0, margin_capacity_after_close=100.0))
    assert abs(lower.open_target.target_quantity) == 5.0
    assert abs(higher.open_target.target_quantity) == 10.0


def test_wrong_second_leg_direction_fails_closed() -> None:
    long_again = make_target_position_action_r1(
        action_id="bad",
        policy_id="p",
        policy_version="v1",
        target_direction=LONG,
        requested_target_risk=0.5,
    )
    try:
        _execute(opposite_action=long_again)
    except RuntimeError as exc:
        assert "OPPOSITE_DIRECTION_REQUIRED" in str(exc)
    else:
        raise AssertionError("same-side second leg must fail closed")


def test_close_leg_is_explicit_and_never_overwritten_by_open_result() -> None:
    rejected = _execute(available_margin_for_new_exposure=0.0)
    assert rejected.close_leg.source_direction == LONG
    assert rejected.close_leg.close_quantity == 10.0
    assert rejected.close_leg.account_state_sha256 == "a" * 64
