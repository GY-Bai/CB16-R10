from __future__ import annotations

from cb16_local_opt.action_contract_r1 import LONG, make_target_position_action_r1
from cb16_local_opt.execution_feasibility_r0 import (
    FEASIBLE,
    REJECT_MAINTENANCE,
    REJECT_MARGIN,
    MechanicalExecutionAuthorityR0,
    evaluate_execution_feasibility_r0,
)
from cb16_local_opt.target_exposure_r1 import (
    make_target_exposure_authority_r1,
    map_action_to_target_exposure_r1,
)


def _target(*, risk=0.5, price=10.0, equity=100.0):
    action = make_target_position_action_r1(
        action_id="a",
        policy_id="p",
        policy_version="v1",
        target_direction=LONG,
        requested_target_risk=risk,
    )
    sizing = make_target_exposure_authority_r1(
        authority_id="size",
        account_id="acct",
        account_state_sha256="a" * 64,
        legal_envelope_id="env",
        equity=equity,
        current_price=price,
        margin_capacity=1000.0,
        max_gross_leverage=2.0,
        initial_margin_rate=0.5,
    )
    return map_action_to_target_exposure_r1(action, sizing)


def _mechanical(**overrides):
    kwargs = dict(
        authority_id="mechanical",
        current_quantity=0.0,
        current_price=10.0,
        available_margin_for_new_exposure=1000.0,
        initial_margin_rate=0.5,
        maintenance_margin_rate=0.1,
        maintenance_collateral=1000.0,
        lot_min_qty=None,
        lot_max_qty=None,
        min_notional=None,
    )
    kwargs.update(overrides)
    return MechanicalExecutionAuthorityR0(**kwargs)


def test_feasibility_derives_actual_delta_after_fresh_sizing() -> None:
    target = _target()
    result = evaluate_execution_feasibility_r0(target, _mechanical(current_quantity=4.0))
    assert target.target_quantity == 10.0
    assert result.delta_quantity == 6.0
    assert result.added_quantity == 6.0
    assert result.added_margin_required == 30.0
    assert result.status == FEASIBLE


def test_increase_rejected_when_added_margin_exceeds_current_capacity() -> None:
    result = evaluate_execution_feasibility_r0(
        _target(),
        _mechanical(current_quantity=4.0, available_margin_for_new_exposure=20.0),
    )
    assert result.status == REJECT_MARGIN
    assert result.reason_codes == ("INSUFFICIENT_NEW_EXPOSURE_MARGIN",)


def test_maintenance_feasibility_uses_target_position_only() -> None:
    result = evaluate_execution_feasibility_r0(
        _target(),
        _mechanical(maintenance_collateral=5.0),
    )
    assert result.status == REJECT_MAINTENANCE
    assert result.maintenance_margin_required == 10.0


def test_reduction_has_zero_added_margin_requirement() -> None:
    target = _target(risk=0.25)
    assert target.target_quantity == 5.0
    result = evaluate_execution_feasibility_r0(
        target,
        _mechanical(current_quantity=10.0, available_margin_for_new_exposure=0.0),
    )
    assert result.delta_quantity == -5.0
    assert result.added_quantity == 0.0
    assert result.added_margin_required == 0.0
    assert result.status == FEASIBLE


def test_feasibility_has_no_strategy_or_profitability_inputs() -> None:
    fields = MechanicalExecutionAuthorityR0.__dataclass_fields__
    forbidden = {"atr", "stop_loss", "take_profit", "expected_return", "confidence", "max_holding_bars"}
    assert forbidden.isdisjoint(fields)


def test_same_inputs_are_deterministic() -> None:
    target = _target()
    authority = _mechanical(current_quantity=2.0)
    first = evaluate_execution_feasibility_r0(target, authority)
    second = evaluate_execution_feasibility_r0(target, authority)
    assert first == second
