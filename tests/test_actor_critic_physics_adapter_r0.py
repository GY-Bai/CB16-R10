from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cb16_local_opt.action_contract_r0 import FLAT, LONG, SHORT, make_target_position_action_r0
from cb16_local_opt.actor_critic_physics_adapter_r0 import (
    CLOSE_TO_FLAT,
    EXECUTED,
    NOOP,
    OPEN_FROM_FLAT,
    RESIZE_INCREASE,
    RESIZE_REDUCE,
    REVERSAL_CLOSE,
    REVERSAL_OPEN,
    execute_target_position_r0,
)
from cb16_local_opt.actor_critic_supervisor_r0 import (
    ACCEPT,
    REVERSAL_CLOSE_THEN_OPEN_R0,
    make_supervisor_authority_state_r0,
    supervise_target_action_r0,
)
from cb16_local_opt.r102_physics import FLAT as V55_FLAT
from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102
from cb16_local_opt.target_exposure_r0 import (
    make_target_exposure_authority_r0,
    map_permission_to_target_exposure_r0,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRICE = 100.0
DECLARED_NOTIONAL_CAP = 10_000.0


@pytest.fixture(scope="module")
def runtime() -> FrozenPhysicsRuntimeR102:
    return FrozenPhysicsRuntimeR102.load(REPO_ROOT)


def _primed_flat_snapshot(runtime: FrozenPhysicsRuntimeR102):
    snapshot, _risk = runtime.initialize("ac014-account", 1.0)
    kernel = runtime.physics.restore_kernel(snapshot, runtime.physics_contract)
    kernel.state.atr = 10.0
    kernel.state.prev_close = PRICE
    kernel.state.last_mark_price = PRICE
    support = copy.deepcopy(snapshot["observation_support_state"])
    return runtime.physics.snapshot_kernel(kernel, runtime.physics_contract, support)


def _direction_from_quantity(quantity: float) -> str:
    if quantity > 1e-12:
        return LONG
    if quantity < -1e-12:
        return SHORT
    return FLAT


def _permission_and_target(
    runtime: FrozenPhysicsRuntimeR102,
    snapshot,
    *,
    direction: str,
    risk: float,
    current_risk: float,
    margin_available: bool = True,
):
    position = float(snapshot["kernel_state"]["position"])
    current_direction = _direction_from_quantity(position)
    action = make_target_position_action_r0(
        action_id=f"ac014:{current_direction}:{current_risk}->{direction}:{risk}",
        policy_id="actor-policy-r0",
        policy_version="actor-policy-v1",
        target_direction=direction,
        requested_target_risk=risk,
    )
    supervisor_authority = make_supervisor_authority_state_r0(
        authority_id="ac014-supervisor-authority",
        account_id="ac014-account",
        current_direction=current_direction,
        current_target_risk=current_risk,
        terminated=False,
        truncated=False,
        margin_available_for_new_exposure=margin_available,
        legal_target_directions=(SHORT, FLAT, LONG),
        max_permitted_target_risk=1.0,
    )
    permission = supervise_target_action_r0(action, supervisor_authority)
    assert permission.outcome == ACCEPT

    kernel = runtime.physics.restore_kernel(snapshot, runtime.physics_contract)
    kernel.state.last_mark_price = PRICE
    equity = float(kernel.state.equity())
    margin_capacity = float(kernel.state.margin_used + kernel.state.available_margin())
    config = runtime.physics_contract["sim_config"]
    target_authority = make_target_exposure_authority_r0(
        authority_id="ac014-target-authority",
        account_id="ac014-account",
        equity=equity,
        current_price=PRICE,
        margin_capacity=margin_capacity,
        max_gross_leverage=float(config["max_leverage"]),
        initial_margin_rate=config["initial_margin_rate"],
        declared_max_legal_notional=DECLARED_NOTIONAL_CAP,
        lot_step_size=config["lot_step_size"],
        lot_min_qty=config["lot_min_qty"],
        lot_max_qty=config["lot_max_qty"],
        min_notional=config["min_notional"],
    )
    target = map_permission_to_target_exposure_r0(permission, target_authority)
    return permission, target


def _execute(
    runtime: FrozenPhysicsRuntimeR102,
    snapshot,
    permission,
    target,
    suffix: str,
):
    return execute_target_position_r0(
        runtime,
        snapshot,
        permission,
        target,
        execution_price=PRICE,
        execution_record_id=f"exec-{suffix}",
        permission_record_id=f"permission-record-{suffix}",
    )


def test_open_from_flat_uses_frozen_pricing_fee_and_margin(runtime) -> None:
    snapshot = _primed_flat_snapshot(runtime)
    before_hash = runtime.physics.sha256_obj(snapshot)
    permission, target = _permission_and_target(
        runtime,
        snapshot,
        direction=LONG,
        risk=0.5,
        current_risk=0.0,
    )
    assert target.target_quantity == pytest.approx(50.0)

    result = _execute(runtime, snapshot, permission, target, "open")
    payload = result.execution_record.payload()
    state = result.snapshot_t1["kernel_state"]

    assert result.status == EXECUTED
    assert result.target_reached is True
    assert state["position"] == pytest.approx(50.0)
    assert payload["legs"][0]["kind"] == OPEN_FROM_FLAT
    expected_fill = PRICE * (1.0 + runtime.physics_contract["sim_config"]["slippage_bps"] / 10_000.0)
    assert payload["legs"][0]["fill_price"] == pytest.approx(expected_fill)
    expected_fee = 50.0 * expected_fill * runtime.physics_contract["sim_config"]["fee_rate"]
    assert payload["legs"][0]["fee"] == pytest.approx(expected_fee)
    assert float(state["margin_used"]) > 0.0
    assert result.execution_record.executed_delta_quantity == pytest.approx(50.0)
    assert runtime.physics.sha256_obj(snapshot) == before_hash


def test_noop_does_not_churn_position_or_charge_fee(runtime) -> None:
    flat = _primed_flat_snapshot(runtime)
    permission, target = _permission_and_target(
        runtime, flat, direction=LONG, risk=0.5, current_risk=0.0
    )
    opened = _execute(runtime, flat, permission, target, "noop-open")

    permission2, target2 = _permission_and_target(
        runtime,
        opened.snapshot_t1,
        direction=LONG,
        risk=0.5,
        current_risk=0.5,
    )
    result = _execute(runtime, opened.snapshot_t1, permission2, target2, "noop")
    leg = result.execution_record.payload()["legs"][0]
    assert result.status == NOOP
    assert result.target_reached is True
    assert result.execution_record.executed_delta_quantity == 0.0
    assert leg["fee"] == 0.0
    assert leg["kind"] == "NOOP"
    assert result.snapshot_t1["kernel_state"]["position"] == pytest.approx(50.0)


def test_same_side_increase_executes_only_delta_not_close_reopen(runtime) -> None:
    flat = _primed_flat_snapshot(runtime)
    p1, t1 = _permission_and_target(runtime, flat, direction=LONG, risk=0.5, current_risk=0.0)
    opened = _execute(runtime, flat, p1, t1, "increase-open")
    turnover_before = float(opened.snapshot_t1["kernel_state"]["turnover_notional"])

    p2, t2 = _permission_and_target(
        runtime,
        opened.snapshot_t1,
        direction=LONG,
        risk=1.0,
        current_risk=0.5,
    )
    assert t2.target_quantity == pytest.approx(100.0)
    increased = _execute(runtime, opened.snapshot_t1, p2, t2, "increase")
    payload = increased.execution_record.payload()
    state = increased.snapshot_t1["kernel_state"]

    assert increased.status == EXECUTED
    assert state["position"] == pytest.approx(100.0)
    assert increased.execution_record.executed_delta_quantity == pytest.approx(50.0)
    assert payload["legs"][0]["kind"] == RESIZE_INCREASE
    assert payload["legs"][0]["delta_quantity"] == pytest.approx(50.0)
    assert float(state["turnover_notional"]) - turnover_before == pytest.approx(
        50.0 * payload["legs"][0]["fill_price"]
    )
    assert int(state["round_trips"]) == 0


def test_same_side_reduce_realizes_only_reduced_delta(runtime) -> None:
    flat = _primed_flat_snapshot(runtime)
    p1, t1 = _permission_and_target(runtime, flat, direction=LONG, risk=1.0, current_risk=0.0)
    opened = _execute(runtime, flat, p1, t1, "reduce-open")
    margin_before = float(opened.snapshot_t1["kernel_state"]["margin_used"])

    p2, t2 = _permission_and_target(
        runtime,
        opened.snapshot_t1,
        direction=LONG,
        risk=0.25,
        current_risk=1.0,
        margin_available=False,
    )
    reduced = _execute(runtime, opened.snapshot_t1, p2, t2, "reduce")
    payload = reduced.execution_record.payload()
    state = reduced.snapshot_t1["kernel_state"]

    assert state["position"] == pytest.approx(25.0)
    assert reduced.execution_record.executed_delta_quantity == pytest.approx(-75.0)
    assert payload["legs"][0]["kind"] == RESIZE_REDUCE
    assert float(state["margin_used"]) == pytest.approx(margin_before * 0.25)
    assert int(state["round_trips"]) == 0


def test_close_to_flat_delegates_full_close_settlement(runtime) -> None:
    flat = _primed_flat_snapshot(runtime)
    p1, t1 = _permission_and_target(runtime, flat, direction=SHORT, risk=0.5, current_risk=0.0)
    opened = _execute(runtime, flat, p1, t1, "close-open")

    p2, t2 = _permission_and_target(
        runtime,
        opened.snapshot_t1,
        direction=FLAT,
        risk=0.0,
        current_risk=0.5,
        margin_available=False,
    )
    closed = _execute(runtime, opened.snapshot_t1, p2, t2, "close")
    payload = closed.execution_record.payload()
    state = closed.snapshot_t1["kernel_state"]

    assert state["position"] == 0.0
    assert state["margin_used"] == 0.0
    assert closed.execution_record.executed_delta_quantity == pytest.approx(50.0)
    assert payload["legs"][0]["kind"] == CLOSE_TO_FLAT
    assert int(state["round_trips"]) == 1
    assert state["last_close_reason"] == "actor_target_close"


def test_reversal_executes_explicit_close_then_open_legs(runtime) -> None:
    flat = _primed_flat_snapshot(runtime)
    p1, t1 = _permission_and_target(runtime, flat, direction=LONG, risk=0.5, current_risk=0.0)
    opened = _execute(runtime, flat, p1, t1, "reverse-open")

    p2, t2 = _permission_and_target(
        runtime,
        opened.snapshot_t1,
        direction=SHORT,
        risk=0.5,
        current_risk=0.5,
    )
    assert p2.execution_contract.contract_kind == REVERSAL_CLOSE_THEN_OPEN_R0
    reversed_result = _execute(runtime, opened.snapshot_t1, p2, t2, "reverse")
    payload = reversed_result.execution_record.payload()
    state = reversed_result.snapshot_t1["kernel_state"]

    assert reversed_result.status == EXECUTED
    assert reversed_result.target_reached is True
    assert state["position"] == pytest.approx(-50.0)
    assert reversed_result.execution_record.executed_delta_quantity == pytest.approx(-100.0)
    assert [leg["kind"] for leg in payload["legs"]] == [REVERSAL_CLOSE, REVERSAL_OPEN]
    assert int(state["round_trips"]) == 1
    assert state["last_close_reason"] == "actor_target_reversal"


def test_adapter_is_deterministic_for_same_snapshot_permission_target(runtime) -> None:
    snapshot = _primed_flat_snapshot(runtime)
    permission, target = _permission_and_target(
        runtime, snapshot, direction=SHORT, risk=0.37, current_risk=0.0
    )
    a = _execute(runtime, snapshot, permission, target, "det")
    b = _execute(runtime, snapshot, permission, target, "det")
    assert a.status == b.status
    assert a.target_reached == b.target_reached
    assert a.snapshot_t1 == b.snapshot_t1
    assert a.execution_record == b.execution_record
    assert a.execution_record.payload() == b.execution_record.payload()


def test_funding_and_bar_lifecycle_remain_owned_by_frozen_physics(runtime) -> None:
    flat = _primed_flat_snapshot(runtime)
    permission, target = _permission_and_target(
        runtime, flat, direction=LONG, risk=0.5, current_risk=0.0
    )
    opened = _execute(runtime, flat, permission, target, "funding-open")
    assert opened.execution_record.payload()["bar_lifecycle_advanced"] is False

    bar = runtime.bar_dict("BTCUSDT", 1_700_000_000_000, [100.0, 100.1, 99.9, 100.0, 10.0])
    stepped = runtime.physics.step_account(
        opened.snapshot_t1,
        {"decision": V55_FLAT},
        {"bar": bar, "funding_rate": 0.001},
        runtime.physics_contract,
    )
    expected_carry = 50.0 * 100.0 * 0.001
    assert stepped["execution_metadata"]["carry_cost"] == pytest.approx(expected_carry)
    assert stepped["termination_type"] == "NORMAL_CONTINUE"
    assert stepped["snapshot_t1"]["kernel_state"]["position"] == pytest.approx(50.0)


def test_implicit_signed_flip_is_rejected_without_reversal_contract(runtime) -> None:
    flat = _primed_flat_snapshot(runtime)
    p1, t1 = _permission_and_target(runtime, flat, direction=LONG, risk=0.5, current_risk=0.0)
    opened = _execute(runtime, flat, p1, t1, "implicit-open")

    # A real opposite-side permission is a reversal contract. Tampering the target
    # sign while retaining a direct contract must fail through permission/target binding.
    direct_permission, direct_target = _permission_and_target(
        runtime,
        opened.snapshot_t1,
        direction=LONG,
        risk=0.25,
        current_risk=0.5,
    )
    bad_target = copy.copy(direct_target)
    object.__setattr__(bad_target, "target_direction", SHORT)
    object.__setattr__(bad_target, "target_quantity", -abs(float(direct_target.target_quantity)))
    with pytest.raises(RuntimeError):
        _execute(runtime, opened.snapshot_t1, direct_permission, bad_target, "implicit")
