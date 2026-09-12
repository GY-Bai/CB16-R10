from __future__ import annotations

"""Actor-Critic R0 target-position adapter over the frozen account Physics.

This module does not replace the frozen bar lifecycle.  It adds only the
position-management primitive missing from the recovered V5.5 kernel:
convert one already-permitted target quantity into the smallest executable
position delta at an authorized execution price.

Pricing, fee rates, margin rates, full-close settlement, protective stop setup,
and capital/liquidation checks delegate to the frozen kernel methods loaded by
``FrozenPhysicsRuntimeR102``. Funding, intrabar SL/TP, liquidation and max-hold
remain owned by the unchanged frozen ``step_account`` bar lifecycle.
"""

from dataclasses import dataclass
import copy
import math
from types import SimpleNamespace
from typing import Any, Mapping

from .action_contract_r0 import FLAT, LONG, SHORT
from .actor_critic_supervisor_r0 import (
    DIRECT_TARGET_EXECUTION_R0,
    REVERSAL_CLOSE_THEN_OPEN_R0,
    SupervisorPermissionResultR0,
)
from .execution_record_r0 import (
    ExecutedDeltaRecordR0,
    make_executed_delta_record_r0,
)
from .r102_physics import FrozenPhysicsRuntimeR102
from .target_exposure_r0 import TargetExposureResultR0


PHYSICS_ADAPTER_VERSION_R0 = "CB16_R11_AC_TARGET_POSITION_PHYSICS_ADAPTER_V1_R0"

EXECUTED = "EXECUTED"
NOOP = "NOOP"
REJECTED_CAPITAL = "REJECTED_CAPITAL"
REVERSAL_CLOSED_OPEN_REJECTED = "REVERSAL_CLOSED_OPEN_REJECTED"
ADAPTER_STATUSES_R0 = (
    EXECUTED,
    NOOP,
    REJECTED_CAPITAL,
    REVERSAL_CLOSED_OPEN_REJECTED,
)

OPEN_FROM_FLAT = "OPEN_FROM_FLAT"
RESIZE_INCREASE = "RESIZE_INCREASE"
RESIZE_REDUCE = "RESIZE_REDUCE"
CLOSE_TO_FLAT = "CLOSE_TO_FLAT"
REVERSAL_CLOSE = "REVERSAL_CLOSE"
REVERSAL_OPEN = "REVERSAL_OPEN"
NOOP_LEG = "NOOP"

EPS = 1e-12


def _require_nonempty_string(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _finite_positive(value: object, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(code)
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(code)
    return number


def _finite_number(value: object, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(code)
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(code)
    return 0.0 if number == 0.0 else number


def _position_direction(quantity: float) -> str:
    if quantity > EPS:
        return LONG
    if quantity < -EPS:
        return SHORT
    return FLAT


def _same_quantity(a: float, b: float) -> bool:
    return abs(a - b) <= EPS


def _snapshot_hash(runtime: FrozenPhysicsRuntimeR102, snapshot: Mapping[str, Any]) -> str:
    return runtime.physics.sha256_obj(snapshot)


def _leg_payload(
    *,
    kind: str,
    delta_quantity: float,
    fill_price: float | None,
    fee: float,
    margin_change: float,
    realized_pnl: float,
    status: str = EXECUTED,
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "kind": kind,
        "status": status,
        "reason": reason,
        "delta_quantity": float(delta_quantity),
        "fill_price": None if fill_price is None else float(fill_price),
        "fee": float(fee),
        "margin_change": float(margin_change),
        "realized_pnl": float(realized_pnl),
    }


@dataclass(frozen=True)
class ActorCriticPhysicsAdapterResultR0:
    adapter_version: str
    status: str
    target_reached: bool
    snapshot_t1: Mapping[str, Any]
    execution_record: ExecutedDeltaRecordR0

    def validate(self) -> None:
        if self.adapter_version != PHYSICS_ADAPTER_VERSION_R0:
            raise RuntimeError("ACPHY_ADAPTER_VERSION_MISMATCH")
        if self.status not in ADAPTER_STATUSES_R0:
            raise RuntimeError("ACPHY_STATUS_INVALID")
        if not isinstance(self.target_reached, bool):
            raise RuntimeError("ACPHY_TARGET_REACHED_INVALID")
        if not isinstance(self.snapshot_t1, Mapping):
            raise RuntimeError("ACPHY_SNAPSHOT_INVALID")
        self.execution_record.validate()


def _ensure_live_snapshot(snapshot: Mapping[str, Any]) -> None:
    term = snapshot.get("termination_state", {})
    if bool(term.get("terminated")):
        raise RuntimeError("ACPHY_TERMINATED_SNAPSHOT")
    if bool(term.get("truncated")):
        raise RuntimeError("ACPHY_TRUNCATED_SNAPSHOT")


def _validate_binding(
    permission: SupervisorPermissionResultR0,
    target: TargetExposureResultR0,
) -> None:
    if not isinstance(permission, SupervisorPermissionResultR0):
        raise RuntimeError("ACPHY_PERMISSION_TYPE_INVALID")
    if not isinstance(target, TargetExposureResultR0):
        raise RuntimeError("ACPHY_TARGET_TYPE_INVALID")
    permission.validate()
    target.validate()
    if target.permission_sha256 != permission.permission_sha256:
        raise RuntimeError("ACPHY_PERMISSION_TARGET_HASH_MISMATCH")
    if target.target_direction != permission.permitted_target_direction:
        raise RuntimeError("ACPHY_TARGET_DIRECTION_MISMATCH")
    if target.target_risk != permission.permitted_target_risk:
        raise RuntimeError("ACPHY_TARGET_RISK_MISMATCH")


def _open_from_flat(
    kernel: Any,
    *,
    target_quantity: float,
    execution_price: float,
    leg_kind: str,
) -> tuple[bool, dict[str, object]]:
    st = kernel.state
    if abs(st.position) > EPS:
        raise RuntimeError("ACPHY_OPEN_REQUIRES_FLAT")
    if abs(target_quantity) <= EPS:
        return True, _leg_payload(
            kind=leg_kind,
            delta_quantity=0.0,
            fill_price=None,
            fee=0.0,
            margin_change=0.0,
            realized_pnl=0.0,
            status=NOOP,
        )

    side = 1 if target_quantity > 0.0 else -1
    st.last_mark_price = execution_price
    fill = float(kernel._effective_price(execution_price, is_buy=side > 0))
    equity_before = float(st.equity())
    if equity_before <= 0.0:
        return False, _leg_payload(
            kind=leg_kind,
            delta_quantity=0.0,
            fill_price=fill,
            fee=0.0,
            margin_change=0.0,
            realized_pnl=0.0,
            status=REJECTED_CAPITAL,
            reason="NONPOSITIVE_EQUITY",
        )

    try:
        stop, _take, _stop_distance = kernel._entry_risk_prices(fill, side)
    except RuntimeError:
        return False, _leg_payload(
            kind=leg_kind,
            delta_quantity=0.0,
            fill_price=fill,
            fee=0.0,
            margin_change=0.0,
            realized_pnl=0.0,
            status=REJECTED_CAPITAL,
            reason="ATR_NOT_PRIMED",
        )

    target_exposure = target_quantity * fill / equity_before
    plan = SimpleNamespace(
        target_exposure=target_exposure,
        target_qty=target_quantity,
        delta_qty=target_quantity,
        fill_price=fill,
    )
    capital_error = kernel._capital_check(plan, stop)
    if capital_error is not None:
        return False, _leg_payload(
            kind=leg_kind,
            delta_quantity=0.0,
            fill_price=fill,
            fee=0.0,
            margin_change=0.0,
            realized_pnl=0.0,
            status=REJECTED_CAPITAL,
            reason=str(capital_error),
        )

    fee = abs(target_quantity) * fill * float(kernel.config.fee_rate)
    margin = float(kernel._initial_margin_for_notional(abs(target_quantity) * fill))
    kernel._apply_entry_plan(plan)
    kernel._refresh_account_fields()
    return True, _leg_payload(
        kind=leg_kind,
        delta_quantity=target_quantity,
        fill_price=fill,
        fee=fee,
        margin_change=margin,
        realized_pnl=0.0,
    )


def _partial_reduce(
    kernel: Any,
    *,
    target_quantity: float,
    execution_price: float,
) -> dict[str, object]:
    st = kernel.state
    current = float(st.position)
    if current * target_quantity <= 0.0 or abs(target_quantity) >= abs(current) - EPS:
        raise RuntimeError("ACPHY_REDUCE_SHAPE_INVALID")

    close_qty = abs(current) - abs(target_quantity)
    is_long = current > 0.0
    fill = float(kernel._effective_price(execution_price, is_buy=not is_long))
    realized = (
        (fill - float(st.avg_entry_price)) * close_qty
        if is_long
        else (float(st.avg_entry_price) - fill) * close_qty
    )
    fee = close_qty * fill * float(kernel.config.fee_rate)
    old_margin = float(st.margin_used)
    release = old_margin * (close_qty / abs(current))

    st.realized_pnl += realized
    st.turnover_notional += close_qty * fill
    st.cash += realized - fee + release
    st.cash = max(0.0, float(st.cash))
    st.margin_used = max(0.0, old_margin - release)
    st.position = target_quantity
    st.last_mark_price = execution_price
    equity = float(st.equity())
    st.last_executed_target = (
        0.0 if equity <= 0.0 else target_quantity * execution_price / equity
    )
    kernel._refresh_account_fields()
    return _leg_payload(
        kind=RESIZE_REDUCE,
        delta_quantity=target_quantity - current,
        fill_price=fill,
        fee=fee,
        margin_change=-release,
        realized_pnl=realized,
    )


def _project_increase_state(
    kernel: Any,
    *,
    target_quantity: float,
    fill: float,
    fee: float,
    added_margin: float,
) -> Any:
    st = copy.deepcopy(kernel.state)
    current = float(st.position)
    delta = target_quantity - current
    new_abs = abs(target_quantity)
    old_abs = abs(current)
    delta_abs = abs(delta)
    st.avg_entry_price = (
        old_abs * float(st.avg_entry_price) + delta_abs * fill
    ) / new_abs
    st.cash -= fee + added_margin
    st.margin_used += added_margin
    st.position = target_quantity
    st.last_mark_price = fill
    if kernel.config.margin_type == "isolated":
        st.isolated_margin = st.margin_used
        st.isolated_wallet = st.margin_used
    return st


def _increase_same_side(
    kernel: Any,
    *,
    target_quantity: float,
    execution_price: float,
) -> tuple[bool, dict[str, object]]:
    st = kernel.state
    current = float(st.position)
    if current * target_quantity <= 0.0 or abs(target_quantity) <= abs(current) + EPS:
        raise RuntimeError("ACPHY_INCREASE_SHAPE_INVALID")

    delta = target_quantity - current
    is_buy = delta > 0.0
    fill = float(kernel._effective_price(execution_price, is_buy=is_buy))
    delta_abs = abs(delta)
    fee = delta_abs * fill * float(kernel.config.fee_rate)
    added_margin = float(kernel._initial_margin_for_notional(delta_abs * fill))
    projected = _project_increase_state(
        kernel,
        target_quantity=target_quantity,
        fill=fill,
        fee=fee,
        added_margin=added_margin,
    )

    available = (
        float(projected.cash)
        if kernel.config.margin_type == "isolated"
        else float(projected.raw_equity() - projected.margin_used)
    )
    if available < -1e-8:
        return False, _leg_payload(
            kind=RESIZE_INCREASE,
            delta_quantity=0.0,
            fill_price=fill,
            fee=0.0,
            margin_change=0.0,
            realized_pnl=0.0,
            status=REJECTED_CAPITAL,
            reason="insufficient_margin",
        )
    maintenance = float(
        kernel._maintenance_margin_for_notional(projected.position * fill)
    )
    if float(kernel._margin_collateral(projected)) <= maintenance:
        return False, _leg_payload(
            kind=RESIZE_INCREASE,
            delta_quantity=0.0,
            fill_price=fill,
            fee=0.0,
            margin_change=0.0,
            realized_pnl=0.0,
            status=REJECTED_CAPITAL,
            reason="maintenance_margin_violation",
        )
    liquidation = kernel._liquidation_price_for(projected)
    stop = float(st.current_stop_price)
    if liquidation is not None and stop > 0.0:
        if projected.position > 0.0 and stop <= liquidation:
            return False, _leg_payload(
                kind=RESIZE_INCREASE,
                delta_quantity=0.0,
                fill_price=fill,
                fee=0.0,
                margin_change=0.0,
                realized_pnl=0.0,
                status=REJECTED_CAPITAL,
                reason="stop_beyond_liquidation",
            )
        if projected.position < 0.0 and stop >= liquidation:
            return False, _leg_payload(
                kind=RESIZE_INCREASE,
                delta_quantity=0.0,
                fill_price=fill,
                fee=0.0,
                margin_change=0.0,
                realized_pnl=0.0,
                status=REJECTED_CAPITAL,
                reason="stop_beyond_liquidation",
            )

    old_abs = abs(current)
    new_abs = abs(target_quantity)
    st.avg_entry_price = (
        old_abs * float(st.avg_entry_price) + delta_abs * fill
    ) / new_abs
    st.cash -= fee + added_margin
    st.margin_used += added_margin
    st.position = target_quantity
    st.last_mark_price = execution_price
    st.turnover_notional += delta_abs * fill
    if kernel.config.margin_type == "isolated":
        st.isolated_margin = st.margin_used
        st.isolated_wallet = st.margin_used
    equity = float(st.equity())
    st.last_executed_target = (
        0.0 if equity <= 0.0 else target_quantity * execution_price / equity
    )
    kernel._refresh_account_fields()
    return True, _leg_payload(
        kind=RESIZE_INCREASE,
        delta_quantity=delta,
        fill_price=fill,
        fee=fee,
        margin_change=added_margin,
        realized_pnl=0.0,
    )


def _full_close(
    kernel: Any,
    *,
    execution_price: float,
    reason: str,
    leg_kind: str,
) -> dict[str, object]:
    st = kernel.state
    current = float(st.position)
    if abs(current) <= EPS:
        return _leg_payload(
            kind=leg_kind,
            delta_quantity=0.0,
            fill_price=None,
            fee=0.0,
            margin_change=0.0,
            realized_pnl=0.0,
            status=NOOP,
        )
    qty = abs(current)
    is_long = current > 0.0
    fill = float(kernel._effective_price(execution_price, is_buy=not is_long))
    realized = (
        (fill - float(st.avg_entry_price)) * qty
        if is_long
        else (float(st.avg_entry_price) - fill) * qty
    )
    fee = qty * fill * float(kernel.config.fee_rate)
    released_margin = float(st.margin_used)
    kernel._force_close_at(execution_price, reason)
    kernel._refresh_account_fields()
    return _leg_payload(
        kind=leg_kind,
        delta_quantity=-current,
        fill_price=fill,
        fee=fee,
        margin_change=-released_margin,
        realized_pnl=realized,
    )


def execute_target_position_r0(
    runtime: FrozenPhysicsRuntimeR102,
    snapshot_t: Mapping[str, Any],
    permission: SupervisorPermissionResultR0,
    target: TargetExposureResultR0,
    *,
    execution_price: float,
    execution_record_id: str,
    permission_record_id: str,
) -> ActorCriticPhysicsAdapterResultR0:
    """Execute the smallest target-position delta without advancing a market bar."""

    if not isinstance(runtime, FrozenPhysicsRuntimeR102):
        raise RuntimeError("ACPHY_RUNTIME_TYPE_INVALID")
    _ensure_live_snapshot(snapshot_t)
    _validate_binding(permission, target)
    price = _finite_positive(execution_price, code="ACPHY_EXECUTION_PRICE_INVALID")
    _require_nonempty_string(execution_record_id, code="ACPHY_EXECUTION_RECORD_ID_INVALID")
    _require_nonempty_string(permission_record_id, code="ACPHY_PERMISSION_RECORD_ID_INVALID")

    before_hash = _snapshot_hash(runtime, snapshot_t)
    kernel = runtime.physics.restore_kernel(snapshot_t, runtime.physics_contract)
    support = copy.deepcopy(snapshot_t["observation_support_state"])
    current_quantity = _finite_number(kernel.state.position, code="ACPHY_CURRENT_QUANTITY_INVALID")
    target_quantity = _finite_number(target.target_quantity, code="ACPHY_TARGET_QUANTITY_INVALID")

    contract = permission.execution_contract
    contract.validate()
    source_direction = _position_direction(current_quantity)
    if contract.source_direction != source_direction:
        raise RuntimeError("ACPHY_EXECUTION_SOURCE_DIRECTION_MISMATCH")

    legs: list[dict[str, object]] = []
    status = EXECUTED
    target_reached = False

    if contract.contract_kind == REVERSAL_CLOSE_THEN_OPEN_R0:
        if source_direction == FLAT:
            raise RuntimeError("ACPHY_REVERSAL_REQUIRES_POSITION")
        legs.append(
            _full_close(
                kernel,
                execution_price=price,
                reason="actor_target_reversal",
                leg_kind=REVERSAL_CLOSE,
            )
        )
        opened, open_leg = _open_from_flat(
            kernel,
            target_quantity=target_quantity,
            execution_price=price,
            leg_kind=REVERSAL_OPEN,
        )
        legs.append(open_leg)
        if not opened:
            status = REVERSAL_CLOSED_OPEN_REJECTED
            target_reached = abs(target_quantity) <= EPS
        else:
            target_reached = _same_quantity(float(kernel.state.position), target_quantity)
    elif contract.contract_kind == DIRECT_TARGET_EXECUTION_R0:
        if _same_quantity(current_quantity, target_quantity):
            status = NOOP
            target_reached = True
            legs.append(
                _leg_payload(
                    kind=NOOP_LEG,
                    delta_quantity=0.0,
                    fill_price=None,
                    fee=0.0,
                    margin_change=0.0,
                    realized_pnl=0.0,
                    status=NOOP,
                )
            )
        elif abs(current_quantity) <= EPS:
            opened, leg = _open_from_flat(
                kernel,
                target_quantity=target_quantity,
                execution_price=price,
                leg_kind=OPEN_FROM_FLAT,
            )
            legs.append(leg)
            if not opened:
                status = REJECTED_CAPITAL
            target_reached = opened and _same_quantity(
                float(kernel.state.position), target_quantity
            )
        elif abs(target_quantity) <= EPS:
            legs.append(
                _full_close(
                    kernel,
                    execution_price=price,
                    reason="actor_target_close",
                    leg_kind=CLOSE_TO_FLAT,
                )
            )
            target_reached = abs(float(kernel.state.position)) <= EPS
        elif current_quantity * target_quantity < 0.0:
            raise RuntimeError("ACPHY_IMPLICIT_REVERSAL_FORBIDDEN")
        elif abs(target_quantity) < abs(current_quantity):
            legs.append(
                _partial_reduce(
                    kernel,
                    target_quantity=target_quantity,
                    execution_price=price,
                )
            )
            target_reached = _same_quantity(float(kernel.state.position), target_quantity)
        else:
            increased, leg = _increase_same_side(
                kernel,
                target_quantity=target_quantity,
                execution_price=price,
            )
            legs.append(leg)
            if not increased:
                status = REJECTED_CAPITAL
            target_reached = increased and _same_quantity(
                float(kernel.state.position), target_quantity
            )
    else:
        raise RuntimeError("ACPHY_EXECUTION_CONTRACT_UNKNOWN")

    snapshot_t1 = runtime.physics.snapshot_kernel(
        kernel,
        runtime.physics_contract,
        support,
        truncated=False,
        truncation_reason=None,
    )
    after_hash = _snapshot_hash(runtime, snapshot_t1)
    if _snapshot_hash(runtime, snapshot_t) != before_hash:
        raise RuntimeError("ACPHY_INPUT_SNAPSHOT_MUTATED")

    final_quantity = float(snapshot_t1["kernel_state"]["position"])
    actual_delta = final_quantity - current_quantity
    payload = {
        "adapter_version": PHYSICS_ADAPTER_VERSION_R0,
        "status": status,
        "target_reached": target_reached,
        "permission_sha256": permission.permission_sha256,
        "target_exposure_sha256": target.semantic_sha256,
        "snapshot_t_sha256": before_hash,
        "snapshot_t1_sha256": after_hash,
        "execution_price": price,
        "current_quantity": current_quantity,
        "target_quantity": target_quantity,
        "requested_net_delta_quantity": target_quantity - current_quantity,
        "actual_net_delta_quantity": actual_delta,
        "legs": legs,
        "bar_lifecycle_advanced": False,
        "funding_owner": "FROZEN_PHYSICS_STEP_ACCOUNT",
        "liquidation_owner": "FROZEN_PHYSICS_STEP_ACCOUNT",
    }
    execution_record = make_executed_delta_record_r0(
        record_id=execution_record_id,
        nominal_action_id=permission.nominal_action_id,
        permission_record_id=permission_record_id,
        executed_delta_quantity=actual_delta,
        execution_payload=payload,
    )
    result = ActorCriticPhysicsAdapterResultR0(
        adapter_version=PHYSICS_ADAPTER_VERSION_R0,
        status=status,
        target_reached=target_reached,
        snapshot_t1=snapshot_t1,
        execution_record=execution_record,
    )
    result.validate()
    return result
