
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Sequence

import numpy as np

from .cc_fast_account_state_r0 import AccountStateSoA
from .cc_fast_wire_r0 import ExecutionLeg, FastEnvironmentTransition, semantic_sha256


@dataclass(frozen=True)
class AccountKernelConfig:
    fee_rate: float = 0.0004
    maintenance_margin_fraction: float = 0.005
    liquidation_fee_rate: float = 0.001

    def validate(self) -> "AccountKernelConfig":
        if not (0 <= self.fee_rate < 1):
            raise ValueError("FEE_RATE_INVALID")
        if not (0 <= self.maintenance_margin_fraction < 1):
            raise ValueError("MAINTENANCE_MARGIN_INVALID")
        if not (0 <= self.liquidation_fee_rate < 1):
            raise ValueError("LIQUIDATION_FEE_INVALID")
        return self


@dataclass(frozen=True)
class KernelStepResult:
    transition: FastEnvironmentTransition
    target_quantity: float
    post_quantity: float
    post_cash: float
    post_liability: float
    post_equity: float


def _truth_hash(*, quantity: float, avg_cost: float, cash: float, liability: float,
                decision_index: int, terminal: bool) -> str:
    return semantic_sha256({
        "quantity": float(quantity),
        "avg_cost": float(avg_cost),
        "cash": float(cash),
        "liability": float(liability),
        "decision_index": int(decision_index),
        "terminal": bool(terminal),
    })


def _realized_close_pnl(old_qty: float, close_qty_abs: float, old_avg_cost: float, price: float) -> float:
    if old_qty > 0:
        return close_qty_abs * (price - old_avg_cost)
    if old_qty < 0:
        return close_qty_abs * (old_avg_cost - price)
    return 0.0


def _apply_trade_scalar(
    *,
    quantity: float,
    avg_cost: float,
    cash: float,
    delta: float,
    price: float,
    fee_rate: float,
    leg_kind: str,
    sequence: int,
) -> tuple[float, float, float, float, ExecutionLeg]:
    if price <= 0 or not math.isfinite(price):
        raise ValueError("PRICE_INVALID")
    if not math.isfinite(delta):
        raise ValueError("DELTA_INVALID")
    old_qty = quantity
    fee = abs(delta * price) * fee_rate
    cash_after = cash - delta * price - fee
    realized = 0.0
    new_qty = old_qty + delta

    if old_qty == 0 or old_qty * delta > 0:
        if new_qty == 0:
            new_avg = 0.0
        else:
            old_notional_cost = abs(old_qty) * avg_cost
            add_notional_cost = abs(delta) * price
            new_avg = (old_notional_cost + add_notional_cost) / abs(new_qty)
    else:
        close_abs = min(abs(delta), abs(old_qty))
        realized = _realized_close_pnl(old_qty, close_abs, avg_cost, price)
        if new_qty == 0:
            new_avg = 0.0
        elif old_qty * new_qty > 0:
            new_avg = avg_cost
        else:
            new_avg = price

    leg = ExecutionLeg(
        sequence=sequence,
        quantity_delta=float(delta),
        price=float(price),
        fee=float(fee),
        kind=leg_kind,
    )
    return new_qty, new_avg, cash_after, realized, leg


def _target_legs(quantity: float, target_quantity: float) -> tuple[tuple[float, str], ...]:
    if target_quantity == quantity:
        return ()
    if quantity == 0:
        return ((target_quantity, "OPEN"),)
    if quantity * target_quantity < 0:
        return (
            (-quantity, "REVERSAL_CLOSE"),
            (target_quantity, "REVERSAL_OPEN"),
        )
    delta = target_quantity - quantity
    if target_quantity == 0:
        return ((delta, "CLOSE"),)
    if abs(target_quantity) < abs(quantity):
        return ((delta, "REDUCE"),)
    return ((delta, "OPEN"),)


def step_account_scalar(
    state: AccountStateSoA,
    i: int,
    *,
    account_lineage_id: str,
    environment_time_before_ns: int,
    environment_time_after_ns: int,
    mark_price_before: float,
    mark_price_after: float,
    target_quantity: float,
    funding_rate: float,
    policy_decision_ref: str,
    permitted_target_direction: str,
    permitted_target_risk: float,
    permission_status: str = "ALLOW",
    permission_reason: str = "SYNTHETIC_ALLOWED",
    boundary_type: str = "NONE",
    config: AccountKernelConfig = AccountKernelConfig(),
) -> KernelStepResult:
    config.validate()
    if state.terminal[i]:
        raise RuntimeError("ACCOUNT_ALREADY_TERMINAL")
    if environment_time_after_ns <= environment_time_before_ns:
        raise ValueError("TIME_NONMONOTONE")
    for p in (mark_price_before, mark_price_after):
        if not math.isfinite(p) or p <= 0:
            raise ValueError("MARK_PRICE_INVALID")
    if not math.isfinite(target_quantity):
        raise ValueError("TARGET_QUANTITY_INVALID")
    if not math.isfinite(funding_rate):
        raise ValueError("FUNDING_INVALID")

    old_qty = float(state.quantity[i])
    old_avg = float(state.avg_cost[i])
    old_cash = float(state.cash[i])
    old_liability = float(state.liability[i])
    old_equity = old_cash + old_qty * mark_price_before - old_liability
    pre_hash = _truth_hash(
        quantity=old_qty, avg_cost=old_avg, cash=old_cash, liability=old_liability,
        decision_index=int(state.decision_index[i]), terminal=False
    )

    qty, avg, cash = old_qty, old_avg, old_cash
    realized_total = 0.0
    fee_total = 0.0
    legs: list[ExecutionLeg] = []
    effective_target = old_qty if permission_status in {"REJECT", "NOOP"} else float(target_quantity)
    for seq, (delta, kind) in enumerate(_target_legs(qty, effective_target)):
        qty, avg, cash, realized, leg = _apply_trade_scalar(
            quantity=qty, avg_cost=avg, cash=cash, delta=delta,
            price=mark_price_before, fee_rate=config.fee_rate,
            leg_kind=kind, sequence=seq,
        )
        legs.append(leg)
        realized_total += realized
        fee_total += leg.fee

    funding_cash = qty * mark_price_after * float(funding_rate)
    cash -= funding_cash
    liability = old_liability
    equity_before_liq = cash + qty * mark_price_after - liability
    terminal = False

    maint = config.maintenance_margin_fraction * abs(qty * mark_price_after)
    if qty != 0 and equity_before_liq <= maint:
        seq = len(legs)
        liq_delta = -qty
        qty, avg, cash, realized, leg = _apply_trade_scalar(
            quantity=qty, avg_cost=avg, cash=cash, delta=liq_delta,
            price=mark_price_after, fee_rate=config.liquidation_fee_rate,
            leg_kind="LIQUIDATION", sequence=seq,
        )
        legs.append(leg)
        realized_total += realized
        fee_total += leg.fee
        terminal = True
        boundary_type = "ECONOMIC_TERMINAL"

    post_equity = cash + qty * mark_price_after - liability
    unrealized_before = old_qty * (mark_price_before - old_avg) if old_qty != 0 else 0.0
    unrealized_after = qty * (mark_price_after - avg) if qty != 0 else 0.0
    unrealized_delta = unrealized_after - unrealized_before
    next_decision_index = int(state.decision_index[i]) + 1

    state.quantity[i] = qty
    state.avg_cost[i] = avg
    state.cash[i] = cash
    state.liability[i] = liability
    state.equity[i] = post_equity
    state.realized_pnl[i] += realized_total
    state.fees_paid[i] += fee_total
    state.funding_paid[i] += funding_cash
    state.decision_index[i] = next_decision_index
    state.terminal[i] = terminal
    post_hash = _truth_hash(
        quantity=qty, avg_cost=avg, cash=cash, liability=liability,
        decision_index=next_decision_index, terminal=terminal
    )

    transition = FastEnvironmentTransition(
        account_lineage_id=account_lineage_id,
        decision_index=next_decision_index - 1,
        environment_time_before_ns=environment_time_before_ns,
        environment_time_after_ns=environment_time_after_ns,
        pre_account_truth_hash=pre_hash,
        policy_decision_ref=policy_decision_ref,
        permission_status=permission_status,
        permission_reason=permission_reason,
        permitted_target_direction=permitted_target_direction,
        permitted_target_risk=float(permitted_target_risk),
        target_quantity=float(effective_target),
        execution_legs=tuple(legs),
        fees=float(fee_total),
        funding=float(funding_cash),
        realized_pnl=float(realized_total),
        unrealized_pnl_delta=float(unrealized_delta),
        liability_delta=float(liability - old_liability),
        post_account_truth_hash=post_hash,
        post_equity=float(post_equity),
        boundary_type=boundary_type,
        mechanical_terminal=terminal,
        external_capital_flow_ref_or_null=None,
    ).validate()

    return KernelStepResult(
        transition=transition,
        target_quantity=effective_target,
        post_quantity=qty,
        post_cash=cash,
        post_liability=liability,
        post_equity=post_equity,
    )


def target_quantity_from_risk(direction: str, risk: float, equity: float, mark_price: float) -> float:
    if direction == "FLAT":
        return 0.0
    if direction not in {"SHORT", "LONG"}:
        raise ValueError("DIRECTION_INVALID")
    if not (0.0 < risk < 1.0):
        raise ValueError("NONFLAT_RISK_MUST_BE_INTERIOR")
    if mark_price <= 0 or not math.isfinite(mark_price):
        raise ValueError("MARK_PRICE_INVALID")
    sign = 1.0 if direction == "LONG" else -1.0
    return sign * (risk * equity / mark_price)
