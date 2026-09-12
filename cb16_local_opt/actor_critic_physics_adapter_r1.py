from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Mapping

from .account_economics_r0 import AccountEconomicsStateR0
from .action_contract_r1 import FLAT, LONG, SHORT, TargetPositionActionR1, make_target_position_action_r1
from .actor_critic_environment_profile_r1 import ActorCriticEnvironmentProfileR1
from .actor_critic_supervisor_r1 import REJECT, SupervisorAuthorityStateR1, supervise_target_action_r1
from .execution_feasibility_r0 import FEASIBLE, MechanicalExecutionAuthorityR0, evaluate_execution_feasibility_r0
from .target_exposure_r1 import make_target_exposure_authority_r1, map_action_to_target_exposure_r1

PHYSICS_ADAPTER_SCHEMA_R1 = "CB16_R11_BC_TARGET_POSITION_EXECUTION_ADAPTER_V1_R1"
EXECUTED = "EXECUTED"
EXECUTION_REJECTED = "EXECUTION_REJECTED"
NO_POSITION_CHANGE = "NO_POSITION_CHANGE"
REVERSAL_OPEN_REJECTED = "REVERSAL_OPEN_REJECTED"


@dataclass(frozen=True)
class Round2MechanicalExecutionConfigR1:
    fee_rate: float
    slippage_bps: float
    initial_margin_rate: float
    maintenance_margin_rate: float
    max_gross_leverage: float
    declared_max_legal_notional: float | None = None
    lot_step_size: float | None = None
    lot_min_qty: float | None = None
    lot_max_qty: float | None = None
    min_notional: float | None = None

    def validate(self) -> None:
        if not 0.0 <= self.fee_rate < 1.0:
            raise RuntimeError("ACPHY_R1_FEE_RATE_INVALID")
        if self.slippage_bps < 0.0:
            raise RuntimeError("ACPHY_R1_SLIPPAGE_INVALID")
        if not 0.0 < self.initial_margin_rate <= 1.0:
            raise RuntimeError("ACPHY_R1_INITIAL_MARGIN_INVALID")
        if not 0.0 <= self.maintenance_margin_rate <= 1.0:
            raise RuntimeError("ACPHY_R1_MAINTENANCE_MARGIN_INVALID")
        if self.max_gross_leverage <= 0.0:
            raise RuntimeError("ACPHY_R1_LEVERAGE_INVALID")


@dataclass(frozen=True)
class TargetExecutionReceiptR1:
    schema_version: str
    status: str
    transition_kind: str
    requested_direction: str
    requested_risk: float
    permitted_direction: str
    permitted_risk: float
    target_quantity: float
    delta_quantity: float
    fill_price: float | None
    fee_paid: float
    realized_pnl: float
    account_before_sha256: str
    account_after_sha256: str
    account_after: AccountEconomicsStateR0
    feasibility_status: str
    reason_codes: tuple[str, ...]
    second_leg_status: str | None = None


def _state_sha(state: AccountEconomicsStateR0) -> str:
    state.validate()
    payload = asdict(state)
    payload["unrealized_pnl"] = state.unrealized_pnl
    payload["equity"] = state.equity
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _fill_price(mark: float, delta_quantity: float, slippage_bps: float) -> float:
    if delta_quantity == 0.0:
        return mark
    direction = 1.0 if delta_quantity > 0.0 else -1.0
    return mark * (1.0 + direction * slippage_bps / 10000.0)


def _available_new_margin(state: AccountEconomicsStateR0) -> float:
    # Resource capacity may floor at zero; the authoritative cash/equity ledger is never mutated/clamped.
    return max(0.0, state.cash)


def _sizing_target(
    *,
    state: AccountEconomicsStateR0,
    action: TargetPositionActionR1,
    mark_price: float,
    legal_envelope_id: str,
    config: Round2MechanicalExecutionConfigR1,
):
    total_margin_capacity = state.margin_collateral + _available_new_margin(state)
    authority = make_target_exposure_authority_r1(
        authority_id="BC_ROUND2_R1_DYNAMIC_SIZING",
        account_id=state.account_id,
        account_state_sha256=_state_sha(state),
        legal_envelope_id=legal_envelope_id,
        equity=max(0.0, state.equity),
        current_price=mark_price,
        margin_capacity=total_margin_capacity,
        max_gross_leverage=config.max_gross_leverage,
        initial_margin_rate=config.initial_margin_rate,
        declared_max_legal_notional=config.declared_max_legal_notional,
        lot_step_size=config.lot_step_size,
        lot_min_qty=config.lot_min_qty,
        lot_max_qty=config.lot_max_qty,
        min_notional=config.min_notional,
    )
    return map_action_to_target_exposure_r1(action, authority)


def _feasibility(state: AccountEconomicsStateR0, target, mark_price: float, config: Round2MechanicalExecutionConfigR1):
    return evaluate_execution_feasibility_r0(
        target,
        MechanicalExecutionAuthorityR0(
            authority_id="BC_ROUND2_R1_MECHANICAL_FEASIBILITY",
            current_quantity=state.position_quantity,
            current_price=mark_price,
            available_margin_for_new_exposure=_available_new_margin(state),
            initial_margin_rate=config.initial_margin_rate,
            maintenance_margin_rate=config.maintenance_margin_rate,
            maintenance_collateral=state.cash + state.margin_collateral,
            lot_min_qty=config.lot_min_qty,
            lot_max_qty=config.lot_max_qty,
            min_notional=config.min_notional,
        ),
    )


def _apply_same_side_delta(state: AccountEconomicsStateR0, target_quantity: float, mark_price: float, config: Round2MechanicalExecutionConfigR1):
    current = state.position_quantity
    delta = target_quantity - current
    if abs(delta) <= 1e-12:
        return state, mark_price, 0.0, 0.0, "NOOP"

    fill = _fill_price(mark_price, delta, config.slippage_bps)
    fee = abs(delta) * fill * config.fee_rate
    same_side = current == 0.0 or target_quantity == 0.0 or current * target_quantity > 0.0
    if not same_side:
        raise RuntimeError("ACPHY_R1_REVERSAL_REQUIRES_TWO_PHASE")

    if current == 0.0 or abs(target_quantity) > abs(current):
        added_margin = abs(delta) * fill * config.initial_margin_rate
        old_abs = abs(current)
        new_abs = abs(target_quantity)
        if old_abs == 0.0:
            new_basis = fill if new_abs > 0.0 else 0.0
        else:
            new_basis = (old_abs * state.position_cost_basis + abs(delta) * fill) / new_abs
        out = replace(
            state,
            cash=state.cash - added_margin - fee,
            position_quantity=target_quantity,
            position_cost_basis=new_basis,
            mark_price=mark_price,
            fees_cumulative=state.fees_cumulative + fee,
            margin_collateral=state.margin_collateral + added_margin,
        )
        out.validate()
        return out, fill, fee, 0.0, "OPEN_OR_INCREASE"

    close_qty = abs(delta)
    sign = 1.0 if current > 0.0 else -1.0
    realized = close_qty * (fill - state.position_cost_basis) * sign
    fraction = close_qty / abs(current)
    released_margin = state.margin_collateral * fraction
    remaining = target_quantity
    out = replace(
        state,
        cash=state.cash + released_margin + realized - fee,
        position_quantity=remaining,
        position_cost_basis=(0.0 if abs(remaining) <= 1e-12 else state.position_cost_basis),
        mark_price=mark_price,
        realized_pnl_cumulative=state.realized_pnl_cumulative + realized,
        fees_cumulative=state.fees_cumulative + fee,
        margin_collateral=state.margin_collateral - released_margin,
    )
    out.validate()
    return out, fill, fee, realized, ("CLOSE" if abs(remaining) <= 1e-12 else "REDUCE")


def execute_target_position_r1(
    *,
    account: AccountEconomicsStateR0,
    action: TargetPositionActionR1,
    supervisor_authority: SupervisorAuthorityStateR1,
    profile: ActorCriticEnvironmentProfileR1,
    taxonomy: Mapping[str, object],
    mark_price: float,
    legal_envelope_id: str,
    config: Round2MechanicalExecutionConfigR1,
) -> TargetExecutionReceiptR1:
    account.validate()
    action.validate()
    supervisor_authority.validate()
    profile.validate(taxonomy)
    config.validate()
    if any(profile.is_enabled(rule) for rule in ("stop_loss", "take_profit", "max_hold", "cooldown", "finalize_behavior")):
        raise RuntimeError("ACPHY_R1_STRATEGY_EXIT_RULE_FORBIDDEN")
    if supervisor_authority.account_id != account.account_id:
        raise RuntimeError("ACPHY_R1_ACCOUNT_ID_MISMATCH")
    before_sha = _state_sha(account)
    permission = supervise_target_action_r1(action, supervisor_authority)
    if permission.outcome == REJECT:
        return TargetExecutionReceiptR1(
            PHYSICS_ADAPTER_SCHEMA_R1, EXECUTION_REJECTED, "SUPERVISOR_REJECT", action.target_direction,
            action.requested_target_risk, permission.permitted_target_direction, permission.permitted_target_risk,
            account.position_quantity, 0.0, None, 0.0, 0.0, before_sha, before_sha, account,
            "NOT_EVALUATED", permission.reason_codes,
        )

    effective_action = make_target_position_action_r1(
        action_id=action.action_id,
        policy_id=action.policy_id,
        policy_version=action.policy_version,
        target_direction=permission.permitted_target_direction,
        requested_target_risk=permission.permitted_target_risk,
    )
    target = _sizing_target(
        state=account,
        action=effective_action,
        mark_price=mark_price,
        legal_envelope_id=legal_envelope_id,
        config=config,
    )

    current = account.position_quantity
    reversal = current != 0.0 and target.target_quantity != 0.0 and current * target.target_quantity < 0.0
    if reversal:
        flat_state, close_fill, close_fee, close_realized, _ = _apply_same_side_delta(account, 0.0, mark_price, config)
        second_target = _sizing_target(
            state=flat_state,
            action=effective_action,
            mark_price=mark_price,
            legal_envelope_id=legal_envelope_id,
            config=config,
        )
        second_feas = _feasibility(flat_state, second_target, mark_price, config)
        if second_feas.status != FEASIBLE:
            after_sha = _state_sha(flat_state)
            return TargetExecutionReceiptR1(
                PHYSICS_ADAPTER_SCHEMA_R1, REVERSAL_OPEN_REJECTED, "REVERSAL_CLOSE_THEN_OPEN",
                action.target_direction, action.requested_target_risk, permission.permitted_target_direction,
                permission.permitted_target_risk, 0.0, -current, close_fill, close_fee, close_realized,
                before_sha, after_sha, flat_state, second_feas.status,
                permission.reason_codes + second_feas.reason_codes, second_leg_status=second_feas.status,
            )
        opened, open_fill, open_fee, _, _ = _apply_same_side_delta(flat_state, second_target.target_quantity, mark_price, config)
        return TargetExecutionReceiptR1(
            PHYSICS_ADAPTER_SCHEMA_R1, EXECUTED, "REVERSAL_CLOSE_THEN_OPEN", action.target_direction,
            action.requested_target_risk, permission.permitted_target_direction, permission.permitted_target_risk,
            second_target.target_quantity, second_target.target_quantity - current, open_fill,
            close_fee + open_fee, close_realized, before_sha, _state_sha(opened), opened, FEASIBLE,
            permission.reason_codes, second_leg_status=FEASIBLE,
        )

    feas = _feasibility(account, target, mark_price, config)
    if feas.status != FEASIBLE:
        return TargetExecutionReceiptR1(
            PHYSICS_ADAPTER_SCHEMA_R1, EXECUTION_REJECTED, "DIRECT_TARGET", action.target_direction,
            action.requested_target_risk, permission.permitted_target_direction, permission.permitted_target_risk,
            target.target_quantity, target.target_quantity - current, None, 0.0, 0.0, before_sha, before_sha,
            account, feas.status, permission.reason_codes + feas.reason_codes,
        )

    after, fill, fee, realized, transition = _apply_same_side_delta(account, target.target_quantity, mark_price, config)
    status = NO_POSITION_CHANGE if transition == "NOOP" else EXECUTED
    return TargetExecutionReceiptR1(
        PHYSICS_ADAPTER_SCHEMA_R1, status, transition, action.target_direction, action.requested_target_risk,
        permission.permitted_target_direction, permission.permitted_target_risk, target.target_quantity,
        target.target_quantity - current, (None if transition == "NOOP" else fill), fee, realized,
        before_sha, _state_sha(after), after, feas.status, permission.reason_codes,
    )
