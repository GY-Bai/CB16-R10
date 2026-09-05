"""Scalar reference implementation of the v5 trading laws.

The policy has exactly three choices while flat: short, wait, or long.  The
environment owns position size and every exit.  A position cannot be resized,
manually closed, or reversed by the policy.
"""
# V5.5 FROZEN KERNEL: numerical semantics must remain parity-locked to audited V5.4.
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

from .actions import TradeDecision, decision_sign, validate_decision
from .bar import CanonicalBar
from .reward import log_equity_reward
from .sim_config import SimConfig
from .state import AccountState


@dataclass(frozen=True)
class _ExecutionPlan:
    target_exposure: float
    target_qty: float
    delta_qty: float
    fill_price: float


@dataclass
class StepResult:
    state: Dict[str, Any]
    step_reward: float
    carry_cost: float
    done: bool
    info: Dict[str, Any]


class FrozenTradingKernel:
    """Single-account, next-bar-open market simulator."""

    def __init__(self, config: Optional[SimConfig] = None) -> None:
        self.config = config or SimConfig()
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> AccountState:
        with self._lock:
            self.state = AccountState(
                cash=self.config.initial_cash,
                margin_type=self.config.margin_type,
            )
            self.state.peak_equity = self.config.initial_cash
            self._last_bar_time: Optional[datetime] = None
            self._symbol: Optional[str] = None
            self._last_carry_cost = 0.0
            return self.state

    def warmup_market(self, bars: Sequence[CanonicalBar]) -> None:
        """Prime ATR from already-visible history without advancing the account."""
        with self._lock:
            for bar in bars:
                self._validate_next_bar(bar)
                self._update_atr(bar)
                self._last_bar_time = bar.bar_start

    @staticmethod
    def action_space() -> Dict[str, Any]:
        return {"decision": [int(x) for x in TradeDecision]}

    def lint_action(self, action: Dict[str, Any]) -> list[str]:
        if not isinstance(action, dict) or "decision" not in action:
            return ["missing_decision"]
        try:
            decision = validate_decision(action["decision"])
        except ValueError:
            return ["decision_out_of_range"]
        if decision is TradeDecision.FLAT:
            return []
        if "risk_multiplier" in action:
            try:
                risk_multiplier = float(action["risk_multiplier"])
            except (TypeError, ValueError):
                return ["risk_multiplier_not_numeric"]
            if not math.isfinite(risk_multiplier):
                return ["risk_multiplier_not_finite"]
            if not 0.0 <= risk_multiplier <= 1.0:
                return ["risk_multiplier_out_of_range"]
        return []

    def _validate_next_bar(self, bar: CanonicalBar) -> None:
        if self._last_bar_time is not None and bar.bar_start <= self._last_bar_time:
            raise ValueError("bars must be strictly increasing by bar_start")
        if self._symbol is None:
            self._symbol = bar.symbol
        elif bar.symbol != self._symbol:
            raise ValueError("one FrozenTradingKernel may contain only one symbol")

    # ------------------------------------------------------------------ #
    # Pricing, risk sizing, and entry execution
    # ------------------------------------------------------------------ #
    def _effective_price(self, price: float, is_buy: bool) -> float:
        slip = price * self.config.slippage_bps / 10_000.0
        effective = price + slip if is_buy else price - slip
        tick = self.config.tick_size
        if tick is None:
            return effective
        units = effective / tick
        return (math.ceil(units - 1e-12) if is_buy else math.floor(units + 1e-12)) * tick

    def _initial_margin_for_notional(self, notional: float) -> float:
        rate = self.config.initial_margin_rate
        if rate is None:
            rate = 1.0 / self.config.max_leverage
        return abs(notional) * rate

    def _maintenance_margin_for_notional(self, notional: float) -> float:
        return abs(notional) * self.config.maintenance_margin_rate

    def _margin_collateral(self, state: Optional[AccountState] = None) -> float:
        st = self.state if state is None else state
        if self.config.margin_type == "isolated":
            return st.isolated_wallet + st.unrealized_pnl()
        return st.raw_equity()

    def _quantize_qty_toward_zero(self, qty: float) -> float:
        step = self.config.lot_step_size
        if step is None:
            return qty
        units = math.floor(abs(qty) / step + 1e-12)
        return 0.0 if units == 0 else math.copysign(units * step, qty)

    def _entry_risk_prices(self, fill_price: float, side: int) -> Tuple[float, float, float]:
        atr = self.state.atr
        if atr <= 0:
            raise RuntimeError("ATR must be primed before an entry decision")
        stop_distance = self.config.stop_atr_mult * atr / fill_price
        stop = fill_price * (1.0 - side * stop_distance)
        take = fill_price * (1.0 + side * self.config.min_risk_reward * stop_distance)
        return stop, take, stop_distance

    def _build_entry_plan(
        self,
        decision: TradeDecision,
        bar: CanonicalBar,
        risk_multiplier: float = 1.0,
    ) -> Tuple[Optional[_ExecutionPlan], Optional[str]]:
        side = decision_sign(decision)
        if side == 0:
            return None, None
        st = self.state
        fill = self._effective_price(bar.open, is_buy=side > 0)
        if fill <= 0:
            return None, "fill_price_not_positive"
        try:
            stop, take, stop_distance = self._entry_risk_prices(fill, side)
        except RuntimeError:
            return None, "atr_not_primed"
        if stop <= 0 or take <= 0 or stop_distance <= 0:
            return None, "invalid_protective_prices"

        equity = st.equity()
        base_exposure = min(
            self.config.max_leverage,
            self.config.risk_fraction_per_trade / stop_distance,
        )
        desired_exposure = float(risk_multiplier) * base_exposure
        margin_rate = self.config.initial_margin_rate
        if margin_rate is None:
            margin_rate = 1.0 / self.config.max_leverage
        slippage_loss_rate = abs(fill - bar.open) / fill
        all_in_rate = margin_rate + self.config.fee_rate + slippage_loss_rate
        feasible_notional = equity / all_in_rate if all_in_rate > 0 else math.inf
        desired_notional = min(equity * desired_exposure, feasible_notional * (1.0 - 1e-12))
        target_qty = self._quantize_qty_toward_zero(side * desired_notional / fill)
        if abs(target_qty) < 1e-12:
            return None, "quantity_quantized_to_zero"
        if self.config.lot_min_qty is not None and abs(target_qty) < self.config.lot_min_qty - 1e-12:
            return None, "target_quantity_below_min"
        if self.config.lot_max_qty is not None and abs(target_qty) > self.config.lot_max_qty + 1e-12:
            return None, "target_quantity_above_max"
        notional = abs(target_qty) * fill
        if self.config.min_notional is not None and notional < self.config.min_notional - 1e-9:
            return None, "order_notional_below_min"
        target_exposure = side * notional / equity
        plan = _ExecutionPlan(target_exposure, target_qty, target_qty, fill)
        capital_error = self._capital_check(plan, stop)
        return (None, capital_error) if capital_error else (plan, None)

    def _capital_check(self, plan: _ExecutionPlan, stop_price: float) -> Optional[str]:
        projected = replace(self.state)
        self._apply_entry_plan(plan, projected, record_metrics=False)
        available = (
            projected.cash
            if self.config.margin_type == "isolated"
            else projected.raw_equity() - projected.margin_used
        )
        if available < -1e-8:
            return "insufficient_margin"
        maintenance = self._maintenance_margin_for_notional(
            projected.position * plan.fill_price
        )
        if self._margin_collateral(projected) <= maintenance:
            return "maintenance_margin_violation"
        liquidation = self._liquidation_price_for(projected)
        if liquidation is not None:
            if projected.position > 0 and stop_price <= liquidation:
                return "stop_beyond_liquidation"
            if projected.position < 0 and stop_price >= liquidation:
                return "stop_beyond_liquidation"
        return None

    def _apply_entry_plan(
        self,
        plan: _ExecutionPlan,
        state: Optional[AccountState] = None,
        *,
        record_metrics: bool = True,
    ) -> None:
        st = self.state if state is None else state
        equity_before = st.equity()
        fee = abs(plan.delta_qty) * plan.fill_price * self.config.fee_rate
        margin = self._initial_margin_for_notional(abs(plan.target_qty) * plan.fill_price)
        st.cash -= fee + margin
        st.margin_used = margin
        st.initial_margin = margin
        if self.config.margin_type == "isolated":
            st.isolated_margin = margin
            st.isolated_wallet = margin
        st.position = plan.target_qty
        st.avg_entry_price = plan.fill_price
        st.last_executed_target = plan.target_exposure
        if not record_metrics:
            return
        st.turnover_notional += abs(plan.target_qty) * plan.fill_price
        st.entry_price = plan.fill_price
        st.entry_equity = equity_before
        st.max_favorable_price = plan.fill_price
        st.max_adverse_price = plan.fill_price
        st.position_age_bars = 0
        st.trade_realized_pnl_start = st.realized_pnl
        self._set_entry_risk(st, plan.fill_price)
        st.trade_phase = "LONG" if st.position > 0 else "SHORT"
        if st.position > 0:
            st.long_entry_count += 1
        else:
            st.short_entry_count += 1

    def _set_entry_risk(self, st: AccountState, price: float) -> None:
        side = 1 if st.position > 0 else -1
        stop, take, distance = self._entry_risk_prices(price, side)
        st.initial_stop_price = stop
        st.initial_take_price = take
        st.current_stop_price = stop
        st.current_take_price = take
        st.initial_risk_pct = distance
        st.initial_entry_qty = abs(st.position)
        st.initial_risk_notional = abs(st.position * (price - stop))

    @staticmethod
    def _clear_trade_risk(st: AccountState) -> None:
        st.initial_stop_price = 0.0
        st.initial_take_price = 0.0
        st.current_stop_price = 0.0
        st.current_take_price = 0.0
        st.initial_risk_pct = 0.0
        st.initial_risk_notional = 0.0
        st.initial_entry_qty = 0.0

    # ------------------------------------------------------------------ #
    # Step lifecycle
    # ------------------------------------------------------------------ #
    def step(
        self,
        bar: CanonicalBar,
        action: Dict[str, Any],
        funding_rate: float = 0.0,
    ) -> StepResult:
        with self._lock:
            return self._step_unlocked(bar, action, funding_rate)

    def finalize(self, price: Optional[float] = None) -> float:
        """Liquidate an unfinished evaluation position and return its log reward."""
        with self._lock:
            st = self.state
            previous = st.equity()
            if abs(st.position) < 1e-12:
                return 0.0
            close_price = st.last_mark_price if price is None else float(price)
            if not math.isfinite(close_price) or close_price <= 0:
                raise ValueError("finalization price must be finite and positive")
            self._force_close_at(close_price, "evaluation_end")
            self._refresh_account_fields()
            st.update_peak()
            return log_equity_reward(previous, st.equity(), self.config.initial_cash)

    def _step_unlocked(
        self,
        bar: CanonicalBar,
        action: Dict[str, Any],
        funding_rate: float,
    ) -> StepResult:
        st = self.state
        if st.terminal:
            return StepResult(
                state=st.to_dict(), step_reward=0.0, carry_cost=0.0, done=True,
                info={
                    "accepted": False,
                    "errors": ["terminal_state"],
                    "equity": st.equity(),
                    "policy_action_mask": False,
                },
            )
        self._validate_next_bar(bar)
        if not math.isfinite(float(funding_rate)):
            raise ValueError("funding_rate must be finite")

        st.steps_survived += 1
        prev_equity = st.equity()
        self._last_carry_cost = 0.0
        errors = self.lint_action(action)
        decision = TradeDecision.FLAT
        risk_multiplier = 1.0
        if not errors:
            decision = validate_decision(action["decision"])
            risk_multiplier = float(action.get("risk_multiplier", 1.0))
        policy_action_mask = (
            abs(st.position) < 1e-12 and not self._cooldown_active()
        )
        info: Dict[str, Any] = {
            "policy_action_mask": policy_action_mask,
            "accepted": not errors,
            "errors": list(errors),
            "order_status": "wait",
        }
        if errors:
            st.invalid_actions += 1
            info["accepted"] = False
            info["order_status"] = "rejected"

        st.last_mark_price = bar.open
        was_positioned = abs(st.position) >= 1e-12
        liquidated_at_open = self._gap_liquidated(bar)
        automatic_reason: Optional[str] = None
        if not liquidated_at_open:
            automatic_reason = self._automatic_close_at_open(bar)
        if liquidated_at_open:
            info.update(accepted=False, errors=["liquidated_at_open"], order_status="liquidation")
        elif automatic_reason is not None:
            info["order_status"] = automatic_reason
            info["close_reason"] = automatic_reason
            if automatic_reason == "stop_loss":
                self._start_cooldown()
        elif was_positioned:
            info["order_status"] = "managed_position"
        elif self._cooldown_active():
            info["order_status"] = "cooldown"
        elif not errors and decision is not TradeDecision.FLAT and risk_multiplier <= 0.0:
            info.update(accepted=True, errors=[], order_status="zero_risk")
        elif not errors and decision is not TradeDecision.FLAT:
            plan, order_error = self._build_entry_plan(
                decision, bar, risk_multiplier=risk_multiplier
            )
            if order_error is not None:
                st.invalid_actions += 1
                info.update(accepted=False, errors=[order_error], order_status="rejected")
            elif plan is not None:
                self._apply_entry_plan(plan)
                info.update(
                    accepted=True,
                    errors=[],
                    order_status="execute",
                    fill_price=plan.fill_price,
                    executed_exposure=plan.target_exposure,
                    risk_multiplier=risk_multiplier,
                )

        if not st.terminal and st.raw_equity() <= 0:
            self._settle_insolvency_at(bar.open)
            info.update(accepted=False, errors=info["errors"] + ["insolvent_after_order"])

        if not st.terminal:
            close_reason = self._check_intrabar_sl_tp(bar)
            if close_reason == "stop_loss":
                self._start_cooldown()
        if not st.terminal:
            self._check_intrabar_liquidation(bar)

        if not st.terminal:
            st.last_mark_price = float(bar.mark_price)
            self._apply_funding(float(funding_rate))
            maintenance = self._maintenance_margin_for_notional(st.position * st.last_mark_price)
            if st.position != 0 and self._margin_collateral() <= maintenance:
                self._force_close_at(st.last_mark_price, reason="liquidation")
            else:
                st.update_peak()
            if not st.terminal and st.drawdown_from_peak() > self.config.max_drawdown:
                self._force_close_at(st.last_mark_price, reason="risk_limit")
                st.terminal = True
                st.terminal_reason = "max_drawdown"

        self._update_trade_memory_for_bar(bar)
        self._last_bar_time = bar.bar_start
        self._refresh_account_fields()
        st.update_peak()
        new_equity = st.equity()
        drawdown = st.drawdown_from_peak()
        st.max_observed_drawdown = max(st.max_observed_drawdown, drawdown)
        st.equity_sum += new_equity
        st.equity_observations += 1
        self._update_atr(bar)
        step_reward = log_equity_reward(prev_equity, new_equity, self.config.initial_cash)
        info.update(
            equity=new_equity,
            drawdown=drawdown,
            invalid_actions=st.invalid_actions,
            mark_price=st.last_mark_price,
            close_reason=st.last_close_reason,
            r_multiple=st.last_r_multiple,
            atr=st.atr,
            trade_phase=st.trade_phase,
        )
        return StepResult(st.to_dict(), step_reward, self._last_carry_cost, st.terminal, info)

    def _cooldown_active(self) -> bool:
        st = self.state
        return (
            abs(st.position) < 1e-12
            and st.stop_cooldown_until >= 0
            and st.steps_survived <= st.stop_cooldown_until
        )

    def _start_cooldown(self) -> None:
        self.state.stop_cooldown_until = (
            self.state.steps_survived + self.config.stop_cooldown_bars
        )

    def _update_trade_memory_for_bar(self, bar: CanonicalBar) -> None:
        st = self.state
        if st.position > 1e-12:
            st.trade_phase = "LONG"
            st.max_favorable_price = max(st.max_favorable_price, float(bar.high))
            st.max_adverse_price = min(st.max_adverse_price, float(bar.low))
            st.position_age_bars += 1
        elif st.position < -1e-12:
            st.trade_phase = "SHORT"
            st.max_favorable_price = min(st.max_favorable_price, float(bar.low))
            st.max_adverse_price = max(st.max_adverse_price, float(bar.high))
            st.position_age_bars += 1
        else:
            st.position_age_bars = 0
            next_step_is_blocked = (
                st.stop_cooldown_until >= 0
                and st.steps_survived + 1 <= st.stop_cooldown_until
            )
            st.trade_phase = "COOLDOWN" if next_step_is_blocked else "FLAT"

    def _update_atr(self, bar: CanonicalBar) -> None:
        st = self.state
        prev_close = st.prev_close if st.prev_close > 0 else bar.open
        tr = max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close))
        st.atr = tr if st.atr <= 0 else (
            st.atr * (self.config.atr_period - 1) + tr
        ) / self.config.atr_period
        st.prev_close = bar.close

    # ------------------------------------------------------------------ #
    # Funding, liquidation, and environment-owned exits
    # ------------------------------------------------------------------ #
    def _apply_funding(self, rate: float) -> None:
        st = self.state
        self._last_carry_cost = st.position * st.last_mark_price * rate
        if self.config.margin_type == "isolated" and st.position != 0:
            st.margin_used -= self._last_carry_cost
            st.isolated_margin = st.margin_used
            st.isolated_wallet = st.margin_used
        else:
            st.cash -= self._last_carry_cost

    def _refresh_account_fields(self) -> None:
        st = self.state
        st.initial_margin = self._initial_margin_for_notional(st.position * st.last_mark_price)
        st.maint_margin = self._maintenance_margin_for_notional(st.position * st.last_mark_price)
        if self.config.margin_type == "isolated":
            st.isolated_margin = st.margin_used
            st.isolated_wallet = st.margin_used
        else:
            st.isolated_margin = 0.0
            st.isolated_wallet = 0.0
        st.open_order_initial_margin = 0.0

    def _liquidation_price_for(self, st: AccountState) -> Optional[float]:
        if st.position == 0 or st.avg_entry_price <= 0:
            return None
        collateral_without_pnl = (
            st.isolated_wallet if self.config.margin_type == "isolated" else st.cash + st.margin_used
        )
        base = collateral_without_pnl - st.position * st.avg_entry_price
        denom = st.position - self.config.maintenance_margin_rate * abs(st.position)
        if abs(denom) < 1e-12:
            return None
        price = -base / denom
        return price if price > 0 and math.isfinite(price) else None

    def _liquidation_price(self) -> Optional[float]:
        return self._liquidation_price_for(self.state)

    def _gap_liquidated(self, bar: CanonicalBar) -> bool:
        st = self.state
        liq = self._liquidation_price()
        if liq is None:
            return False
        hit = (st.position > 0 and bar.open <= liq) or (st.position < 0 and bar.open >= liq)
        if hit:
            self._force_close_at(bar.open, "liquidation")
        return hit

    def _automatic_close_at_open(self, bar: CanonicalBar) -> Optional[str]:
        st = self.state
        if abs(st.position) < 1e-12:
            return None
        reason: Optional[str] = None
        if st.position > 0:
            if bar.open <= st.current_stop_price:
                reason = "stop_loss"
            elif bar.open >= st.current_take_price:
                reason = "take_profit"
        else:
            if bar.open >= st.current_stop_price:
                reason = "stop_loss"
            elif bar.open <= st.current_take_price:
                reason = "take_profit"
        if reason is None and st.position_age_bars >= self.config.max_holding_bars:
            reason = "time_stop"
        if reason is not None:
            self._force_close_at(bar.open, reason)
        return reason

    def _check_intrabar_sl_tp(self, bar: CanonicalBar) -> Optional[str]:
        st = self.state
        if abs(st.position) < 1e-12:
            return None
        if st.position > 0:
            stop_hit = bar.low <= st.current_stop_price
            take_hit = bar.high >= st.current_take_price
        else:
            stop_hit = bar.high >= st.current_stop_price
            take_hit = bar.low <= st.current_take_price
        if stop_hit:
            self._force_close_at(st.current_stop_price, "stop_loss")
            return "stop_loss"
        if take_hit:
            self._force_close_at(st.current_take_price, "take_profit")
            return "take_profit"
        return None

    def _check_intrabar_liquidation(self, bar: CanonicalBar) -> bool:
        st = self.state
        liq = self._liquidation_price()
        if liq is None:
            return False
        hit = (st.position > 0 and bar.low <= liq) or (st.position < 0 and bar.high >= liq)
        if hit:
            self._force_close_at(liq, "liquidation")
        return hit

    def _record_trade_close(self, st: AccountState, reason: str) -> None:
        trade_pnl = st.equity() - st.entry_equity
        r_multiple = trade_pnl / st.initial_risk_notional if st.initial_risk_notional > 1e-12 else 0.0
        st.last_close_reason = reason
        st.last_r_multiple = r_multiple
        st.r_multiples.append(r_multiple)
        st.last_trade_bonus = 0.0
        if reason == "stop_loss":
            st.stop_loss_count += 1
        elif reason == "take_profit":
            st.take_profit_count += 1
        elif reason == "liquidation":
            st.liquidation_count += 1
        elif reason == "time_stop":
            st.time_stop_count += 1
        elif reason == "risk_limit":
            st.risk_limit_count += 1
        elif reason == "evaluation_end":
            st.evaluation_end_count += 1

    def _settle_insolvency_at(self, price: float) -> None:
        st = self.state
        if st.raw_equity() > 0:
            return
        if abs(st.position) > 1e-12:
            self._force_close_at(price, "liquidation")
            return
        st.cash = 0.0
        st.margin_used = 0.0
        st.trade_phase = "FLAT"
        st.liquidation_hit = True
        st.terminal = True
        st.terminal_reason = "liquidation"

    def _force_close_at(self, price: float, reason: str) -> None:
        st = self.state
        qty = abs(st.position)
        if qty <= 1e-12:
            return
        is_long = st.position > 0
        fill = self._effective_price(price, is_buy=not is_long)
        realized = (fill - st.avg_entry_price) * qty if is_long else (st.avg_entry_price - fill) * qty
        fee = qty * fill * self.config.fee_rate
        st.realized_pnl += realized
        st.turnover_notional += qty * fill
        holding_bars = st.position_age_bars
        st.round_trips += 1
        st.total_holding_bars += holding_bars
        st.record_holding_bars(holding_bars)
        st.cash += realized - fee + st.margin_used
        st.cash = max(0.0, st.cash)
        st.margin_used = 0.0
        st.position = 0.0
        st.avg_entry_price = 0.0
        st.last_exit_bar = st.steps_survived - 1
        st.last_executed_target = 0.0
        st.trade_count += 1
        self._record_trade_close(st, reason)
        st.position_age_bars = 0
        st.entry_price = 0.0
        st.entry_equity = 0.0
        st.max_favorable_price = 0.0
        st.max_adverse_price = 0.0
        self._clear_trade_risk(st)
        st.trade_phase = "FLAT"
        if reason == "liquidation":
            st.liquidation_hit = True
            st.terminal = True
            st.terminal_reason = reason
