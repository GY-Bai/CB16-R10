from __future__ import annotations

"""
New vectorized linear-perpetual account engine for CB16 local R&D.

This is versioned R2 local Physics.  It is intentionally NOT presented as byte-parity
with the historical V5.5 scalar kernel.  Its semantics are explicit, vectorized and
auditable, and any scientific run must bind this version/config hash.

Important inherited principles:
- realized account ledger is economic truth;
- fees/funding are charged exactly once;
- Supervisor executable action is the only action reaching Physics;
- FLAT => risk 0;
- counterfactual branches over one realized bar remain one dependence group.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

PHYSICS_VERSION = "CB16_LOCAL_VECTOR_PHYSICS_R2"


def _canonical(obj: Any) -> bytes:
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


@dataclass(frozen=True)
class VectorPhysicsConfig:
    initial_equity: float = 100_000.0
    max_gross_leverage: float = 3.0
    fee_rate: float = 0.0004
    slippage_bps: float = 2.0
    maintenance_margin_rate: float = 0.10
    max_holding_bars: int = 72
    stop_loss_fraction: float | None = None
    take_profit_fraction: float | None = None
    both_hit_policy: str = "STOP_FIRST"  # deterministic conservative tie rule
    bankruptcy_floor_ratio: float = 1e-12

    def validate(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive")
        if self.fee_rate < 0 or self.slippage_bps < 0:
            raise ValueError("fee/slippage must be nonnegative")
        if not 0 < self.maintenance_margin_rate < 1:
            raise ValueError("maintenance_margin_rate must be in (0,1)")
        if self.max_holding_bars <= 0:
            raise ValueError("max_holding_bars must be positive")
        if self.both_hit_policy not in {"STOP_FIRST", "TAKE_FIRST"}:
            raise ValueError("unsupported both_hit_policy")
        for x in (self.stop_loss_fraction, self.take_profit_fraction):
            if x is not None and x <= 0:
                raise ValueError("stop/take fractions must be positive")

    @property
    def config_hash(self) -> str:
        self.validate()
        return _hash(self)


@dataclass(frozen=True)
class MarketBar:
    open: float
    high: float
    low: float
    close: float
    funding_rate: float = 0.0

    def validate(self) -> None:
        vals = (self.open, self.high, self.low, self.close)
        if not all(np.isfinite(vals)) or min(vals) <= 0:
            raise ValueError("OHLC must be finite positive")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC ordering invalid")
        if not np.isfinite(self.funding_rate):
            raise ValueError("funding_rate must be finite")


@dataclass
class AccountBatchState:
    account_ids: np.ndarray
    balance: np.ndarray
    position_qty: np.ndarray
    entry_price: np.ndarray
    peak_equity: np.ndarray
    realized_pnl: np.ndarray
    margin_used: np.ndarray
    holding_bars: np.ndarray
    risk_budget_remaining: np.ndarray
    risk_budget_capacity: np.ndarray
    terminated: np.ndarray
    last_mark_price: np.ndarray
    trade_count: np.ndarray
    liquidation_count: np.ndarray
    stop_count: np.ndarray
    take_count: np.ndarray
    time_exit_count: np.ndarray

    @classmethod
    def empty(
        cls,
        n: int,
        config: VectorPhysicsConfig,
        *,
        risk_budget_capacity: float = 1.0,
        account_prefix: str = "A",
    ) -> "AccountBatchState":
        config.validate()
        if n <= 0:
            raise ValueError("n must be positive")
        ids = np.array([f"{account_prefix}{i:08d}" for i in range(n)], dtype=object)
        f = lambda v: np.full(n, v, dtype=np.float64)
        i = lambda v: np.full(n, v, dtype=np.int64)
        b = lambda v: np.full(n, v, dtype=bool)
        return cls(
            account_ids=ids,
            balance=f(config.initial_equity),
            position_qty=f(0.0),
            entry_price=f(0.0),
            peak_equity=f(config.initial_equity),
            realized_pnl=f(0.0),
            margin_used=f(0.0),
            holding_bars=i(0),
            risk_budget_remaining=f(risk_budget_capacity),
            risk_budget_capacity=f(risk_budget_capacity),
            terminated=b(False),
            last_mark_price=f(0.0),
            trade_count=i(0),
            liquidation_count=i(0),
            stop_count=i(0),
            take_count=i(0),
            time_exit_count=i(0),
        )

    @property
    def n(self) -> int:
        return int(self.balance.shape[0])

    def copy(self) -> "AccountBatchState":
        return AccountBatchState(**{f.name: getattr(self, f.name).copy() for f in dataclasses.fields(self)})

    def validate(self) -> None:
        n = self.n
        for f in dataclasses.fields(self):
            a = getattr(self, f.name)
            if not isinstance(a, np.ndarray) or len(a) != n:
                raise ValueError(f"state field mismatch: {f.name}")
        if np.any(self.risk_budget_capacity <= 0):
            raise ValueError("risk budget capacity must be positive")
        if np.any(self.risk_budget_remaining < -1e-12):
            raise ValueError("risk budget remaining negative")
        if np.any(self.risk_budget_remaining - self.risk_budget_capacity > 1e-12):
            raise ValueError("risk budget remaining exceeds capacity")


@dataclass(frozen=True)
class StepReceipt:
    physics_version: str
    physics_config_hash: str
    equity_before: np.ndarray
    equity_after: np.ndarray
    requested_direction: np.ndarray
    executable_direction: np.ndarray
    executable_risk: np.ndarray
    turnover_notional: np.ndarray
    fee: np.ndarray
    funding_cashflow: np.ndarray
    realized_pnl_step: np.ndarray
    unrealized_pnl_after: np.ndarray
    log_equity_reward: np.ndarray
    exit_reason: np.ndarray
    dependence_group_count: int


def _unrealized(qty: np.ndarray, entry: np.ndarray, mark: float) -> np.ndarray:
    active = qty != 0
    out = np.zeros_like(qty, dtype=np.float64)
    out[active] = qty[active] * (float(mark) - entry[active])
    return out


def mark_equity(state: AccountBatchState, mark: float) -> np.ndarray:
    return state.balance + _unrealized(state.position_qty, state.entry_price, mark)


def _fill_price(open_price: float, delta_qty: np.ndarray, slippage_bps: float) -> np.ndarray:
    slip = slippage_bps * 1e-4
    # Buy delta pays up, sell delta receives down.
    return open_price * (1.0 + np.sign(delta_qty) * slip)


def _realize_and_rebase(
    old_qty: np.ndarray,
    old_entry: np.ndarray,
    target_qty: np.ndarray,
    fill: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized linear futures position transition.

    Returns (realized_pnl, new_entry, closed_qty_abs).
    """
    realized = np.zeros_like(old_qty, dtype=np.float64)
    new_entry = old_entry.copy()
    same_sign = np.sign(old_qty) == np.sign(target_qty)
    reducing_same = same_sign & (np.abs(target_qty) < np.abs(old_qty)) & (old_qty != 0)
    reversing = (old_qty != 0) & (target_qty != 0) & (~same_sign)
    closing = (old_qty != 0) & (target_qty == 0)

    closed_abs = np.zeros_like(old_qty, dtype=np.float64)
    for mask in (reducing_same, reversing, closing):
        closed_abs[mask] = np.minimum(np.abs(old_qty[mask]), np.abs(old_qty[mask] - target_qty[mask]))

    close_mask = closed_abs > 0
    realized[close_mask] = (
        np.sign(old_qty[close_mask])
        * closed_abs[close_mask]
        * (fill[close_mask] - old_entry[close_mask])
    )

    # Entry logic.
    flat_new = target_qty == 0
    open_from_flat = (old_qty == 0) & (target_qty != 0)
    increase_same = same_sign & (old_qty != 0) & (np.abs(target_qty) > np.abs(old_qty))
    reverse_new = reversing

    new_entry[flat_new] = 0.0
    new_entry[open_from_flat | reverse_new] = fill[open_from_flat | reverse_new]

    if np.any(increase_same):
        old_notional_units = np.abs(old_qty[increase_same])
        add_units = np.abs(target_qty[increase_same]) - old_notional_units
        new_entry[increase_same] = (
            old_notional_units * old_entry[increase_same] + add_units * fill[increase_same]
        ) / np.abs(target_qty[increase_same])

    # Reductions retain the original entry.
    return realized, new_entry, closed_abs


def _barrier_exit_masks(
    qty: np.ndarray,
    entry: np.ndarray,
    bar: MarketBar,
    cfg: VectorPhysicsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(qty)
    stop = np.zeros(n, dtype=bool)
    take = np.zeros(n, dtype=bool)
    active = qty != 0
    if cfg.stop_loss_fraction is not None:
        long = active & (qty > 0)
        short = active & (qty < 0)
        stop[long] = bar.low <= entry[long] * (1.0 - cfg.stop_loss_fraction)
        stop[short] = bar.high >= entry[short] * (1.0 + cfg.stop_loss_fraction)
    if cfg.take_profit_fraction is not None:
        long = active & (qty > 0)
        short = active & (qty < 0)
        take[long] = bar.high >= entry[long] * (1.0 + cfg.take_profit_fraction)
        take[short] = bar.low <= entry[short] * (1.0 - cfg.take_profit_fraction)
    both = stop & take
    if np.any(both):
        if cfg.both_hit_policy == "STOP_FIRST":
            take[both] = False
        else:
            stop[both] = False
    return stop, take


class VectorizedPhysics:
    def __init__(self, config: VectorPhysicsConfig):
        config.validate()
        self.config = config

    def account_observation6(self, state: AccountBatchState, mark: float) -> np.ndarray:
        """Vectorized equivalent of the six conceptual account-observation channels."""
        eq = mark_equity(state, mark)
        gross_notional = state.position_qty * mark
        max_lev = self.config.max_gross_leverage
        signed_exp = np.divide(gross_notional, np.maximum(eq, 1e-12)) / max_lev
        has_pos = state.position_qty != 0
        entry_log = np.zeros(state.n, dtype=np.float64)
        entry_log[has_pos] = np.log(mark / state.entry_price[has_pos])
        remaining = np.ones(state.n, dtype=np.float64)
        remaining[has_pos] = (self.config.max_holding_bars - state.holding_bars[has_pos]) / self.config.max_holding_bars
        drawdown = (state.peak_equity - eq) / np.maximum(state.peak_equity, 1e-12)
        risk_frac = state.risk_budget_remaining / state.risk_budget_capacity
        margin_util = state.margin_used / np.maximum(eq, 1e-12)
        return np.stack(
            [signed_exp, entry_log, remaining, drawdown, risk_frac, margin_util],
            axis=1,
        ).astype(np.float32, copy=False)

    def step(
        self,
        state: AccountBatchState,
        bar: MarketBar,
        executable_direction: np.ndarray,
        executable_risk: np.ndarray,
        *,
        requested_direction: np.ndarray | None = None,
        dependence_group_count: int | None = None,
    ) -> StepReceipt:
        state.validate()
        bar.validate()
        n = state.n
        d = np.asarray(executable_direction, dtype=np.int8).reshape(n)
        r = np.asarray(executable_risk, dtype=np.float64).reshape(n)
        if requested_direction is None:
            requested_direction = d
        requested_direction = np.asarray(requested_direction, dtype=np.int8).reshape(n)

        if np.any(~np.isin(d, (-1, 0, 1))):
            raise ValueError("direction must be -1/0/+1")
        if np.any((r < 0) | (r > 1) | ~np.isfinite(r)):
            raise ValueError("risk must be finite in [0,1]")
        if np.any((d == 0) & (np.abs(r) > 1e-12)):
            raise ValueError("FLAT_REQUIRES_ZERO_RISK")

        live = ~state.terminated
        r = np.where(live, r, 0.0)
        d = np.where(live, d, 0)

        equity_before = mark_equity(state, bar.open)
        # Bankruptcy/terminated accounts cannot request exposure.
        viable = live & (equity_before > 0)
        d = np.where(viable, d, 0)
        r = np.where(viable, r, 0.0)

        # Risk budget acts as a hard exposure cap but does not mutate the Trader request.
        risk_cap_fraction = np.clip(state.risk_budget_remaining / state.risk_budget_capacity, 0.0, 1.0)
        r_exec = np.minimum(r, risk_cap_fraction)

        target_notional = d.astype(np.float64) * r_exec * self.config.max_gross_leverage * np.maximum(equity_before, 0.0)
        raw_target_qty = target_notional / bar.open
        delta_guess = raw_target_qty - state.position_qty
        fill = _fill_price(bar.open, delta_guess, self.config.slippage_bps)
        target_qty = np.divide(target_notional, fill, out=np.zeros_like(target_notional), where=fill != 0)
        delta = target_qty - state.position_qty
        fill = _fill_price(bar.open, delta, self.config.slippage_bps)

        realized, new_entry, closed_abs = _realize_and_rebase(
            state.position_qty,
            state.entry_price,
            target_qty,
            fill,
        )
        turnover = np.abs(delta) * fill
        fee = turnover * self.config.fee_rate

        old_nonzero = state.position_qty != 0
        new_nonzero = target_qty != 0
        opened_or_reversed = new_nonzero & ((~old_nonzero) | (np.sign(state.position_qty) != np.sign(target_qty)))
        closed_to_flat = ~new_nonzero

        state.balance += realized - fee
        state.realized_pnl += realized
        state.position_qty = target_qty
        state.entry_price = new_entry
        state.trade_count += opened_or_reversed.astype(np.int64)
        state.holding_bars[opened_or_reversed] = 0
        state.holding_bars[new_nonzero & ~opened_or_reversed] += 1
        state.holding_bars[closed_to_flat] = 0

        # Funding is applied once, after the open execution and before close marking.
        funding_cashflow = -state.position_qty * bar.close * float(bar.funding_rate)
        state.balance += funding_cashflow
        state.realized_pnl += funding_cashflow

        exit_reason = np.full(n, "NONE", dtype=object)

        # Optional deterministic intrabar barrier exits.
        stop, take = _barrier_exit_masks(state.position_qty, state.entry_price, bar, self.config)
        if np.any(stop | take):
            exit_mask = stop | take
            long = state.position_qty > 0
            exit_px = np.full(n, bar.close, dtype=np.float64)
            if self.config.stop_loss_fraction is not None:
                exit_px[stop & long] = state.entry_price[stop & long] * (1 - self.config.stop_loss_fraction)
                exit_px[stop & ~long] = state.entry_price[stop & ~long] * (1 + self.config.stop_loss_fraction)
            if self.config.take_profit_fraction is not None:
                exit_px[take & long] = state.entry_price[take & long] * (1 + self.config.take_profit_fraction)
                exit_px[take & ~long] = state.entry_price[take & ~long] * (1 - self.config.take_profit_fraction)
            q = state.position_qty.copy()
            pnl = q * (exit_px - state.entry_price)
            exit_fee = np.abs(q) * exit_px * self.config.fee_rate
            state.balance[exit_mask] += pnl[exit_mask] - exit_fee[exit_mask]
            state.realized_pnl[exit_mask] += pnl[exit_mask]
            fee[exit_mask] += exit_fee[exit_mask]
            turnover[exit_mask] += np.abs(q[exit_mask]) * exit_px[exit_mask]
            state.position_qty[exit_mask] = 0.0
            state.entry_price[exit_mask] = 0.0
            state.holding_bars[exit_mask] = 0
            state.stop_count += stop.astype(np.int64)
            state.take_count += take.astype(np.int64)
            exit_reason[stop] = "STOP"
            exit_reason[take] = "TAKE"

        # Max holding forced exit at close.
        time_exit = (state.position_qty != 0) & (state.holding_bars >= self.config.max_holding_bars)
        if np.any(time_exit):
            q = state.position_qty.copy()
            px = float(bar.close)
            pnl = q * (px - state.entry_price)
            exit_fee = np.abs(q) * px * self.config.fee_rate
            state.balance[time_exit] += pnl[time_exit] - exit_fee[time_exit]
            state.realized_pnl[time_exit] += pnl[time_exit]
            fee[time_exit] += exit_fee[time_exit]
            turnover[time_exit] += np.abs(q[time_exit]) * px
            state.position_qty[time_exit] = 0
            state.entry_price[time_exit] = 0
            state.holding_bars[time_exit] = 0
            state.time_exit_count += time_exit.astype(np.int64)
            exit_reason[time_exit] = "MAX_HOLDING"

        # Close mark, maintenance margin and liquidation.
        unreal = _unrealized(state.position_qty, state.entry_price, bar.close)
        equity_mid = state.balance + unreal
        notional = np.abs(state.position_qty) * bar.close
        maint = notional * self.config.maintenance_margin_rate
        liquidate = (state.position_qty != 0) & (equity_mid <= maint)
        if np.any(liquidate):
            q = state.position_qty.copy()
            px = float(bar.close)
            pnl = q * (px - state.entry_price)
            liq_fee = np.abs(q) * px * self.config.fee_rate
            state.balance[liquidate] += pnl[liquidate] - liq_fee[liquidate]
            state.realized_pnl[liquidate] += pnl[liquidate]
            fee[liquidate] += liq_fee[liquidate]
            turnover[liquidate] += np.abs(q[liquidate]) * px
            state.position_qty[liquidate] = 0
            state.entry_price[liquidate] = 0
            state.holding_bars[liquidate] = 0
            state.terminated[liquidate] = True
            state.liquidation_count += liquidate.astype(np.int64)
            exit_reason[liquidate] = "LIQUIDATION"

        unreal_after = _unrealized(state.position_qty, state.entry_price, bar.close)
        equity_after = state.balance + unreal_after
        state.peak_equity = np.maximum(state.peak_equity, equity_after)
        state.margin_used = np.abs(state.position_qty) * bar.close / self.config.max_gross_leverage
        state.last_mark_price[:] = bar.close

        floor = max(self.config.initial_equity * self.config.bankruptcy_floor_ratio, 1e-300)
        denom = np.maximum(equity_before, floor)
        numer = np.maximum(equity_after, floor)
        log_reward = np.log(numer / denom)

        return StepReceipt(
            physics_version=PHYSICS_VERSION,
            physics_config_hash=self.config.config_hash,
            equity_before=equity_before.copy(),
            equity_after=equity_after.copy(),
            requested_direction=requested_direction.copy(),
            executable_direction=d.copy(),
            executable_risk=r_exec.copy(),
            turnover_notional=turnover.copy(),
            fee=fee.copy(),
            funding_cashflow=funding_cashflow.copy(),
            realized_pnl_step=realized.copy(),
            unrealized_pnl_after=unreal_after.copy(),
            log_equity_reward=log_reward.copy(),
            exit_reason=exit_reason.copy(),
            dependence_group_count=int(n if dependence_group_count is None else dependence_group_count),
        )

    def counterfactual_grid(
        self,
        state: AccountBatchState,
        bar: MarketBar,
        risk_levels: Iterable[float],
        directions: Iterable[int] = (-1, 0, 1),
    ) -> dict[str, Any]:
        """Evaluate the same bar/future under an action grid.

        Each original account is one dependence group regardless of branch count.
        """
        dirs = np.asarray(tuple(directions), dtype=np.int8)
        risks = np.asarray(tuple(risk_levels), dtype=np.float64)
        if dirs.ndim != 1 or risks.ndim != 1:
            raise ValueError("directions/risk_levels must be 1-D")
        combos = [(int(d), float(0.0 if d == 0 else r)) for d in dirs for r in risks]
        n, k = state.n, len(combos)

        def repeat(a: np.ndarray) -> np.ndarray:
            return np.repeat(a, k, axis=0)

        branch_state = AccountBatchState(**{
            f.name: repeat(getattr(state, f.name))
            for f in dataclasses.fields(state)
        })
        d = np.tile(np.array([c[0] for c in combos], dtype=np.int8), n)
        r = np.tile(np.array([c[1] for c in combos], dtype=np.float64), n)
        receipt = self.step(
            branch_state,
            bar,
            d,
            r,
            requested_direction=d,
            dependence_group_count=n,
        )
        return {
            "shape": (n, len(dirs), len(risks)),
            "directions": dirs,
            "risk_levels": risks,
            "equity_after": receipt.equity_after.reshape(n, len(dirs), len(risks)),
            "log_equity_reward": receipt.log_equity_reward.reshape(n, len(dirs), len(risks)),
            "dependence_group_count": n,
            "branch_count": n * k,
        }
