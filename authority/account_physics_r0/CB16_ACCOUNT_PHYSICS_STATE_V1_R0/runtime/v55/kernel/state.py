"""Single-asset account state."""
# V5.5 FROZEN KERNEL: numerical semantics must remain parity-locked to audited V5.4.
from __future__ import annotations

from bisect import insort
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AccountState:
    """Account state for one traded asset.

    Accounting model:
      - cash: free cash available for new margin/withdrawal
      - margin_used: locked margin for open positions
      - position: signed quantity (positive=long, negative=short)
      - equity = cash + margin_used + unrealized_pnl
    """

    cash: float = 100_000.0
    margin_used: float = 0.0
    position: float = 0.0          # positive = long, negative = short
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    last_mark_price: float = 0.0
    peak_equity: float = 100_000.0
    liquidation_hit: bool = False
    terminal: bool = False
    terminal_reason: Optional[str] = None
    steps_survived: int = 0
    invalid_actions: int = 0
    position_side: str = "BOTH"
    margin_type: str = "cross"
    isolated_margin: float = 0.0
    isolated_wallet: float = 0.0
    maint_margin: float = 0.0
    initial_margin: float = 0.0
    open_order_initial_margin: float = 0.0
    trade_phase: str = "FLAT"
    position_age_bars: int = 0
    entry_price: float = 0.0
    entry_equity: float = 0.0
    max_favorable_price: float = 0.0
    max_adverse_price: float = 0.0
    last_exit_bar: int = -1
    last_executed_target: float = 0.0
    trade_count: int = 0
    long_entry_count: int = 0
    short_entry_count: int = 0
    round_trips: int = 0
    total_holding_bars: int = 0
    holding_bars_history: List[int] = field(default_factory=list)
    turnover_notional: float = 0.0
    max_observed_drawdown: float = 0.0
    equity_sum: float = 0.0
    equity_observations: int = 0
    initial_stop_price: float = 0.0
    initial_take_price: float = 0.0
    current_stop_price: float = 0.0
    current_take_price: float = 0.0
    initial_risk_pct: float = 0.0
    initial_risk_notional: float = 0.0
    initial_entry_qty: float = 0.0
    trade_realized_pnl_start: float = 0.0
    last_close_reason: Optional[str] = None
    stop_loss_count: int = 0
    take_profit_count: int = 0
    liquidation_count: int = 0
    time_stop_count: int = 0
    risk_limit_count: int = 0
    evaluation_end_count: int = 0
    r_multiples: List[float] = field(default_factory=list)
    last_r_multiple: float = 0.0
    last_trade_bonus: float = 0.0
    atr: float = 0.0
    prev_close: float = 0.0
    stop_cooldown_until: int = -1

    def __post_init__(self) -> None:
        # Keep peak consistent with the actual initial cash.
        if self.peak_equity == 100_000.0 and self.cash != 100_000.0:
            self.peak_equity = self.cash

    def unrealized_pnl(self) -> float:
        if self.position == 0 or self.last_mark_price == 0:
            return 0.0
        return self.position * (self.last_mark_price - self.avg_entry_price)

    def raw_equity(self) -> float:
        return self.cash + self.margin_used + self.unrealized_pnl()

    def equity(self) -> float:
        """Limited-liability account equity exposed to callers."""
        return max(0.0, self.raw_equity())

    @property
    def median_holding_bars(self) -> float:
        if not self.holding_bars_history:
            return 0.0
        middle = len(self.holding_bars_history) // 2
        if len(self.holding_bars_history) % 2:
            return float(self.holding_bars_history[middle])
        return (
            self.holding_bars_history[middle - 1]
            + self.holding_bars_history[middle]
        ) / 2.0

    def record_holding_bars(self, holding_bars: int) -> None:
        """Insert a completed duration while retaining exact online median data."""
        insort(self.holding_bars_history, int(holding_bars))

    @property
    def turnover(self) -> float:
        if self.equity_observations == 0 or self.equity_sum <= 0:
            return 0.0
        average_equity = self.equity_sum / self.equity_observations
        return self.turnover_notional / average_equity

    @property
    def unfinished_trades(self) -> int:
        return int(abs(self.position) >= 1e-12)

    @property
    def avg_r_multiple(self) -> float:
        if not self.r_multiples:
            return 0.0
        return sum(self.r_multiples) / len(self.r_multiples)

    @property
    def winning_trade_count(self) -> int:
        return sum(value > 0.0 for value in self.r_multiples)

    @property
    def losing_trade_count(self) -> int:
        return sum(value < 0.0 for value in self.r_multiples)

    @property
    def breakeven_trade_count(self) -> int:
        return len(self.r_multiples) - self.winning_trade_count - self.losing_trade_count

    @property
    def win_rate(self) -> float:
        return self.winning_trade_count / len(self.r_multiples) if self.r_multiples else 0.0

    @property
    def winning_r_sum(self) -> float:
        return sum(value for value in self.r_multiples if value > 0.0)

    @property
    def losing_r_sum(self) -> float:
        return sum(value for value in self.r_multiples if value < 0.0)

    @property
    def avg_winning_r(self) -> float:
        return self.winning_r_sum / self.winning_trade_count if self.winning_trade_count else 0.0

    @property
    def avg_losing_r(self) -> float:
        return self.losing_r_sum / self.losing_trade_count if self.losing_trade_count else 0.0

    @property
    def close_reason_counts(self) -> dict:
        return {
            "stop_loss": self.stop_loss_count,
            "take_profit": self.take_profit_count,
            "liquidation": self.liquidation_count,
            "time_stop": self.time_stop_count,
            "risk_limit": self.risk_limit_count,
            "evaluation_end": self.evaluation_end_count,
        }

    def drawdown_from_peak(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity()) / self.peak_equity)

    def update_peak(self) -> None:
        self.peak_equity = max(self.peak_equity, self.equity())

    def available_margin(self) -> float:
        if self.margin_type == "isolated":
            # Position PnL remains inside the isolated wallet and cannot
            # collateralize unrelated orders or withdrawals.
            return max(0.0, self.cash)
        # Unrealized PnL is collateral in the cross-margined single-account
        # model, so free collateral is equity less locked initial margin.
        return max(0.0, self.equity() - self.margin_used)

    @property
    def isolated_unrealized_pnl(self) -> float:
        return self.unrealized_pnl() if self.margin_type == "isolated" else 0.0

    @property
    def max_withdraw_amount(self) -> float:
        return self.available_margin()

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "margin_used": self.margin_used,
            "available_margin": self.available_margin(),
            "position": self.position,
            "avg_entry_price": self.avg_entry_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl(),
            "equity": self.equity(),
            "peak_equity": self.peak_equity,
            "drawdown_from_peak": self.drawdown_from_peak(),
            "liquidation_hit": self.liquidation_hit,
            "terminal": self.terminal,
            "terminal_reason": self.terminal_reason,
            "steps_survived": self.steps_survived,
            "invalid_actions": self.invalid_actions,
            "position_side": self.position_side,
            "margin_type": self.margin_type,
            "isolated_margin": self.isolated_margin,
            "isolated_wallet": self.isolated_wallet,
            "isolated_unrealized_pnl": self.isolated_unrealized_pnl,
            "maint_margin": self.maint_margin,
            "initial_margin": self.initial_margin,
            "open_order_initial_margin": self.open_order_initial_margin,
            "max_withdraw_amount": self.max_withdraw_amount,
            "trade_phase": self.trade_phase,
            "position_age_bars": self.position_age_bars,
            "entry_price": self.entry_price,
            "entry_equity": self.entry_equity,
            "max_favorable_price": self.max_favorable_price,
            "max_adverse_price": self.max_adverse_price,
            "last_exit_bar": self.last_exit_bar,
            "last_executed_target": self.last_executed_target,
            "trade_count": self.trade_count,
            "long_entry_count": self.long_entry_count,
            "short_entry_count": self.short_entry_count,
            "round_trips": self.round_trips,
            "total_holding_bars": self.total_holding_bars,
            "holding_bars_history": list(self.holding_bars_history),
            "turnover_notional": self.turnover_notional,
            "median_holding_bars": self.median_holding_bars,
            "turnover": self.turnover,
            "max_observed_drawdown": self.max_observed_drawdown,
            "unfinished_trades": self.unfinished_trades,
            "initial_stop_price": self.initial_stop_price,
            "initial_take_price": self.initial_take_price,
            "current_stop_price": self.current_stop_price,
            "current_take_price": self.current_take_price,
            "initial_risk_pct": self.initial_risk_pct,
            "initial_risk_notional": self.initial_risk_notional,
            "initial_entry_qty": self.initial_entry_qty,
            "last_close_reason": self.last_close_reason,
            "stop_loss_count": self.stop_loss_count,
            "take_profit_count": self.take_profit_count,
            "liquidation_count": self.liquidation_count,
            "time_stop_count": self.time_stop_count,
            "risk_limit_count": self.risk_limit_count,
            "evaluation_end_count": self.evaluation_end_count,
            "close_reason_counts": self.close_reason_counts,
            "r_multiples": list(self.r_multiples),
            "last_r_multiple": self.last_r_multiple,
            "avg_r_multiple": self.avg_r_multiple,
            "winning_trade_count": self.winning_trade_count,
            "losing_trade_count": self.losing_trade_count,
            "breakeven_trade_count": self.breakeven_trade_count,
            "win_rate": self.win_rate,
            "winning_r_sum": self.winning_r_sum,
            "losing_r_sum": self.losing_r_sum,
            "avg_winning_r": self.avg_winning_r,
            "avg_losing_r": self.avg_losing_r,
            "last_trade_bonus": self.last_trade_bonus,
            "atr": self.atr,
            "prev_close": self.prev_close,
            "stop_cooldown_until": self.stop_cooldown_until,
        }
