from __future__ import annotations

"""
H72/H73 multi-bar counterfactual trajectory compiler.

Economic clock used by R6:
- decision anchor: bar t has fully closed and is Student-visible;
- first executable future bar: t+1;
- H72 future horizon: bars t+1 ... t+72 inclusive;
- H73 timestamp support: anchor t plus 72 future bars = 73 chronological bar timestamps;
- terminal accounting: any still-open position is forcibly closed at the t+72 close,
  including terminal fee/slippage, before terminal log-equity utility is computed.

This compiler preserves one dependence group per decision anchor.  The entire action/risk
counterfactual grid shares the exact same future market path and must not be counted as
independent market samples.
"""

import dataclasses
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from .vectorized_physics import (
    AccountBatchState,
    MarketBar,
    VectorPhysicsConfig,
    VectorizedPhysics,
    mark_equity,
)


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class EconomicClockR6:
    horizon_bars: int = 72
    execution_offset_bars: int = 1

    def validate(self):
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        if self.execution_offset_bars != 1:
            raise ValueError("R6 currently requires next-bar execution")

    @property
    def required_timestamp_count(self) -> int:
        # Decision anchor + H future bars.
        return self.horizon_bars + 1

    @property
    def maturity_offset_bars(self) -> int:
        return self.horizon_bars

    @property
    def clock_id(self) -> str:
        self.validate()
        return f"H{self.horizon_bars}_H{self.required_timestamp_count}_NEXT_BAR_EXEC"


@dataclass(frozen=True)
class MarketPathR6:
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    funding_rate: np.ndarray | None = None

    def validate(self):
        n = len(self.timestamp)
        if n <= 1:
            raise ValueError("market path too short")
        for name in ("open", "high", "low", "close", "volume"):
            a = np.asarray(getattr(self, name))
            if len(a) != n:
                raise ValueError(f"market column length mismatch:{name}")
        ts = np.asarray(self.timestamp, dtype=np.int64)
        if np.any(np.diff(ts) <= 0):
            raise RuntimeError("MARKET_PATH_NON_INCREASING_TIME")
        o = np.asarray(self.open, dtype=np.float64)
        h = np.asarray(self.high, dtype=np.float64)
        l = np.asarray(self.low, dtype=np.float64)
        c = np.asarray(self.close, dtype=np.float64)
        v = np.asarray(self.volume, dtype=np.float64)
        if np.any(~np.isfinite(o)) or np.any(~np.isfinite(h)) or np.any(~np.isfinite(l)) or np.any(~np.isfinite(c)):
            raise RuntimeError("MARKET_PATH_NONFINITE_PRICE")
        if np.any(o <= 0) or np.any(h <= 0) or np.any(l <= 0) or np.any(c <= 0):
            raise RuntimeError("MARKET_PATH_NONPOSITIVE_PRICE")
        if np.any(l > np.minimum(o, c)) or np.any(h < np.maximum(o, c)):
            raise RuntimeError("MARKET_PATH_OHLC_ORDER")
        if np.any(v < 0) or np.any(~np.isfinite(v)):
            raise RuntimeError("MARKET_PATH_BAD_VOLUME")
        if self.funding_rate is not None:
            f = np.asarray(self.funding_rate, dtype=np.float64)
            if len(f) != n or np.any(~np.isfinite(f)):
                raise RuntimeError("MARKET_PATH_BAD_FUNDING")

    @property
    def rows(self) -> int:
        return int(len(self.timestamp))


@dataclass(frozen=True)
class InitialAccountSnapshotR6:
    balance: float
    position_qty: float = 0.0
    entry_price: float = 0.0
    peak_equity: float | None = None
    realized_pnl: float = 0.0
    margin_used: float = 0.0
    holding_bars: int = 0
    risk_budget_remaining: float = 1.0
    risk_budget_capacity: float = 1.0
    terminated: bool = False
    last_mark_price: float = 0.0

    @classmethod
    def flat(cls, initial_equity: float) -> "InitialAccountSnapshotR6":
        return cls(balance=float(initial_equity), peak_equity=float(initial_equity))

    def validate(self):
        if self.balance <= 0 or not np.isfinite(self.balance):
            raise ValueError("initial balance must be finite positive")
        if self.position_qty == 0 and self.entry_price != 0:
            raise ValueError("flat account must have zero entry")
        if self.position_qty != 0 and self.entry_price <= 0:
            raise ValueError("open position requires positive entry")
        if self.risk_budget_capacity <= 0:
            raise ValueError("risk budget capacity")
        if not 0 <= self.risk_budget_remaining <= self.risk_budget_capacity:
            raise ValueError("risk budget remaining")
        if self.holding_bars < 0:
            raise ValueError("holding bars")


@dataclass(frozen=True)
class DecisionAnchorR6:
    parent_id: str
    student_context_object_id: str
    decision_index: int
    context_features: tuple[float, ...]
    dependence_group_id: str
    market_lineage_hash: str
    initial_account: InitialAccountSnapshotR6

    def validate(self):
        if not self.parent_id or not self.student_context_object_id or not self.dependence_group_id:
            raise ValueError("anchor ids required")
        if self.decision_index < 0:
            raise ValueError("decision_index")
        x = np.asarray(self.context_features, dtype=np.float64)
        if x.ndim != 1 or not len(x) or np.any(~np.isfinite(x)):
            raise ValueError("bad context features")
        self.initial_account.validate()


@dataclass(frozen=True)
class ActionCandidateR6:
    direction: int
    requested_risk: float

    def validate(self):
        if self.direction not in {-1, 0, 1}:
            raise ValueError("direction")
        if not 0 <= self.requested_risk <= 1:
            raise ValueError("risk")
        if self.direction == 0 and self.requested_risk != 0:
            raise ValueError("FLAT_REQUIRES_ZERO_RISK")

    @property
    def candidate_id(self) -> str:
        self.validate()
        return f"D{self.direction:+d}_R{self.requested_risk:.8f}"


@dataclass(frozen=True)
class HorizonExitReceiptR6:
    terminal_mark_price: float
    exit_fill_price: tuple[float, ...]
    exit_fee: tuple[float, ...]
    realized_terminal_pnl: tuple[float, ...]
    terminal_equity: tuple[float, ...]
    closed_positions: tuple[bool, ...]


@dataclass(frozen=True)
class TrajectoryBranchResultR6:
    parent_id: str
    student_context_object_id: str
    candidate: ActionCandidateR6
    dependence_group_id: str
    decision_timestamp: int
    first_future_timestamp: int
    maturity_timestamp: int
    clock_id: str
    anchor_equity: float
    terminal_equity: float
    truth_log_utility: float
    total_fee: float
    total_turnover: float
    total_funding_cashflow: float
    liquidated: bool
    stopped: bool
    took_profit: bool
    forced_horizon_exit: bool
    path_hash: str
    physics_config_hash: str

    @property
    def content_hash(self) -> str:
        # Infinity is allowed in economic truth but canonical JSON is not. Encode explicitly.
        payload = asdict(self)
        u = payload["truth_log_utility"]
        if not math.isfinite(u):
            payload["truth_log_utility"] = "-Infinity" if u < 0 else "+Infinity"
        return canonical_hash(payload)


@dataclass(frozen=True)
class CompiledAnchorTrajectoriesR6:
    parent_id: str
    dependence_group_id: str
    decision_timestamp: int
    maturity_timestamp: int
    clock_id: str
    branches: tuple[TrajectoryBranchResultR6, ...]
    counterfactual_branch_count: int
    independent_market_group_count: int
    future_path_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash({
            "parent_id": self.parent_id,
            "dependence_group_id": self.dependence_group_id,
            "decision_timestamp": self.decision_timestamp,
            "maturity_timestamp": self.maturity_timestamp,
            "clock_id": self.clock_id,
            "branch_hashes": [b.content_hash for b in self.branches],
            "counterfactual_branch_count": self.counterfactual_branch_count,
            "independent_market_group_count": self.independent_market_group_count,
            "future_path_hash": self.future_path_hash,
        })


def default_action_grid_r6(
    risk_levels: Sequence[float] = (0.25, 0.50, 0.75, 1.0),
) -> tuple[ActionCandidateR6, ...]:
    risks = tuple(float(x) for x in risk_levels)
    if any(x <= 0 or x > 1 for x in risks):
        raise ValueError("non-flat risk levels must be in (0,1]")
    out = [ActionCandidateR6(-1, r) for r in risks]
    out.append(ActionCandidateR6(0, 0.0))
    out.extend(ActionCandidateR6(1, r) for r in risks)
    return tuple(out)


class MultiBarTrajectoryCompilerR6:
    def __init__(
        self,
        physics_config: VectorPhysicsConfig,
        *,
        clock: EconomicClockR6 | None = None,
        terminal_slippage_bps: float | None = None,
    ):
        self.physics_config = physics_config
        self.physics_config.validate()
        self.clock = clock or EconomicClockR6()
        self.clock.validate()
        self.terminal_slippage_bps = (
            physics_config.slippage_bps
            if terminal_slippage_bps is None
            else float(terminal_slippage_bps)
        )
        if self.terminal_slippage_bps < 0:
            raise ValueError("terminal slippage")

    def _state_from_anchor(
        self,
        anchor: DecisionAnchorR6,
        n: int,
    ) -> AccountBatchState:
        a = anchor.initial_account
        a.validate()
        ids = np.asarray(
            [f"{anchor.parent_id}:{i}" for i in range(n)],
            dtype=object,
        )
        f = lambda x: np.full(n, float(x), dtype=np.float64)
        i = lambda x: np.full(n, int(x), dtype=np.int64)
        b = lambda x: np.full(n, bool(x), dtype=bool)
        peak = a.balance if a.peak_equity is None else a.peak_equity
        return AccountBatchState(
            account_ids=ids,
            balance=f(a.balance),
            position_qty=f(a.position_qty),
            entry_price=f(a.entry_price),
            peak_equity=f(peak),
            realized_pnl=f(a.realized_pnl),
            margin_used=f(a.margin_used),
            holding_bars=i(a.holding_bars),
            risk_budget_remaining=f(a.risk_budget_remaining),
            risk_budget_capacity=f(a.risk_budget_capacity),
            terminated=b(a.terminated),
            last_mark_price=f(a.last_mark_price),
            trade_count=i(0),
            liquidation_count=i(0),
            stop_count=i(0),
            take_count=i(0),
            time_exit_count=i(0),
        )

    def _hold_current_quantity_action(
        self,
        state: AccountBatchState,
        next_open: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Translate 'hold current quantity' into the target-exposure Physics interface."""
        eq = mark_equity(state, next_open)
        qty = state.position_qty
        direction = np.sign(qty).astype(np.int8)
        notional = np.abs(qty) * float(next_open)
        denom = self.physics_config.max_gross_leverage * np.maximum(eq, 1e-12)
        risk = np.divide(notional, denom, out=np.zeros_like(notional), where=denom > 0)
        # If losses imply current quantity exceeds the configured leverage cap, Physics
        # deleverages to its legal cap rather than inventing risk > 1.
        risk = np.clip(risk, 0.0, 1.0)
        direction[state.terminated] = 0
        risk[state.terminated] = 0.0
        direction[qty == 0] = 0
        risk[qty == 0] = 0.0
        return direction, risk

    def _terminal_close(
        self,
        state: AccountBatchState,
        terminal_mark: float,
    ) -> HorizonExitReceiptR6:
        qty = state.position_qty.copy()
        active = (qty != 0) & (~state.terminated)
        slip = self.terminal_slippage_bps * 1e-4
        # Closing long is a sell (downward fill); closing short is a buy (upward fill).
        fill = np.full(state.n, float(terminal_mark), dtype=np.float64)
        fill[active] = float(terminal_mark) * (
            1.0 - np.sign(qty[active]) * slip
        )
        pnl = np.zeros(state.n, dtype=np.float64)
        pnl[active] = qty[active] * (fill[active] - state.entry_price[active])
        fee = np.zeros(state.n, dtype=np.float64)
        fee[active] = np.abs(qty[active]) * fill[active] * self.physics_config.fee_rate
        state.balance[active] += pnl[active] - fee[active]
        state.realized_pnl[active] += pnl[active]
        state.position_qty[active] = 0.0
        state.entry_price[active] = 0.0
        state.margin_used[active] = 0.0
        state.holding_bars[active] = 0
        terminal_eq = state.balance.copy()
        return HorizonExitReceiptR6(
            terminal_mark_price=float(terminal_mark),
            exit_fill_price=tuple(float(x) for x in fill),
            exit_fee=tuple(float(x) for x in fee),
            realized_terminal_pnl=tuple(float(x) for x in pnl),
            terminal_equity=tuple(float(x) for x in terminal_eq),
            closed_positions=tuple(bool(x) for x in active),
        )

    def compile_anchor(
        self,
        market: MarketPathR6,
        anchor: DecisionAnchorR6,
        candidates: Sequence[ActionCandidateR6] | None = None,
    ) -> CompiledAnchorTrajectoriesR6:
        market.validate()
        anchor.validate()
        candidates = tuple(candidates or default_action_grid_r6())
        if not candidates:
            raise ValueError("empty action grid")
        for c in candidates:
            c.validate()
        if len({c.candidate_id for c in candidates}) != len(candidates):
            raise ValueError("duplicate action candidate")

        start = anchor.decision_index + self.clock.execution_offset_bars
        stop_exclusive = anchor.decision_index + self.clock.horizon_bars + 1
        if stop_exclusive > market.rows:
            raise RuntimeError("ANCHOR_NOT_MATURE_H72")
        future_slice = slice(start, stop_exclusive)
        future_hash = canonical_hash({
            "timestamp": [int(x) for x in np.asarray(market.timestamp)[future_slice]],
            "open": [float(x) for x in np.asarray(market.open)[future_slice]],
            "high": [float(x) for x in np.asarray(market.high)[future_slice]],
            "low": [float(x) for x in np.asarray(market.low)[future_slice]],
            "close": [float(x) for x in np.asarray(market.close)[future_slice]],
            "funding": (
                None
                if market.funding_rate is None
                else [float(x) for x in np.asarray(market.funding_rate)[future_slice]]
            ),
        })

        n = len(candidates)
        state = self._state_from_anchor(anchor, n)
        anchor_mark = float(market.close[anchor.decision_index])
        anchor_eq_all = mark_equity(state, anchor_mark)
        if not np.allclose(anchor_eq_all, anchor_eq_all[0], atol=1e-12, rtol=0):
            raise RuntimeError("COUNTERFACTUAL_INITIAL_STATE_NOT_IDENTICAL")
        anchor_equity = float(anchor_eq_all[0])
        if anchor_equity <= 0:
            raise RuntimeError("ANCHOR_EQUITY_NONPOSITIVE")

        engine = VectorizedPhysics(self.physics_config)
        initial_d = np.asarray([c.direction for c in candidates], dtype=np.int8)
        initial_r = np.asarray([c.requested_risk for c in candidates], dtype=np.float64)
        total_fee = np.zeros(n, dtype=np.float64)
        total_turnover = np.zeros(n, dtype=np.float64)
        total_funding = np.zeros(n, dtype=np.float64)
        ever_liq = np.zeros(n, dtype=bool)
        ever_stop = np.zeros(n, dtype=bool)
        ever_take = np.zeros(n, dtype=bool)

        for j, row in enumerate(range(start, stop_exclusive)):
            bar = MarketBar(
                open=float(market.open[row]),
                high=float(market.high[row]),
                low=float(market.low[row]),
                close=float(market.close[row]),
                funding_rate=(
                    0.0
                    if market.funding_rate is None
                    else float(market.funding_rate[row])
                ),
            )
            if j == 0:
                d, r = initial_d, initial_r
                requested = initial_d
            else:
                d, r = self._hold_current_quantity_action(state, bar.open)
                requested = d
            rec = engine.step(
                state,
                bar,
                executable_direction=d,
                executable_risk=r,
                requested_direction=requested,
                dependence_group_count=1,
            )
            total_fee += rec.fee
            total_turnover += rec.turnover_notional
            total_funding += rec.funding_cashflow
            ever_liq |= rec.exit_reason == "LIQUIDATION"
            ever_stop |= rec.exit_reason == "STOP"
            ever_take |= rec.exit_reason == "TAKE"

        terminal = self._terminal_close(
            state,
            float(market.close[stop_exclusive - 1]),
        )
        terminal_eq = np.asarray(terminal.terminal_equity, dtype=np.float64)
        terminal_fee = np.asarray(terminal.exit_fee, dtype=np.float64)
        total_fee += terminal_fee
        terminal_closed = np.asarray(terminal.closed_positions, dtype=bool)

        branches = []
        for i, candidate in enumerate(candidates):
            if terminal_eq[i] <= 0:
                utility = float("-inf")
            else:
                utility = float(math.log(terminal_eq[i] / anchor_equity))
            branches.append(TrajectoryBranchResultR6(
                parent_id=anchor.parent_id,
                student_context_object_id=anchor.student_context_object_id,
                candidate=candidate,
                dependence_group_id=anchor.dependence_group_id,
                decision_timestamp=int(market.timestamp[anchor.decision_index]),
                first_future_timestamp=int(market.timestamp[start]),
                maturity_timestamp=int(market.timestamp[stop_exclusive - 1]),
                clock_id=self.clock.clock_id,
                anchor_equity=anchor_equity,
                terminal_equity=float(terminal_eq[i]),
                truth_log_utility=utility,
                total_fee=float(total_fee[i]),
                total_turnover=float(total_turnover[i]),
                total_funding_cashflow=float(total_funding[i]),
                liquidated=bool(ever_liq[i]),
                stopped=bool(ever_stop[i]),
                took_profit=bool(ever_take[i]),
                forced_horizon_exit=bool(terminal_closed[i]),
                path_hash=future_hash,
                physics_config_hash=self.physics_config.config_hash,
            ))

        return CompiledAnchorTrajectoriesR6(
            parent_id=anchor.parent_id,
            dependence_group_id=anchor.dependence_group_id,
            decision_timestamp=int(market.timestamp[anchor.decision_index]),
            maturity_timestamp=int(market.timestamp[stop_exclusive - 1]),
            clock_id=self.clock.clock_id,
            branches=tuple(branches),
            counterfactual_branch_count=len(branches),
            independent_market_group_count=1,
            future_path_hash=future_hash,
        )

    def compile_many(
        self,
        market: MarketPathR6,
        anchors: Sequence[DecisionAnchorR6],
        candidates: Sequence[ActionCandidateR6] | None = None,
        *,
        skip_immature: bool = True,
    ) -> list[CompiledAnchorTrajectoriesR6]:
        market.validate()
        out = []
        seen = set()
        for a in sorted(anchors, key=lambda x: (x.decision_index, x.parent_id)):
            if a.parent_id in seen:
                raise RuntimeError("DUPLICATE_PARENT_ID")
            seen.add(a.parent_id)
            try:
                out.append(self.compile_anchor(market, a, candidates))
            except RuntimeError as exc:
                if skip_immature and str(exc) == "ANCHOR_NOT_MATURE_H72":
                    continue
                raise
        return out

    @staticmethod
    def to_teacher_samples(
        compiled: Sequence[CompiledAnchorTrajectoriesR6],
        anchors_by_parent: dict[str, DecisionAnchorR6],
    ) -> list[CounterfactualBranchSampleR5]:
        """Convert finite economic truth branches to the R5/R6 probabilistic Teacher input.

        Bankruptcy truth remains -Infinity. It is NOT silently clipped. If encountered,
        the caller must define an explicit censored/bankruptcy Teacher protocol.
        """
        out = []
        for group in compiled:
            anchor = anchors_by_parent[group.parent_id]
            for branch in group.branches:
                if not math.isfinite(branch.truth_log_utility):
                    raise RuntimeError(
                        "BANKRUPT_BRANCH_REQUIRES_EXPLICIT_TEACHER_CENSORING_POLICY"
                    )
                out.append(CounterfactualBranchSampleR5(
                    parent_id=group.parent_id,
                    student_context_object_id=branch.student_context_object_id,
                    timestamp=branch.decision_timestamp,
                    context_features=anchor.context_features,
                    direction=branch.candidate.direction,
                    requested_risk=branch.candidate.requested_risk,
                    realized_utility=branch.truth_log_utility,
                    dependence_group_id=group.dependence_group_id,
                    market_lineage_hash=anchor.market_lineage_hash,
                ))
        return out
