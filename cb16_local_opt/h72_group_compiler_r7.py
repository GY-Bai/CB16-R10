from __future__ import annotations

"""
Vectorized H72 compiler over many AccountState anchors sharing one future market path.

R6 vectorized the action/risk branches of one parent account. R7 adds the next axis:

    K account anchors
    × C action/risk candidates
    × H72 future bars

are flattened into one `AccountBatchState` with K*C rows and stepped by one
`VectorizedPhysics` call per future bar.

Scientific support remains ONE dependence group for the shared future path.

This module is intentionally independent of multiprocessing; `h72_anchor_farm_r7.py`
runs many dependence groups in spawn workers.
"""

import dataclasses
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from .trajectory_compiler_r6 import (
    ActionCandidateR6,
    DecisionAnchorR6,
    EconomicClockR6,
    HorizonExitReceiptR6,
    InitialAccountSnapshotR6,
    MarketPathR6,
    default_action_grid_r6,
)
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
class GroupAnchorBatchR7:
    dependence_group_id: str
    decision_index: int
    parent_ids: tuple[str, ...]
    student_context_object_ids: tuple[str, ...]
    context_features: np.ndarray       # [K,F]
    balance: np.ndarray                # [K]
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
    market_lineage_hash: str

    @property
    def accounts(self) -> int:
        return len(self.parent_ids)

    def validate(self):
        k = self.accounts
        if k <= 0:
            raise ValueError("empty group anchor")
        if len(set(self.parent_ids)) != k:
            raise ValueError("duplicate parent id")
        if len(self.student_context_object_ids) != k:
            raise ValueError("context id count")
        if self.context_features.ndim != 2 or self.context_features.shape[0] != k:
            raise ValueError("context feature shape")
        if np.any(~np.isfinite(self.context_features)):
            raise ValueError("nonfinite context")
        for name in (
            "balance","position_qty","entry_price","peak_equity","realized_pnl",
            "margin_used","holding_bars","risk_budget_remaining",
            "risk_budget_capacity","terminated","last_mark_price",
        ):
            if len(np.asarray(getattr(self, name))) != k:
                raise ValueError(f"anchor array length:{name}")
        if np.any(np.asarray(self.balance) <= 0):
            raise ValueError("nonpositive anchor balance")
        if np.any(np.asarray(self.risk_budget_capacity) <= 0):
            raise ValueError("risk budget capacity")
        if not self.dependence_group_id:
            raise ValueError("dependence group id")


@dataclass(frozen=True)
class GroupBranchMatrixReceiptR7:
    dependence_group_id: str
    decision_timestamp: int
    maturity_timestamp: int
    clock_id: str
    accounts: int
    candidates: int
    branch_count: int
    independent_market_group_count: int
    future_path_hash: str
    physics_config_hash: str
    parent_hash: str
    candidate_grid_hash: str
    utility_sha256: str
    terminal_equity_sha256: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass
class CompiledGroupR7:
    anchor: GroupAnchorBatchR7
    candidates: tuple[ActionCandidateR6, ...]
    decision_timestamp: int
    maturity_timestamp: int
    clock_id: str
    future_path_hash: str
    truth_log_utility: np.ndarray      # [K,C]
    terminal_equity: np.ndarray        # [K,C]
    total_fee: np.ndarray              # [K,C]
    total_turnover: np.ndarray         # [K,C]
    total_funding_cashflow: np.ndarray # [K,C]
    liquidated: np.ndarray             # [K,C] bool
    stopped: np.ndarray
    took_profit: np.ndarray
    forced_horizon_exit: np.ndarray
    receipt: GroupBranchMatrixReceiptR7

    def to_teacher_samples(self) -> list[CounterfactualBranchSampleR5]:
        out = []
        k, c = self.truth_log_utility.shape
        if (k, c) != (self.anchor.accounts, len(self.candidates)):
            raise RuntimeError("GROUP_RESULT_SHAPE_MISMATCH")
        for i in range(k):
            for j, candidate in enumerate(self.candidates):
                u = float(self.truth_log_utility[i, j])
                if not math.isfinite(u):
                    raise RuntimeError(
                        "BANKRUPT_BRANCH_REQUIRES_EXPLICIT_TEACHER_CENSORING_POLICY"
                    )
                out.append(CounterfactualBranchSampleR5(
                    parent_id=self.anchor.parent_ids[i],
                    student_context_object_id=self.anchor.student_context_object_ids[i],
                    timestamp=self.decision_timestamp,
                    context_features=tuple(
                        float(x) for x in self.anchor.context_features[i]
                    ),
                    direction=candidate.direction,
                    requested_risk=candidate.requested_risk,
                    realized_utility=u,
                    dependence_group_id=self.anchor.dependence_group_id,
                    market_lineage_hash=self.anchor.market_lineage_hash,
                ))
        return out


class VectorizedGroupH72CompilerR7:
    def __init__(
        self,
        physics_config: VectorPhysicsConfig,
        *,
        clock: EconomicClockR6 | None = None,
        terminal_slippage_bps: float | None = None,
    ):
        physics_config.validate()
        self.physics_config = physics_config
        self.clock = clock or EconomicClockR6(horizon_bars=72)
        self.clock.validate()
        self.terminal_slippage_bps = (
            physics_config.slippage_bps
            if terminal_slippage_bps is None
            else float(terminal_slippage_bps)
        )

    @staticmethod
    def from_decision_anchors(
        dependence_group_id: str,
        anchors: Sequence[DecisionAnchorR6],
    ) -> GroupAnchorBatchR7:
        if not anchors:
            raise ValueError("no anchors")
        idx = anchors[0].decision_index
        market_lineage = anchors[0].market_lineage_hash
        if any(a.decision_index != idx for a in anchors):
            raise RuntimeError("GROUP_ANCHORS_DIFFERENT_DECISION_INDEX")
        if any(a.dependence_group_id != dependence_group_id for a in anchors):
            raise RuntimeError("GROUP_DEPENDENCE_ID_MISMATCH")
        if any(a.market_lineage_hash != market_lineage for a in anchors):
            raise RuntimeError("GROUP_MARKET_LINEAGE_MISMATCH")
        for a in anchors:
            a.validate()
        return GroupAnchorBatchR7(
            dependence_group_id=dependence_group_id,
            decision_index=idx,
            parent_ids=tuple(a.parent_id for a in anchors),
            student_context_object_ids=tuple(
                a.student_context_object_id for a in anchors
            ),
            context_features=np.asarray(
                [a.context_features for a in anchors], dtype=np.float32
            ),
            balance=np.asarray([a.initial_account.balance for a in anchors], dtype=np.float64),
            position_qty=np.asarray([a.initial_account.position_qty for a in anchors], dtype=np.float64),
            entry_price=np.asarray([a.initial_account.entry_price for a in anchors], dtype=np.float64),
            peak_equity=np.asarray([
                a.initial_account.balance
                if a.initial_account.peak_equity is None
                else a.initial_account.peak_equity
                for a in anchors
            ], dtype=np.float64),
            realized_pnl=np.asarray([a.initial_account.realized_pnl for a in anchors], dtype=np.float64),
            margin_used=np.asarray([a.initial_account.margin_used for a in anchors], dtype=np.float64),
            holding_bars=np.asarray([a.initial_account.holding_bars for a in anchors], dtype=np.int64),
            risk_budget_remaining=np.asarray([a.initial_account.risk_budget_remaining for a in anchors], dtype=np.float64),
            risk_budget_capacity=np.asarray([a.initial_account.risk_budget_capacity for a in anchors], dtype=np.float64),
            terminated=np.asarray([a.initial_account.terminated for a in anchors], dtype=bool),
            last_mark_price=np.asarray([a.initial_account.last_mark_price for a in anchors], dtype=np.float64),
            market_lineage_hash=market_lineage,
        )

    def _expanded_state(
        self,
        anchor: GroupAnchorBatchR7,
        candidate_count: int,
    ) -> AccountBatchState:
        k = anchor.accounts
        c = candidate_count
        rep = lambda x: np.repeat(np.asarray(x), c)
        ids = np.asarray([
            f"{anchor.parent_ids[i]}:CF{j:03d}"
            for i in range(k)
            for j in range(c)
        ], dtype=object)
        zeros_i = np.zeros(k*c, dtype=np.int64)
        return AccountBatchState(
            account_ids=ids,
            balance=rep(anchor.balance).astype(np.float64, copy=True),
            position_qty=rep(anchor.position_qty).astype(np.float64, copy=True),
            entry_price=rep(anchor.entry_price).astype(np.float64, copy=True),
            peak_equity=rep(anchor.peak_equity).astype(np.float64, copy=True),
            realized_pnl=rep(anchor.realized_pnl).astype(np.float64, copy=True),
            margin_used=rep(anchor.margin_used).astype(np.float64, copy=True),
            holding_bars=rep(anchor.holding_bars).astype(np.int64, copy=True),
            risk_budget_remaining=rep(anchor.risk_budget_remaining).astype(np.float64, copy=True),
            risk_budget_capacity=rep(anchor.risk_budget_capacity).astype(np.float64, copy=True),
            terminated=rep(anchor.terminated).astype(bool, copy=True),
            last_mark_price=rep(anchor.last_mark_price).astype(np.float64, copy=True),
            trade_count=zeros_i.copy(),
            liquidation_count=zeros_i.copy(),
            stop_count=zeros_i.copy(),
            take_count=zeros_i.copy(),
            time_exit_count=zeros_i.copy(),
        )

    def _hold_action(self, state: AccountBatchState, next_open: float):
        eq = mark_equity(state, next_open)
        qty = state.position_qty
        d = np.sign(qty).astype(np.int8)
        notional = np.abs(qty) * float(next_open)
        denom = (
            self.physics_config.max_gross_leverage
            * np.maximum(eq, 1e-12)
        )
        r = np.divide(
            notional, denom,
            out=np.zeros_like(notional),
            where=denom > 0,
        )
        r = np.clip(r, 0, 1)
        d[(qty == 0) | state.terminated] = 0
        r[(qty == 0) | state.terminated] = 0
        return d, r

    def _terminal_close(
        self,
        state: AccountBatchState,
        terminal_mark: float,
    ):
        qty = state.position_qty.copy()
        active = (qty != 0) & (~state.terminated)
        slip = self.terminal_slippage_bps * 1e-4
        fill = np.full(state.n, float(terminal_mark), dtype=np.float64)
        fill[active] = float(terminal_mark) * (
            1.0 - np.sign(qty[active]) * slip
        )
        pnl = np.zeros(state.n, dtype=np.float64)
        pnl[active] = qty[active] * (
            fill[active] - state.entry_price[active]
        )
        fee = np.zeros(state.n, dtype=np.float64)
        fee[active] = (
            np.abs(qty[active]) * fill[active]
            * self.physics_config.fee_rate
        )
        state.balance[active] += pnl[active] - fee[active]
        state.realized_pnl[active] += pnl[active]
        state.position_qty[active] = 0
        state.entry_price[active] = 0
        state.margin_used[active] = 0
        state.holding_bars[active] = 0
        return state.balance.copy(), fee, active

    def compile(
        self,
        market: MarketPathR6,
        anchor: GroupAnchorBatchR7,
        candidates: Sequence[ActionCandidateR6] | None = None,
    ) -> CompiledGroupR7:
        market.validate()
        anchor.validate()
        candidates = tuple(candidates or default_action_grid_r6())
        if not candidates:
            raise ValueError("empty candidate grid")
        for c in candidates:
            c.validate()
        if len({c.candidate_id for c in candidates}) != len(candidates):
            raise ValueError("duplicate candidate")
        k, c = anchor.accounts, len(candidates)
        start = anchor.decision_index + self.clock.execution_offset_bars
        stop = anchor.decision_index + self.clock.horizon_bars + 1
        if stop > market.rows:
            raise RuntimeError("ANCHOR_GROUP_NOT_MATURE_H72")

        state = self._expanded_state(anchor, c)
        anchor_eq = mark_equity(
            state,
            float(market.close[anchor.decision_index]),
        ).reshape(k, c)
        # Every counterfactual candidate for the same parent must start identically.
        if np.any(np.ptp(anchor_eq, axis=1) > 1e-10):
            raise RuntimeError("COUNTERFACTUAL_INITIAL_EQUITY_MISMATCH")
        anchor_eq_parent = anchor_eq[:, 0].copy()

        d0 = np.tile(
            np.asarray([x.direction for x in candidates], dtype=np.int8),
            k,
        )
        r0 = np.tile(
            np.asarray([x.requested_risk for x in candidates], dtype=np.float64),
            k,
        )
        engine = VectorizedPhysics(self.physics_config)
        total_fee = np.zeros(k*c)
        total_turnover = np.zeros(k*c)
        total_funding = np.zeros(k*c)
        ever_liq = np.zeros(k*c, dtype=bool)
        ever_stop = np.zeros(k*c, dtype=bool)
        ever_take = np.zeros(k*c, dtype=bool)

        for j, row in enumerate(range(start, stop)):
            bar = MarketBar(
                open=float(market.open[row]),
                high=float(market.high[row]),
                low=float(market.low[row]),
                close=float(market.close[row]),
                funding_rate=(
                    0.0 if market.funding_rate is None
                    else float(market.funding_rate[row])
                ),
            )
            if j == 0:
                d, r = d0, r0
            else:
                d, r = self._hold_action(state, bar.open)
            rec = engine.step(
                state,
                bar,
                executable_direction=d,
                executable_risk=r,
                requested_direction=d,
                dependence_group_count=1,
            )
            total_fee += rec.fee
            total_turnover += rec.turnover_notional
            total_funding += rec.funding_cashflow
            ever_liq |= rec.exit_reason == "LIQUIDATION"
            ever_stop |= rec.exit_reason == "STOP"
            ever_take |= rec.exit_reason == "TAKE"

        terminal_eq, terminal_fee, terminal_closed = self._terminal_close(
            state, float(market.close[stop-1])
        )
        total_fee += terminal_fee

        terminal_eq = terminal_eq.reshape(k, c)
        u = np.full((k, c), -np.inf, dtype=np.float64)
        viable = terminal_eq > 0
        u[viable] = np.log(
            terminal_eq[viable]
            / np.repeat(anchor_eq_parent, c).reshape(k, c)[viable]
        )

        future_hash = canonical_hash({
            "timestamp": [int(x) for x in np.asarray(market.timestamp)[start:stop]],
            "open": [float(x) for x in np.asarray(market.open)[start:stop]],
            "high": [float(x) for x in np.asarray(market.high)[start:stop]],
            "low": [float(x) for x in np.asarray(market.low)[start:stop]],
            "close": [float(x) for x in np.asarray(market.close)[start:stop]],
            "funding": (
                None if market.funding_rate is None
                else [float(x) for x in np.asarray(market.funding_rate)[start:stop]]
            ),
        })
        receipt = GroupBranchMatrixReceiptR7(
            dependence_group_id=anchor.dependence_group_id,
            decision_timestamp=int(market.timestamp[anchor.decision_index]),
            maturity_timestamp=int(market.timestamp[stop-1]),
            clock_id=self.clock.clock_id,
            accounts=k,
            candidates=c,
            branch_count=k*c,
            independent_market_group_count=1,
            future_path_hash=future_hash,
            physics_config_hash=self.physics_config.config_hash,
            parent_hash=canonical_hash(anchor.parent_ids),
            candidate_grid_hash=canonical_hash([
                {"direction": x.direction, "risk": x.requested_risk}
                for x in candidates
            ]),
            utility_sha256=hashlib.sha256(
                np.ascontiguousarray(u).tobytes()
            ).hexdigest(),
            terminal_equity_sha256=hashlib.sha256(
                np.ascontiguousarray(terminal_eq).tobytes()
            ).hexdigest(),
        )
        return CompiledGroupR7(
            anchor=anchor,
            candidates=candidates,
            decision_timestamp=receipt.decision_timestamp,
            maturity_timestamp=receipt.maturity_timestamp,
            clock_id=receipt.clock_id,
            future_path_hash=future_hash,
            truth_log_utility=u,
            terminal_equity=terminal_eq,
            total_fee=total_fee.reshape(k,c),
            total_turnover=total_turnover.reshape(k,c),
            total_funding_cashflow=total_funding.reshape(k,c),
            liquidated=ever_liq.reshape(k,c),
            stopped=ever_stop.reshape(k,c),
            took_profit=ever_take.reshape(k,c),
            forced_horizon_exit=terminal_closed.reshape(k,c),
            receipt=receipt,
        )
