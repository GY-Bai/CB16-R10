from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .r102_common import HOUR_MS
from .r102_physics import FLAT, LONG, SHORT, FrozenPhysicsRuntimeR102

MINUTE_MS = 60_000
MINUTES_PER_HOUR = 60
SNAPSHOT_SCHEMA = "CB16_R11_MINUTE_PHYSICS_BINDING_SNAPSHOT_R2_V1"


def _ms(dt: datetime) -> int:
    return int(round(dt.astimezone(timezone.utc).timestamp() * 1000.0))


def _hour_start_ms(ts_ms: int) -> int:
    return (int(ts_ms) // HOUR_MS) * HOUR_MS


def _minute_kernel_class(runtime: FrozenPhysicsRuntimeR102):
    base_probe = runtime.physics.make_kernel(runtime.physics_contract)
    Base = type(base_probe)
    CanonicalBar = runtime.physics.CanonicalBar

    class WallClockMinuteKernelR2(Base):
        """Minute-resolution adapter preserving hourly temporal semantics.

        The frozen financial equations and environment-owned exit logic are inherited.
        Only bar-count clocks are adapted:
        - ATR updates once per completed UTC hour using the exact frozen ATR update.
        - stop cooldown counts 60 minute steps per frozen hourly bar.
        - position_age_bars remains elapsed completed hours for the frozen time-stop.
        Minute bars resolve threshold ordering causally; same-minute double touches keep
        the frozen stop-first tie-break because Base._check_intrabar_sl_tp is reused.
        """

        def __init__(self, config=None):
            super().__init__(config)
            self._r2_current_time_ms: int | None = None
            self._r2_entry_time_ms: int | None = None
            self._r2_hour_bucket: dict[str, Any] | None = None

        def r2_load_hidden(self, hidden: Mapping[str, Any]) -> None:
            self._r2_entry_time_ms = (
                None if hidden.get("entry_time_ms") is None else int(hidden["entry_time_ms"])
            )
            bucket = hidden.get("hour_bucket")
            self._r2_hour_bucket = None if bucket is None else copy.deepcopy(dict(bucket))
            self._r2_current_time_ms = None

        def r2_hidden(self) -> dict[str, Any]:
            return {
                "entry_time_ms": self._r2_entry_time_ms,
                "hour_bucket": copy.deepcopy(self._r2_hour_bucket),
            }

        def r2_prepare_bar(self, bar: Any) -> None:
            if str(bar.timeframe) != "1m":
                raise ValueError("R2_REQUIRES_1M_BAR")
            now = _ms(bar.bar_start)
            bucket_start = _hour_start_ms(now)
            b = self._r2_hour_bucket
            if b is not None and bucket_start != int(b["start_ms"]):
                expected = int(b["start_ms"]) + HOUR_MS
                if bucket_start != expected:
                    raise RuntimeError(
                        f"R2_MINUTE_BUCKET_GAP:{int(b['start_ms'])}->{bucket_start}"
                    )
                hourly = CanonicalBar(
                    symbol=str(b["symbol"]),
                    bar_start=datetime.fromtimestamp(int(b["start_ms"]) / 1000, tz=timezone.utc),
                    timeframe="1h",
                    open=float(b["open"]),
                    high=float(b["high"]),
                    low=float(b["low"]),
                    close=float(b["close"]),
                    volume=float(b["volume"]),
                    mark_price=float(b["mark_price"]),
                    index_price=float(b["index_price"]),
                )
                Base._update_atr(self, hourly)
                self._r2_hour_bucket = None
            self._r2_current_time_ms = now

        def r2_flush_completed_hour(self, next_hour_start_ms: int) -> None:
            b = self._r2_hour_bucket
            if b is None:
                return
            expected = int(b["start_ms"]) + HOUR_MS
            if int(next_hour_start_ms) != expected:
                raise RuntimeError(
                    f"R2_FLUSH_REQUIRES_NEXT_HOUR_BOUNDARY:{expected}!={int(next_hour_start_ms)}"
                )
            hourly = CanonicalBar(
                symbol=str(b["symbol"]),
                bar_start=datetime.fromtimestamp(int(b["start_ms"]) / 1000, tz=timezone.utc),
                timeframe="1h",
                open=float(b["open"]),
                high=float(b["high"]),
                low=float(b["low"]),
                close=float(b["close"]),
                volume=float(b["volume"]),
                mark_price=float(b["mark_price"]),
                index_price=float(b["index_price"]),
            )
            Base._update_atr(self, hourly)
            self._r2_hour_bucket = None

        def _update_atr(self, bar: Any) -> None:
            if str(bar.timeframe) != "1m":
                raise RuntimeError("R2_ATR_EXPECTS_MINUTE_BAR_DURING_STEP")
            now = _ms(bar.bar_start)
            hs = _hour_start_ms(now)
            b = self._r2_hour_bucket
            if b is None:
                self._r2_hour_bucket = {
                    "symbol": bar.symbol,
                    "start_ms": hs,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "mark_price": float(bar.mark_price),
                    "index_price": float(bar.index_price),
                    "rows": 1,
                }
                return
            if int(b["start_ms"]) != hs:
                raise RuntimeError("R2_BUCKET_NOT_PREPARED_BEFORE_STEP")
            b["high"] = max(float(b["high"]), float(bar.high))
            b["low"] = min(float(b["low"]), float(bar.low))
            b["close"] = float(bar.close)
            b["volume"] = float(b["volume"]) + float(bar.volume)
            b["mark_price"] = float(bar.mark_price)
            b["index_price"] = float(bar.index_price)
            b["rows"] = int(b["rows"]) + 1

        def _start_cooldown(self) -> None:
            self.state.stop_cooldown_until = (
                int(self.state.steps_survived)
                + int(self.config.stop_cooldown_bars) * MINUTES_PER_HOUR
            )

        def _apply_entry_plan(self, plan, state=None, *, record_metrics: bool = True) -> None:
            Base._apply_entry_plan(self, plan, state, record_metrics=record_metrics)
            st = self.state if state is None else state
            if (
                record_metrics
                and state is None
                and abs(float(st.position)) >= 1e-12
                and self._r2_entry_time_ms is None
            ):
                if self._r2_current_time_ms is None:
                    raise RuntimeError("R2_ENTRY_WITHOUT_CURRENT_MINUTE_CLOCK")
                self._r2_entry_time_ms = int(self._r2_current_time_ms)

        def _force_close_at(self, price: float, reason: str) -> None:
            Base._force_close_at(self, price, reason)
            if abs(float(self.state.position)) < 1e-12:
                self._r2_entry_time_ms = None

        def _update_trade_memory_for_bar(self, bar: Any) -> None:
            st = self.state
            if self._r2_current_time_ms is None:
                raise RuntimeError("R2_TRADE_MEMORY_WITHOUT_CURRENT_CLOCK")
            if st.position > 1e-12:
                st.trade_phase = "LONG"
                st.max_favorable_price = max(st.max_favorable_price, float(bar.high))
                st.max_adverse_price = min(st.max_adverse_price, float(bar.low))
                if self._r2_entry_time_ms is None:
                    raise RuntimeError("R2_POSITION_WITHOUT_ENTRY_CLOCK")
                elapsed_end = (
                    int(self._r2_current_time_ms) + MINUTE_MS - int(self._r2_entry_time_ms)
                )
                st.position_age_bars = max(0, int(elapsed_end // HOUR_MS))
            elif st.position < -1e-12:
                st.trade_phase = "SHORT"
                st.max_favorable_price = min(st.max_favorable_price, float(bar.low))
                st.max_adverse_price = max(st.max_adverse_price, float(bar.high))
                if self._r2_entry_time_ms is None:
                    raise RuntimeError("R2_POSITION_WITHOUT_ENTRY_CLOCK")
                elapsed_end = (
                    int(self._r2_current_time_ms) + MINUTE_MS - int(self._r2_entry_time_ms)
                )
                st.position_age_bars = max(0, int(elapsed_end // HOUR_MS))
            else:
                st.position_age_bars = 0
                next_step_is_blocked = (
                    st.stop_cooldown_until >= 0
                    and st.steps_survived + 1 <= st.stop_cooldown_until
                )
                st.trade_phase = "COOLDOWN" if next_step_is_blocked else "FLAT"

    return WallClockMinuteKernelR2, Base, CanonicalBar


@dataclass
class MinutePhysicsSessionR2:
    runtime: FrozenPhysicsRuntimeR102
    kernel: Any
    support_state: dict[str, Any]
    risk_authority: dict[str, Any]
    Base: Any
    CanonicalBar: Any

    @classmethod
    def initialize(
        cls,
        runtime: FrozenPhysicsRuntimeR102,
        *,
        account_id: str,
        risk_fraction: float = 1.0,
    ) -> "MinutePhysicsSessionR2":
        snap, risk = runtime.initialize(account_id, risk_fraction)
        K, Base, Canon = _minute_kernel_class(runtime)
        base = runtime.physics.restore_kernel(snap, runtime.physics_contract)
        k = K(base.config)
        k.state = copy.deepcopy(base.state)
        k._last_bar_time = base._last_bar_time
        k._symbol = base._symbol
        k._last_carry_cost = base._last_carry_cost
        return cls(
            runtime=runtime,
            kernel=k,
            support_state=copy.deepcopy(snap["observation_support_state"]),
            risk_authority=copy.deepcopy(risk),
            Base=Base,
            CanonicalBar=Canon,
        )

    @classmethod
    def restore(
        cls,
        runtime: FrozenPhysicsRuntimeR102,
        state: Mapping[str, Any],
        risk_authority: Mapping[str, Any],
    ) -> "MinutePhysicsSessionR2":
        if state.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("R2_WRAPPER_SNAPSHOT_SCHEMA_MISMATCH")
        base_snap = state["base_snapshot"]
        K, Base, Canon = _minute_kernel_class(runtime)
        base = runtime.physics.restore_kernel(base_snap, runtime.physics_contract)
        k = K(base.config)
        k.state = copy.deepcopy(base.state)
        k._last_bar_time = base._last_bar_time
        k._symbol = base._symbol
        k._last_carry_cost = base._last_carry_cost
        k.r2_load_hidden(state["minute_hidden"])
        return cls(
            runtime=runtime,
            kernel=k,
            support_state=copy.deepcopy(base_snap["observation_support_state"]),
            risk_authority=copy.deepcopy(dict(risk_authority)),
            Base=Base,
            CanonicalBar=Canon,
        )

    def warmup_hourly(self, bars: Sequence[Any]) -> None:
        if abs(float(self.kernel.state.position)) >= 1e-12:
            raise RuntimeError("R2_WARMUP_REQUIRES_FLAT_ACCOUNT")
        for bar in bars:
            if str(bar.timeframe) != "1h":
                raise ValueError("R2_WARMUP_REQUIRES_1H")
            self.kernel._validate_next_bar(bar)
            self.Base._update_atr(self.kernel, bar)
            self.kernel._last_bar_time = bar.bar_start

    def base_snapshot(self) -> dict[str, Any]:
        return self.runtime.physics.snapshot_kernel(
            self.kernel,
            self.runtime.physics_contract,
            self.support_state,
        )

    def export_state(self) -> dict[str, Any]:
        return {
            "schema": SNAPSHOT_SCHEMA,
            "base_snapshot": self.base_snapshot(),
            "minute_hidden": self.kernel.r2_hidden(),
        }

    def account6(self, mark: float):
        return self.runtime.account6(self.base_snapshot(), float(mark))

    def equity_at_mark(self, mark: float) -> float:
        return self.runtime.equity_at_mark(self.base_snapshot(), float(mark))

    def minute_bar(
        self,
        *,
        symbol: str,
        open_time_ms: int,
        ohlcv: Sequence[float],
    ):
        o, h, l, c, v = [float(x) for x in ohlcv]
        return self.CanonicalBar(
            symbol=symbol,
            bar_start=datetime.fromtimestamp(int(open_time_ms) / 1000, tz=timezone.utc),
            timeframe="1m",
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            mark_price=c,
            index_price=c,
        )

    def step_intent(
        self,
        *,
        direction_v55: int,
        risk: float,
        symbol: str,
        open_time_ms: int,
        ohlcv: Sequence[float],
        funding_rate: float,
        trace_id: str,
    ) -> dict[str, Any]:
        bar = self.minute_bar(
            symbol=symbol,
            open_time_ms=open_time_ms,
            ohlcv=ohlcv,
        )
        self.kernel.r2_prepare_bar(bar)
        snapshot_t = self.base_snapshot()
        intent = self.runtime.intent(direction_v55, risk, trace_id=trace_id)
        decision = self.runtime.supervisor.supervise(
            intent,
            snapshot_t,
            self.risk_authority,
            self.runtime.physics_contract,
        )
        executable = self.runtime.supervisor.executable_action(
            decision, self.runtime.physics_contract
        )
        before = copy.deepcopy(self.kernel.state)
        result = self.kernel.step(
            bar,
            {
                "decision": int(executable["direction"]),
                "risk_multiplier": float(executable["risk_multiplier"]),
            },
            float(funding_rate),
        )
        snapshot_t1 = self.base_snapshot()
        term = self.runtime.physics._termination_type(
            before, self.kernel.state, truncate_after_step=False
        )
        return {
            "intent": intent,
            "supervisor_decision": decision,
            "executable_action": executable,
            "snapshot_t": snapshot_t,
            "snapshot_t1": snapshot_t1,
            "account_observation_t1": self.runtime.account6(
                snapshot_t1, float(bar.mark_price)
            ).tolist(),
            "termination_type": term,
            "execution_metadata": {
                "step_reward": float(result.step_reward),
                "carry_cost": float(result.carry_cost),
                "kernel_done": bool(result.done),
                "kernel_info": copy.deepcopy(result.info),
            },
        }

    def finalize(self, close_price: float) -> dict[str, Any]:
        before = float(self.kernel.state.position)
        reward = 0.0
        if abs(before) >= 1e-12:
            reward = float(self.kernel.finalize(float(close_price)))
        return {
            "used": abs(before) >= 1e-12,
            "position_before": before,
            "reward": reward,
            "snapshot": self.base_snapshot(),
        }


def economic_state_projection(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    st = snapshot["kernel_state"]
    keys = (
        "cash",
        "margin_used",
        "position",
        "avg_entry_price",
        "last_mark_price",
        "realized_pnl",
        "atr",
        "prev_close",
        "initial_stop_price",
        "initial_take_price",
        "current_stop_price",
        "current_take_price",
        "position_age_bars",
        "trade_count",
        "round_trips",
        "stop_loss_count",
        "take_profit_count",
        "liquidation_count",
        "time_stop_count",
        "risk_limit_count",
        "evaluation_end_count",
        "terminal",
        "terminal_reason",
    )
    return {k: copy.deepcopy(st[k]) for k in keys}


def canonical_hash(obj: Any) -> str:
    import json

    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def direction_teacher_value(direction_v55: int) -> int:
    if int(direction_v55) == SHORT:
        return -1
    if int(direction_v55) == FLAT:
        return 0
    if int(direction_v55) == LONG:
        return 1
    raise ValueError("DIRECTION_OUT_OF_DOMAIN")
