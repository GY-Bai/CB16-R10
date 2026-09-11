from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from cb16_local_opt.minute_physics_binding_r2 import (
    MinutePhysicsSessionR2,
    economic_state_projection,
)
from cb16_local_opt.r102_common import HOUR_MS
from cb16_local_opt.r102_physics import FLAT, LONG, FrozenPhysicsRuntimeR102

ROOT = Path(__file__).resolve().parents[1]
MINUTE_MS = 60_000


def runtime():
    return FrozenPhysicsRuntimeR102.load(ROOT)


def hbar(session, t_ms, *, o=100.0, h=101.0, l=99.0, c=100.0, v=60.0):
    return session.CanonicalBar(
        symbol="BTCUSDT",
        bar_start=datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc),
        timeframe="1h",
        open=o, high=h, low=l, close=c, volume=v,
        mark_price=c, index_price=c,
    )


def warmup(session, start_ms):
    bars = [
        hbar(session, start_ms + i * HOUR_MS)
        for i in range(16)
    ]
    session.warmup_hourly(bars)
    return bars


def minute_ohlcv(px=100.0, *, high=None, low=None):
    return (
        px,
        px if high is None else high,
        px if low is None else low,
        px,
        1.0,
    )


def test_checkpoint_restore_exact_inside_partial_hour():
    rt = runtime()
    s = MinutePhysicsSessionR2.initialize(rt, account_id="E6R2:CP")
    t0 = 1_700_000_000_000 // HOUR_MS * HOUR_MS
    warmup(s, t0 - 16 * HOUR_MS)
    for j in range(17):
        s.step_intent(
            direction_v55=LONG if j == 0 else FLAT,
            risk=0.5 if j == 0 else 0.0,
            symbol="BTCUSDT",
            open_time_ms=t0 + j * MINUTE_MS,
            ohlcv=minute_ohlcv(),
            funding_rate=0.0,
            trace_id=f"CP:{j}",
        )
    state = s.export_state()
    restored = MinutePhysicsSessionR2.restore(rt, state, s.risk_authority)
    assert restored.export_state() == state


def test_unambiguous_hourly_economic_parity():
    rt = runtime()
    s = MinutePhysicsSessionR2.initialize(rt, account_id="E6R2:PARITY")
    t0 = 1_700_000_000_000 // HOUR_MS * HOUR_MS
    warm = warmup(s, t0 - 16 * HOUR_MS)

    base = rt.physics.make_kernel(rt.physics_contract)
    base.warmup_market(warm)
    base.step(
        hbar(s, t0, o=100.0, h=100.0, l=100.0, c=100.0, v=60.0),
        {"decision": LONG, "risk_multiplier": 0.5},
        0.0,
    )

    for j in range(60):
        s.step_intent(
            direction_v55=LONG if j == 0 else FLAT,
            risk=0.5 if j == 0 else 0.0,
            symbol="BTCUSDT",
            open_time_ms=t0 + j * MINUTE_MS,
            ohlcv=minute_ohlcv(),
            funding_rate=0.0,
            trace_id=f"PARITY:{j}",
        )
    s.kernel.r2_flush_completed_hour(t0 + HOUR_MS)
    minute_snap = s.base_snapshot()
    base_snap = rt.physics.snapshot_kernel(
        base, rt.physics_contract, s.support_state
    )
    assert economic_state_projection(minute_snap) == economic_state_projection(base_snap)


def test_minute_chronology_resolves_take_before_later_stop():
    rt = runtime()
    s = MinutePhysicsSessionR2.initialize(rt, account_id="E6R2:FIRSTHIT")
    t0 = 1_700_000_000_000 // HOUR_MS * HOUR_MS
    warmup(s, t0 - 16 * HOUR_MS)
    s.step_intent(
        direction_v55=LONG, risk=0.5, symbol="BTCUSDT",
        open_time_ms=t0, ohlcv=minute_ohlcv(), funding_rate=0.0, trace_id="ENTRY",
    )
    stop = float(s.kernel.state.current_stop_price)
    take = float(s.kernel.state.current_take_price)
    assert stop < 100.0 < take
    s.step_intent(
        direction_v55=FLAT, risk=0.0, symbol="BTCUSDT",
        open_time_ms=t0 + MINUTE_MS,
        ohlcv=(100.0, take + 0.01, 100.0, take, 1.0),
        funding_rate=0.0, trace_id="TAKE_FIRST",
    )
    assert s.kernel.state.take_profit_count == 1
    assert s.kernel.state.stop_loss_count == 0
    assert abs(float(s.kernel.state.position)) < 1e-12


def test_same_minute_double_touch_keeps_frozen_stop_first_tie_break():
    rt = runtime()
    s = MinutePhysicsSessionR2.initialize(rt, account_id="E6R2:DOUBLE")
    t0 = 1_700_000_000_000 // HOUR_MS * HOUR_MS
    warmup(s, t0 - 16 * HOUR_MS)
    s.step_intent(
        direction_v55=LONG, risk=0.5, symbol="BTCUSDT",
        open_time_ms=t0, ohlcv=minute_ohlcv(), funding_rate=0.0, trace_id="ENTRY",
    )
    stop = float(s.kernel.state.current_stop_price)
    take = float(s.kernel.state.current_take_price)
    s.step_intent(
        direction_v55=FLAT, risk=0.0, symbol="BTCUSDT",
        open_time_ms=t0 + MINUTE_MS,
        ohlcv=(100.0, take + 0.01, stop - 0.01, 100.0, 1.0),
        funding_rate=0.0, trace_id="DOUBLE",
    )
    assert s.kernel.state.stop_loss_count == 1
    assert s.kernel.state.take_profit_count == 0
    assert s.kernel.state.last_close_reason == "stop_loss"


def test_72h_time_stop_uses_elapsed_hours_not_72_minutes():
    rt = runtime()
    s = MinutePhysicsSessionR2.initialize(rt, account_id="E6R2:H72")
    t0 = 1_700_000_000_000 // HOUR_MS * HOUR_MS
    warmup(s, t0 - 16 * HOUR_MS)
    for j in range(72 * 60):
        s.step_intent(
            direction_v55=LONG if j == 0 else FLAT,
            risk=0.25 if j == 0 else 0.0,
            symbol="BTCUSDT",
            open_time_ms=t0 + j * MINUTE_MS,
            ohlcv=minute_ohlcv(),
            funding_rate=0.0,
            trace_id=f"H72:{j}",
        )
    assert s.kernel.state.time_stop_count == 0
    assert s.kernel.state.position_age_bars == 72
    assert abs(float(s.kernel.state.position)) > 1e-12

    s.step_intent(
        direction_v55=FLAT, risk=0.0, symbol="BTCUSDT",
        open_time_ms=t0 + 72 * 60 * MINUTE_MS,
        ohlcv=minute_ohlcv(),
        funding_rate=0.0,
        trace_id="H72:EXIT",
    )
    assert s.kernel.state.time_stop_count == 1
    assert s.kernel.state.last_close_reason == "time_stop"
    assert abs(float(s.kernel.state.position)) < 1e-12
