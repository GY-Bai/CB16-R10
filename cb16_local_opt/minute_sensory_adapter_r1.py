from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from .binance_archive_input_r10 import (
    AggregatedKline,
    HOUR_MS,
    MINUTE_MS,
    SensoryDecisionFrameR10,
    ordered4h30_from_hourly,
    stamps_from_open_times_ms,
)
from .full_minute_long_trajectory_r1 import MinuteEnvelopeR1


@dataclass(frozen=True)
class MinuteSensoryAuditR1:
    decision_time_ms: int
    current_hour_start_ms: int
    latest_completed_hour_start_ms: int
    micro_latest_time_ms: int
    macro_latest_hour_start_ms: int
    medium_latest_hour_start_ms: int


def _hour_start(ts: int) -> int:
    return (int(ts) // HOUR_MS) * HOUR_MS


def _finalize_hour(rows: list[MinuteEnvelopeR1], hour_start: int) -> AggregatedKline:
    if len(rows) != 60:
        raise RuntimeError(f"R1_HOUR_NOT_60_ROWS:{hour_start}:{len(rows)}")
    expected = [hour_start + i * MINUTE_MS for i in range(60)]
    got = [x.timestamp for x in rows]
    if got != expected:
        raise RuntimeError(f"R1_HOUR_NOT_CONTIGUOUS:{hour_start}")
    rr = [x.record for x in rows]
    return AggregatedKline(
        interval="1h",
        open_time=hour_start,
        open=float(rr[0].open),
        high=float(max(x.high for x in rr)),
        low=float(min(x.low for x in rr)),
        close=float(rr[-1].close),
        volume=float(sum(x.volume for x in rr)),
        close_time=int(rr[-1].close_time),
        quote_asset_volume=float(sum(x.quote_asset_volume for x in rr)),
        number_of_trades=int(sum(x.number_of_trades for x in rr)),
        taker_buy_base_asset_volume=float(sum(x.taker_buy_base_asset_volume for x in rr)),
        taker_buy_quote_asset_volume=float(sum(x.taker_buy_quote_asset_volume for x in rr)),
        source_rows=60,
    )


def iter_minute_sensory_frames_r1(
    symbol: str,
    minutes: Iterable[MinuteEnvelopeR1],
) -> Iterator[tuple[SensoryDecisionFrameR10, MinuteSensoryAuditR1]]:
    """Emit one causal sensory frame per eligible minute.

    Micro60 ends at the current minute.  Macro32 and Medium64 use only fully
    completed hours; the in-progress current hour is never aggregated into those
    lanes.  Thus the slow lanes are latched throughout an hour while Micro60 rolls.
    """
    micro: deque[MinuteEnvelopeR1] = deque(maxlen=60)
    hourly: deque[AggregatedKline] = deque(maxlen=64)
    cur_hour: int | None = None
    hour_rows: list[MinuteEnvelopeR1] = []
    prev_ts: int | None = None

    for env in minutes:
        ts = env.timestamp
        if prev_ts is not None and ts - prev_ts != MINUTE_MS:
            raise RuntimeError(f"R1_MINUTE_SENSORY_INPUT_NOT_CONTIGUOUS:{prev_ts}->{ts}")
        h = _hour_start(ts)
        if cur_hour is None:
            cur_hour = h
        elif h != cur_hour:
            if h != cur_hour + HOUR_MS:
                raise RuntimeError(f"R1_HOUR_JUMP_AFTER_HALT_EXPANSION:{cur_hour}->{h}")
            hourly.append(_finalize_hour(hour_rows, cur_hour))
            hour_rows = []
            cur_hour = h

        # Current minute is visible to Micro only after it has arrived.
        micro.append(env)
        hour_rows.append(env)
        prev_ts = ts

        if len(micro) < 60 or len(hourly) < 64:
            continue

        mi = np.asarray([[x.record.open, x.record.high, x.record.low, x.record.close, x.record.volume] for x in micro], dtype=np.float32)
        mits = np.asarray([x.timestamp for x in micro], dtype=np.int64)
        hh = list(hourly)
        hv = np.asarray([[x.open, x.high, x.low, x.close, x.volume] for x in hh], dtype=np.float32)
        hts = np.asarray([x.open_time for x in hh], dtype=np.int64)
        frame = SensoryDecisionFrameR10(
            symbol=symbol,
            decision_time_ms=ts,
            micro_1m_60x5=mi,
            micro_stamps_60x5=stamps_from_open_times_ms(mits),
            hourly_64x5=hv,
            hourly_stamps_64x5=stamps_from_open_times_ms(hts),
            ordered4h30=ordered4h30_from_hourly(hv[-24:]),
        )
        latest_hour = int(hts[-1])
        if latest_hour >= _hour_start(ts):
            raise RuntimeError("R1_UNFINISHED_HOUR_LEAK")
        yield frame, MinuteSensoryAuditR1(
            decision_time_ms=ts,
            current_hour_start_ms=_hour_start(ts),
            latest_completed_hour_start_ms=latest_hour,
            micro_latest_time_ms=int(mits[-1]),
            macro_latest_hour_start_ms=int(hts[-1]),
            medium_latest_hour_start_ms=int(hts[-1]),
        )
