from __future__ import annotations

"""Execution entry for Training Admission R0 with exact frozen sensory alignment.

The preregistered Student decision event is HH:59 and execution is the next minute
open HH+1:00. Frozen SensoryDecisionFrameR10 is nominally stamped HH+1:00, while
both micro60 and the final hourly64 bar end at HH:59. The hourly64 window therefore
includes the just-completed current hour; it is not shifted back by one hour.
"""

import numpy as np

from cb16_local_opt.binance_archive_input_r10 import (
    MINUTE_MS,
    SensoryDecisionFrameR10,
    ordered4h30_from_hourly,
    stamps_from_open_times_ms,
)
from cb16_local_opt.r102_common import HOUR_MS
from scripts import r11_longtraj_training_admission_r0 as impl


def build_sensory_frame_exact_r0(records, parent_index: int, symbol: str) -> SensoryDecisionFrameR10:
    idx = int(parent_index)
    impl.require(int(records[idx].open_time) % HOUR_MS == HOUR_MS - MINUTE_MS, "ADMISSION_PARENT_NOT_HH59")
    current_start = idx - 59
    hourly_start = current_start - 63 * 60
    impl.require(hourly_start >= 0, "ADMISSION_SENSORY_PREHISTORY_MISSING")
    current = records[current_start:idx + 1]
    hourly_minutes = records[hourly_start:idx + 1]
    impl.require(len(current) == 60, "ADMISSION_MICRO_SLICE_LENGTH")
    impl.require(len(hourly_minutes) == 64 * 60, "ADMISSION_HOURLY64_SLICE_LENGTH")
    impl.require(
        all(int(b.open_time) - int(a.open_time) == MINUTE_MS for a, b in zip(hourly_minutes, hourly_minutes[1:])),
        "ADMISSION_SENSORY_GAP",
    )

    hourly = np.stack(
        [impl._aggregate_hour(hourly_minutes[i:i + 60]) for i in range(0, len(hourly_minutes), 60)],
        axis=0,
    )
    hourly_open_ms = np.asarray(
        [int(hourly_minutes[i].open_time) for i in range(0, len(hourly_minutes), 60)],
        dtype=np.int64,
    )
    micro = np.stack([r.ohlcv() for r in current], axis=0).astype(np.float32)
    micro_ms = np.asarray([int(r.open_time) for r in current], dtype=np.int64)
    parent_ms = int(records[idx].open_time)
    nominal_h = parent_ms + MINUTE_MS
    impl.require(int(micro_ms[-1]) == parent_ms, "ADMISSION_MICRO_VISIBILITY_DRIFT")
    impl.require(int(hourly_open_ms[-1]) + HOUR_MS - MINUTE_MS == parent_ms, "ADMISSION_HOURLY_VISIBILITY_DRIFT")
    impl.require(np.array_equal(hourly[-1], impl._aggregate_hour(current)), "ADMISSION_CURRENT_HOUR_NOT_LAST_HOURLY_BAR")

    return SensoryDecisionFrameR10(
        symbol=symbol,
        decision_time_ms=nominal_h,
        micro_1m_60x5=micro,
        micro_stamps_60x5=stamps_from_open_times_ms(micro_ms),
        hourly_64x5=hourly,
        hourly_stamps_64x5=stamps_from_open_times_ms(hourly_open_ms),
        ordered4h30=ordered4h30_from_hourly(hourly[-24:]),
    )


# Bind the exact frozen-frame semantics before entering the preregistered runner.
impl.build_sensory_frame_r0 = build_sensory_frame_exact_r0


if __name__ == "__main__":
    raise SystemExit(impl.main())
