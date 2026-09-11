from __future__ import annotations

import numpy as np

from cb16_local_opt.binance_archive_input_r10 import KlineRecord, MINUTE_MS
from scripts.r11_longtraj_training_admission_r0 import SENSORY_PREFIX_MINUTES
from scripts.r11_longtraj_training_admission_r0_entry import build_sensory_frame_exact_r0


def _records(n: int):
    return [
        KlineRecord(
            open_time=i * MINUTE_MS,
            open=100.0 + i * 0.001,
            high=101.0 + i * 0.001,
            low=99.0 + i * 0.001,
            close=100.1 + i * 0.001,
            volume=10.0 + i % 5,
            close_time=(i + 1) * MINUTE_MS - 1,
            number_of_trades=10 + i % 3,
        )
        for i in range(n)
    ]


def test_hourly64_includes_just_completed_current_hour_and_never_h_open():
    rows = _records(SENSORY_PREFIX_MINUTES)
    idx = len(rows) - 1
    frame = build_sensory_frame_exact_r0(rows, idx, "BTCUSDT")
    assert frame.decision_time_ms == rows[idx].open_time + MINUTE_MS
    assert frame.micro_1m_60x5.shape == (60, 5)
    assert frame.hourly_64x5.shape == (64, 5)
    assert np.array_equal(frame.hourly_64x5[-1], np.asarray([
        rows[idx - 59].open,
        max(r.high for r in rows[idx - 59:idx + 1]),
        min(r.low for r in rows[idx - 59:idx + 1]),
        rows[idx].close,
        sum(r.volume for r in rows[idx - 59:idx + 1]),
    ], dtype=np.float32))
    assert np.isclose(frame.micro_1m_60x5[-1, 3], rows[idx].close)
