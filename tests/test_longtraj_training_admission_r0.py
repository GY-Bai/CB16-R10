from __future__ import annotations

import numpy as np
import torch

from cb16_local_opt.binance_archive_input_r10 import KlineRecord, MINUTE_MS
from cb16_local_opt.r102_physics import CANDIDATES_R102, FLAT, LONG, SHORT
from scripts.r11_longtraj_training_admission_r0 import (
    DIRECTION_TO_TEACHER,
    SENSORY_PREFIX_MINUTES,
    _tree_exact_equal,
    build_sensory_frame_r0,
)


def _records(n: int):
    out = []
    for i in range(n):
        px = 100.0 + 0.001 * i
        out.append(KlineRecord(
            open_time=i * MINUTE_MS,
            open=px,
            high=px + 1.0,
            low=px - 1.0,
            close=px + 0.1,
            volume=10.0 + i % 7,
            close_time=(i + 1) * MINUTE_MS - 1,
            number_of_trades=10 + i % 3,
        ))
    return out


def test_full_grid_is_existing_canonical_nine_arm_grid():
    assert tuple(CANDIDATES_R102) == (
        (FLAT, 0.0),
        (SHORT, 0.25), (SHORT, 0.5), (SHORT, 0.75), (SHORT, 1.0),
        (LONG, 0.25), (LONG, 0.5), (LONG, 0.75), (LONG, 1.0),
    )
    assert DIRECTION_TO_TEACHER == {SHORT: -1, FLAT: 0, LONG: 1}


def test_sensory_frame_nominal_h_contains_nothing_after_h_minus_1m():
    rows = _records(SENSORY_PREFIX_MINUTES)
    parent_idx = len(rows) - 1
    frame = build_sensory_frame_r0(rows, parent_idx, "BTCUSDT")
    assert frame.decision_time_ms == rows[parent_idx].open_time + MINUTE_MS
    assert frame.micro_1m_60x5.shape == (60, 5)
    assert frame.hourly_64x5.shape == (64, 5)
    assert frame.ordered4h30.shape == (30,)
    assert np.isclose(frame.micro_1m_60x5[-1, 3], rows[parent_idx].close)
    assert np.isclose(frame.hourly_64x5[-1, 3], rows[parent_idx - 60].close)


def test_checkpoint_tree_exact_comparison_is_bitwise_for_tensors():
    a = {"x": torch.tensor([1.0, 2.0], dtype=torch.float32), "nested": [3, ("a", 4)]}
    b = {"x": a["x"].clone(), "nested": [3, ("a", 4)]}
    c = {"x": torch.tensor([1.0, 2.0001], dtype=torch.float32), "nested": [3, ("a", 4)]}
    assert _tree_exact_equal(a, b)
    assert not _tree_exact_equal(a, c)
