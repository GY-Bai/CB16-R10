from __future__ import annotations

import numpy as np
import pytest

from scripts.r11_m_series_h5_16_fake_env_clock_marginal_repair_r0 import (
    shift_clock_environment_clock_marginal_repair_h516,
)


def _clock_labels(ts: np.ndarray, env: np.ndarray) -> np.ndarray:
    clocks = np.asarray(sorted(set(int(x) for x in ts.tolist())), dtype=np.int64)
    return np.asarray([int(env[np.flatnonzero(ts == c)[0]]) for c in clocks], dtype=np.int8)


def test_repair_preserves_defined_clock_label_marginal_even_when_row_marginal_changes():
    # Six clocks, but clock 30 contains three future groups.  A +1 clock-label
    # rotation therefore preserves the three-one / three-zero CLOCK marginal
    # while legitimately changing the FUTURE-GROUP-ROW marginal.
    ts = np.asarray([10, 20, 30, 30, 30, 40, 50, 60], dtype=np.int64)
    env = np.asarray([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    shifted = shift_clock_environment_clock_marginal_repair_h516(ts, env, 1)

    before_clock = _clock_labels(ts, env)
    after_clock = _clock_labels(ts, shifted)
    assert int(before_clock.sum()) == 3
    assert int(after_clock.sum()) == 3
    assert sorted(before_clock.tolist()) == sorted(after_clock.tolist())

    # Unequal row multiplicity means this is not a required invariant.
    assert int(env.sum()) == 3
    assert int(shifted.sum()) == 5

    # All future groups at one decision clock still receive exactly one label.
    assert np.unique(shifted[ts == 30]).tolist() == [1]
    assert not np.array_equal(before_clock, after_clock)


def test_repair_rejects_identity_on_defined_unique_clock_sequence():
    ts = np.asarray([10, 10, 20, 30, 40, 50, 60], dtype=np.int64)
    env = np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    with pytest.raises(RuntimeError, match="H516_FAKE_ENV_SHIFT_IDENTITY_INDEX"):
        shift_clock_environment_clock_marginal_repair_h516(ts, env, 6)


def test_repair_is_deterministic_and_does_not_modify_inputs():
    ts = np.asarray([10, 20, 30, 30, 40, 50, 60], dtype=np.int64)
    env = np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    ts0 = ts.copy()
    env0 = env.copy()
    a = shift_clock_environment_clock_marginal_repair_h516(ts, env, 1)
    b = shift_clock_environment_clock_marginal_repair_h516(ts, env, 1)
    assert np.array_equal(a, b)
    assert np.array_equal(ts, ts0)
    assert np.array_equal(env, env0)
