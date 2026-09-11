from __future__ import annotations

import numpy as np

from scripts.r11_longtraj_generalization_r0_evaluate import paired_bootstrap, smooth_l1_constant_minimizer


def _smooth_l1_sum(x: float, y: np.ndarray, beta: float = 0.05) -> float:
    d = np.abs(float(x) - y)
    z = np.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta)
    return float(z.sum())


def test_smooth_l1_constant_minimizer_is_convex_optimum() -> None:
    y = np.asarray([0.05, 0.12, 0.19, 0.31, 0.44, 0.71], dtype=np.float64)
    x = smooth_l1_constant_minimizer(y)
    assert 0.0 <= x <= 1.0
    eps = 1e-7
    assert _smooth_l1_sum(x, y) <= _smooth_l1_sum(max(0.0, x - eps), y) + 1e-12
    assert _smooth_l1_sum(x, y) <= _smooth_l1_sum(min(1.0, x + eps), y) + 1e-12


def test_paired_bootstrap_is_deterministic_and_preserves_sign() -> None:
    diff = np.linspace(-0.4, -0.1, 48, dtype=np.float64)
    a = paired_bootstrap(diff, seed=20260911, reps=10000)
    b = paired_bootstrap(diff, seed=20260911, reps=10000)
    assert a == b
    assert a["point"] < 0.0
    assert a["ci_high"] < 0.0
    assert a["replicates"] == 10000


def test_cyclic_control_family_has_exactly_47_nonidentity_shifts() -> None:
    n = 48
    base = np.arange(n)
    shifted = [tuple(((base + k) % n).tolist()) for k in range(1, n)]
    assert len(shifted) == 47
    assert len(set(shifted)) == 47
    assert all(s != tuple(base.tolist()) for s in shifted)
