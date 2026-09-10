#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import cb16_local_opt.full_state_nonlinear_utility_invariance_h516 as h516

REPAIR_GATE_BLOB = "6eeb2a9ad91af8ca3e7906bca5fbd2effa60d8ba"
FROZEN_HELPER_BLOB = "bfa28be3ad6676f26676404bb170a73b474b6192"
FROZEN_EXECUTOR_BLOB = "f7351e08f6a4c22c009843fc4da60661685f80d5"


def shift_clock_environment_clock_marginal_repair_h516(
    ts: np.ndarray | list[int] | tuple[int, ...],
    env: np.ndarray | list[int] | tuple[int, ...],
    shift: int,
) -> np.ndarray:
    """Exact H5.16 fake-environment shift with the preregistered clock-level marginal guard.

    The scientific operation is unchanged from the frozen helper: shift the binary
    environment label vector on sorted unique decision clocks, then broadcast the
    shifted clock label to every future-group row at that clock.  Only the final
    marginal-preservation assertion is repaired: it is evaluated on the defined
    per-clock label vector rather than on a row vector with unequal clock
    multiplicities.
    """
    t = np.asarray(ts, dtype=np.int64).reshape(-1)
    e = np.asarray(env, dtype=np.int8).reshape(-1)
    clocks = np.asarray(sorted(set(int(x) for x in t.tolist())), dtype=np.int64)
    h516.require(len(clocks) > 0, "H516_FAKE_ENV_EMPTY_CLOCK_SEQUENCE")
    base = np.asarray([int(e[np.flatnonzero(t == c)[0]]) for c in clocks], dtype=np.int8)
    for c, v in zip(clocks.tolist(), base.tolist()):
        h516.require(np.all(e[t == c] == v), f"H516_CLOCK_ENV_DRIFT:{c}")
    k = int(shift) % len(clocks)
    h516.require(k != 0, f"H516_FAKE_ENV_SHIFT_IDENTITY_INDEX:{shift}:{len(clocks)}")
    shifted = base[(np.arange(len(base), dtype=np.int64) + k) % len(base)]
    h516.require(not np.array_equal(base, shifted), f"H516_FAKE_ENV_LABEL_IDENTITY:{shift}:{len(clocks)}")
    h516.require(int(np.sum(shifted)) == int(np.sum(base)), "H516_FAKE_ENV_CLOCK_MARGINAL_DRIFT")
    mapping = {int(c): int(v) for c, v in zip(clocks.tolist(), shifted.tolist())}
    return np.asarray([mapping[int(v)] for v in t.tolist()], dtype=np.int8)


def install_repair_h516() -> None:
    h516._shift_clock_environment_h516 = shift_clock_environment_clock_marginal_repair_h516


def main() -> int:
    install_repair_h516()
    # Import only after installing the repair.  The frozen executor and all of its
    # scientific functions remain byte-identical; run_environment_arm_h516 looks
    # up the helper-module global at execution time.
    from scripts import r11_m_series_h5_16_full_state_nonlinear_utility_invariance_r0 as frozen_executor

    return int(frozen_executor.main())


if __name__ == "__main__":
    raise SystemExit(main())
