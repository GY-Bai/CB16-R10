"""The single v5 reward definition: account log-equity growth."""
# V5.5 FROZEN KERNEL: numerical semantics must remain parity-locked to audited V5.4.
from __future__ import annotations

import math


NUMERICAL_EQUITY_FLOOR_RATIO = 1e-12


def log_equity_reward(
    previous_equity: float,
    new_equity: float,
    initial_equity: float,
) -> float:
    """Return log(new/previous), with only a numerical floor at bankruptcy.

    Zero equity has a mathematically infinite log loss.  The fixed floor keeps
    the optimizer finite; it is not a tunable reward term and is derived only
    from the account's initial unit scale.
    """
    if previous_equity <= 0:
        return 0.0
    floor = max(float(initial_equity) * NUMERICAL_EQUITY_FLOOR_RATIO, 1e-300)
    return math.log(max(float(new_equity), floor) / float(previous_equity))
