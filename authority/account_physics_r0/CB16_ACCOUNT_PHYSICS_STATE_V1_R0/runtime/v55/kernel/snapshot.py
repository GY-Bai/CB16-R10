"""Non-numerical snapshot/restore helpers around the frozen trading kernel."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

from .engine import FrozenTradingKernel
from .state import AccountState


@dataclass(frozen=True)
class KernelSnapshot:
    state: AccountState
    last_bar_time: datetime | None
    symbol: str | None
    last_carry_cost: float


def take_snapshot(kernel: FrozenTradingKernel) -> KernelSnapshot:
    return KernelSnapshot(
        state=deepcopy(kernel.state),
        last_bar_time=kernel._last_bar_time,
        symbol=kernel._symbol,
        last_carry_cost=float(kernel._last_carry_cost),
    )


def restore_snapshot(kernel: FrozenTradingKernel, snapshot: KernelSnapshot) -> None:
    kernel.state = deepcopy(snapshot.state)
    kernel._last_bar_time = snapshot.last_bar_time
    kernel._symbol = snapshot.symbol
    kernel._last_carry_cost = float(snapshot.last_carry_cost)


def clone_from_snapshot(source: FrozenTradingKernel, snapshot: KernelSnapshot) -> FrozenTradingKernel:
    clone = FrozenTradingKernel(source.config)
    restore_snapshot(clone, snapshot)
    return clone
