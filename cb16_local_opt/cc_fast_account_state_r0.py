
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

import numpy as np

ACCOUNT_SOA_SCHEMA = "CB16_R11_CC_FAST_ACCOUNT_SOA_V1"


def lineage_u64(lineage_id: str) -> np.uint64:
    if not lineage_id:
        raise ValueError("LINEAGE_ID_EMPTY")
    return np.uint64(int.from_bytes(hashlib.sha256(lineage_id.encode("utf-8")).digest()[:8], "big"))


@dataclass
class AccountStateSoA:
    lineage: np.ndarray
    decision_index: np.ndarray
    quantity: np.ndarray
    avg_cost: np.ndarray
    cash: np.ndarray
    liability: np.ndarray
    equity: np.ndarray
    realized_pnl: np.ndarray
    fees_paid: np.ndarray
    funding_paid: np.ndarray
    terminal: np.ndarray

    @classmethod
    def allocate(
        cls,
        lineage_ids: Sequence[str],
        *,
        initial_equity: float = 10_000.0,
    ) -> "AccountStateSoA":
        n = len(lineage_ids)
        if n <= 0:
            raise ValueError("ACCOUNT_COUNT_INVALID")
        if not np.isfinite(initial_equity) or initial_equity <= 0:
            raise ValueError("INITIAL_EQUITY_INVALID")
        lineage = np.fromiter((lineage_u64(x) for x in lineage_ids), dtype=np.uint64, count=n)
        return cls(
            lineage=lineage,
            decision_index=np.zeros(n, dtype=np.uint64),
            quantity=np.zeros(n, dtype=np.float64),
            avg_cost=np.zeros(n, dtype=np.float64),
            cash=np.full(n, float(initial_equity), dtype=np.float64),
            liability=np.zeros(n, dtype=np.float64),
            equity=np.full(n, float(initial_equity), dtype=np.float64),
            realized_pnl=np.zeros(n, dtype=np.float64),
            fees_paid=np.zeros(n, dtype=np.float64),
            funding_paid=np.zeros(n, dtype=np.float64),
            terminal=np.zeros(n, dtype=np.bool_),
        )

    @property
    def count(self) -> int:
        return int(self.quantity.shape[0])

    @property
    def nbytes(self) -> int:
        return int(sum(x.nbytes for x in (
            self.lineage, self.decision_index, self.quantity, self.avg_cost, self.cash,
            self.liability, self.equity, self.realized_pnl, self.fees_paid,
            self.funding_paid, self.terminal,
        )))

    def validate(self) -> "AccountStateSoA":
        arrays = (
            self.lineage, self.decision_index, self.quantity, self.avg_cost, self.cash,
            self.liability, self.equity, self.realized_pnl, self.fees_paid,
            self.funding_paid, self.terminal,
        )
        n = len(self.quantity)
        if n == 0 or any(len(x) != n for x in arrays):
            raise ValueError("SOA_LENGTH_MISMATCH")
        for arr in (self.quantity, self.avg_cost, self.cash, self.liability, self.equity,
                    self.realized_pnl, self.fees_paid, self.funding_paid):
            if not np.all(np.isfinite(arr)):
                raise ValueError("SOA_NONFINITE")
            if not arr.flags.c_contiguous:
                raise ValueError("SOA_NONCONTIGUOUS")
        return self

    def snapshot(self, i: int) -> dict[str, float | int | bool]:
        if i < 0 or i >= self.count:
            raise IndexError(i)
        return {
            "lineage": int(self.lineage[i]),
            "decision_index": int(self.decision_index[i]),
            "quantity": float(self.quantity[i]),
            "avg_cost": float(self.avg_cost[i]),
            "cash": float(self.cash[i]),
            "liability": float(self.liability[i]),
            "equity": float(self.equity[i]),
            "realized_pnl": float(self.realized_pnl[i]),
            "fees_paid": float(self.fees_paid[i]),
            "funding_paid": float(self.funding_paid[i]),
            "terminal": bool(self.terminal[i]),
        }

    def copy(self) -> "AccountStateSoA":
        return AccountStateSoA(**{name: getattr(self, name).copy() for name in self.__dataclass_fields__})
