"""The only policy action contract used by v5."""
# V5.5 FROZEN KERNEL: numerical semantics must remain parity-locked to audited V5.4.
from __future__ import annotations

from enum import IntEnum
from numbers import Integral


class TradeDecision(IntEnum):
    SHORT = 0
    FLAT = 1
    LONG = 2


NUM_DECISIONS = len(TradeDecision)


def validate_decision(value: object) -> TradeDecision:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("decision must be an integer in [0, 2]")
    try:
        return TradeDecision(int(value))
    except ValueError as exc:
        raise ValueError("decision must be an integer in [0, 2]") from exc


def decision_sign(decision: int | TradeDecision) -> int:
    checked = validate_decision(decision)
    return -1 if checked is TradeDecision.SHORT else int(checked is TradeDecision.LONG)
