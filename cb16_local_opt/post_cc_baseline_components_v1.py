from __future__ import annotations

from dataclasses import dataclass
import math

BASELINE_COMPONENT_CONTRACT_ID = "CB16_R11_POST_CC_BASELINE_COMPONENTS_V1"
PASS = "PASS"
FAIL = "FAIL"
TIE = "TIE"
NOT_APPLICABLE = "NOT_APPLICABLE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
_VALID = {PASS, FAIL, TIE, NOT_APPLICABLE, INSUFFICIENT_EVIDENCE}


@dataclass(frozen=True)
class BaselineComponentV1:
    status: str
    delta: float | None

    def validate(self) -> "BaselineComponentV1":
        if self.status not in _VALID:
            raise ValueError("invalid baseline component status")
        if self.delta is not None and not math.isfinite(float(self.delta)):
            raise ValueError("baseline delta must be finite when present")
        if self.status in {PASS, FAIL, TIE} and self.delta is None:
            raise ValueError("numeric status requires delta")
        if self.status == PASS and not float(self.delta) > 0:
            raise ValueError("PASS requires positive delta")
        if self.status == FAIL and not float(self.delta) < 0:
            raise ValueError("FAIL requires negative delta")
        if self.status == TIE and float(self.delta) != 0.0:
            raise ValueError("TIE requires zero delta")
        return self


@dataclass(frozen=True)
class BaselineComponentsV1:
    contract_id: str
    buy_and_hold: BaselineComponentV1
    flat: BaselineComponentV1

    def validate(self) -> "BaselineComponentsV1":
        if self.contract_id != BASELINE_COMPONENT_CONTRACT_ID:
            raise ValueError("unexpected baseline component contract")
        self.buy_and_hold.validate()
        self.flat.validate()
        return self


def _from_delta(delta: float) -> BaselineComponentV1:
    if not math.isfinite(float(delta)):
        raise ValueError("baseline delta must be finite")
    status = PASS if delta > 0 else FAIL if delta < 0 else TIE
    return BaselineComponentV1(status, float(delta)).validate()


def from_deltas(*, buy_hold_delta: float, flat_delta: float) -> BaselineComponentsV1:
    return BaselineComponentsV1(
        BASELINE_COMPONENT_CONTRACT_ID,
        _from_delta(buy_hold_delta),
        _from_delta(flat_delta),
    ).validate()
