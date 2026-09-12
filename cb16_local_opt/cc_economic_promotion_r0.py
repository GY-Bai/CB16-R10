from __future__ import annotations
from dataclasses import dataclass
from .cc_experience_wire_r0 import CCEconomicResultV1

UNRESOLVED_OWNER_DECISION="UNRESOLVED_OWNER_DECISION"
@dataclass(frozen=True)
class PromotionAssessment:
    status: str
    mean_arithmetic_return: float
    buy_hold_delta: float
    flat_delta: float

def assess(result: CCEconomicResultV1) -> PromotionAssessment:
    result.validate()
    if result.buy_hold_delta > 0 and result.flat_delta > 0: status="QUALIFIES_BOTH_BASELINES"
    elif result.buy_hold_delta < 0 and result.flat_delta < 0: status="FAILS_BOTH_BASELINES"
    elif result.buy_hold_delta == 0 and result.flat_delta == 0: status="TIED_BOTH_BASELINES"
    else: status=UNRESOLVED_OWNER_DECISION
    return PromotionAssessment(status,result.mean_arithmetic_return,result.buy_hold_delta,result.flat_delta)
