from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class BaselineResult:
    baseline: str
    cohort_id: str
    common_horizon_id: str
    capital_denominator_id: str
    mean_arithmetic_return: float


def compute_baselines(*, cohort_id: str, common_horizon_id: str, capital_denominator_id: str,
                      weights: Mapping[str,float], buy_hold_returns: Mapping[str,float]) -> tuple[BaselineResult,BaselineResult]:
    if set(weights)!=set(buy_hold_returns): raise ValueError("baseline must use identical cohort")
    denom=sum(weights.values())
    if denom<=0: raise ValueError("invalid weights")
    bh=sum(weights[k]*buy_hold_returns[k] for k in weights)/denom
    return (BaselineResult("BUY_HOLD",cohort_id,common_horizon_id,capital_denominator_id,bh),
            BaselineResult("FLAT",cohort_id,common_horizon_id,capital_denominator_id,0.0))
