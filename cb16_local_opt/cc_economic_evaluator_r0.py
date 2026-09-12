from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
from .cc_economic_cohort_r0 import EconomicCohort
from .cc_experience_wire_r0 import CCEconomicResultV1

@dataclass(frozen=True)
class AccountEconomicOutcome:
    account_lineage_id: str
    arithmetic_return: float
    failed: bool=False
    liquidated: bool=False
    debt: bool=False
    truncated: bool=False


def _quantile(xs: list[float], q: float) -> float:
    ys=sorted(xs); pos=(len(ys)-1)*q; lo=int(pos); hi=min(lo+1,len(ys)-1); frac=pos-lo
    return ys[lo]*(1-frac)+ys[hi]*frac


def evaluate(*, evaluation_id: str, cohort: EconomicCohort, outcomes: Sequence[AccountEconomicOutcome],
             buy_hold_return: float, flat_return: float=0.0, result_scope: str="SYNTHETIC_COMPONENT") -> CCEconomicResultV1:
    cohort.validate(); by_id={o.account_lineage_id:o for o in outcomes}
    expected=set(cohort.account_start_states)
    if set(by_id)!=expected or len(by_id)!=len(outcomes): raise ValueError("evaluation must contain every preregistered account exactly once")
    total_w=sum(cohort.weights.values())
    mean=sum(cohort.weights[k]*by_id[k].arithmetic_return for k in expected)/total_w
    xs=[by_id[k].arithmetic_return for k in expected]
    failures={"failed":sum(o.failed for o in outcomes),"liquidated":sum(o.liquidated for o in outcomes),"debt":sum(o.debt for o in outcomes),"truncated":sum(o.truncated for o in outcomes)}
    tails={"median":_quantile(xs,.5),"q05":_quantile(xs,.05),"q95":_quantile(xs,.95),"survival":sum(not o.failed for o in outcomes)/len(outcomes)}
    accounts=tuple({"account_lineage_id":o.account_lineage_id,"arithmetic_return":o.arithmetic_return,"failed":o.failed,"liquidated":o.liquidated,"debt":o.debt,"truncated":o.truncated} for o in outcomes)
    return CCEconomicResultV1(evaluation_id,cohort.policy_object_type,cohort.policy_identity,cohort.cohort_id,
        cohort.common_horizon_id,cohort.capital_denominator_id,accounts,mean,mean-buy_hold_return,mean-flat_return,failures,tails,result_scope).validate()
