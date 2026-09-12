from dataclasses import dataclass
from typing import Mapping
from .cc_economic_cohort_r0 import EconomicCohort
class BaselineError(ValueError): pass
@dataclass(frozen=True)
class BaselineResult:
    baseline_type:str; cohort_id:str; common_horizon_id:str; capital_denominator_id:str; account_returns:Mapping[str,float]; mean_arithmetic_return:float; continued_through_common_horizon:bool
def _mean(v,c):
    z=sum(float(c.weights[k]) for k in v); return sum(float(v[k])*float(c.weights[k]) for k in v)/z
def compute_baselines(cohort:EconomicCohort,*,start_prices:Mapping[str,float],end_prices:Mapping[str,float]):
    ids=cohort.account_ids
    if set(start_prices)!=ids or set(end_prices)!=ids: raise BaselineError("baseline must cover fixed cohort")
    bh={}
    for k in ids:
        if float(start_prices[k])<=0: raise BaselineError("nonpositive start")
        bh[k]=float(end_prices[k])/float(start_prices[k])-1
    flat={k:0.0 for k in ids}; common=(cohort.cohort_id,cohort.common_horizon_id,cohort.capital_denominator_id)
    return BaselineResult("buy_and_hold",*common,bh,_mean(bh,cohort),True),BaselineResult("flat",*common,flat,0.0,True)
