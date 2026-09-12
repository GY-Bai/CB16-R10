from dataclasses import dataclass
from statistics import median
from math import isfinite
from typing import Mapping,Sequence
from .cc_economic_cohort_r0 import EconomicCohort
from .cc_economic_baselines_r0 import BaselineResult
class EvaluationIntegrityError(ValueError): pass
@dataclass(frozen=True)
class AccountEconomicResult: account_id:str; arithmetic_return:float; terminal_status:str; debt_or_liquidated:bool=False; truncated:bool=False
@dataclass(frozen=True)
class EconomicEvaluation:
    evaluation_id:str; cohort_id:str; mean_arithmetic_return:float; median_return:float; quantiles:Mapping[str,float]; liquidation_or_debt_frequency:float; survival_frequency:float; truncation_count:int; buy_hold_delta:float; flat_delta:float; account_results:tuple[AccountEconomicResult,...]
    def to_w05(self,cohort:EconomicCohort):
        failures={}
        for x in self.account_results:
            if x.terminal_status not in {"alive","complete"}: failures[x.terminal_status]=failures.get(x.terminal_status,0)+1
        return {"evaluation_id":self.evaluation_id,"policy_object_type":cohort.policy_object_type,"policy_identity":cohort.policy_identity,"cohort_id":cohort.cohort_id,"common_horizon_id":cohort.common_horizon_id,"capital_denominator_id":cohort.capital_denominator_id,"account_results":[x.__dict__ for x in self.account_results],"mean_arithmetic_return":self.mean_arithmetic_return,"buy_hold_delta":self.buy_hold_delta,"flat_delta":self.flat_delta,"failure_counts":failures,"tail_diagnostics":{"median":self.median_return,"quantiles":dict(self.quantiles),"liquidation_or_debt_frequency":self.liquidation_or_debt_frequency,"survival_frequency":self.survival_frequency,"truncation_count":self.truncation_count},"result_scope":"THREAD_C_LOCAL_SYNTHETIC_COMPONENT"}
def _q(vals,q):
    s=sorted(vals)
    if len(s)==1:return s[0]
    pos=q*(len(s)-1); lo=int(pos); hi=min(lo+1,len(s)-1); f=pos-lo; return s[lo]*(1-f)+s[hi]*f
def evaluate_arithmetic_returns(evaluation_id:str,cohort:EconomicCohort,account_results:Sequence[AccountEconomicResult],buy_hold:BaselineResult,flat:BaselineResult):
    by={x.account_id:x for x in account_results}
    if not evaluation_id or len(by)!=len(account_results) or set(by)!=cohort.account_ids: raise EvaluationIntegrityError("frozen cohort exact coverage required; survivor filtering forbidden")
    if any(not isfinite(float(x.arithmetic_return)) for x in account_results): raise EvaluationIntegrityError("nonfinite return")
    for b in (buy_hold,flat):
        if (b.cohort_id,b.common_horizon_id,b.capital_denominator_id)!=(cohort.cohort_id,cohort.common_horizon_id,cohort.capital_denominator_id): raise EvaluationIntegrityError("baseline mismatch")
    z=sum(float(cohort.weights[k]) for k in cohort.account_ids); mean=sum(float(by[k].arithmetic_return)*float(cohort.weights[k]) for k in cohort.account_ids)/z; vals=[float(x.arithmetic_return) for x in account_results]; bad=sum(x.debt_or_liquidated for x in account_results)
    return EconomicEvaluation(evaluation_id,cohort.cohort_id,mean,median(vals),{"q05":_q(vals,.05),"q50":_q(vals,.5),"q95":_q(vals,.95)},bad/len(vals),(len(vals)-bad)/len(vals),sum(x.truncated for x in account_results),mean-buy_hold.mean_arithmetic_return,mean-flat.mean_arithmetic_return,tuple(account_results))
