from .cc_economic_baselines_r0 import compute_baselines
from .cc_economic_capital_r0 import CapitalEvent,summarize_capital
from .cc_economic_cohort_r0 import EconomicCohort
from .cc_economic_evaluator_r0 import AccountEconomicResult,EvaluationIntegrityError,evaluate_arithmetic_returns
from .cc_economic_policy_identity_r0 import identity_from_ownership,PolicyIdentityError,EconomicPolicyIdentity
from .cc_economic_promotion_r0 import decide_promotion
def toy_cohort(resolved=True):return EconomicCohort("toy-cohort",{"a":"s0","b":"s0","c":"s0"},{"a":1,"b":1,"c":1},"capital-v1","horizon-v1","generation_chain","toy-policy","realized",resolved)
def high_bankruptcy_higher_arithmetic_expectation():
    c=toy_cohort(); bh,fl=compute_baselines(c,start_prices={k:100 for k in "abc"},end_prices={k:105 for k in "abc"}); rs=[AccountEconomicResult("a",4,"complete"),AccountEconomicResult("b",-1,"liquidated",True),AccountEconomicResult("c",-1,"debt",True)]; e=evaluate_arithmetic_returns("toy-high",c,rs,bh,fl); return {"mean":e.mean_arithmetic_return,"bankruptcy_frequency":e.liquidation_or_debt_frequency,"decision":decide_promotion(e,c).decision.value}
def survivor_filter_is_rejected():
    c=toy_cohort(); bh,fl=compute_baselines(c,start_prices={k:100 for k in "abc"},end_prices={k:100 for k in "abc"})
    try:evaluate_arithmetic_returns("cheat",c,[AccountEconomicResult("a",.5,"complete")],bh,fl)
    except EvaluationIntegrityError:return True
    return False
def capital_injection_changes_denominator():return summarize_capital({"a":100},[]).gross_contributed_denominator,summarize_capital({"a":100},[CapitalEvent("e1","a2","lineage_restart",50)]).gross_contributed_denominator
def mixed_generation_not_final_checkpoint():
    i=identity_from_ownership([("G3","P3"),("G4","P4"),("G5","P5")])
    try:EconomicPolicyIdentity("frozen_checkpoint","G5:P5",i.ordered_generation_policy_refs)
    except PolicyIdentityError:return i.policy_object_type=="generation_chain"
    return False
def baseline_disagreement_is_unresolved():
    c=toy_cohort(); bh,fl=compute_baselines(c,start_prices={k:100 for k in "abc"},end_prices={k:150 for k in "abc"}); e=evaluate_arithmetic_returns("conflict",c,[AccountEconomicResult(k,.2,"complete") for k in "abc"],bh,fl); return decide_promotion(e,c).decision.value
