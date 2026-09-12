import pytest
from cb16_local_opt.cc_economic_cohort_r0 import EconomicCohort
from cb16_local_opt.cc_economic_baselines_r0 import compute_baselines
from cb16_local_opt.cc_economic_evaluator_r0 import *

def setup():
    c=EconomicCohort("c",{"a":"s","b":"s","c":"s"},{"a":1,"b":1,"c":1},"cap","h","generation_chain","chain","realized",True); bh,fl=compute_baselines(c,start_prices={k:100 for k in "abc"},end_prices={k:100 for k in "abc"}); return c,bh,fl
def test_arithmetic_primary_and_failures_remain_denominator():
    c,bh,fl=setup(); e=evaluate_arithmetic_returns("e",c,[AccountEconomicResult("a",1.,"complete"),AccountEconomicResult("b",-1.,"liquidated",True),AccountEconomicResult("c",0.,"complete")],bh,fl); assert e.mean_arithmetic_return==0 and e.liquidation_or_debt_frequency==pytest.approx(1/3); assert e.to_w05(c)["mean_arithmetic_return"]==0
def test_deleting_failed_account_changes_result_and_is_rejected():
    c,bh,fl=setup()
    with pytest.raises(EvaluationIntegrityError): evaluate_arithmetic_returns("e",c,[AccountEconomicResult("a",1.,"complete"),AccountEconomicResult("c",0.,"complete")],bh,fl)
