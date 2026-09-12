import pytest
from cb16_local_opt.cc_economic_cohort_r0 import EconomicCohort
from cb16_local_opt.cc_economic_evaluator_r0 import AccountEconomicOutcome, evaluate

def cohort(): return EconomicCohort('c',{'dead':'s','alive':'s'},{'dead':1,'alive':1},'cap','T','frozen_checkpoint','p','REALIZED')

def test_arithmetic_expected_return_keeps_failures_debt_and_truncation():
    r=evaluate(evaluation_id='e',cohort=cohort(),outcomes=[AccountEconomicOutcome('dead',-1,True,True,True),AccountEconomicOutcome('alive',2,truncated=True)],buy_hold_return=.25)
    assert r.mean_arithmetic_return==.5 and r.failure_counts=={'failed':1,'liquidated':1,'debt':1,'truncated':1}

def test_survivor_deletion_fails_integrity():
    with pytest.raises(ValueError): evaluate(evaluation_id='e',cohort=cohort(),outcomes=[AccountEconomicOutcome('alive',2)],buy_hold_return=0)
