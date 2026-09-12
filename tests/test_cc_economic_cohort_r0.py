import pytest
from cb16_local_opt.cc_economic_cohort_r0 import *
def test_cohort_preregisters_population_weight_capital_horizon_and_status():
    c=EconomicCohort("c",{"a":"s0","b":"s1"},{"a":1,"b":2},"cap","h","generation_chain","G1->G2","truncated",False); assert len(c.frozen_hash)==64 and not c.formal_horizon_owner_rule_resolved
def test_weight_population_mismatch_rejected():
    with pytest.raises(CohortIntegrityError): EconomicCohort("c",{"a":"s"},{"b":1},"cap","h","frozen_checkpoint","p","realized",True)
