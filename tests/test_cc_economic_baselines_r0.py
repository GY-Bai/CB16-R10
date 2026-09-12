from cb16_local_opt.cc_economic_cohort_r0 import EconomicCohort
from cb16_local_opt.cc_economic_baselines_r0 import compute_baselines
def test_bh_and_flat_share_common_cohort_capital_horizon_and_continue():
    c=EconomicCohort("c",{"a":"s","b":"s"},{"a":1,"b":1},"cap","h","frozen_checkpoint","p","realized",True); bh,fl=compute_baselines(c,start_prices={"a":100,"b":100},end_prices={"a":120,"b":80}); assert bh.common_horizon_id==fl.common_horizon_id=="h" and bh.cohort_id==fl.cohort_id=="c" and bh.continued_through_common_horizon and abs(bh.mean_arithmetic_return)<1e-12 and fl.mean_arithmetic_return==0
