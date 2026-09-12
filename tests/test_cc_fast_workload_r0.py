import numpy as np
from cb16_local_opt.cc_fast_workload_r0 import *
def test_workload_deterministic_and_offline():
    a=build_workload(WorkloadConfig(seed=1,market_steps=128,account_count=8)); b=build_workload(WorkloadConfig(seed=1,market_steps=128,account_count=8)); assert a.workload_id==b.workload_id; assert np.array_equal(a.market,b.market); assert a.accesses_final_or_fresh_data is False; assert not a.market.flags.writeable; assert len(a.ready_accounts(3))>0
