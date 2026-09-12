import pytest
from cb16_local_opt.cc_fast_memory_budget_r0 import *
def budget(): return MemoryBudgets(host_limit_bytes=16<<30,workers_bytes=4<<30,shared_market_bytes=2<<30,queues_bytes=1<<30,policy_broker_bytes=1<<30,replay_io_cache_bytes=1<<30,learner_overlap_bytes=1<<30,os_reserve_bytes=4<<30)
def test_budget_and_pressure_scale_down():
    b=budget().validate(); p=linux_memory_pressure(budget=b,rss_bytes=b.active_budget_bytes+1); assert p.under_pressure; assert scale_down_worker_candidate(6,p)==4
def test_overcommit_rejected():
    with pytest.raises(ValueError): MemoryBudgets(10,5,5,5,5,5,5,1).validate()
