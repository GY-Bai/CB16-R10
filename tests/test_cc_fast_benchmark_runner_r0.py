from cb16_local_opt.cc_fast_benchmark_runner_r0 import _batch_distribution,_policy_requests,_policy_fixture
from cb16_local_opt.cc_fast_policy_benchmark_r0 import benchmark_cpu
def test_batch_distribution_is_explicit():
    d=_batch_distribution((1,2,2,4)); assert d["count"]==4.0; assert d["min"]==1.0; assert d["max"]==4.0; assert d["mean"]==2.25
def test_policy_microbenchmark_checksum_is_deterministic():
    req=_policy_requests(8); a=benchmark_cpu(_policy_fixture(),req,repeat=1); b=benchmark_cpu(_policy_fixture(),list(reversed(req)),repeat=1); assert a.available and b.available; assert a.requests==b.requests==8; assert a.decisions_per_s>0 and b.decisions_per_s>0
