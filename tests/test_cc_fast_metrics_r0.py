import time
from cb16_local_opt.cc_fast_metrics_r0 import *
def test_monotonic_phase_breakdown():
    m=FastMetrics()
    with m.phase("account_kernel"): time.sleep(.001)
    snap=m.snapshot_system(); assert snap["phase_ns"]["account_kernel"]>0; assert snap["phase_count"]["account_kernel"]==1; assert snap["monotonic_ns"]>=m.started_ns
