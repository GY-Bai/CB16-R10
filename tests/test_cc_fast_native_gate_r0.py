from cb16_local_opt.cc_fast_native_gate_r0 import *
def test_profile_first_native_gate_and_amdahl():
    low=evaluate_native_gate(hotspot="json",wall_fraction=.05,estimated_speedup=10,boundary_overhead_fraction=.1,semantic_harness_available=True); assert not low.trigger_numba and low.reason=="HOTSPOT_NOT_MATERIAL"; n=evaluate_native_gate(hotspot="kernel",wall_fraction=.4,estimated_speedup=3,boundary_overhead_fraction=.05,semantic_harness_available=True); assert n.trigger_numba; assert n.estimated_amdahl_speedup>1
