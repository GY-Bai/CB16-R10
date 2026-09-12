from cb16_local_opt.cc_fast_native_gate_r0 import evaluate_native_gate
def test_go_only_for_measured_service_scheduling_hotspot():
    numeric=evaluate_native_gate(hotspot="actor",wall_fraction=.5,estimated_speedup=2,boundary_overhead_fraction=.05,semantic_harness_available=True,hotspot_kind="numeric_cpu"); assert numeric.trigger_go is False; svc=evaluate_native_gate(hotspot="scheduler",wall_fraction=.4,estimated_speedup=2,boundary_overhead_fraction=.05,semantic_harness_available=True,hotspot_kind="scheduling_service"); assert svc.trigger_go is True
