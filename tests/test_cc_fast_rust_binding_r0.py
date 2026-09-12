from cb16_local_opt.cc_fast_native_gate_r0 import evaluate_native_gate
def test_rust_not_triggered_without_measured_remaining_hotspot():
    d=evaluate_native_gate(hotspot="account",wall_fraction=.2,estimated_speedup=2,boundary_overhead_fraction=.02,semantic_harness_available=True,hotspot_kind="numeric_cpu"); assert d.trigger_rust is False
