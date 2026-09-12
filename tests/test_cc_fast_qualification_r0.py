from cb16_local_opt.cc_fast_qualification_r0 import compile_qualification
def test_thread_d_component_qualification_passes_without_claiming_measurement():
    q=compile_qualification(performance_measured=False); assert q.verdict=="PASS",[k for k,v in q.checks.items() if not v]; assert q.strongest_evidence=="COMPONENT"; assert q.performance_measured is False; assert "SHANXI_PERFORMANCE_MEASUREMENT_NOT_RUN" in q.unresolved
