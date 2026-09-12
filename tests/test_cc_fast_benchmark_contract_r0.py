import pytest
from cb16_local_opt.cc_fast_benchmark_contract_r0 import *
def make(verdict="PASS"):
    return BenchmarkReport(schema_version=BENCHMARK_SCHEMA,workload_identity="w",code_identity="c",science_identity="s",topology={"workers":2},account_count=16,decision_rate=10,batch_size_distribution={"p50":8},compliant_transitions_per_s=100,policy_decisions_per_s=100,wall_clock_s=1.0,cpu_per_core_utilization=(10.0,20.0),pss_bytes=1,cgroup_memory_bytes=None,page_faults=0,swap_in_bytes=0,swap_out_bytes=0,disk_read_bytes_per_s=0,disk_write_bytes_per_s=0,io_wait_fraction=0,gpu_vram_bytes=None,gpu_kernel_ms=None,gpu_transfer_ms=None,queue_bytes=0,queue_depth=0,queue_oldest_age_s=0,correctness_checksum="x"*64,semantic_verdict=verdict)
def test_benchmark_requires_semantic_pass():
    assert make().validate().semantic_verdict=="PASS"
    with pytest.raises(ValueError): make("FAIL").validate()
