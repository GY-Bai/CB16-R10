from pathlib import Path
from cb16_local_opt.cc_fast_collector_r0 import *
from cb16_local_opt.cc_fast_policy_broker_r0 import PolicySpec
from cb16_local_opt.cc_fast_workload_r0 import *
def test_end_to_end_cc_fast_collector_uses_persistent_workers_and_durable_chunks(tmp_path):
    w=build_workload(WorkloadConfig(seed=4,market_steps=72,account_count=8)); p=PolicySpec(1,"p","1"*64,(-.1,-.5,.3)); r=run_collector(w,p,output_root=str(tmp_path),config=CollectorConfig(worker_count=2,queue_max_bytes=1<<20,writer_chunk_facts=16,max_steps=48)); assert r.transitions>0; assert r.policy_decisions==r.transitions; assert len(r.worker_pids)==2; assert all(x is False for x in r.worker_cuda_initialized); assert all(x.durable for x in r.chunk_receipts); assert all(Path(x.path).exists() for x in r.chunk_receipts); assert len(r.semantic_checksum)==64
