import hashlib
from cb16_local_opt.cc_fast_policy_broker_r0 import *
from cb16_local_opt.cc_fast_policy_benchmark_r0 import *
def test_cpu_gpu_candidate_contract_and_selection():
    p=PolicySpec(1,"p","1"*64,(0,0,0)); reqs=[PolicyRequest(f"a{i}",0,0,1,"1"*64,(1.0,2.0),hashlib.sha256(str(i).encode()).hexdigest(),"n",f"s{i}",0) for i in range(8)]; cpu=benchmark_cpu(p,reqs,repeat=1); gpu=benchmark_cuda_if_available(p,reqs,repeat=1); assert cpu.available and cpu.transfer_queue_included; assert gpu.mode=="GPU_BROKER"; assert select_policy_mode(cpu,gpu) in {"CPU","GPU_BROKER"}
