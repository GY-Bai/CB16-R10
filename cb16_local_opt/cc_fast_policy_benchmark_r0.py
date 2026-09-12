from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,time
from typing import Sequence
from .cc_fast_policy_broker_r0 import BatchedPolicyBroker,PolicyRequest,PolicySpec

@dataclass(frozen=True)
class PolicyModeBenchmark:
    mode:str; available:bool; requests:int; batch_size:int; elapsed_s:float; decisions_per_s:float; transfer_queue_included:bool; semantic_checksum:str; note:str

def _checksum(responses):
    payload=[(x.account_lineage_id,x.decision_index,x.policy_generation,x.policy_sha256,x.nominal_direction,round(x.nominal_target_risk,15),round(x.log_mu,15),x.rng_stream_id,x.rng_counter) for x in responses]
    return hashlib.sha256(json.dumps(payload,separators=(",",":"),sort_keys=False).encode()).hexdigest()

def benchmark_cpu(policy:PolicySpec,requests:Sequence[PolicyRequest],*,repeat:int=5):
    broker=BatchedPolicyBroker([policy],execution_mode="CPU"); t0=time.perf_counter(); response=None
    for _ in range(repeat): response=broker.infer(requests)
    elapsed=time.perf_counter()-t0; n=len(requests)*repeat
    return PolicyModeBenchmark("CPU",True,n,len(requests),elapsed,n/elapsed if elapsed else float("inf"),True,_checksum(response or []),"CPU broker includes batching and request/response object overhead.")

def benchmark_cuda_if_available(policy:PolicySpec,requests:Sequence[PolicyRequest],*,repeat:int=5):
    try:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("CUDA_UNAVAILABLE")
        device=torch.device("cuda:0"); x_cpu=torch.tensor([r.observation for r in requests],dtype=torch.float32)
        x=x_cpu.to(device,non_blocking=False); y=torch.tanh(x); _=y.cpu(); torch.cuda.synchronize(device)
        t0=time.perf_counter()
        for _ in range(repeat):
            x=x_cpu.to(device,non_blocking=False); y=torch.tanh(x); _=y.cpu(); torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-t0; responses=BatchedPolicyBroker([policy],execution_mode="CUDA_SINGLE_OWNER").infer(requests); n=len(requests)*repeat
        return PolicyModeBenchmark("GPU_BROKER",True,n,len(requests),elapsed,n/elapsed if elapsed else float("inf"),True,_checksum(responses),"Representative CUDA microbatch timing includes host/device transfer and synchronize.")
    except Exception as exc:
        return PolicyModeBenchmark("GPU_BROKER",False,0,len(requests),0.0,0.0,True,"",f"EXECUTION_BLOCKED/HARDWARE_LIMIT:{type(exc).__name__}")

def select_policy_mode(cpu:PolicyModeBenchmark,gpu:PolicyModeBenchmark):
    if not cpu.available: raise RuntimeError("CPU_POLICY_CANDIDATE_REQUIRED")
    if gpu.available and gpu.semantic_checksum==cpu.semantic_checksum and gpu.decisions_per_s>cpu.decisions_per_s: return "GPU_BROKER"
    return "CPU"
