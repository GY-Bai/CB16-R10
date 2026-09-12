from __future__ import annotations
from dataclasses import asdict,dataclass
import importlib,inspect,json
from typing import Mapping
from .cc_fast_semantic_harness_r0 import run_semantic_harness
from .cc_fast_router_r0 import CC_FAST_ROUTE,resolve_fast_route
REQUIRED_MODULES=("cc_fast_wire_r0","cc_fast_benchmark_contract_r0","cc_fast_benchmark_runner_r0","cc_fast_workload_r0","cc_fast_market_cache_r0","cc_fast_policy_broker_r0","cc_fast_policy_benchmark_r0","cc_fast_account_state_r0","cc_fast_account_kernel_r0","cc_fast_account_workers_r0","cc_fast_scheduler_r0","cc_fast_collector_r0","cc_fast_fact_queue_r0","cc_fast_writer_r0","cc_fast_storage_tier_r0","cc_fast_io_benchmark_r0","cc_fast_memory_budget_r0","cc_fast_metrics_r0","cc_fast_topology_search_r0","cc_fast_native_gate_r0","cc_fast_semantic_harness_r0","cc_fast_router_r0","cc_fast_selection_r0")
LEGACY_IMPORTS=("cb16_local_opt.gpu_inference_broker","cb16_local_opt.multiprocess_trajectory_farm","cb16_local_opt.vectorized_physics")
@dataclass(frozen=True)
class QualificationResult:
    schema_version:str; verdict:str; strongest_evidence:str; performance_measured:bool; semantic_checksum:str; checks:Mapping[str,bool]; unresolved:tuple[str,...]
    def to_json(self): return json.dumps(asdict(self),sort_keys=True,separators=(",",":"))
def compile_qualification(*,performance_measured=False):
    checks={}
    for name in REQUIRED_MODULES: checks[f"module:{name}"]=importlib.import_module(f"cb16_local_opt.{name}") is not None
    sem=run_semantic_harness(); checks["semantic_harness"]=sem.verdict=="PASS"; checks["single_route"]=resolve_fast_route(CC_FAST_ROUTE)==CC_FAST_ROUTE
    for name in REQUIRED_MODULES:
        source=inspect.getsource(importlib.import_module(f"cb16_local_opt.{name}")); checks[f"no_legacy_import:{name}"]=not any(x in source for x in LEGACY_IMPORTS)
    verdict="PASS" if all(checks.values()) else "FAIL"; unresolved=() if performance_measured else ("SHANXI_PERFORMANCE_MEASUREMENT_NOT_RUN",)
    return QualificationResult("CB16_R11_CC_THREAD_D_QUALIFICATION_V1",verdict,"COMPONENT",performance_measured,sem.checksum,checks,unresolved)
