from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class NativeGateDecision:
    hotspot:str
    wall_fraction:float
    estimated_speedup:float
    boundary_overhead_fraction:float
    semantic_harness_available:bool
    trigger_numba:bool
    trigger_rust:bool
    trigger_go:bool
    estimated_amdahl_speedup:float
    reason:str

def evaluate_native_gate(*,hotspot:str,wall_fraction:float,estimated_speedup:float,boundary_overhead_fraction:float,semantic_harness_available:bool,hotspot_kind:str="numeric_cpu",material_fraction:float=0.15,min_amdahl_speedup:float=1.05):
    if not hotspot or not (0<=wall_fraction<=1) or estimated_speedup<1: raise ValueError("NATIVE_GATE_INPUT_INVALID")
    if not (0<=boundary_overhead_fraction<1): raise ValueError("BOUNDARY_OVERHEAD_INVALID")
    effective_speedup=max(1.0,1.0/(boundary_overhead_fraction+(1.0-boundary_overhead_fraction)/estimated_speedup))
    amdahl=1.0/((1.0-wall_fraction)+wall_fraction/effective_speedup)
    material=wall_fraction>=material_fraction and amdahl>=min_amdahl_speedup and semantic_harness_available
    trigger_numba=material and hotspot_kind=="numeric_cpu"
    trigger_rust=material and hotspot_kind=="numeric_cpu_remaining" and wall_fraction>=0.25
    trigger_go=material and hotspot_kind in {"scheduling_service","network_service"} and wall_fraction>=0.25
    if not semantic_harness_available: reason="NO_SEMANTIC_HARNESS"
    elif wall_fraction<material_fraction: reason="HOTSPOT_NOT_MATERIAL"
    elif amdahl<min_amdahl_speedup: reason="INSUFFICIENT_END_TO_END_AMDAHL_IMPACT"
    elif trigger_numba: reason="NUMBA_CANDIDATE_TRIGGERED"
    elif trigger_rust: reason="RUST_COARSE_GRAINED_CANDIDATE_TRIGGERED"
    elif trigger_go: reason="GO_SERVICE_CANDIDATE_TRIGGERED"
    else: reason="NO_LANGUAGE_MIGRATION_TRIGGER"
    return NativeGateDecision(hotspot,wall_fraction,estimated_speedup,boundary_overhead_fraction,semantic_harness_available,trigger_numba,trigger_rust,trigger_go,amdahl,reason)
