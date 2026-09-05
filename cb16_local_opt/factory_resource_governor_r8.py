from __future__ import annotations

"""Resource admission for the continuous R8 factory.

The governor may delay work, reduce concurrency, or request maintenance. It must never drop
scientific objects or modify scientific thresholds.
"""

import dataclasses, json, shutil, subprocess, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


def _mem() -> tuple[float,int]:
    vals={};p=Path('/proc/meminfo')
    if p.exists():
        for line in p.read_text().splitlines():
            if ':' in line:
                k,v=line.split(':',1)
                try:vals[k]=int(v.strip().split()[0])*1024
                except Exception:pass
    total=vals.get('MemTotal',1);avail=vals.get('MemAvailable',total);swap=max(0,vals.get('SwapTotal',0)-vals.get('SwapFree',vals.get('SwapTotal',0)))
    return max(0.0,min(1.0,1-avail/total)),swap


def _gpu() -> tuple[float,float|None]:
    try:
        import torch
        if torch.cuda.is_available():
            total=torch.cuda.get_device_properties(0).total_memory
            # nvidia-smi is process-wide; torch allocated is local process only. Prefer smi.
        if shutil.which('nvidia-smi'):
            p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
            if p.returncode==0:
                x=[float(v.strip()) for v in p.stdout.strip().split(',')];return (0.0 if x[1]<=0 else x[0]/x[1]),x[2]
    except Exception:pass
    return 0.0,None


@dataclass(frozen=True)
class FactoryResourceLimitsR8:
    ram_high: float=0.85
    ram_hard: float=0.92
    vram_high: float=0.88
    vram_hard: float=0.96
    gpu_temp_high_c: float=82.0
    gpu_temp_hard_c: float=90.0
    min_ssd_free_gib: float=10.0
    min_hdd_free_gib: float=20.0
    max_running_gpu_jobs: int=1
    max_running_cpu_heavy_jobs: int=1
    max_running_cpu_io_jobs: int=2
    max_running_transfer_jobs: int=1
    max_running_maintenance_jobs: int=1


@dataclass(frozen=True)
class ResourceSnapshotR8:
    timestamp_unix: float
    ram_used_fraction: float
    swap_used_bytes: int
    vram_used_fraction: float
    gpu_temp_c: float|None
    ssd_free_gib: float
    hdd_free_gib: float
    running_by_resource: Mapping[str,int]


@dataclass(frozen=True)
class AdmissionDecisionR8:
    allowed: bool
    reasons: tuple[str,...]
    hard_stop: bool
    suggested_sleep_seconds: float
    scientific_semantics_changed: bool=False


class FactoryResourceGovernorR8:
    def __init__(self,*,ssd_root:str|Path,hdd_root:str|Path,limits:FactoryResourceLimitsR8|None=None):
        self.ssd=Path(ssd_root);self.hdd=Path(hdd_root);self.limits=limits or FactoryResourceLimitsR8()

    def snapshot(self,running_by_resource:Mapping[str,int]|None=None)->ResourceSnapshotR8:
        ram,swap=_mem();vram,temp=_gpu()
        def free(p):
            try:return shutil.disk_usage(p).free/1024**3
            except Exception:return 0.0
        return ResourceSnapshotR8(time.time(),ram,swap,vram,temp,free(self.ssd),free(self.hdd),dict(running_by_resource or {}))

    def decide(self,resource_class:str,s:ResourceSnapshotR8)->AdmissionDecisionR8:
        L=self.limits;hard=[];soft=[]
        if s.ram_used_fraction>=L.ram_hard:hard.append('RAM_HARD_LIMIT')
        elif s.ram_used_fraction>=L.ram_high:soft.append('RAM_BACKPRESSURE')
        if s.vram_used_fraction>=L.vram_hard:hard.append('VRAM_HARD_LIMIT')
        elif s.vram_used_fraction>=L.vram_high and resource_class=='GPU':soft.append('VRAM_BACKPRESSURE')
        if s.gpu_temp_c is not None and s.gpu_temp_c>=L.gpu_temp_hard_c:hard.append('GPU_TEMP_HARD_LIMIT')
        elif s.gpu_temp_c is not None and s.gpu_temp_c>=L.gpu_temp_high_c and resource_class=='GPU':soft.append('GPU_TEMP_BACKPRESSURE')
        if s.ssd_free_gib<L.min_ssd_free_gib:hard.append('SSD_FREE_SPACE_HARD_LIMIT')
        if s.hdd_free_gib<L.min_hdd_free_gib:soft.append('HDD_FREE_SPACE_LOW')
        limits={'GPU':L.max_running_gpu_jobs,'CPU_HEAVY':L.max_running_cpu_heavy_jobs,'CPU_IO':L.max_running_cpu_io_jobs,'TRANSFER':L.max_running_transfer_jobs,'MAINTENANCE':L.max_running_maintenance_jobs}
        if s.running_by_resource.get(resource_class,0)>=limits.get(resource_class,1):soft.append('RESOURCE_CLASS_CONCURRENCY_LIMIT')
        if hard:return AdmissionDecisionR8(False,tuple(hard+soft),True,30.0)
        if soft:return AdmissionDecisionR8(False,tuple(soft),False,5.0)
        return AdmissionDecisionR8(True,(),False,0.0)
