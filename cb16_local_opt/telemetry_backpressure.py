from __future__ import annotations

"""
Runtime telemetry and bounded performance backpressure.

Backpressure may tune ONLY performance knobs such as producer sleep, queue admission,
micro-batch target and max in-flight work. It must not mutate scientific semantics:
dataset split, Teacher law, Physics, Supervisor, promotion rule, loss definition, etc.
"""

import collections
import dataclasses
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Mapping

try:
    import torch
except Exception:
    torch=None


@dataclass(frozen=True)
class RuntimeSample:
    timestamp_unix: float
    queue_fill_fraction: float
    ram_used_fraction: float
    gpu_vram_used_fraction: float
    gpu_utilization_fraction: float | None
    cpu_load_fraction: float | None
    rollout_states_per_s: float | None
    training_samples_per_s: float | None
    teacher_jobs_per_s: float | None
    disk_write_mb_s: float | None


@dataclass(frozen=True)
class PerformanceKnobs:
    producer_sleep_ms: float
    max_in_flight: int
    target_inference_batch: int
    loader_prefetch: int

    def validate(self):
        if self.producer_sleep_ms < 0:
            raise ValueError("negative sleep")
        if self.max_in_flight <= 0 or self.target_inference_batch <= 0 or self.loader_prefetch <= 0:
            raise ValueError("invalid performance knob")


@dataclass(frozen=True)
class BackpressureLimits:
    min_in_flight: int=2
    max_in_flight: int=16
    min_inference_batch: int=1024
    max_inference_batch: int=32768
    min_prefetch: int=1
    max_prefetch: int=4
    max_producer_sleep_ms: float=50.0
    high_queue: float=0.85
    low_queue: float=0.25
    high_ram: float=0.85
    high_vram: float=0.88


@dataclass(frozen=True)
class BackpressureDecision:
    previous: PerformanceKnobs
    updated: PerformanceKnobs
    reasons: tuple[str,...]
    scientific_semantics_changed: bool=False


class TelemetryWindow:
    def __init__(self,maxlen:int=120):
        self.samples:Deque[RuntimeSample]=collections.deque(maxlen=maxlen)

    def append(self,s:RuntimeSample)->None:
        self.samples.append(s)

    def mean(self,field:str)->float|None:
        vals=[getattr(s,field) for s in self.samples if getattr(s,field) is not None]
        return None if not vals else sum(vals)/len(vals)

    def latest(self)->RuntimeSample|None:
        return self.samples[-1] if self.samples else None


class BoundedBackpressureController:
    def __init__(self,limits:BackpressureLimits|None=None):
        self.limits=limits or BackpressureLimits()

    def decide(self,knobs:PerformanceKnobs,sample:RuntimeSample)->BackpressureDecision:
        knobs.validate()
        L=self.limits
        sleep=knobs.producer_sleep_ms
        inflight=knobs.max_in_flight
        batch=knobs.target_inference_batch
        prefetch=knobs.loader_prefetch
        reasons=[]

        # Memory pressure has priority over throughput chasing.
        if sample.ram_used_fraction >= L.high_ram:
            inflight=max(L.min_in_flight,inflight-1)
            prefetch=max(L.min_prefetch,prefetch-1)
            sleep=min(L.max_producer_sleep_ms,max(1.0,sleep*1.5 if sleep else 2.0))
            reasons.append("HIGH_RAM_PRESSURE")
        if sample.gpu_vram_used_fraction >= L.high_vram:
            batch=max(L.min_inference_batch,batch//2)
            reasons.append("HIGH_VRAM_PRESSURE")

        if sample.queue_fill_fraction >= L.high_queue:
            # Consumers are behind. Slow producers rather than dropping scientific objects.
            sleep=min(L.max_producer_sleep_ms,max(1.0,sleep+2.0))
            inflight=max(L.min_in_flight,inflight-1)
            reasons.append("QUEUE_BACKPRESSURE")
        elif (
            sample.queue_fill_fraction <= L.low_queue
            and sample.ram_used_fraction < L.high_ram*0.9
            and sample.gpu_vram_used_fraction < L.high_vram*0.9
        ):
            # Pipeline is starved; cautiously feed more work.
            sleep=max(0.0,sleep-1.0)
            inflight=min(L.max_in_flight,inflight+1)
            batch=min(L.max_inference_batch,batch*2)
            reasons.append("PIPELINE_STARVATION")

        updated=PerformanceKnobs(
            producer_sleep_ms=sleep,
            max_in_flight=inflight,
            target_inference_batch=batch,
            loader_prefetch=prefetch,
        )
        return BackpressureDecision(knobs,updated,tuple(reasons),False)


def read_ram_fraction()->float:
    p=Path("/proc/meminfo")
    if not p.exists():
        return 0.0
    vals={}
    for line in p.read_text().splitlines():
        if ":" in line:
            k,v=line.split(":",1)
            try: vals[k]=int(v.strip().split()[0])
            except Exception: pass
    total=vals.get("MemTotal",0)
    avail=vals.get("MemAvailable",total)
    return 0.0 if total<=0 else max(0.0,min(1.0,1-avail/total))


def read_gpu_vram_fraction()->float:
    if torch is None or not torch.cuda.is_available():
        return 0.0
    total=torch.cuda.get_device_properties(0).total_memory
    used=torch.cuda.memory_allocated(0)
    return 0.0 if total<=0 else float(used/total)


class JsonlTelemetryWriter:
    def __init__(self,path:str|Path,fsync_every:int=50):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.fsync_every=max(1,int(fsync_every)); self._n=0

    def write(self,obj:RuntimeSample|BackpressureDecision|Mapping[str,Any]):
        if dataclasses.is_dataclass(obj): payload=asdict(obj)
        else: payload=dict(obj)
        with self.path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(payload,sort_keys=True,separators=(",",":"))+"\n")
            self._n+=1
            if self._n%self.fsync_every==0:
                f.flush(); os.fsync(f.fileno())
