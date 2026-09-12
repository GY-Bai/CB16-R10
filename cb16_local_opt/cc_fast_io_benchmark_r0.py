from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Sequence

@dataclass(frozen=True)
class SampleLocation:
    sample_id:str
    path:str
    offset:int
    length:int

def read_preselected(selections:Sequence[SampleLocation],*,physical_strategy:str)->list[bytes]:
    if physical_strategy not in {"naive","grouped_sequential","ssd_active","bounded_prefetch"}: raise ValueError("IO_STRATEGY_INVALID")
    if len({x.sample_id for x in selections})!=len(selections): raise ValueError("DUPLICATE_SAMPLE_ID")
    indexed=list(enumerate(selections)); physical=indexed if physical_strategy=="naive" else sorted(indexed,key=lambda z:(z[1].path,z[1].offset))
    out=[None]*len(selections)
    for logical_i,loc in physical:
        with open(loc.path,"rb") as fh:
            fh.seek(loc.offset); data=fh.read(loc.length)
        if len(data)!=loc.length: raise RuntimeError("SHORT_SAMPLE_READ")
        out[logical_i]=data
    return [x for x in out if x is not None]

def benchmark_strategies(selections:Sequence[SampleLocation])->dict[str,float]:
    result={}; reference=None
    for strategy in ("naive","grouped_sequential","ssd_active","bounded_prefetch"):
        t0=time.perf_counter(); data=read_preselected(selections,physical_strategy=strategy); elapsed=time.perf_counter()-t0
        if reference is None: reference=data
        elif data!=reference: raise RuntimeError("IO_STRATEGY_CHANGED_LOGICAL_SAMPLES")
        result[strategy]=elapsed
    return result
