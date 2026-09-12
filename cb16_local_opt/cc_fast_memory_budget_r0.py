from __future__ import annotations
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class MemoryBudgets:
    host_limit_bytes:int
    workers_bytes:int
    shared_market_bytes:int
    queues_bytes:int
    policy_broker_bytes:int
    replay_io_cache_bytes:int
    learner_overlap_bytes:int
    os_reserve_bytes:int
    @property
    def active_budget_bytes(self): return self.workers_bytes+self.shared_market_bytes+self.queues_bytes+self.policy_broker_bytes+self.replay_io_cache_bytes+self.learner_overlap_bytes
    def validate(self):
        if any((not isinstance(v,int) or v<0) for v in self.__dict__.values()): raise ValueError("MEMORY_BUDGET_INVALID")
        if self.host_limit_bytes<=0: raise ValueError("HOST_LIMIT_INVALID")
        if self.active_budget_bytes+self.os_reserve_bytes>self.host_limit_bytes: raise ValueError("MEMORY_BUDGET_OVERCOMMITTED")
        return self

@dataclass(frozen=True)
class MemoryPressure:
    rss_bytes:int
    swap_in_pages:int
    swap_out_pages:int
    under_pressure:bool
    reason:str

def linux_memory_pressure(*,budget:MemoryBudgets,rss_bytes:int|None=None):
    budget.validate()
    if rss_bytes is None:
        try:
            pages=int(open("/proc/self/statm",encoding="utf-8").read().split()[1]); rss_bytes=pages*os.sysconf("SC_PAGE_SIZE")
        except Exception: rss_bytes=0
    sin=sout=0
    try:
        for line in open("/proc/vmstat",encoding="utf-8"):
            k,v=line.split()
            if k=="pswpin": sin=int(v)
            elif k=="pswpout": sout=int(v)
    except Exception: pass
    pressure=int(rss_bytes)>budget.active_budget_bytes
    return MemoryPressure(int(rss_bytes),sin,sout,pressure,"RSS_OVER_ACTIVE_BUDGET" if pressure else "OK")

def scale_down_worker_candidate(worker_count:int,pressure:MemoryPressure):
    if worker_count not in {2,4,6}: raise ValueError("WORKER_COUNT_INVALID")
    if not pressure.under_pressure: return worker_count
    return {6:4,4:2,2:2}[worker_count]
