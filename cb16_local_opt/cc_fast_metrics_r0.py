from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import resource
import subprocess
import time
from typing import Dict, Iterator

try:
    import psutil
except Exception:
    psutil=None

PHASES=("market_cache","policy_queue_wait","policy_inference","account_kernel","serialization_write","io_read","idle_backpressure")

def _cgroup_memory_current():
    for path in ("/sys/fs/cgroup/memory.current","/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return int(open(path,"r",encoding="utf-8").read().strip())
        except Exception: pass
    return None

@dataclass
class FastMetrics:
    started_ns:int=field(default_factory=time.monotonic_ns)
    phase_ns:Dict[str,int]=field(default_factory=lambda:{x:0 for x in PHASES})
    phase_count:Dict[str,int]=field(default_factory=lambda:{x:0 for x in PHASES})
    transitions:int=0
    policy_decisions:int=0
    _disk_read_start:int=field(init=False,default=0,repr=False)
    _disk_write_start:int=field(init=False,default=0,repr=False)
    _swap_in_start:int=field(init=False,default=0,repr=False)
    _swap_out_start:int=field(init=False,default=0,repr=False)
    def __post_init__(self):
        if psutil is not None:
            try:
                d=psutil.disk_io_counters(); self._disk_read_start=int(d.read_bytes); self._disk_write_start=int(d.write_bytes)
            except Exception: pass
            try:
                s=psutil.swap_memory(); self._swap_in_start=int(getattr(s,"sin",0)); self._swap_out_start=int(getattr(s,"sout",0))
            except Exception: pass
            try: psutil.cpu_percent(interval=None,percpu=True)
            except Exception: pass
    @contextmanager
    def phase(self,name):
        if name not in self.phase_ns: raise ValueError("METRIC_PHASE_INVALID")
        t0=time.monotonic_ns()
        try: yield
        finally:
            self.phase_ns[name]+=time.monotonic_ns()-t0; self.phase_count[name]+=1
    @property
    def elapsed_s(self): return (time.monotonic_ns()-self.started_ns)/1e9
    def snapshot_system(self):
        elapsed=max(self.elapsed_s,1e-12); usage=resource.getrusage(resource.RUSAGE_SELF); rss=int(usage.ru_maxrss)
        if hasattr(os,"uname") and os.uname().sysname=="Linux": rss*=1024
        cpu_per_core=(); pss=rss; disk_read_rate=disk_write_rate=0.0; swap_in=swap_out=0; io_wait_fraction=0.0
        if psutil is not None:
            try: cpu_per_core=tuple(float(x) for x in psutil.cpu_percent(interval=None,percpu=True))
            except Exception: pass
            try: pss=int(getattr(psutil.Process().memory_full_info(),"pss",rss))
            except Exception: pass
            try:
                d=psutil.disk_io_counters(); disk_read_rate=max(0,int(d.read_bytes)-self._disk_read_start)/elapsed; disk_write_rate=max(0,int(d.write_bytes)-self._disk_write_start)/elapsed
            except Exception: pass
            try:
                s=psutil.swap_memory(); swap_in=max(0,int(getattr(s,"sin",0))-self._swap_in_start); swap_out=max(0,int(getattr(s,"sout",0))-self._swap_out_start)
            except Exception: pass
            try: io_wait_fraction=max(0.0,float(getattr(psutil.cpu_times_percent(interval=None,percpu=False),"iowait",0.0))/100.0)
            except Exception: pass
        return {"monotonic_ns":time.monotonic_ns(),"elapsed_s":elapsed,"cpu_per_core_utilization":cpu_per_core,"rss_or_peak_bytes":rss,"pss_bytes":pss,"cgroup_memory_bytes":_cgroup_memory_current(),"minor_faults":int(usage.ru_minflt),"major_faults":int(usage.ru_majflt),"swap_in_bytes":swap_in,"swap_out_bytes":swap_out,"disk_read_bytes_per_s":disk_read_rate,"disk_write_bytes_per_s":disk_write_rate,"io_wait_fraction":io_wait_fraction,"phase_ns":dict(self.phase_ns),"phase_count":dict(self.phase_count),"transitions":self.transitions,"policy_decisions":self.policy_decisions,"gpu":query_nvidia_smi()}

def query_nvidia_smi():
    try:
        proc=subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True,timeout=2)
        line=proc.stdout.strip().splitlines()[0]; mem_mib,util=[x.strip() for x in line.split(",")[:2]]
        return {"vram_used_bytes":int(float(mem_mib)*1024*1024),"utilization_percent":float(util)}
    except Exception: return None
