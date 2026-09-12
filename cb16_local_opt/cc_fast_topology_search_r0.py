from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Sequence
@dataclass(frozen=True)
class TopologyCandidate:
    worker_count:int; active_accounts:int; policy_mode:str; writer_chunk_facts:int; prefetch_depth:int; market_reuse:bool=True
    def validate(self):
        if self.worker_count not in {2,4,6}: raise ValueError("WORKER_COUNT_INVALID")
        if self.active_accounts not in {16,32,64}: raise ValueError("ACTIVE_ACCOUNTS_INVALID")
        if self.policy_mode not in {"CPU","GPU_BROKER"}: raise ValueError("POLICY_MODE_INVALID")
        if self.writer_chunk_facts<=0 or self.prefetch_depth<0: raise ValueError("TOPOLOGY_IO_INVALID")
        if not self.market_reuse: raise ValueError("CC_FAST_REQUIRES_MARKET_REUSE")
        return self
def candidate_matrix(*,worker_counts:Sequence[int]=(2,4,6),active_accounts:Sequence[int]=(16,32,64),policy_modes:Sequence[str]=("CPU","GPU_BROKER"),writer_chunk_facts:Sequence[int]=(64,256),prefetch_depths:Sequence[int]=(0,2)):
    return tuple(TopologyCandidate(*values).validate() for values in product(worker_counts,active_accounts,policy_modes,writer_chunk_facts,prefetch_depths))
