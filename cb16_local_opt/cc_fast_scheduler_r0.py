from __future__ import annotations
from dataclasses import dataclass
from typing import Dict,Iterable
@dataclass(frozen=True)
class ScheduledAccountTask:
    account_lineage_id:str; decision_index:int; policy_generation:int; enqueue_tick:int
class AccountSerialScheduler:
    def __init__(self,account_ids:Iterable[str],*,max_starvation_ticks:int=64):
        ids=tuple(account_ids)
        if not ids or len(set(ids))!=len(ids): raise ValueError("ACCOUNT_IDS_INVALID")
        if max_starvation_ticks<=0: raise ValueError("STARVATION_BOUND_INVALID")
        self.expected={x:0 for x in ids}; self.inflight={}; self.last_served_tick={x:0 for x in ids}; self.max_starvation_ticks=max_starvation_ticks
    def submit(self,task:ScheduledAccountTask,*,now_tick:int):
        if task.account_lineage_id not in self.expected: raise KeyError("UNKNOWN_ACCOUNT")
        if task.account_lineage_id in self.inflight: raise RuntimeError("DUPLICATE_ACCOUNT_INFLIGHT")
        if task.decision_index!=self.expected[task.account_lineage_id]: raise RuntimeError("ACCOUNT_DECISION_ORDER_VIOLATION")
        if task.enqueue_tick>now_tick: raise ValueError("ENQUEUE_TICK_IN_FUTURE")
        self.inflight[task.account_lineage_id]=task
    def commit(self,account_lineage_id:str,decision_index:int,*,now_tick:int):
        task=self.inflight.get(account_lineage_id)
        if task is None: raise RuntimeError("ACCOUNT_COMMIT_WITHOUT_INFLIGHT")
        if task.decision_index!=decision_index: raise RuntimeError("ACCOUNT_COMMIT_INDEX_MISMATCH")
        del self.inflight[account_lineage_id]; self.expected[account_lineage_id]+=1; self.last_served_tick[account_lineage_id]=now_tick
    def cancel_fail_closed(self,account_lineage_id:str):
        try:return self.inflight.pop(account_lineage_id)
        except KeyError as exc: raise RuntimeError("ACCOUNT_CANCEL_WITHOUT_INFLIGHT") from exc
    def assert_fairness(self,*,now_tick:int,active_accounts:Iterable[str]):
        for account_id in active_accounts:
            if account_id not in self.expected: raise KeyError("UNKNOWN_ACCOUNT")
            if now_tick-self.last_served_tick[account_id]>self.max_starvation_ticks: raise RuntimeError("ACCOUNT_STARVATION_BOUND_EXCEEDED")
