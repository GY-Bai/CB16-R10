from dataclasses import dataclass
from typing import Mapping,Iterable
class CapitalLedgerError(ValueError): pass
@dataclass(frozen=True)
class CapitalEvent: event_id:str; account_lineage_id:str; event_type:str; amount:float
@dataclass(frozen=True)
class CapitalSummary: initial_allocated_capital:float; external_deposits:float; external_withdrawals:float; new_account_or_restart_capital:float; gross_contributed_denominator:float
def summarize_capital(initial_allocations:Mapping[str,float],events:Iterable[CapitalEvent]):
    if not initial_allocations or any(float(v)<0 for v in initial_allocations.values()): raise CapitalLedgerError("invalid initial capital")
    initial=sum(map(float,initial_allocations.values())); dep=wd=restart=0.; seen=set()
    for e in events:
        if not e.event_id or e.event_id in seen or e.amount<0: raise CapitalLedgerError("invalid capital event")
        seen.add(e.event_id)
        if e.event_type=="deposit": dep+=e.amount
        elif e.event_type=="withdrawal": wd+=e.amount
        elif e.event_type in {"new_account_allocation","lineage_restart"}: restart+=e.amount
        else: raise CapitalLedgerError("unknown event")
    return CapitalSummary(initial,dep,wd,restart,initial+dep+restart)
