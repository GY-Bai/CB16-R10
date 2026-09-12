from __future__ import annotations
class CCReferenceSchedulerR0:
    def __init__(self,runtimes:dict[str,object]): self.runtimes=dict(runtimes)
    def run(self,work:dict[str,list],callbacks:dict[str,object],order:list[str]):
        out={k:[] for k in work}; cursors={k:0 for k in work}; remaining=sum(len(v) for v in work.values())
        while remaining:
            progressed=False
            for k in order:
                if k not in work or cursors[k]>=len(work[k]): continue
                rt=self.runtimes[k]; out[k].append(rt.step(work[k][cursors[k]],callbacks.get(k),expected_predecessor_token=rt.predecessor_token)); cursors[k]+=1; remaining-=1; progressed=True
            if not progressed: raise RuntimeError("CCREFSCHED_ORDER_CANNOT_PROGRESS")
        return out
