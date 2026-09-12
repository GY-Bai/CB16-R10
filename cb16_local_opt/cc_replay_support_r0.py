from dataclasses import dataclass
from collections import Counter
from math import exp,isfinite
class ReplaySupportError(ValueError): pass
@dataclass(frozen=True)
class SupportSample: log_mu:float|None; log_pi:float; generation:str; source:str
@dataclass(frozen=True)
class ReplaySupportHealth:
    ratios:tuple[float,...]; finite:bool; nonfinite_count:int; clipping_fraction:float; low_support_frequency:float; effective_sample_size:float; effective_support_fraction:float; generation_contribution:dict[str,int]; source_contribution:dict[str,int]
def support_health(samples,*,clip_rho:float=1.0,low_support_ratio:float=.05)->ReplaySupportHealth:
    items=tuple(samples)
    if not items: raise ReplaySupportError("samples required")
    ratios=[]; bad=0
    for x in items:
        if x.log_mu is None: raise ReplaySupportError("missing log_mu cannot be fabricated")
        if not isfinite(float(x.log_mu)) or not isfinite(float(x.log_pi)): bad+=1; continue
        try:r=exp(float(x.log_pi)-float(x.log_mu))
        except OverflowError: bad+=1; continue
        if not isfinite(r): bad+=1; continue
        ratios.append(r)
    gc=dict(Counter(x.generation for x in items)); sc=dict(Counter(x.source for x in items))
    if not ratios:return ReplaySupportHealth((),False,bad,1,1,0,0,gc,sc)
    w=[min(r,clip_rho) for r in ratios]; den=sum(x*x for x in w); ess=(sum(w)**2/den) if den else 0
    return ReplaySupportHealth(tuple(ratios),bad==0,bad,sum(r>clip_rho for r in ratios)/len(ratios),sum(r<low_support_ratio for r in ratios)/len(ratios),ess,ess/len(ratios),gc,sc)
