from dataclasses import dataclass
import hashlib,random
from typing import Mapping,Sequence
class ReplaySelectionError(ValueError): pass
@dataclass(frozen=True)
class ReplayCandidate:
    sequence_id:str; source_class:str; generation_class:str; sampling_weight:float; support_health_summary:Mapping[str,float|int|bool]; outcome_conditioned_or_survivor_filtered:bool=False
@dataclass(frozen=True)
class ReplaySelectionMetadata:
    sequence_id:str; selection_rng_identity:str; sampling_probability:float; sampling_weight:float; source_class:str; generation_class:str; support_health_summary:Mapping[str,float|int|bool]; selection_policy_version:str; draw_index:int
def select_sequences(candidates:Sequence[ReplayCandidate],*,count:int,rng_identity:str,selection_policy_version:str):
    if count<0 or count>len(candidates): raise ReplaySelectionError("invalid count")
    if not rng_identity or not selection_policy_version: raise ReplaySelectionError("identities required")
    if any(c.outcome_conditioned_or_survivor_filtered for c in candidates): raise ReplaySelectionError("survivor/outcome filtering forbidden")
    if len({c.sequence_id for c in candidates})!=len(candidates) or any(c.sampling_weight<=0 for c in candidates): raise ReplaySelectionError("invalid candidates")
    r=random.Random(int.from_bytes(hashlib.sha256(rng_identity.encode()).digest()[:8],"big")); pool=list(candidates); out=[]
    for draw in range(count):
        total=sum(c.sampling_weight for c in pool); target=r.random()*total; acc=0; idx=len(pool)-1
        for i,c in enumerate(pool):
            acc+=c.sampling_weight
            if target<=acc:idx=i;break
        c=pool.pop(idx); out.append(ReplaySelectionMetadata(c.sequence_id,rng_identity,c.sampling_weight/total,c.sampling_weight,c.source_class,c.generation_class,dict(c.support_health_summary),selection_policy_version,draw))
    return tuple(out)
