from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from .cc_experience_transition_r0 import CCExperienceTransitionR0
class SequenceValidationError(ValueError): pass
@dataclass(frozen=True)
class GenerationSwitch:
    decision_index:int; from_generation:str; to_generation:str; from_policy:str; to_policy:str
@dataclass(frozen=True)
class CCExperienceSequenceR0:
    sequence_id:str; transitions:tuple[CCExperienceTransitionR0,...]; terminal:bool; chunk_boundary:bool; lineage_events:tuple[tuple[str,str],...]=()
    def __post_init__(self):
        if not self.sequence_id or not self.transitions: raise SequenceValidationError("sequence required")
        ids=[t.transition_id for t in self.transitions]
        if len(ids)!=len(set(ids)): raise SequenceValidationError("duplicate transition")
        ix=[int(t.w02["decision_index"]) for t in self.transitions]
        if any(b!=a+1 for a,b in zip(ix,ix[1:])): raise SequenceValidationError("reorder/drop detected: decision indices must be contiguous")
        allowed=set(self.lineage_events)
        for a,b in zip(self.transitions,self.transitions[1:]):
            if a.account_lineage_id!=b.account_lineage_id and (a.account_lineage_id,b.account_lineage_id) not in allowed: raise SequenceValidationError("cross-account splice")
            if a.post_account_truth_hash!=b.pre_account_truth_hash: raise SequenceValidationError("account hash discontinuity")
        if self.terminal and self.chunk_boundary: raise SequenceValidationError("chunk boundary distinct from terminal")
    @property
    def generation_ownership(self): return tuple((int(t.w02["decision_index"]),t.generation_id,t.policy_id) for t in self.transitions)
    @property
    def generation_switches(self):
        out=[]
        for a,b in zip(self.transitions,self.transitions[1:]):
            if (a.generation_id,a.policy_id)!=(b.generation_id,b.policy_id): out.append(GenerationSwitch(int(b.w02["decision_index"]),a.generation_id,b.generation_id,a.policy_id,b.policy_id))
        return tuple(out)
    @property
    def content_hash(self):
        p={"sequence_id":self.sequence_id,"transition_hashes":[t.content_hash for t in self.transitions],"terminal":self.terminal,"chunk_boundary":self.chunk_boundary,"lineage_events":self.lineage_events}
        return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()
