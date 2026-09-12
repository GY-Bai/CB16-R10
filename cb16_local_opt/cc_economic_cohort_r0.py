from dataclasses import dataclass
import hashlib,json
from typing import Mapping
class CohortIntegrityError(ValueError): pass
@dataclass(frozen=True)
class EconomicCohort:
    cohort_id:str; account_start_states:Mapping[str,str]; weights:Mapping[str,float]; capital_denominator_id:str; common_horizon_id:str; policy_object_type:str; policy_identity:str; realization_status:str; formal_horizon_owner_rule_resolved:bool
    def __post_init__(self):
        if not self.cohort_id or not self.account_start_states or not self.capital_denominator_id or not self.common_horizon_id or not self.policy_identity: raise CohortIntegrityError("identity required")
        if set(self.weights)!=set(self.account_start_states) or any(float(w)<=0 for w in self.weights.values()): raise CohortIntegrityError("weights must exactly cover cohort")
        if self.policy_object_type not in {"frozen_checkpoint","generation_chain","baseline"}: raise CohortIntegrityError("invalid policy type")
        if self.realization_status not in {"realized","truncated","estimated"}: raise CohortIntegrityError("invalid realization status")
    @property
    def account_ids(self):return frozenset(self.account_start_states)
    @property
    def frozen_hash(self):
        p={k:(dict(v) if k in {"account_start_states","weights"} else v) for k,v in self.__dict__.items()}; return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()
