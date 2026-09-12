from dataclasses import dataclass
import hashlib,json
from typing import Mapping
class EvaluationRescueAttempt(RuntimeError): pass
@dataclass(frozen=True)
class EvaluationFreeze:
    version_id:str; cohort_hash:str; horizon_id:str; weights:Mapping[str,float]; capital_denominator_id:str; baseline_definition_ids:tuple[str,...]; policy_identity:str; data_lineage:str
    @property
    def fingerprint(self):
        p={**self.__dict__,"weights":dict(self.weights)}; return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    def assert_same_scoring_contract(self,other):
        a={k:v for k,v in self.__dict__.items() if k!="version_id"}; b={k:v for k,v in other.__dict__.items() if k!="version_id"}
        if a!=b and self.version_id==other.version_id: raise EvaluationRescueAttempt("in-place rescue forbidden; create new evaluation version")
