from dataclasses import dataclass
from typing import Mapping
class ViewIntegrityError(ValueError): pass
@dataclass(frozen=True)
class ExperienceViews:
    raw_fact_ids:frozenset[str]; replay_admissible_ids:frozenset[str]; demonstration_ids:frozenset[str]; evaluation_cohort_ids:frozenset[str]; preregistered_evaluation_ids:frozenset[str]; replay_weights:Mapping[str,float]; evaluation_weights:Mapping[str,float]
    def __post_init__(self):
        for n,v in (("replay",self.replay_admissible_ids),("demonstration",self.demonstration_ids),("evaluation",self.evaluation_cohort_ids)):
            if not v.issubset(self.raw_fact_ids): raise ViewIntegrityError(f"{n} contains non-raw fact")
        if self.evaluation_cohort_ids!=self.preregistered_evaluation_ids: raise ViewIntegrityError("survivor-filtered evaluation cohort")
        if set(self.evaluation_weights)!=set(self.evaluation_cohort_ids): raise ViewIntegrityError("evaluation weight mismatch")
        if not set(self.replay_weights).issubset(self.replay_admissible_ids): raise ViewIntegrityError("replay weight mismatch")
        if any(float(x)<0 for x in (*self.replay_weights.values(),*self.evaluation_weights.values())): raise ViewIntegrityError("negative weight")
