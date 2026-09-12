from __future__ import annotations
from dataclasses import dataclass, asdict
from .cc_economic_toys_r0 import run_known_answer_toys

@dataclass(frozen=True)
class ThreadCQualification:
    immutable_facts_sequences: bool
    failure_retention: bool
    four_view_separation: bool
    historical_source_classification: bool
    replay_compatibility_support: bool
    no_age_expiration: bool
    generation_attribution: bool
    idempotent_persistence: bool
    economic_known_answers: bool
    synthetic_local_only: bool
    def passed(self)->bool: return all(asdict(self).values())

def compile_qualification(**evidence: bool) -> ThreadCQualification:
    required=[f.name for f in ThreadCQualification.__dataclass_fields__.values()]
    missing=[k for k in required if k not in evidence]
    if missing: raise ValueError(f"missing qualification evidence: {missing}")
    q=ThreadCQualification(**{k:bool(evidence[k]) for k in required})
    if not all(run_known_answer_toys().values()):
        raise ValueError("economic known-answer toys failed")
    return q
