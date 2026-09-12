from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
@dataclass(frozen=True)
class CandidateResult:
    candidate_id:str; semantic_pass:bool; memory_safety_pass:bool; queue_safety_pass:bool; compliant_transitions_per_s:float; wall_clock_s:float; measured:bool; notes:tuple[str,...]=()
def select_fast_candidate(results:Sequence[CandidateResult])->CandidateResult:
    eligible=[r for r in results if r.semantic_pass and r.memory_safety_pass and r.queue_safety_pass and r.measured]
    if not eligible: raise RuntimeError("NO_MEASURED_SEMANTICALLY_VALID_CANDIDATE")
    return max(eligible,key=lambda r:(r.compliant_transitions_per_s,-r.wall_clock_s))
