from dataclasses import dataclass
from typing import Mapping
from .cc_economic_toys_r0 import high_bankruptcy_higher_arithmetic_expectation,survivor_filter_is_rejected,capital_injection_changes_denominator,mixed_generation_not_final_checkpoint,baseline_disagreement_is_unresolved
@dataclass(frozen=True)
class ThreadCQualification:
    gates:Mapping[str,bool]; evidence_level:str="COMPONENT_LOCAL_SYNTHETIC"; final_holdout_opened:bool=False; fresh_market_data_used:bool=False
    @property
    def passed(self):return all(self.gates.values()) and not self.final_holdout_opened and not self.fresh_market_data_used
def compile_thread_c_qualification(*,semantic_gates:Mapping[str,bool]):
    before,after=capital_injection_changes_denominator(); g={"KNOWN_ANSWER_HIGH_BANKRUPTCY_ARITHMETIC":high_bankruptcy_higher_arithmetic_expectation()["mean"]>0,"SURVIVOR_FILTER_REJECTED":survivor_filter_is_rejected(),"CAPITAL_RESTART_NOT_FREE":after>before,"MIXED_GENERATION_NOT_FINAL_CHECKPOINT":mixed_generation_not_final_checkpoint(),"BASELINE_CONFLICT_UNRESOLVED":baseline_disagreement_is_unresolved()=="UNRESOLVED_OWNER_DECISION"}; g.update(dict(semantic_gates)); return ThreadCQualification(g)
