from dataclasses import dataclass
from enum import Enum
from .cc_economic_cohort_r0 import EconomicCohort
from .cc_economic_evaluator_r0 import EconomicEvaluation
class PromotionDecision(str,Enum): PASS="PASS"; FAIL="FAIL"; UNRESOLVED_OWNER_DECISION="UNRESOLVED_OWNER_DECISION"
@dataclass(frozen=True)
class PromotionArtifact: evaluation_id:str; expected_arithmetic_return:float; buy_hold_delta:float; flat_delta:float; decision:PromotionDecision; reason:str; evidence_level:str="COMPONENT_LOCAL_SYNTHETIC"
def decide_promotion(e:EconomicEvaluation,c:EconomicCohort):
    if not c.formal_horizon_owner_rule_resolved:return PromotionArtifact(e.evaluation_id,e.mean_arithmetic_return,e.buy_hold_delta,e.flat_delta,PromotionDecision.UNRESOLVED_OWNER_DECISION,"FORMAL_HORIZON_OR_MASTER_RANKING_OWNER_RULE_UNRESOLVED")
    bh,fl=e.buy_hold_delta,e.flat_delta
    if bh>0 and fl>0:d,r=PromotionDecision.PASS,"POSITIVE_VS_BOTH_DECLARED_BASELINES"
    elif bh<=0 and fl<=0:d,r=PromotionDecision.FAIL,"NONPOSITIVE_VS_BOTH_DECLARED_BASELINES"
    else:d,r=PromotionDecision.UNRESOLVED_OWNER_DECISION,"B_AND_H_FLAT_COMPARATOR_CONFLICT"
    return PromotionArtifact(e.evaluation_id,e.mean_arithmetic_return,bh,fl,d,r)
