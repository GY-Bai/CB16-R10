from __future__ import annotations
from .cc_economic_cohort_r0 import EconomicCohort
from .cc_economic_evaluator_r0 import AccountEconomicOutcome, evaluate
from .cc_economic_capital_r0 import CapitalLedger, CapitalFlow
from .cc_economic_policy_identity_r0 import EconomicPolicyIdentity
from .cc_economic_promotion_r0 import assess, UNRESOLVED_OWNER_DECISION

def _cohort(ids=("a","b"), policy_type="frozen_checkpoint", policy_id="p"):
    return EconomicCohort("c",{i:f"s-{i}" for i in ids},{i:1 for i in ids},"cap","h",policy_type,policy_id,"REALIZED")

def run_known_answer_toys() -> dict[str,bool]:
    # 1: arithmetic expectation can favor high-bankruptcy strategy.
    risky=evaluate(evaluation_id="r",cohort=_cohort(),outcomes=[AccountEconomicOutcome("a",-1,failed=True,liquidated=True),AccountEconomicOutcome("b",3)],buy_hold_return=.4)
    safe=evaluate(evaluation_id="s",cohort=_cohort(),outcomes=[AccountEconomicOutcome("a",.3),AccountEconomicOutcome("b",.3)],buy_hold_return=.4)
    t1=risky.mean_arithmetic_return > safe.mean_arithmetic_return and risky.failure_counts["liquidated"]==1
    # 2: survivor deletion is rejected by evaluator cohort integrity.
    try:
        evaluate(evaluation_id="bad",cohort=_cohort(),outcomes=[AccountEconomicOutcome("b",3)],buy_hold_return=0); t2=False
    except ValueError: t2=True
    # 3: injected new money enters gross denominator identity.
    l0=CapitalLedger(100); l1=CapitalLedger(100,(CapitalFlow("a","NEW_ACCOUNT_ALLOCATION",50,"f1"),))
    t3=l1.gross_capital_denominator==150 and l0.denominator_id!=l1.denominator_id
    # 4: mixed generations cannot masquerade as final checkpoint.
    chain=EconomicPolicyIdentity.generation_chain("deploy",("G3","G4","G5"))
    try: chain.assert_pure_generation("G5"); t4=False
    except ValueError: t4=True
    # 5: baseline conflict remains unresolved.
    conflict=evaluate(evaluation_id="x",cohort=_cohort(),outcomes=[AccountEconomicOutcome("a",.2),AccountEconomicOutcome("b",.2)],buy_hold_return=.3,flat_return=0)
    t5=assess(conflict).status==UNRESOLVED_OWNER_DECISION
    return {"high_bankruptcy_higher_expectation":t1,"survivor_mean_rejected":t2,"capital_injection_changes_denominator":t3,"mixed_generation_not_final_checkpoint":t4,"baseline_disagreement_unresolved":t5}
