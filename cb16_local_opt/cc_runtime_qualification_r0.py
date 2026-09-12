from __future__ import annotations
REQUIRED=("wire_contract_valid","four_clocks_distinct","continuous_runtime","no_order_environment_progression","account_serialization","pause_resume_exact","generation_switch_continuity","hostile_economics_matrix")
def compile_thread_a_qualification_r0(claims:dict[str,bool],*,test_command:str)->dict:
    unknown=set(claims)-set(REQUIRED)
    if unknown: raise RuntimeError("CCQUAL_UNAUTHORIZED_CLAIM:"+",".join(sorted(unknown)))
    normalized={k:bool(claims.get(k,False)) for k in REQUIRED}
    return {"schema":"CB16_R11_CC_THREAD_A_QUALIFICATION_V1_R0","thread":"A","status":"PASS" if all(normalized.values()) else "FAIL","evidence_level":"THREAD_LOCAL_CLOSED_LOOP" if all(normalized.values()) else "COMPONENT","claims":normalized,"test_command":test_command,"not_claimed":["learner","experience_storage","economic_improvement","high_throughput"]}
