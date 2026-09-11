#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, multiprocessing as mp, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
import numpy as np, torch
from cb16_local_opt.frozen_sensory_stack_r10 import FrozenSensoryStackR10
from cb16_local_opt.full_minute_long_trajectory_r1 import FINAL_HOLDOUT_START_MS
from cb16_local_opt.longtraj_infra_closure_r0 import find_contiguous_prefinal_run_r0, funding_events_by_minute_r0
from cb16_local_opt.minute_physics_binding_r2 import canonical_hash
from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.r102_evidence_cache import ParentContextR102
from cb16_local_opt.r102_learning import evidence_summary
from cb16_local_opt.r102_physics import CANDIDATES_R102, FrozenPhysicsRuntimeR102
from cb16_local_opt.r11_teacher_authority_candidate import R11_TRAIN_TEACHER_CONFIG, R11_VALIDATION_TEACHER_CONFIG
from cb16_local_opt.science_feedback_diagnostics_r11 import validate_teacher_targets_r11
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.training_runtime_r11 import EvaluationRuntimeR11, PreparedEvidenceR11, policy_hash_r11, prepare_evidence_campaign_r11
from scripts import r11_longtraj_generalization_r0_evaluate as base
from scripts import r11_longtraj_training_admission_r0 as adm
from scripts import r11_longtraj_training_campaign_r0 as camp
from scripts.r11_longtraj_training_admission_r0_entry import build_sensory_frame_exact_r0
from scripts import r11_science_g0_historical_r1 as r1
from scripts.r11_science_g0_canonical_historical_learning_baseline_r4 import clone_state, state_equal

ROOT=Path(__file__).resolve().parents[1]
REPAIR=ROOT/"authority/rearchitecture_r11/CB16_R11_LONGTRAJ_GENERALIZATION_R0_EVALUATOR_R1_REPAIR_SPEC_V1.json"
OLD_VAL="0ac7a0490e0df3c97150c14e84ca378fe3cc33d767ec26a907aad3a781297131"

def main()->int:
    p=argparse.ArgumentParser()
    for a in ("raw-root","package-root","challenger-checkpoint","work-root","output"): p.add_argument("--"+a,required=True)
    p.add_argument("--g0-root",default=os.environ.get("CB16_G0_ROOT","/cb16/g0")); p.add_argument("--symbol",default="BTCUSDT")
    p.add_argument("--workers",type=int,default=8); p.add_argument("--device",default="cuda")
    a=p.parse_args(); q=base.require
    out=Path(a.output).resolve(); Path(a.work_root).mkdir(parents=True,exist_ok=True); pkg=Path(a.package_root).resolve(); g0=Path(a.g0_root).resolve()
    spec=json.loads(base.SPEC.read_text()); cohort=json.loads(base.COHORT.read_text()); sel=json.loads(base.SELECTION_RECEIPT.read_text()); repair=json.loads(REPAIR.read_text())
    q(spec["status"]=="FROZEN_BEFORE_FIRST_EVALUATION_EXECUTION","GZR1_SPEC"); q(cohort["status"]=="FROZEN_EXACT_COHORT","GZR1_COHORT")
    q(sel["status"]=="FROZEN_AND_QUALIFIED","GZR1_SELECTION"); q(repair["status"]=="FROZEN_BEFORE_R1_REPLAY","GZR1_REPAIR")
    q(repair["scientific_gates_unchanged"] is True and repair["G1_G2_G3_unchanged"] is True,"GZR1_GATE_DRIFT")
    q(a.symbol=="BTCUSDT" and a.workers==8 and a.device=="cuda","GZR1_RUNTIME_SCOPE")
    q(base.sha256_file(base.COHORT)==spec["selection_authority"]["cohort_file_sha256"],"GZR1_COHORT_HASH")
    q(cohort["cohort"]["decision_times_canonical_sha256"]==spec["selection_authority"]["cohort_decision_times_sha256"],"GZR1_COHORT_DECISIONS")
    q(cohort["cohort"]["zero_overlap_with_prior_consumed_tail"] is True and int(cohort["cohort"]["dependence_groups"])==48,"GZR1_COHORT_GEOMETRY")
    q(int(cohort["cohort"]["future_end_times_ms"][-1])<FINAL_HOLDOUT_START_MS,"GZR1_FINAL")

    src=adm.BinanceUSDMArchiveSourceR10(a.raw_root); q(a.symbol in src.validate_layout()["symbols"],"GZR1_SYMBOL")
    start=int(cohort["cohort"]["used_segments"][0]["start_ms"]); last=int(cohort["cohort"]["future_end_times_ms"][-1])
    rec=find_contiguous_prefinal_run_r0(src,a.symbol,required_rows=(last-start)//adm.MINUTE_MS+1)
    q(int(rec[0].open_time)==start and int(rec[-1].open_time)>=last,"GZR1_SEGMENT")
    def ix(t:int)->int:
        d=t-int(rec[0].open_time); q(d>=0 and d%adm.MINUTE_MS==0,f"GZR1_TIME:{t}"); i=d//adm.MINUTE_MS
        q(0<=i<len(rec) and int(rec[i].open_time)==t,f"GZR1_MISSING:{t}"); return int(i)

    ci=adm.select_parent_indices_r0(rec,adm.TOTAL_GROUPS); cd=[int(rec[i].open_time) for i in ci]
    q(camp.canonical_json_sha256(cd)==base.EXPECTED_CAMPAIGN_DECISION_TIMES,"GZR1_CAMPAIGN_COHORT")
    train_t=cd[:48]; old_val_t=cd[48:]; new_t=[int(x) for x in cohort["cohort"]["decision_times_ms"]]
    q(len(train_t)==48 and len(old_val_t)==12 and len(new_t)==48,"GZR1_COUNTS")
    q(max(old_val_t)+adm.H72_MINUTES*adm.MINUTE_MS<int(cohort["cohort"]["first_new_sensory_start_ms"]),"GZR1_OVERLAP")

    entries=[]
    for j,i in enumerate(ci): entries.append(("TRAIN" if j<48 else "VALIDATION",int(i),f"ADMR0:{a.symbol}:{int(rec[i].open_time)}","FLAT_MINUTE_R2_ADMISSION"))
    for t in new_t: entries.append(("VALIDATION",ix(t),f"GENR0:{a.symbol}:{t}","GENERALIZATION_R0_UNCONSUMED"))
    q(len(entries)==108,"GZR1_ENTRIES")
    first=min(e[1] for e in entries)-(adm.SENSORY_PREFIX_MINUTES-1); fstart=int(rec[first].open_time); fend=max(int(rec[e[1]+adm.H72_MINUTES].open_time) for e in entries)
    funding=funding_events_by_minute_r0(src,a.symbol,start_ms=fstart,end_ms=fend); physics=FrozenPhysicsRuntimeR102.load(pkg)
    frames=[build_sensory_frame_exact_r0(rec,e[1],a.symbol) for e in entries]
    sensory=FrozenSensoryStackR10(pkg,device=a.device,verify_hashes=True); enc={}
    for s in range(0,len(frames),8):
        z=sensory.encode_frames(frames[s:s+8])
        for j in range(len(frames[s:s+8])): enc[s+j]=(z.operator48[j].copy(),z.medium48[j].copy(),z.ordered4h30[j].copy())
    q(sum(int(p.requires_grad) for m in (sensory.operator.tok,sensory.operator.model,sensory.medium.model) for p in m.parameters())==0,"GZR1_SENSORY")
    del sensory
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    parents={}; payloads=[]
    for n,(split,i,pid,scenario) in enumerate(entries):
        row=rec[i]; fut=tuple(rec[i+1:i+1+adm.H72_MINUTES]); q(len(fut)==adm.H72_MINUTES,"GZR1_H72")
        q(all(int(x.open_time)<FINAL_HOLDOUT_START_MS for x in fut),"GZR1_FUT_FINAL")
        pref=tuple(rec[i-(adm.e6.PARENT_PREFIX_MINUTES-1):i+1]); st,ra,acc=adm.e6._build_parent_state(physics,symbol=a.symbol,account_id=pid,prefix_rows=pref,funding=funding)
        lh=adm.e6._future_hash(a.symbol,int(row.open_time),fut,funding); op,med,ord4=enc[n]; dep=f"FUT:{a.symbol}:{int(row.open_time)}:{lh[:16]}"
        parents[pid]=ParentContextR102(parent_id=pid,dependence_group_id=dep,symbol=a.symbol,decision_time_ms=int(row.open_time),split=split,scenario=scenario,operator48=tuple(float(x) for x in op),medium48=tuple(float(x) for x in med),account6=tuple(float(x) for x in acc),ordered4h30=tuple(float(x) for x in ord4),current_mark=float(row.close),snapshot_sha256=canonical_hash(st),eligible_for_economic_evidence=True,market_lineage_hash=lh)
        payloads.append({"ordinal":n,"package_root":str(pkg),"symbol":a.symbol,"parent_id":pid,"decision_time_ms":int(row.open_time),"parent_state":st,"risk_authority":ra,"future_rows":fut,"funding":dict(funding)})
    q(len(parents)==108,"GZR1_PARENT_UNIQUE")
    causal=adm.e6._prefix_causality_canaries(str(pkg),a.symbol,payloads[60]["parent_state"],payloads[60]["risk_authority"],payloads[60]["future_rows"],funding); q(all(causal.values()),f"GZR1_CAUSAL:{causal}")
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context("spawn")) as pool: ex=list(pool.map(adm._simulate_parent,payloads))
    ex.sort(key=lambda x:int(x["ordinal"])); q(len(ex)==108 and all(len(x["branches"])==len(CANDIDATES_R102) for x in ex),"GZR1_GRID"); ex={str(x["parent_id"]):x for x in ex}

    def samples(ids):
        res=[]
        for pid in ids:
            pc=parents[pid]
            for b in ex[pid]["branches"]:
                z=CounterfactualBranchSampleR5(parent_id=pid,student_context_object_id=pc.student_context_object_id,timestamp=int(pc.decision_time_ms),context_features=tuple(float(v) for v in pc.student_features),direction=camp.DIRECTION_TO_TEACHER[int(b["direction"])],requested_risk=float(b["requested_risk"]),realized_utility=float(b["utility"]),dependence_group_id=pc.dependence_group_id,market_lineage_hash=pc.market_lineage_hash)
                z.validate(); res.append(z)
        return res

    old_ids=[e[2] for e in entries[:60]]; train_ids=old_ids[:48]; consumed_ids=old_ids[48:]; new_ids=[e[2] for e in entries[60:]]
    q(set(train_ids).isdisjoint(new_ids) and set(consumed_ids).isdisjoint(new_ids),"GZR1_ID_OVERLAP")
    pa={x:parents[x] for x in old_ids}; ta,va,sta=compile_teacher_evidence_r11(samples=samples(old_ids),parents=pa,train_config=R11_TRAIN_TEACHER_CONFIG,val_config=R11_VALIDATION_TEACHER_CONFIG,workers=8,block_targets=32)
    tsa,vsa=evidence_summary(ta),evidence_summary(va); q(tsa["admitted_dependence_groups"]==48 and tsa["rejected"]==0,f"GZR1_A_TRAIN:{tsa}"); q(vsa["admitted_dependence_groups"]==12 and vsa["rejected"]==0,f"GZR1_A_VAL:{vsa}")
    ca=prepare_evidence_campaign_r11(train_evidence=ta,validation_evidence=va,parents=pa,device=a.device)
    q(ca.train.evidence_hash==base.EXPECTED_TRAIN_EVIDENCE,"GZR1_A_TRAIN_HASH"); q(ca.validation.evidence_hash==OLD_VAL,"GZR1_A_VAL_HASH")
    train_audit=validate_teacher_targets_r11(ca.train); q(train_audit["independent_dependence_groups"]==48,"GZR1_A_AUDIT")

    b_ids=train_ids+new_ids; pb={x:parents[x] for x in b_ids}; tb,vb,stb=compile_teacher_evidence_r11(samples=samples(b_ids),parents=pb,train_config=R11_TRAIN_TEACHER_CONFIG,val_config=R11_VALIDATION_TEACHER_CONFIG,workers=8,block_targets=32)
    tsb,vsb=evidence_summary(tb),evidence_summary(vb)
    if not (vsb["admitted_dependence_groups"]==48 and vsb["rejected"]==0): return base._scientific_support_fail(out,spec=spec,cohort=cohort,train_summary=tsb,validation_summary=vsb,teacher_stats=stb)
    val=PreparedEvidenceR11.from_evidence(vb,pb,device=a.device); q(val.rows==48 and tuple(val.parent_ids)==tuple(new_ids),"GZR1_B_ROWS")
    val_audit=validate_teacher_targets_r11(val); q(val_audit["independent_dependence_groups"]==48 and bool(torch.all(val.group_weight==1.0).item()),"GZR1_B_AUDIT")

    tp=ca.train.direction_target_probs.detach().cpu().numpy().astype(np.float64); tr=ca.train.requested_risk_target.detach().cpu().numpy().astype(np.float64)
    cp=tp.mean(axis=0); cp=cp/cp.sum(); cr=base.smooth_l1_constant_minimizer(tr)
    ch1=r1.load_bootstrap_model(g0,a.device); ch2=r1.load_bootstrap_model(g0,a.device); q(policy_hash_r11(ch1)==base.EXPECTED_CHAMPION and policy_hash_r11(ch2)==base.EXPECTED_CHAMPION,"GZR1_CHAMP")
    x1=base.load_challenger(Path(a.challenger_checkpoint).resolve(),ch1,a.device); x2=base.load_challenger(Path(a.challenger_checkpoint).resolve(),ch2,a.device); cs,xis=clone_state(ch1),clone_state(x1)
    for m in (ch1,ch2,x1,x2): m.eval()
    with torch.inference_mode(): o1=ch1(val.operator48,val.medium48,val.account6); o2=ch2(val.operator48,val.medium48,val.account6); y1=x1(val.operator48,val.medium48,val.account6); y2=x2(val.operator48,val.medium48,val.account6)
    q(base.outputs_exact(o1,o2) and base.outputs_exact(y1,y2),"GZR1_RELOAD")
    ev=EvaluationRuntimeR11(enable_cuda_graph=False); ce=ev.evaluate(ch1,val,use_cache=False); xe=ev.evaluate(x1,val,use_cache=False)
    tgt=val.direction_target_probs.detach(); risk=val.requested_risk_target.detach(); cl=base.row_losses(o1,tgt,risk); xl=base.row_losses(y1,tgt,risk)
    q(abs(float(cl["total"].mean())-float(ce["loss"]))<2e-6 and abs(float(xl["total"].mean())-float(xe["loss"]))<2e-6,"GZR1_LOSS")
    cpt=torch.tensor(cp,dtype=torch.float32,device=tgt.device).clamp_min(1e-12); crt=torch.tensor(float(cr),dtype=torch.float32,device=risk.device)
    clim=(-(tgt*torch.log(cpt)[None,:]).sum(dim=-1)+base.smooth_l1_rows(torch.full_like(risk,crt),risk)).detach().cpu().numpy().astype(np.float64)
    g1=base.paired_bootstrap(xl["total"]-cl["total"],seed=20260911,reps=10000); g2=base.paired_bootstrap(xl["total"]-clim,seed=20260912,reps=10000)
    shifts=[]; idx=torch.arange(48,device=tgt.device)
    for k in range(1,48):
        j=(idx+k)%48; shifts.append(float((base.row_losses(y1,tgt[j],risk[j])["total"]-base.row_losses(o1,tgt[j],risk[j])["total"]).mean()))
    ident=float(g1["point"]); nle=sum(v<=ident for v in shifts); ep=(1+nle)/48.0
    p1=g1["point"]<0 and g1["ci_high"]<0; p2=g2["point"]<0 and g2["ci_high"]<0; p3=nle==0 and ident<min(shifts) and abs(ep-1/48)<1e-15
    if not p1: status,klass="SCIENTIFIC_FAIL","SCIENTIFIC_FAIL__FROZEN_SHADOW_CHALLENGER_DID_NOT_GENERALIZE_VS_G0_CHAMPION"
    elif not p2: status,klass="SCIENTIFIC_FAIL","SCIENTIFIC_FAIL__CHALLENGER_IMPROVEMENT_DID_NOT_BEAT_PRIOR_NO_STATE_CLIMATOLOGY"
    elif not p3: status,klass="SCIENTIFIC_FAIL","SCIENTIFIC_FAIL__CHALLENGER_ADVANTAGE_NOT_SPECIFIC_TO_TRUE_STATE_TARGET_CORRESPONDENCE"
    else: status,klass="PASS","GENERALIZATION_TARGET_CORRESPONDENCE_QUALIFIED_FOR_SEPARATE_NEXT_GATE__NO_PROMOTION__NO_MARKET_VERDICT"
    q(state_equal(cs,ch1) and state_equal(xis,x1),"GZR1_MUTATION")
    result={"schema":base.SCHEMA,"status":status,"classification":klass,"spec_sha256":base.sha256_file(base.SPEC),"repair_spec_sha256":base.sha256_file(REPAIR),"executor_revision":"R1_TWO_TEACHER_UNIVERSES","cohort":{"path":str(base.COHORT.relative_to(ROOT)),"file_sha256":base.sha256_file(base.COHORT),"decision_times_sha256":cohort["cohort"]["decision_times_canonical_sha256"],"dependence_groups":48,"first_decision_time_ms":new_t[0],"last_decision_time_ms":new_t[-1],"cohort_resampled":False},"teacher":{"train_protocol_hash":R11_TRAIN_TEACHER_CONFIG.content_hash,"validation_protocol_hash":R11_VALIDATION_TEACHER_CONFIG.content_hash,"prior_train_evidence_hash":ca.train.evidence_hash,"new_validation_evidence_hash":val.evidence_hash,"validation_summary":vsb,"train_summary":tsa,"universe_a_consumed_validation_summary":vsa,"universe_b_non_authoritative_train_summary":tsb,"train_target_audit":train_audit,"validation_target_audit":val_audit,"consumed_validation_scored_in_generalization":False,"rebound_train_outputs_used_for_G2":False,"support_thresholds_relaxed":False,"outcome_as_label_used":False,"runtime_stats_a":str(sta),"runtime_stats_b":str(stb)},"models":{"champion_policy_hash":policy_hash_r11(ch1),"challenger_policy_hash":policy_hash_r11(x1),"challenger_checkpoint_sha256":base.sha256_file(Path(a.challenger_checkpoint).resolve()),"independent_champion_reload_outputs_exact":True,"independent_challenger_reload_outputs_exact":True,"training_or_tuning_on_generalization_cohort":False},"canonical_true_alignment":{"champion":{"loss":float(ce["loss"]),"direction_loss":float(ce["direction_loss"]),"sizing_loss":float(ce["sizing_loss"]),"behavior_fingerprint":ce["behavior_fingerprint"]},"challenger":{"loss":float(xe["loss"]),"direction_loss":float(xe["direction_loss"]),"sizing_loss":float(xe["sizing_loss"]),"behavior_fingerprint":xe["behavior_fingerprint"]}},"prior_train_climatology":{"source":"UNIVERSE_A_EXACT_FROZEN_CAMPAIGN_TRAIN_TARGETS","direction_probs":[float(x) for x in cp],"requested_risk":float(cr),"new_cohort_total_loss":float(clim.mean())},"G1":{**g1,"pass":bool(p1),"metric":"challenger_minus_champion_mean_total_loss"},"G2":{**g2,"pass":bool(p2),"metric":"challenger_minus_prior_train_climatology_mean_total_loss"},"G3":{"pass":bool(p3),"identity_delta_challenger_minus_champion":ident,"nonidentity_cyclic_shift_deltas":shifts,"best_nonidentity_shift_delta":float(min(shifts)),"worst_nonidentity_shift_delta":float(max(shifts)),"nonidentity_shifts_with_delta_le_identity":int(nle),"exact_one_sided_rank_p":float(ep),"permutation_family":"ALL_47_NONIDENTITY_CYCLIC_TARGET_ROW_SHIFTS","models_refit":False},"causality_canaries":causal,"firewalls":{"final_holdout_touched":False,"fresh_market_data_downloaded":False,"canonical_promotion_authorized":False,"canonical_generation_advance_authorized":False,"new_market_information_verdict":False,"scientific_market_verdict":None},"next_gate":"SEPARATE_PROMOTION_OR_REPLICATION_DESIGN_PREREGISTRATION_REQUIRED" if status=="PASS" else "STOP__NO_POST_RESULT_RESCUE_UNDER_R0"}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
