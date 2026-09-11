#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, multiprocessing as mp, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
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
REPAIR=ROOT/"authority/rearchitecture_r11/CB16_R11_LONGTRAJ_GENERALIZATION_R0_EVALUATOR_R2_REPAIR_SPEC_V1.json"
OLD_VAL="0ac7a0490e0df3c97150c14e84ca378fe3cc33d767ec26a907aad3a781297131"

def _samples(ids,parents,executed):
    out=[]
    for pid in ids:
        pc=parents[pid]; x=executed[pid]
        for b in x["branches"]:
            s=CounterfactualBranchSampleR5(parent_id=pid,student_context_object_id=pc.student_context_object_id,timestamp=int(pc.decision_time_ms),context_features=tuple(float(v) for v in pc.student_features),direction=camp.DIRECTION_TO_TEACHER[int(b["direction"])],requested_risk=float(b["requested_risk"]),realized_utility=float(b["utility"]),dependence_group_id=pc.dependence_group_id,market_lineage_hash=pc.market_lineage_hash)
            s.validate(); out.append(s)
    return out

def _run_pool(payloads,workers):
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn")) as pool: rows=list(pool.map(adm._simulate_parent,payloads))
    rows.sort(key=lambda x:int(x["ordinal"])); return {str(x["parent_id"]):x for x in rows}

def _support_fail(out,spec,cohort,tsa,vsa,tsb,vsb,sta,stb):
    x={"schema":base.SCHEMA,"status":"SCIENTIFIC_FAIL","classification":"SCIENTIFIC_FAIL__FROZEN_GENERALIZATION_COHORT_TEACHER_SUPPORT_NOT_QUALIFIED","executor_revision":"R2_EXACT_CAMPAIGN_REPRODUCTION","spec_sha256":base.sha256_file(base.SPEC),"repair_spec_sha256":base.sha256_file(REPAIR),"cohort":{"path":str(base.COHORT.relative_to(ROOT)),"file_sha256":base.sha256_file(base.COHORT),"decision_times_sha256":cohort["cohort"]["decision_times_canonical_sha256"],"dependence_groups":48,"cohort_resampled":False},"teacher":{"prior_train_evidence_hash":base.EXPECTED_TRAIN_EVIDENCE,"universe_a_train_summary":tsa,"universe_a_consumed_validation_summary":vsa,"universe_b_non_authoritative_train_summary":tsb,"validation_summary":vsb,"runtime_stats_a":str(sta),"runtime_stats_b":str(stb),"support_thresholds_relaxed":False},"models_loaded":False,"models_scored":False,"firewalls":{"final_holdout_touched":False,"fresh_market_data_downloaded":False,"canonical_promotion_authorized":False,"canonical_generation_advance_authorized":False,"new_market_information_verdict":False,"scientific_market_verdict":None},"next_gate":"STOP__NO_RESCUE_OR_COHORT_REPLACEMENT_UNDER_R0"}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n"); print(json.dumps(x,indent=2,sort_keys=True)); return 0

def main()->int:
    p=argparse.ArgumentParser()
    for k in ("raw-root","package-root","challenger-checkpoint","work-root","output"): p.add_argument("--"+k,required=True)
    p.add_argument("--g0-root",default=os.environ.get("CB16_G0_ROOT","/cb16/g0")); p.add_argument("--symbol",default="BTCUSDT"); p.add_argument("--workers",type=int,default=8); p.add_argument("--device",default="cuda")
    a=p.parse_args(); q=base.require; out=Path(a.output).resolve(); Path(a.work_root).mkdir(parents=True,exist_ok=True); pkg=Path(a.package_root).resolve(); g0=Path(a.g0_root).resolve()
    spec=json.loads(base.SPEC.read_text()); cohort=json.loads(base.COHORT.read_text()); sel=json.loads(base.SELECTION_RECEIPT.read_text()); repair=json.loads(REPAIR.read_text())
    q(spec["status"]=="FROZEN_BEFORE_FIRST_EVALUATION_EXECUTION","GZR2_SPEC"); q(cohort["status"]=="FROZEN_EXACT_COHORT","GZR2_COHORT"); q(sel["status"]=="FROZEN_AND_QUALIFIED","GZR2_SELECTION"); q(repair["status"]=="FROZEN_BEFORE_R2_REPLAY","GZR2_REPAIR")
    q(repair["scientific_gates_unchanged"] is True and repair["G1_G2_G3_unchanged"] is True,"GZR2_GATE_DRIFT"); q(a.symbol=="BTCUSDT" and a.workers==8 and a.device=="cuda","GZR2_RUNTIME")
    q(base.sha256_file(base.COHORT)==spec["selection_authority"]["cohort_file_sha256"],"GZR2_COHORT_HASH"); q(cohort["cohort"]["decision_times_canonical_sha256"]==spec["selection_authority"]["cohort_decision_times_sha256"],"GZR2_DECISION_HASH"); q(cohort["cohort"]["zero_overlap_with_prior_consumed_tail"] is True and int(cohort["cohort"]["dependence_groups"])==48,"GZR2_GEOMETRY"); q(int(cohort["cohort"]["future_end_times_ms"][-1])<FINAL_HOLDOUT_START_MS,"GZR2_FINAL")

    src=adm.BinanceUSDMArchiveSourceR10(a.raw_root); q(a.symbol in src.validate_layout()["symbols"],"GZR2_SYMBOL"); physics=FrozenPhysicsRuntimeR102.load(pkg)
    old_required=adm.SENSORY_PREFIX_MINUTES+(adm.TOTAL_GROUPS-1)*73*60+adm.H72_MINUTES+180
    old_rec=find_contiguous_prefinal_run_r0(src,a.symbol,required_rows=old_required); q(all(int(x.open_time)<FINAL_HOLDOUT_START_MS for x in old_rec),"GZR2_OLD_FINAL")
    old_ix=adm.select_parent_indices_r0(old_rec,adm.TOTAL_GROUPS); old_times=[int(old_rec[i].open_time) for i in old_ix]; q(camp.canonical_json_sha256(old_times)==base.EXPECTED_CAMPAIGN_DECISION_TIMES,"GZR2_OLD_COHORT")
    old_first=old_ix[0]-(adm.SENSORY_PREFIX_MINUTES-1); old_fstart=int(old_rec[old_first].open_time); old_fend=int(old_rec[old_ix[-1]+adm.H72_MINUTES].open_time); old_funding=funding_events_by_minute_r0(src,a.symbol,start_ms=old_fstart,end_ms=old_fend)

    new_start=int(cohort["cohort"]["used_segments"][0]["start_ms"]); new_last=int(cohort["cohort"]["future_end_times_ms"][-1]); new_rec=find_contiguous_prefinal_run_r0(src,a.symbol,required_rows=(new_last-new_start)//adm.MINUTE_MS+1)
    q(int(new_rec[0].open_time)==new_start and int(new_rec[-1].open_time)>=new_last,"GZR2_NEW_SEGMENT")
    def nix(t):
        d=int(t)-int(new_rec[0].open_time); q(d>=0 and d%adm.MINUTE_MS==0,f"GZR2_NEW_TIME:{t}"); i=d//adm.MINUTE_MS; q(0<=i<len(new_rec) and int(new_rec[i].open_time)==int(t),f"GZR2_NEW_MISSING:{t}"); return int(i)
    new_times=[int(x) for x in cohort["cohort"]["decision_times_ms"]]; new_ix=[nix(t) for t in new_times]; q(len(new_ix)==48,"GZR2_NEW_COUNT"); q(old_times[-1]+adm.H72_MINUTES*adm.MINUTE_MS<int(cohort["cohort"]["first_new_sensory_start_ms"]),"GZR2_OVERLAP")
    new_first=min(new_ix)-(adm.SENSORY_PREFIX_MINUTES-1); new_fstart=int(new_rec[new_first].open_time); new_fend=max(int(new_rec[i+adm.H72_MINUTES].open_time) for i in new_ix); new_funding=funding_events_by_minute_r0(src,a.symbol,start_ms=new_fstart,end_ms=new_fend)

    old_frames=[build_sensory_frame_exact_r0(old_rec,i,a.symbol) for i in old_ix]; new_frames=[build_sensory_frame_exact_r0(new_rec,i,a.symbol) for i in new_ix]
    sensory=FrozenSensoryStackR10(pkg,device=a.device,verify_hashes=True); old_enc={}; new_enc={}
    for s in range(0,60,8):
        chunk=old_frames[s:s+8]; z=sensory.encode_frames(chunk)
        for j in range(len(chunk)): old_enc[s+j]=(z.operator48[j].copy(),z.medium48[j].copy(),z.ordered4h30[j].copy())
    q([len(old_frames[s:s+8]) for s in range(0,60,8)]==[8,8,8,8,8,8,8,4],"GZR2_OLD_BATCH_GEOMETRY")
    for s in range(0,48,8):
        chunk=new_frames[s:s+8]; z=sensory.encode_frames(chunk)
        for j in range(len(chunk)): new_enc[s+j]=(z.operator48[j].copy(),z.medium48[j].copy(),z.ordered4h30[j].copy())
    q(sum(int(p.requires_grad) for m in (sensory.operator.tok,sensory.operator.model,sensory.medium.model) for p in m.parameters())==0,"GZR2_SENSORY"); del sensory
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    old_parents={}; old_payloads=[]
    for n,i in enumerate(old_ix):
        row=old_rec[i]; fut=tuple(old_rec[i+1:i+1+adm.H72_MINUTES]); q(len(fut)==adm.H72_MINUTES,"GZR2_OLD_H72"); pid=f"ADMR0:{a.symbol}:{int(row.open_time)}"; pref=tuple(old_rec[i-(adm.e6.PARENT_PREFIX_MINUTES-1):i+1]); st,ra,acc=adm.e6._build_parent_state(physics,symbol=a.symbol,account_id=pid,prefix_rows=pref,funding=old_funding); lh=adm.e6._future_hash(a.symbol,int(row.open_time),fut,old_funding); op,med,ord4=old_enc[n]; dep=f"FUT:{a.symbol}:{int(row.open_time)}:{lh[:16]}"; split="TRAIN" if n<48 else "VALIDATION"
        old_parents[pid]=ParentContextR102(parent_id=pid,dependence_group_id=dep,symbol=a.symbol,decision_time_ms=int(row.open_time),split=split,scenario="FLAT_MINUTE_R2_ADMISSION",operator48=tuple(float(x) for x in op),medium48=tuple(float(x) for x in med),account6=tuple(float(x) for x in acc),ordered4h30=tuple(float(x) for x in ord4),current_mark=float(row.close),snapshot_sha256=canonical_hash(st),eligible_for_economic_evidence=True,market_lineage_hash=lh)
        old_payloads.append({"ordinal":n,"package_root":str(pkg),"symbol":a.symbol,"parent_id":pid,"decision_time_ms":int(row.open_time),"parent_state":st,"risk_authority":ra,"future_rows":fut,"funding":dict(old_funding)})
    old_causal=adm.e6._prefix_causality_canaries(str(pkg),a.symbol,old_payloads[0]["parent_state"],old_payloads[0]["risk_authority"],old_payloads[0]["future_rows"],old_funding); q(all(old_causal.values()),f"GZR2_OLD_CAUSAL:{old_causal}")
    old_exec=_run_pool(old_payloads,8); q(len(old_exec)==60 and all(len(x["branches"])==len(CANDIDATES_R102) for x in old_exec.values()),"GZR2_OLD_GRID")
    old_ids=[f"ADMR0:{a.symbol}:{t}" for t in old_times]; train_ids=old_ids[:48]; consumed_ids=old_ids[48:]; old_samples=_samples(old_ids,old_parents,old_exec)
    ta,va,sta=compile_teacher_evidence_r11(samples=old_samples,parents=old_parents,train_config=R11_TRAIN_TEACHER_CONFIG,val_config=R11_VALIDATION_TEACHER_CONFIG,workers=8,block_targets=32); tsa,vsa=evidence_summary(ta),evidence_summary(va)
    q(tsa["admitted_dependence_groups"]==48 and tsa["rejected"]==0,f"GZR2_A_TRAIN:{tsa}"); q(vsa["admitted_dependence_groups"]==12 and vsa["rejected"]==0,f"GZR2_A_VAL:{vsa}")
    ca=prepare_evidence_campaign_r11(train_evidence=ta,validation_evidence=va,parents=old_parents,device=a.device); q(ca.train.evidence_hash==base.EXPECTED_TRAIN_EVIDENCE,f"GZR2_A_TRAIN_HASH:{ca.train.evidence_hash}"); q(ca.validation.evidence_hash==OLD_VAL,f"GZR2_A_VAL_HASH:{ca.validation.evidence_hash}")
    train_audit=validate_teacher_targets_r11(ca.train); q(train_audit["independent_dependence_groups"]==48,"GZR2_A_AUDIT")

    new_parents={}; new_payloads=[]
    for n,i in enumerate(new_ix):
        row=new_rec[i]; fut=tuple(new_rec[i+1:i+1+adm.H72_MINUTES]); q(len(fut)==adm.H72_MINUTES and all(int(x.open_time)<FINAL_HOLDOUT_START_MS for x in fut),"GZR2_NEW_H72"); pid=f"GENR0:{a.symbol}:{int(row.open_time)}"; pref=tuple(new_rec[i-(adm.e6.PARENT_PREFIX_MINUTES-1):i+1]); st,ra,acc=adm.e6._build_parent_state(physics,symbol=a.symbol,account_id=pid,prefix_rows=pref,funding=new_funding); lh=adm.e6._future_hash(a.symbol,int(row.open_time),fut,new_funding); op,med,ord4=new_enc[n]; dep=f"FUT:{a.symbol}:{int(row.open_time)}:{lh[:16]}"
        new_parents[pid]=ParentContextR102(parent_id=pid,dependence_group_id=dep,symbol=a.symbol,decision_time_ms=int(row.open_time),split="VALIDATION",scenario="GENERALIZATION_R0_UNCONSUMED",operator48=tuple(float(x) for x in op),medium48=tuple(float(x) for x in med),account6=tuple(float(x) for x in acc),ordered4h30=tuple(float(x) for x in ord4),current_mark=float(row.close),snapshot_sha256=canonical_hash(st),eligible_for_economic_evidence=True,market_lineage_hash=lh)
        new_payloads.append({"ordinal":n,"package_root":str(pkg),"symbol":a.symbol,"parent_id":pid,"decision_time_ms":int(row.open_time),"parent_state":st,"risk_authority":ra,"future_rows":fut,"funding":dict(new_funding)})
    new_causal=adm.e6._prefix_causality_canaries(str(pkg),a.symbol,new_payloads[0]["parent_state"],new_payloads[0]["risk_authority"],new_payloads[0]["future_rows"],new_funding); q(all(new_causal.values()),f"GZR2_NEW_CAUSAL:{new_causal}")
    new_exec=_run_pool(new_payloads,8); q(len(new_exec)==48 and all(len(x["branches"])==len(CANDIDATES_R102) for x in new_exec.values()),"GZR2_NEW_GRID"); new_ids=[f"GENR0:{a.symbol}:{t}" for t in new_times]; q(set(old_ids).isdisjoint(new_ids),"GZR2_ID_OVERLAP")

    b_parents={pid:old_parents[pid] for pid in train_ids}; b_parents.update(new_parents); b_exec={pid:old_exec[pid] for pid in train_ids}; b_exec.update(new_exec); b_ids=train_ids+new_ids; b_samples=_samples(b_ids,b_parents,b_exec)
    tb,vb,stb=compile_teacher_evidence_r11(samples=b_samples,parents=b_parents,train_config=R11_TRAIN_TEACHER_CONFIG,val_config=R11_VALIDATION_TEACHER_CONFIG,workers=8,block_targets=32); tsb,vsb=evidence_summary(tb),evidence_summary(vb)
    if not (vsb["admitted_dependence_groups"]==48 and vsb["rejected"]==0): return _support_fail(out,spec,cohort,tsa,vsa,tsb,vsb,sta,stb)
    val=PreparedEvidenceR11.from_evidence(vb,b_parents,device=a.device); q(val.rows==48 and tuple(val.parent_ids)==tuple(new_ids),"GZR2_B_ROWS"); val_audit=validate_teacher_targets_r11(val); q(val_audit["independent_dependence_groups"]==48 and bool(torch.all(val.group_weight==1.0).item()),"GZR2_B_AUDIT")

    tp=ca.train.direction_target_probs.detach().cpu().numpy().astype(np.float64); tr=ca.train.requested_risk_target.detach().cpu().numpy().astype(np.float64); cp=tp.mean(axis=0); cp=cp/cp.sum(); cr=base.smooth_l1_constant_minimizer(tr)
    ch1=r1.load_bootstrap_model(g0,a.device); ch2=r1.load_bootstrap_model(g0,a.device); q(policy_hash_r11(ch1)==base.EXPECTED_CHAMPION and policy_hash_r11(ch2)==base.EXPECTED_CHAMPION,"GZR2_CHAMP")
    x1=base.load_challenger(Path(a.challenger_checkpoint).resolve(),ch1,a.device); x2=base.load_challenger(Path(a.challenger_checkpoint).resolve(),ch2,a.device); cs,xis=clone_state(ch1),clone_state(x1)
    for m in (ch1,ch2,x1,x2): m.eval()
    with torch.inference_mode(): o1=ch1(val.operator48,val.medium48,val.account6); o2=ch2(val.operator48,val.medium48,val.account6); y1=x1(val.operator48,val.medium48,val.account6); y2=x2(val.operator48,val.medium48,val.account6)
    q(base.outputs_exact(o1,o2) and base.outputs_exact(y1,y2),"GZR2_RELOAD")
    ev=EvaluationRuntimeR11(enable_cuda_graph=False); ce=ev.evaluate(ch1,val,use_cache=False); xe=ev.evaluate(x1,val,use_cache=False); tgt=val.direction_target_probs.detach(); risk=val.requested_risk_target.detach(); cl=base.row_losses(o1,tgt,risk); xl=base.row_losses(y1,tgt,risk)
    q(abs(float(cl["total"].mean())-float(ce["loss"]))<2e-6 and abs(float(xl["total"].mean())-float(xe["loss"]))<2e-6,"GZR2_LOSS")
    cpt=torch.tensor(cp,dtype=torch.float32,device=tgt.device).clamp_min(1e-12); crt=torch.tensor(float(cr),dtype=torch.float32,device=risk.device); clim=(-(tgt*torch.log(cpt)[None,:]).sum(dim=-1)+base.smooth_l1_rows(torch.full_like(risk,crt),risk)).detach().cpu().numpy().astype(np.float64)
    g1=base.paired_bootstrap(xl["total"]-cl["total"],seed=20260911,reps=10000); g2=base.paired_bootstrap(xl["total"]-clim,seed=20260912,reps=10000); shifts=[]; ii=torch.arange(48,device=tgt.device)
    for k in range(1,48):
        j=(ii+k)%48; shifts.append(float((base.row_losses(y1,tgt[j],risk[j])["total"]-base.row_losses(o1,tgt[j],risk[j])["total"]).mean()))
    ident=float(g1["point"]); nle=sum(v<=ident for v in shifts); ep=(1+nle)/48.0; p1=g1["point"]<0 and g1["ci_high"]<0; p2=g2["point"]<0 and g2["ci_high"]<0; p3=nle==0 and ident<min(shifts) and abs(ep-1/48)<1e-15
    if not p1: status,klass="SCIENTIFIC_FAIL","SCIENTIFIC_FAIL__FROZEN_SHADOW_CHALLENGER_DID_NOT_GENERALIZE_VS_G0_CHAMPION"
    elif not p2: status,klass="SCIENTIFIC_FAIL","SCIENTIFIC_FAIL__CHALLENGER_IMPROVEMENT_DID_NOT_BEAT_PRIOR_NO_STATE_CLIMATOLOGY"
    elif not p3: status,klass="SCIENTIFIC_FAIL","SCIENTIFIC_FAIL__CHALLENGER_ADVANTAGE_NOT_SPECIFIC_TO_TRUE_STATE_TARGET_CORRESPONDENCE"
    else: status,klass="PASS","GENERALIZATION_TARGET_CORRESPONDENCE_QUALIFIED_FOR_SEPARATE_NEXT_GATE__NO_PROMOTION__NO_MARKET_VERDICT"
    q(state_equal(cs,ch1) and state_equal(xis,x1),"GZR2_MUTATION"); q(policy_hash_r11(ch1)==base.EXPECTED_CHAMPION and policy_hash_r11(x1)==base.EXPECTED_CHALLENGER,"GZR2_POST_HASH")
    result={"schema":base.SCHEMA,"status":status,"classification":klass,"spec_sha256":base.sha256_file(base.SPEC),"repair_spec_sha256":base.sha256_file(REPAIR),"executor_revision":"R2_EXACT_CAMPAIGN_REPRODUCTION","cohort":{"path":str(base.COHORT.relative_to(ROOT)),"file_sha256":base.sha256_file(base.COHORT),"decision_times_sha256":cohort["cohort"]["decision_times_canonical_sha256"],"dependence_groups":48,"first_decision_time_ms":new_times[0],"last_decision_time_ms":new_times[-1],"cohort_resampled":False},"teacher":{"train_protocol_hash":R11_TRAIN_TEACHER_CONFIG.content_hash,"validation_protocol_hash":R11_VALIDATION_TEACHER_CONFIG.content_hash,"prior_train_evidence_hash":ca.train.evidence_hash,"reproduced_consumed_validation_evidence_hash":ca.validation.evidence_hash,"new_validation_evidence_hash":val.evidence_hash,"train_summary":tsa,"reproduced_consumed_validation_summary":vsa,"universe_b_non_authoritative_train_summary":tsb,"validation_summary":vsb,"train_target_audit":train_audit,"validation_target_audit":val_audit,"old_campaign_sensory_batch_geometry":[8,8,8,8,8,8,8,4],"new_validation_sensory_batch_geometry":[8,8,8,8,8,8],"consumed_validation_scored_in_generalization":False,"rebound_train_outputs_used_for_G2":False,"support_thresholds_relaxed":False,"outcome_as_label_used":False,"runtime_stats_a":str(sta),"runtime_stats_b":str(stb)},"models":{"champion_policy_hash":policy_hash_r11(ch1),"challenger_policy_hash":policy_hash_r11(x1),"challenger_checkpoint_sha256":base.sha256_file(Path(a.challenger_checkpoint).resolve()),"independent_champion_reload_outputs_exact":True,"independent_challenger_reload_outputs_exact":True,"training_or_tuning_on_generalization_cohort":False},"canonical_true_alignment":{"champion":{"loss":float(ce["loss"]),"direction_loss":float(ce["direction_loss"]),"sizing_loss":float(ce["sizing_loss"]),"behavior_fingerprint":ce["behavior_fingerprint"]},"challenger":{"loss":float(xe["loss"]),"direction_loss":float(xe["direction_loss"]),"sizing_loss":float(xe["sizing_loss"]),"behavior_fingerprint":xe["behavior_fingerprint"]}},"prior_train_climatology":{"source":"UNIVERSE_A_EXACT_FROZEN_CAMPAIGN_TRAIN_TARGETS","direction_probs":[float(x) for x in cp],"requested_risk":float(cr),"new_cohort_total_loss":float(clim.mean())},"G1":{**g1,"pass":bool(p1),"metric":"challenger_minus_champion_mean_total_loss"},"G2":{**g2,"pass":bool(p2),"metric":"challenger_minus_prior_train_climatology_mean_total_loss"},"G3":{"pass":bool(p3),"identity_delta_challenger_minus_champion":ident,"nonidentity_cyclic_shift_deltas":shifts,"best_nonidentity_shift_delta":float(min(shifts)),"worst_nonidentity_shift_delta":float(max(shifts)),"nonidentity_shifts_with_delta_le_identity":int(nle),"exact_one_sided_rank_p":float(ep),"permutation_family":"ALL_47_NONIDENTITY_CYCLIC_TARGET_ROW_SHIFTS","models_refit":False},"causality_canaries":{"original_campaign_universe":old_causal,"new_validation_universe":new_causal},"firewalls":{"final_holdout_touched":False,"fresh_market_data_downloaded":False,"canonical_promotion_authorized":False,"canonical_generation_advance_authorized":False,"new_market_information_verdict":False,"scientific_market_verdict":None},"next_gate":"SEPARATE_PROMOTION_OR_REPLICATION_DESIGN_PREREGISTRATION_REQUIRED" if status=="PASS" else "STOP__NO_POST_RESULT_RESCUE_UNDER_R0"}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
