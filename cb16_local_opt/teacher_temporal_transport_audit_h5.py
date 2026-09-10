from __future__ import annotations

"""H5 Student-free temporal transport audit for the frozen R11 probabilistic Teacher."""

from dataclasses import replace
import hashlib
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical_state_alignment_falsification_r41 import R41_SCENARIOS, R41_SHIFTS
from .independent_purge_alignment_replication_r5 import compile_validation_targets_only_r5
from .probabilistic_teacher_r5 import weighted_quantile
from .r11_teacher_authority_candidate import R11_VALIDATION_TEACHER_CONFIG
from .teacher_balanced_runtime_r11 import prepare_support_regime_balanced_r11
from .teacher_vectorized_r11 import (
    _compile_block_r11,
    build_columnar_teacher_index_r11,
    group_targets_by_support_r11,
)

H5_RUNTIME="CB16_R11_H5_TEACHER_TEMPORAL_TRANSPORT_AUDIT_R0_V1"
H5_SHIFTS=tuple(int(x) for x in R41_SHIFTS)
H5_SCENARIOS=tuple(str(x) for x in R41_SCENARIOS)
H5_QUANTILES=tuple(float(x) for x in R11_VALIDATION_TEACHER_CONFIG.quantile_levels)


def require(cond:bool,code:str)->None:
    if not cond:
        raise RuntimeError(code)


def _bytes_multiset_hash(rows:np.ndarray)->str:
    x=np.ascontiguousarray(rows,dtype=np.float64)
    payload=sorted(bytes(np.ascontiguousarray(row).tobytes(order="C")) for row in x)
    h=hashlib.sha256(b"CB16_R11_H5_FEATURE_MULTISET_V1\0")
    for row in payload:
        h.update(row)
    return h.hexdigest()


def pinball_score_h5(y:float,quantiles:Sequence[float],levels:Sequence[float]=H5_QUANTILES)->float:
    q=np.asarray(tuple(quantiles),dtype=np.float64)
    tau=np.asarray(tuple(levels),dtype=np.float64)
    require(q.shape==tau.shape and q.ndim==1 and len(q)>0,"H5_QUANTILE_SHAPE")
    require(np.isfinite(q).all() and np.isfinite(float(y)),"H5_NONFINITE_PINBALL_INPUT")
    err=float(y)-q
    loss=np.maximum(tau*err,(tau-1.0)*err)
    value=float(np.mean(loss))
    require(np.isfinite(value) and value>=0.0,"H5_INVALID_PINBALL_SCORE")
    return value


def _fold_material_h5(*,fold_spec:Mapping[str,Any],all_train_parents:Mapping[str,Any],all_train_samples:Sequence[Any]):
    train_clock_set={int(x) for x in fold_spec["train_clocks"]}
    eval_clock_set={int(x) for x in fold_spec["eval_clocks"]}
    train_parents={pid:p for pid,p in all_train_parents.items() if int(p.decision_time_ms) in train_clock_set}
    eval_original={pid:p for pid,p in all_train_parents.items() if int(p.decision_time_ms) in eval_clock_set}
    eval_parents={pid:replace(p,split="VALIDATION") for pid,p in eval_original.items()}
    require(train_parents and eval_parents,f"H5_EMPTY_FOLD:{fold_spec['fold']}")
    train_ids=set(train_parents); eval_ids=set(eval_parents)
    train_samples=[x for x in all_train_samples if x.parent_id in train_ids]
    eval_samples=[x for x in all_train_samples if x.parent_id in eval_ids]
    require(len(train_samples)==len(train_parents)*9,f"H5_TRAIN_GRID_DRIFT:{fold_spec['fold']}")
    require(len(eval_samples)==len(eval_parents)*9,f"H5_EVAL_GRID_DRIFT:{fold_spec['fold']}")
    parents=dict(train_parents); parents.update(eval_parents)
    samples=list(train_samples)+list(eval_samples)
    return train_parents,eval_parents,parents,train_samples,eval_samples,samples


def _support_and_aligned_h5(*,parents:Mapping[str,Any],samples:Sequence[Any],eval_parent_ids:Sequence[str],block_targets:int=64):
    aligned,execution=compile_validation_targets_only_r5(
        samples=samples,parents=parents,target_parent_ids=eval_parent_ids,block_targets=block_targets
    )
    require(all(x.admission.admitted for x in aligned),"H5_ALIGNED_EVIDENCE_NOT_ADMITTED")
    index=build_columnar_teacher_index_r11(samples)
    eligible_train_groups={p.dependence_group_id for p in parents.values() if p.split=="TRAIN"}
    support_groups=group_targets_by_support_r11(
        target_parent_ids=tuple(eval_parent_ids),index=index,config=R11_VALIDATION_TEACHER_CONFIG,
        eligible_train_dependence_groups=eligible_train_groups,
    )
    require(len(support_groups)==1,f"H5_EXPECTED_ONE_OUTER_SUPPORT_REGIME:{len(support_groups)}")
    dep_rows,regime_targets=support_groups[0]
    require(set(regime_targets)==set(eval_parent_ids),"H5_SUPPORT_REGIME_TARGET_SET_DRIFT")
    regime=prepare_support_regime_balanced_r11(dep_rows=np.asarray(dep_rows,dtype=np.int32),index=index)
    manual=[]
    ordered=list(regime_targets)
    for start in range(0,len(ordered),int(block_targets)):
        manual.extend(_compile_block_r11(
            target_parent_ids=ordered[start:start+int(block_targets)],index=index,regime=regime,
            config=R11_VALIDATION_TEACHER_CONFIG,
        ))
    by_a={x.parent_id:x for x in aligned}; by_m={x.parent_id:x for x in manual}
    require(set(by_a)==set(by_m)==set(eval_parent_ids),"H5_ALIGNED_MANUAL_TARGET_SET_DRIFT")
    mismatched=[pid for pid in eval_parent_ids if by_a[pid].content_hash!=by_m[pid].content_hash]
    require(not mismatched,f"H5_ALIGNED_TEACHER_BYTE_SEMANTIC_MISMATCH:{len(mismatched)}")
    ordered_aligned=[by_a[pid] for pid in eval_parent_ids]
    return index,regime,ordered_aligned,execution


def _eval_group_scenario_rows_h5(*,index,eval_parents:Mapping[str,Any],eval_parent_ids:Sequence[str]):
    rows:dict[str,dict[str,int]]={}
    dep_ts:dict[str,int]={}
    for pid in eval_parent_ids:
        p=eval_parents[pid]
        gid=str(p.dependence_group_id); scenario=str(p.scenario)
        require(scenario in H5_SCENARIOS,f"H5_UNKNOWN_SCENARIO:{scenario}")
        rows.setdefault(gid,{})
        require(scenario not in rows[gid],f"H5_DUPLICATE_SCENARIO:{gid}:{scenario}")
        rows[gid][scenario]=int(index.parent_row_by_id[pid])
        dep_ts[gid]=int(p.decision_time_ms)
    expected=set(H5_SCENARIOS)
    for gid,m in rows.items():
        require(set(m)==expected,f"H5_SCENARIO_SET_DRIFT:{gid}")
    order=sorted(rows,key=lambda g:(dep_ts[g],g))
    return order,rows


def shuffled_target_feature_index_h5(*,index,eval_parents:Mapping[str,Any],eval_parent_ids:Sequence[str],shift:int):
    k0=int(shift)
    require(k0 in H5_SHIFTS,f"H5_UNREGISTERED_SHIFT:{k0}")
    order,rows=_eval_group_scenario_rows_h5(index=index,eval_parents=eval_parents,eval_parent_ids=eval_parent_ids)
    n=len(order); require(n>=2,"H5_TOO_FEW_EVAL_GROUPS_FOR_SHUFFLE")
    k=k0%n; require(k!=0,"H5_IDENTITY_SHUFFLE")
    eval_rows=np.asarray([index.parent_row_by_id[p] for p in eval_parent_ids],dtype=np.int32)
    before_hash=_bytes_multiset_hash(index.features[eval_rows])
    out=np.array(index.features,copy=True)
    mapping=[]
    for dst_pos,dst_gid in enumerate(order):
        src_gid=order[(dst_pos+k)%n]
        require(src_gid!=dst_gid,"H5_SHUFFLE_FIXED_POINT")
        for scenario in H5_SCENARIOS:
            out[rows[dst_gid][scenario]]=index.features[rows[src_gid][scenario]]
        mapping.append({"destination_group":dst_gid,"source_group":src_gid})
    after_hash=_bytes_multiset_hash(out[eval_rows])
    require(before_hash==after_hash,"H5_TARGET_FEATURE_MULTISET_DRIFT")
    train_rows=np.asarray([i for i,pid in enumerate(index.parent_ids) if pid not in set(eval_parent_ids)],dtype=np.int32)
    require(np.array_equal(out[train_rows],index.features[train_rows]),"H5_TRAIN_FEATURES_CHANGED_BY_SHUFFLE")
    return replace(index,features=np.ascontiguousarray(out)),{
        "shift":k0,"future_group_count":n,"scenario_identity_preserved":True,
        "target_feature_multiset_sha256_before":before_hash,"target_feature_multiset_sha256_after":after_hash,
        "target_feature_multiset_preserved":True,"train_features_byte_identical":True,"fixed_points":0,
        "mapping_hash":hashlib.sha256(repr(mapping).encode("utf-8")).hexdigest(),
    }


def compile_shuffled_evidence_h5(*,index,regime,eval_parents:Mapping[str,Any],eval_parent_ids:Sequence[str],shift:int,block_targets:int=64):
    shuffled_index,receipt=shuffled_target_feature_index_h5(
        index=index,eval_parents=eval_parents,eval_parent_ids=eval_parent_ids,shift=shift
    )
    out=[]
    for start in range(0,len(eval_parent_ids),int(block_targets)):
        out.extend(_compile_block_r11(
            target_parent_ids=eval_parent_ids[start:start+int(block_targets)],index=shuffled_index,
            regime=regime,config=R11_VALIDATION_TEACHER_CONFIG,
        ))
    require(len(out)==len(eval_parent_ids),"H5_SHUFFLED_TARGET_COUNT_DRIFT")
    require(all(x.admission.admitted for x in out),f"H5_SHUFFLED_EVIDENCE_NOT_ADMITTED:{shift}")
    return out,receipt


def climatology_laws_h5(*,index,regime)->dict[str,Any]:
    dep_rows=np.asarray(regime.dep_rows,dtype=np.int32)
    group_action=[]
    for dep_row in dep_rows:
        raw=np.asarray(index.dep_parent_rows[int(dep_row)],dtype=np.int32)
        raw=raw[raw>=0]
        by_context={}
        for row0 in raw:
            row=int(row0); key=str(index.student_context_ids[row])
            prior=by_context.get(key)
            if prior is None or index.parent_ids[row]<index.parent_ids[prior]:
                by_context[key]=row
        rows=np.asarray([by_context[k] for k in sorted(by_context)],dtype=np.int32)
        require(len(rows)>0,"H5_EMPTY_CLIMATOLOGY_GROUP")
        group_action.append(np.asarray(index.utilities[rows],dtype=np.float64).mean(axis=0))
    y=np.asarray(group_action,dtype=np.float64)
    require(y.ndim==2 and y.shape[1]==index.action_count,"H5_CLIMATOLOGY_SHAPE")
    w=np.full(len(y),1.0/len(y),dtype=np.float64)
    levels=np.asarray(H5_QUANTILES,dtype=np.float64)
    quant=np.stack([weighted_quantile(y[:,a],w,levels) for a in range(index.action_count)],axis=0)
    return {
        "quantiles":quant,"means":np.mean(y,axis=0),"stds":np.std(y,axis=0,ddof=0),
        "train_dependence_groups":int(len(y)),"action_count":int(index.action_count),
        "one_value_per_future_group":True,"state_features_used":False,
    }


def _aggregate_parent_rows_h5(parent_rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
    require(bool(parent_rows),"H5_EMPTY_PARENT_METRICS")
    by_group:dict[str,list[Mapping[str,Any]]]={}
    for row in parent_rows:
        by_group.setdefault(str(row["dependence_group_id"]),[]).append(row)
    group_rows=[]
    for gid,rows in by_group.items():
        require(len(rows)==6,f"H5_GROUP_PARENT_COUNT:{gid}:{len(rows)}")
        group_rows.append({
            "dependence_group_id":gid,
            "qscore":float(np.mean([float(x["qscore"]) for x in rows])),
            "mean_utility_mse":float(np.mean([float(x["mean_utility_mse"]) for x in rows])),
            "coverage_10_90":float(np.mean([float(x["coverage_10_90"]) for x in rows])),
            "coverage_05_95":float(np.mean([float(x["coverage_05_95"]) for x in rows])),
        })
    return {
        "future_dependence_groups":len(group_rows),
        "parent_contexts":len(parent_rows),
        "qscore":float(np.mean([x["qscore"] for x in group_rows])),
        "mean_utility_mse":float(np.mean([x["mean_utility_mse"] for x in group_rows])),
        "coverage_10_90":float(np.mean([x["coverage_10_90"] for x in group_rows])),
        "coverage_05_95":float(np.mean([x["coverage_05_95"] for x in group_rows])),
    }


def score_teacher_evidence_h5(*,evidence:Sequence[Any],index,eval_parents:Mapping[str,Any])->dict[str,Any]:
    rows=[]; nearest=[]; effective=[]
    for ev in evidence:
        require(ev.admission.admitted,f"H5_SCORE_UNADMITTED:{ev.parent_id}")
        row_idx=int(index.parent_row_by_id[ev.parent_id])
        utilities=np.asarray(index.utilities[row_idx],dtype=np.float64)
        require(len(ev.action_laws)==index.action_count,"H5_ACTION_LAW_COUNT_DRIFT")
        scores=[]; sq=[]; c1090=[]; c0595=[]
        for a,law in enumerate(ev.action_laws):
            require(int(law.direction)==int(index.action_directions[a]),"H5_ACTION_DIRECTION_ORDER_DRIFT")
            require(abs(float(law.requested_risk)-float(index.action_risks[a]))<=1e-12,"H5_ACTION_RISK_ORDER_DRIFT")
            y=float(utilities[a]); q=tuple(float(x) for x in law.quantiles)
            scores.append(pinball_score_h5(y,q,law.quantile_levels))
            sq.append((y-float(law.mean_utility))**2)
            qmap={round(float(t),8):float(v) for t,v in zip(law.quantile_levels,q)}
            c1090.append(float(qmap[0.1]<=y<=qmap[0.9]))
            c0595.append(float(qmap[0.05]<=y<=qmap[0.95]))
            nearest.append(float(law.nearest_distance)); effective.append(float(law.effective_dependence_n))
        p=eval_parents[ev.parent_id]
        rows.append({
            "parent_id":ev.parent_id,"dependence_group_id":str(p.dependence_group_id),
            "scenario":str(p.scenario),"qscore":float(np.mean(scores)),"mean_utility_mse":float(np.mean(sq)),
            "coverage_10_90":float(np.mean(c1090)),"coverage_05_95":float(np.mean(c0595)),
        })
    agg=_aggregate_parent_rows_h5(rows)
    agg.update({
        "median_nearest_distance":float(np.median(np.asarray(nearest,dtype=np.float64))),
        "median_effective_dependence_n":float(np.median(np.asarray(effective,dtype=np.float64))),
        "all_evidence_admitted":True,
    })
    return agg


def score_climatology_h5(*,climatology:Mapping[str,Any],index,eval_parent_ids:Sequence[str],eval_parents:Mapping[str,Any])->dict[str,Any]:
    quant=np.asarray(climatology["quantiles"],dtype=np.float64); means=np.asarray(climatology["means"],dtype=np.float64)
    require(quant.shape==(index.action_count,len(H5_QUANTILES)),"H5_CLIM_QUANTILE_SHAPE")
    rows=[]
    for pid in eval_parent_ids:
        row_idx=int(index.parent_row_by_id[pid]); util=np.asarray(index.utilities[row_idx],dtype=np.float64)
        scores=[]; sq=[]; c1090=[]; c0595=[]
        for a,y0 in enumerate(util):
            y=float(y0); q=quant[a]
            scores.append(pinball_score_h5(y,q,H5_QUANTILES)); sq.append((y-float(means[a]))**2)
            c1090.append(float(float(q[1])<=y<=float(q[5]))); c0595.append(float(float(q[0])<=y<=float(q[6])))
        p=eval_parents[pid]
        rows.append({"parent_id":pid,"dependence_group_id":str(p.dependence_group_id),"scenario":str(p.scenario),
                     "qscore":float(np.mean(scores)),"mean_utility_mse":float(np.mean(sq)),
                     "coverage_10_90":float(np.mean(c1090)),"coverage_05_95":float(np.mean(c0595))})
    agg=_aggregate_parent_rows_h5(rows)
    agg.update({"state_features_used":False,"train_dependence_groups":int(climatology["train_dependence_groups"])})
    return agg


def run_fold_h5(*,fold_spec:Mapping[str,Any],all_train_parents:Mapping[str,Any],all_train_samples:Sequence[Any],block_targets:int=64)->dict[str,Any]:
    fold=int(fold_spec["fold"])
    train_parents,eval_parents,parents,train_samples,eval_samples,samples=_fold_material_h5(
        fold_spec=fold_spec,all_train_parents=all_train_parents,all_train_samples=all_train_samples
    )
    eval_parent_ids=tuple(sorted(eval_parents))
    index,regime,aligned,execution=_support_and_aligned_h5(
        parents=parents,samples=samples,eval_parent_ids=eval_parent_ids,block_targets=block_targets
    )
    aligned_score=score_teacher_evidence_h5(evidence=aligned,index=index,eval_parents=eval_parents)
    climate_law=climatology_laws_h5(index=index,regime=regime)
    climate_score=score_climatology_h5(
        climatology=climate_law,index=index,eval_parent_ids=eval_parent_ids,eval_parents=eval_parents
    )
    shuffled={}; shuffle_receipts={}
    for shift in H5_SHIFTS:
        ev,receipt=compile_shuffled_evidence_h5(
            index=index,regime=regime,eval_parents=eval_parents,eval_parent_ids=eval_parent_ids,
            shift=shift,block_targets=block_targets,
        )
        shuffled[int(shift)]=score_teacher_evidence_h5(evidence=ev,index=index,eval_parents=eval_parents)
        shuffle_receipts[int(shift)]=receipt
    med=float(statistics.median(float(shuffled[s]["qscore"]) for s in H5_SHIFTS))
    return {
        "fold":fold,
        "train_parent_contexts":len(train_parents),"eval_parent_contexts":len(eval_parents),
        "train_dependence_groups":int(regime.group_count),"eval_dependence_groups":int(aligned_score["future_dependence_groups"]),
        "train_clock_count":int(fold_spec["train_clock_count"]),"eval_clock_count":int(fold_spec["eval_clock_count"]),
        "train_to_eval_gap_hours":float(fold_spec["train_to_eval_gap_hours"]),
        "teacher_protocol_hash":R11_VALIDATION_TEACHER_CONFIG.content_hash,
        "aligned":aligned_score,"climatology":climate_score,"shuffles":shuffled,
        "median_shuffle_qscore":med,
        "aligned_minus_climatology_qscore":float(aligned_score["qscore"]-climate_score["qscore"]),
        "aligned_minus_median_shuffle_qscore":float(aligned_score["qscore"]-med),
        "aligned_lt_climatology":bool(float(aligned_score["qscore"])<float(climate_score["qscore"])),
        "aligned_lt_median_shuffle":bool(float(aligned_score["qscore"])<med),
        "aligned_lt_each_shuffle_count":int(sum(float(aligned_score["qscore"])<float(shuffled[s]["qscore"]) for s in H5_SHIFTS)),
        "exact_r6_aligned_teacher_identity_guard":"PASS",
        "teacher_execution":execution,
        "shuffle_receipts":shuffle_receipts,
    }


def adjudicate_h5(fold_results:Sequence[Mapping[str,Any]])->dict[str,Any]:
    rows=sorted(fold_results,key=lambda x:int(x["fold"])); require(tuple(int(x["fold"]) for x in rows)==(1,2,3,4,5),"H5_FOLD_SET_DRIFT")
    clim=sum(bool(x["aligned_lt_climatology"]) for x in rows)
    shuf=sum(bool(x["aligned_lt_median_shuffle"]) for x in rows)
    pair=sum(int(x["aligned_lt_each_shuffle_count"]) for x in rows)
    late=all(bool(rows[f-1]["aligned_lt_climatology"]) and bool(rows[f-1]["aligned_lt_median_shuffle"]) for f in (4,5))
    passed=bool(clim>=4 and shuf>=4 and late and pair>=20)
    return {
        "schema":"CB16_R11_M_SERIES_H5_TEACHER_TEMPORAL_TRANSPORT_ADJUDICATION_R0_V1",
        "overall_pass":passed,"aligned_lt_climatology_fold_count":int(clim),
        "aligned_lt_median_shuffle_fold_count":int(shuf),"aligned_lt_each_shuffle_pair_count_of_25":int(pair),
        "folds_4_and_5_both_pass_climatology_and_shuffle":bool(late),
        "conclusion":(
            "H5_TEACHER_TEMPORAL_TRANSPORT_MECHANISTICALLY_SUPPORTED_ON_CONSUMED_TRAIN__H5_PRIMARY_BOTTLENECK_WEAKENED__OPEN_H2_STUDENT_UPDATE_INTERFERENCE_AUDIT"
            if passed else
            "H5_TEACHER_TEMPORAL_TRANSPORT_NOT_MECHANISTICALLY_SUPPORTED_ON_CONSUMED_TRAIN__PRIORITIZE_TEACHER_SUPPORT_CALIBRATION_TRANSPORT_DIAGNOSTICS__DO_NOT_OPEN_NEW_DIRECTION_LOSS"
        ),
        "market_information_verdict":False,"canonical_change_authorized":False,"promotion_authorized":False,
        "r7_evaluation_authorized":False,"final_opening_authorized":False,"another_direction_loss_authorized":False,
        "per_fold":[{
            "fold":int(x["fold"]),"aligned_qscore":float(x["aligned"]["qscore"]),
            "climatology_qscore":float(x["climatology"]["qscore"]),"median_shuffle_qscore":float(x["median_shuffle_qscore"]),
            "aligned_minus_climatology_qscore":float(x["aligned_minus_climatology_qscore"]),
            "aligned_minus_median_shuffle_qscore":float(x["aligned_minus_median_shuffle_qscore"]),
            "aligned_lt_climatology":bool(x["aligned_lt_climatology"]),"aligned_lt_median_shuffle":bool(x["aligned_lt_median_shuffle"]),
            "aligned_lt_each_shuffle_count":int(x["aligned_lt_each_shuffle_count"]),
        } for x in rows],
    }
