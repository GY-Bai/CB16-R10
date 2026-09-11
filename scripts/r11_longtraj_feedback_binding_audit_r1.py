from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from cb16_local_opt.probabilistic_teacher_r5 import (
    CounterfactualBranchSampleR5,
    CrossFitProbabilisticTeacherR5,
    CrossFitTeacherConfigR5,
)

ROOT = Path(__file__).resolve().parents[1]


def make_samples(*, mutate_future_after: int | None = None):
    rows=[]
    t0=1_700_000_000_000
    rng=np.random.default_rng(20260911)
    for i in range(80):
        t=t0+i*60_000
        # Prefix-only synthetic market/account coordinates.
        market=np.sin(i/9.0)
        account_equity=1.0+0.001*i
        account_drawdown=max(0.0,0.02*np.sin(i/17.0))
        context=(float(market),float(account_equity),float(account_drawdown))
        parent=f"minute-parent-{i:04d}"
        dep=f"same-future-minute-{t}"
        lineage=f"synthetic-prefix-lineage-{i:04d}"
        for direction,risk in [(-1,0.5),(0,0.0),(1,0.5)]:
            # Stochastic evidence, not a winner label. All branches for one parent
            # share the same future dependence group.
            utility=0.002*direction*market - 0.0003*abs(direction)*risk + float(rng.normal(0,0.0002))
            if mutate_future_after is not None and i>mutate_future_after:
                utility += 10.0*(direction+2) + i
            rows.append(CounterfactualBranchSampleR5(
                parent_id=parent,
                student_context_object_id=f"student-prefix-{i:04d}",
                timestamp=t,
                context_features=context,
                direction=direction,
                requested_risk=risk,
                realized_utility=float(utility),
                dependence_group_id=dep,
                market_lineage_hash=lineage,
            ))
    return rows


def teacher_minute_api_canary():
    cfg=CrossFitTeacherConfigR5(
        mode='PREQUENTIAL',
        k_neighbors=8,
        min_train_groups=8,
        min_effective_n=2.0,
        max_nearest_distance=100.0,
        distance_temperature=5.0,
        direction_softmax_temperature=0.01,
    )
    teacher=CrossFitProbabilisticTeacherR5(cfg)
    base=teacher.compile_all(make_samples())
    mutated=teacher.compile_all(make_samples(mutate_future_after=40))
    b={e.parent_id:e for e in base}; m={e.parent_id:e for e in mutated}
    # In PREQUENTIAL mode, mutating only later parents must not rewrite evidence
    # for an earlier target parent.
    early='minute-parent-0035'
    future_invariant=(b[early].content_hash==m[early].content_hash)
    # Same parent branches are one dependence group by construction.
    samples=make_samples()
    dep_ok=True
    by={}
    for s in samples: by.setdefault(s.parent_id,set()).add(s.dependence_group_id)
    dep_ok=all(len(v)==1 for v in by.values())
    context_ok=all(len({r.context_features for r in samples if r.parent_id==p})==1 for p in by)
    return {
        'teacher_mode':'PREQUENTIAL',
        'parents':80,
        'branches_per_parent':3,
        'evidence_rows':len(base),
        'future_parent_utility_mutation_does_not_change_earlier_evidence':future_invariant,
        'same_future_dependence_group_per_parent':dep_ok,
        'one_prefix_context_per_parent':context_ok,
        'target_is_realized_winner_label':False,
    }


def find_real_binding_candidates():
    hits=[]
    skip={'.git','.venv','__pycache__'}
    for p in ROOT.rglob('*.py'):
        if any(x in skip for x in p.parts):
            continue
        try: text=p.read_text(encoding='utf-8',errors='ignore')
        except Exception: continue
        if 'CounterfactualBranchSampleR5' not in text:
            continue
        signals={
            'counterfactual_sample':True,
            'simulator_snapshot':('SimulatorStateSnapshotV1' in text or 'snapshot_t' in text),
            'physics_step':('step_account(' in text or 'FrozenTradingKernel' in text),
            'minute_cadence':('60_000' in text or '1m' in text or 'minute' in text.lower()),
            'h72_or_72h':('72h' in text.lower() or 'h72' in text.lower()),
        }
        hits.append({'path':str(p.relative_to(ROOT)),**signals})
    qualified=[h for h in hits if h['simulator_snapshot'] and h['physics_step'] and h['minute_cadence'] and h['h72_or_72h']]
    return hits,qualified


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); args=ap.parse_args()
    api=teacher_minute_api_canary()
    hits,qualified=find_real_binding_candidates()
    api_pass=all([
        api['future_parent_utility_mutation_does_not_change_earlier_evidence'],
        api['same_future_dependence_group_per_parent'],
        api['one_prefix_context_per_parent'],
        not api['target_is_realized_winner_label'],
    ])
    if not api_pass:
        status='EXECUTION_BLOCKED__TEACHER_MINUTE_CAUSALITY_CANARY_FAIL'
    elif qualified:
        status='REAL_PER_MINUTE_BINDING_CANDIDATE_FOUND__REQUIRES_RUNTIME_QUALIFICATION'
    else:
        status='EXECUTION_BLOCKED__REAL_PER_MINUTE_COUNTERFACTUAL_PHYSICS_ADAPTER_MISSING'
    out={
        'schema':'CB16_R11_LONGTRAJ_E6_FEEDBACK_BINDING_AUDIT_R1_RESULT_V1',
        'status':status,
        'teacher_minute_api_canary':api,
        'source_candidates':hits,
        'qualified_source_candidates':qualified,
        'real_market_payload_opened':False,
        'final_holdout_payload_opened':False,
        'outcome_as_label_used':False,
        'long_gradient_training_authorized': status=='REAL_PER_MINUTE_BINDING_CANDIDATE_FOUND__REQUIRES_RUNTIME_QUALIFICATION',
        'note':'This audit never infers a real feedback bridge from Teacher API compatibility alone.',
    }
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps(out,indent=2,sort_keys=True))
    if not api_pass: raise SystemExit(2)

if __name__=='__main__': main()
