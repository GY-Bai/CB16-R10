from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from cb16_local_opt.binance_archive_input_r10 import BinanceUSDMArchiveSourceR10,MINUTE_MS
from cb16_local_opt.r102_common import HOUR_MS
ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'authority/rearchitecture_r11/CB16_R11_LONGTRAJ_GENERALIZATION_R0_SELECTION_SPEC_V1.json'

def req(x,c):
    if not x: raise RuntimeError(c)

def h(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def segments(src,symbol,final_ms):
    out=[]; start=prev=None; n=0
    for r in src.iter_1m(symbol,verify_checksums=False,strict_chronology=True):
        t=int(r.open_time)
        if t>=final_ms: break
        if prev is None or t-prev!=MINUTE_MS:
            if prev is not None: out.append((start,prev,n))
            start=t; n=1
        else: n+=1
        prev=t
    if prev is not None: out.append((start,prev,n))
    return out

def hh59(t):
    x=(t//HOUR_MS)*HOUR_MS+HOUR_MS-MINUTE_MS
    return x if x>=t else x+HOUR_MS

def choose(segs,spec):
    s=spec['selection']; b=spec['prior_consumption_boundary']
    need=int(s['dependence_groups']); spacing=int(s['parent_spacing_hours_minimum'])*HOUR_MS
    back=(int(s['sensory_history_minutes'])-1)*MINUTE_MS; fut=int(s['horizon_minutes'])*MINUTE_MS
    out=[]; used=[]; last=None
    for a,z,n in segs:
        lo=max(int(b['minimum_new_decision_time_ms']),a+back,last+spacing if last is not None else 0)
        t=hh59(lo); touched=False
        while t+fut<=z and len(out)<need:
            if last is None or t-last>=spacing: out.append(t); last=t; touched=True
            t+=spacing
        if touched: used.append({'start_ms':a,'end_ms':z,'rows':n})
        if len(out)==need: break
    req(len(out)==need,f'GENERALIZATION_SELECTION_INSUFFICIENT:{len(out)}<{need}')
    return out,used

def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw-root',required=True); p.add_argument('--output',required=True); p.add_argument('--symbol',default='BTCUSDT'); a=p.parse_args()
    spec=json.loads(SPEC.read_text()); req(spec['status']=='FROZEN_BEFORE_SELECTION_EXECUTION','SPEC_NOT_FROZEN'); req(a.symbol==spec['selection']['symbol'],'SYMBOL_DRIFT')
    src=BinanceUSDMArchiveSourceR10(a.raw_root); req(a.symbol in src.validate_layout()['symbols'],'SYMBOL_MISSING')
    final=int(spec['final_holdout']['start_ms']); segs=segments(src,a.symbol,final); req(bool(segs),'NO_PREFINAL_SEGMENTS')
    times,used=choose(segs,spec); s=spec['selection']; b=spec['prior_consumption_boundary']
    prior_end=int(b['prior_cohort_last_decision_time_ms'])+int(b['prior_horizon_minutes'])*MINUTE_MS
    first_hist=times[0]-(int(s['sensory_history_minutes'])-1)*MINUTE_MS
    ends=[t+int(s['horizon_minutes'])*MINUTE_MS for t in times]
    req(first_hist>prior_end,'PRIOR_CONSUMED_TAIL_OVERLAP'); req(all(x<final for x in ends),'FINAL_TOUCHED')
    result={'schema':'CB16_R11_LONGTRAJ_GENERALIZATION_R0_SELECTION_RESULT_V1','status':'PASS','classification':'GENERALIZATION_COHORT_SELECTED_MODEL_BLIND__NOT_YET_SCORED','spec_sha256':hashlib.sha256(SPEC.read_bytes()).hexdigest(),'symbol':a.symbol,'dependence_groups':len(times),'decision_times_ms':times,'decision_times_canonical_sha256':h(times),'first_decision_time_ms':times[0],'last_decision_time_ms':times[-1],'future_end_times_ms':ends,'future_end_times_canonical_sha256':h(ends),'used_segments':used,'all_prefinal_segment_lineage_sha256':h(segs),'prior_future_end_ms':prior_end,'first_new_sensory_start_ms':first_hist,'zero_overlap_with_prior_consumed_tail':True,'model_loaded':False,'model_scored':False,'teacher_targets_compiled':False,'final_holdout_touched':False,'fresh_market_data_downloaded':False,'scientific_verdict':None,'next_gate':'FREEZE_EXACT_GENERALIZATION_COHORT_THEN_PREREGISTER_CHAMPION_VS_CHALLENGER_EVALUATION_WITH_NEGATIVE_AND_SHUFFLE_CONTROLS'}
    q=Path(a.output); q.parent.mkdir(parents=True,exist_ok=True); q.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
