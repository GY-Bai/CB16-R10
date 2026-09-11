from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from cb16_local_opt.binance_archive_input_r10 import BinanceUSDMArchiveSourceR10, KlineRecord, MINUTE_MS
from cb16_local_opt.full_minute_long_trajectory_r1 import MinuteEnvelopeR1, expand_internal_halts_r1
from cb16_local_opt.longtraj_archive_r1 import iter_prefinal_observed_1m_r1
from cb16_local_opt.minute_sensory_adapter_r1 import iter_minute_sensory_frames_r1


def rec(t, p):
    return KlineRecord(
        open_time=t, open=p, high=p*1.001, low=p*0.999, close=p,
        volume=100.0, close_time=t+MINUTE_MS-1,
        quote_asset_volume=100.0*p, number_of_trades=10,
        taker_buy_base_asset_volume=50.0, taker_buy_quote_asset_volume=50.0*p,
    )


def synth(n=66*60):
    t0=1_700_000_000_000
    return [MinuteEnvelopeR1(rec(t0+i*MINUTE_MS, 100.0+0.01*i), False) for i in range(n)]


def frame_key(frame):
    return (
        frame.decision_time_ms,
        frame.micro_1m_60x5.tobytes(), frame.micro_stamps_60x5.tobytes(),
        frame.hourly_64x5.tobytes(), frame.hourly_stamps_64x5.tobytes(),
        frame.ordered4h30.tobytes(),
    )


def synthetic_causality():
    base=synth()
    frames=list(iter_minute_sensory_frames_r1('SYNTH', base))
    if len(frames)<30: raise RuntimeError('NOT_ENOUGH_SYNTH_FRAMES')
    target_frame,target_audit=frames[20]
    target_ts=target_frame.decision_time_ms

    # Mutate only strictly future minutes, including the unfinished suffix of the same hour.
    mutated=[]
    for x in base:
        if x.timestamp > target_ts:
            r=x.record
            mutated.append(MinuteEnvelopeR1(rec(r.open_time, r.close*7.0+123.0), False))
        else:
            mutated.append(x)
    mut_frames={f.decision_time_ms:(f,a) for f,a in iter_minute_sensory_frames_r1('SYNTH', mutated)}
    mf,ma=mut_frames[target_ts]
    future_invariant = frame_key(target_frame)==frame_key(mf) and target_audit==ma

    # Within one hour slow lanes must latch while Micro rolls.
    same_hour=[(f,a) for f,a in frames if a.current_hour_start_ms==target_audit.current_hour_start_ms]
    if len(same_hour)<2: raise RuntimeError('NO_SAME_HOUR_PAIR')
    f0,a0=same_hour[0]; f1,a1=same_hour[-1]
    slow_latched = (
        np.array_equal(f0.hourly_64x5,f1.hourly_64x5)
        and np.array_equal(f0.hourly_stamps_64x5,f1.hourly_stamps_64x5)
        and np.array_equal(f0.ordered4h30,f1.ordered4h30)
    )
    micro_rolls = not np.array_equal(f0.micro_1m_60x5,f1.micro_1m_60x5)
    endpoint_rules = all(
        a.micro_latest_time_ms==a.decision_time_ms
        and a.latest_completed_hour_start_ms < a.current_hour_start_ms
        for _,a in frames
    )
    return {
        'frames':len(frames),
        'target_time_ms':target_ts,
        'future_suffix_mutation_invariant':future_invariant,
        'slow_lanes_latched_within_hour':slow_latched,
        'micro_rolls_within_hour':micro_rolls,
        'all_frame_endpoint_rules':endpoint_rules,
    }


def real_btc_canary(raw_root, max_minutes=6000):
    src=BinanceUSDMArchiveSourceR10(raw_root)
    observed=iter_prefinal_observed_1m_r1(src,'BTCUSDT')
    expanded=expand_internal_halts_r1(observed)
    bounded=itertools.islice(expanded,max_minutes)
    n=0; first=None; last=None; slow_hashes={}; within_hour_latch=True
    for f,a in iter_minute_sensory_frames_r1('BTCUSDT',bounded):
        n+=1; first=f.decision_time_ms if first is None else first; last=f.decision_time_ms
        h=a.current_hour_start_ms
        sh=(f.hourly_64x5.tobytes(),f.ordered4h30.tobytes())
        if h in slow_hashes and slow_hashes[h]!=sh: within_hour_latch=False
        slow_hashes.setdefault(h,sh)
        if not (a.micro_latest_time_ms==a.decision_time_ms and a.latest_completed_hour_start_ms<a.current_hour_start_ms):
            raise RuntimeError('REAL_BTC_CAUSAL_ENDPOINT_FAIL')
    if n==0: raise RuntimeError('NO_REAL_BTC_MINUTE_FRAMES')
    return {
        'input_minutes_cap':max_minutes,
        'frames':n,
        'first_frame_ms':first,
        'last_frame_ms':last,
        'slow_lanes_latched_within_hour':within_hour_latch,
        'final_holdout_payload_opened':False,
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-root',default='/cb16/raw'); ap.add_argument('--output',required=True); args=ap.parse_args()
    syn=synthetic_causality(); real=real_btc_canary(args.raw_root)
    gates={
        'future_suffix_mutation_invariant':syn['future_suffix_mutation_invariant'],
        'slow_lanes_latched_within_hour':syn['slow_lanes_latched_within_hour'] and real['slow_lanes_latched_within_hour'],
        'micro_rolls_within_hour':syn['micro_rolls_within_hour'],
        'all_frame_endpoint_rules':syn['all_frame_endpoint_rules'],
    }
    status='PASS' if all(gates.values()) else 'FAIL'
    out={
        'schema':'CB16_R11_LONGTRAJ_E5_MINUTE_SENSORY_CAUSALITY_R1_RESULT_V1',
        'status':status,
        'semantic':'Micro60 updates every minute; Macro32/Medium64 contain completed hours only and latch inside the current hour',
        'synthetic':syn,'real_btc_bounded':real,'gates':gates,
        'future_market_in_frame':False,'halt_flag_student_feature':False,
        'market_learning_claim':'NONE__SENSORY_CAUSALITY_ONLY',
    }
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps(out,indent=2,sort_keys=True))
    if status!='PASS': raise SystemExit(2)

if __name__=='__main__': main()
