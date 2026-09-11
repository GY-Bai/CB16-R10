from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

from cb16_local_opt.binance_archive_input_r10 import (
    BinanceUSDMArchiveSourceR10,
    KlineRecord,
    MINUTE_MS,
)
from cb16_local_opt.full_minute_historical_replay_r0 import FINAL_HOLDOUT_START_MS
from cb16_local_opt.longtraj_minpipe_r0 import prepare_lane_from_observed_r0, time_major_batch_scan_r0

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_MINPIPE_R0_SPEC_V1.json"
OUT = ROOT / "artifacts/r11_longtraj_minpipe_r0"


def bar(t: int, p: float, v: float = 1.0) -> KlineRecord:
    return KlineRecord(
        open_time=t, open=p, high=p, low=p, close=p, volume=v,
        close_time=t + MINUTE_MS - 1,
        quote_asset_volume=p * v,
        number_of_trades=1 if v else 0,
        taker_buy_base_asset_volume=v / 2,
        taker_buy_quote_asset_volume=p * v / 2,
    )


def synthetic_qualification() -> tuple[dict, dict]:
    t0 = 1_700_700_000_000
    lane_a = prepare_lane_from_observed_r0(
        "A",
        [bar(t0, 100.0), bar(t0 + 3 * MINUTE_MS, 103.0), bar(t0 + 4 * MINUTE_MS, 104.0), bar(t0 + 5 * MINUTE_MS, 105.0)],
        {"steps": 0, "score": 0.0},
    )
    lane_b = prepare_lane_from_observed_r0(
        "B",
        [bar(t0 + i * MINUTE_MS, 200.0 + i) for i in range(6)],
        {"steps": 0, "score": 0.0},
    )
    audit = []
    policy_batch_sizes = []

    def hook(stage, t, ids):
        audit.append((stage, int(t), tuple(ids)))

    def policy(obs):
        policy_batch_sizes.append(len(obs))
        for x in obs:
            if max(int(r[0]) for r in x["market_window"]) != int(x["decision_time_ms"]):
                raise RuntimeError("QUAL_FUTURE_MARKET_VISIBLE")
            forbidden = {"next_bar", "account_state_t1", "teacher", "target", "future"}
            if forbidden.intersection(x):
                raise RuntimeError("QUAL_FORBIDDEN_POLICY_KEY")
        return [{"x": 1.0 if x["lane_id"] == "A" else 2.0} for x in obs]

    def transition(account, action, current, nxt, lane_id):
        account = dict(account)
        account["steps"] += 1
        account["score"] += float(action["x"])
        return account

    events = list(
        time_major_batch_scan_r0(
            [lane_a, lane_b],
            lookback_minutes=1,
            policy_batch=policy,
            scalar_transition=transition,
            audit_hook=hook,
        )
    )
    clocks = sorted({e.decision_time_ms for e in events})
    ordering_ok = True
    for t in clocks:
        stages = [x[0] for x in audit if x[1] == t]
        if stages[:3] != ["OBSERVATIONS_READY", "POLICY_BATCH_BEGIN", "ACTIONS_FROZEN"]:
            ordering_ok = False
        if "TRANSITION_BEGIN" in stages and stages.index("ACTIONS_FROZEN") > stages.index("TRANSITION_BEGIN"):
            ordering_ok = False

    halts = [x for x in lane_a.minutes if x.halt_imputed]
    halt_ok = (
        len(halts) == 2
        and all(x.record.open == 100.0 and x.record.high == 100.0 and x.record.low == 100.0 and x.record.close == 100.0 for x in halts)
        and all(x.record.volume == 0.0 and x.record.number_of_trades == 0 for x in halts)
    )
    a_events = [e for e in events if e.lane_id == "A"]
    gates = {
        "HALT_PREFIX_ONLY_CANARY_PASS": halt_ok,
        "ACCOUNT_CONTINUITY_ACROSS_HALT_PASS": [e.account_state_t["steps"] for e in a_events] == list(range(len(a_events))),
        "NO_FUTURE_MARKET_IN_POLICY_PASS": all(e.observation_max_time_ms == e.decision_time_ms for e in events),
        "ALL_ACTIONS_FROZEN_BEFORE_ANY_T_PLUS_1_TRANSITION_PASS": ordering_ok,
        "SAME_CLOCK_MULTI_LANE_POLICY_BATCH_PASS": max(policy_batch_sizes, default=0) >= 2,
        "ONE_MINUTE_TRANSITION_CLOCK_PASS": all(e.transition_time_ms == e.decision_time_ms + MINUTE_MS for e in events),
    }
    return gates, {
        "event_count": len(events),
        "decision_clock_count": len(clocks),
        "max_policy_batch_size": max(policy_batch_sizes),
        "lane_a_halt_minutes": len(halts),
    }


def real_archive_minpipe_canary(raw_root: str, rows: int = 2048) -> dict:
    source = BinanceUSDMArchiveSourceR10(raw_root)
    layout = source.validate_layout()
    if "BTCUSDT" not in layout["symbols"]:
        raise RuntimeError("MINPIPE_REAL_BTC_MISSING")
    prefix = list(itertools.islice(
        source.iter_1m("BTCUSDT", verify_checksums=False, strict_chronology=False), rows
    ))
    if len(prefix) != rows:
        raise RuntimeError(f"MINPIPE_REAL_PREFIX_TOO_SHORT:{len(prefix)}!={rows}")
    if any(r.open_time >= FINAL_HOLDOUT_START_MS for r in prefix):
        raise RuntimeError("MINPIPE_REAL_FINAL_TOUCHED")

    # Two independent accounts replay the identical real BTC prefix.  This proves
    # that a real archive can drive same-clock batched policy while each account
    # remains a separate recurrent lane.
    lanes = [
        prepare_lane_from_observed_r0("BTC_REPLICA_0", prefix, {"steps": 0, "score": 0.0}),
        prepare_lane_from_observed_r0("BTC_REPLICA_1", prefix, {"steps": 0, "score": 0.0}),
    ]
    batch_sizes = []

    def policy(obs):
        batch_sizes.append(len(obs))
        for x in obs:
            if max(int(r[0]) for r in x["market_window"]) != int(x["decision_time_ms"]):
                raise RuntimeError("MINPIPE_REAL_FUTURE_MARKET_VISIBLE")
            if int(x["decision_time_ms"]) >= FINAL_HOLDOUT_START_MS:
                raise RuntimeError("MINPIPE_REAL_FINAL_OBSERVATION")
        return [{"x": 0.0} for _ in obs]

    def transition(account, action, current, nxt, lane_id):
        if int(nxt.record.open_time) >= FINAL_HOLDOUT_START_MS:
            raise RuntimeError("MINPIPE_REAL_FINAL_TRANSITION")
        account = dict(account)
        account["steps"] += 1
        return account

    events = list(time_major_batch_scan_r0(
        lanes,
        lookback_minutes=60,
        policy_batch=policy,
        scalar_transition=transition,
        final_holdout_start_ms=FINAL_HOLDOUT_START_MS,
    ))
    if not events:
        raise RuntimeError("MINPIPE_REAL_NO_EVENTS")
    if max(batch_sizes) != 2:
        raise RuntimeError(f"MINPIPE_REAL_BATCH_NOT_TWO:{max(batch_sizes)}")
    if any(e.observation_max_time_ms > e.decision_time_ms for e in events):
        raise RuntimeError("MINPIPE_REAL_CAUSALITY_FAIL")
    return {
        "symbol": "BTCUSDT",
        "raw_rows": len(prefix),
        "expanded_rows_per_lane": len(lanes[0].minutes),
        "halt_imputed_rows": sum(int(x.halt_imputed) for x in lanes[0].minutes),
        "events": len(events),
        "decision_clocks": len({e.decision_time_ms for e in events}),
        "max_policy_batch_size": max(batch_sizes),
        "first_observed_ms": prefix[0].open_time,
        "last_observed_ms": prefix[-1].open_time,
        "final_holdout_touched": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/cb16/raw")
    ap.add_argument("--require-raw-root", action="store_true")
    args = ap.parse_args()

    spec_bytes = SPEC.read_bytes()
    spec = json.loads(spec_bytes)
    gates, synth = synthetic_qualification()
    if not all(gates.values()):
        raise SystemExit("MINPIPE_QUALIFICATION_FAIL:" + json.dumps(gates, sort_keys=True))

    raw_path = Path(args.raw_root)
    if args.require_raw_root and not raw_path.is_dir():
        raise SystemExit("MINPIPE_RAW_ROOT_REQUIRED_BUT_MISSING")
    real = real_archive_minpipe_canary(args.raw_root) if raw_path.is_dir() else None
    if args.require_raw_root and real is None:
        raise SystemExit("MINPIPE_REAL_ARCHIVE_CANARY_REQUIRED")

    gates["REAL_ARCHIVE_MINPIPE_PASS"] = real is not None
    if args.require_raw_root and not gates["REAL_ARCHIVE_MINPIPE_PASS"]:
        raise SystemExit("MINPIPE_REAL_ARCHIVE_CANARY_FAIL")

    result = {
        "schema": "CB16_R11_LONGTRAJ_MINPIPE_R0_RESULT_V1",
        "classification": "INFRA_MINPIPE_QUALIFIED",
        "branch": os.environ.get("GITHUB_REF_NAME"),
        "head_sha": os.environ.get("GITHUB_SHA"),
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "base_sha": spec["base_sha"],
        "synthetic_canary": synth,
        "real_archive_canary": real,
        "gates": gates,
        "final_holdout_touched": False,
        "scientific_verdict": None,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    (OUT / "RESULT.json").write_bytes(raw + b"\n")
    (OUT / "RESULT.sha256").write_text(hashlib.sha256(raw + b"\n").hexdigest() + "  RESULT.json\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
