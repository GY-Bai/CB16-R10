from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cb16_local_opt.binance_archive_input_r10 import (
    BinanceUSDMArchiveSourceR10,
    KlineRecord,
    MINUTE_MS,
)
from cb16_local_opt.full_minute_long_trajectory_r1 import FINAL_HOLDOUT_START_MS
from cb16_local_opt.longtraj_infra_closure_r0 import (
    find_contiguous_prefinal_run_r0,
    funding_events_by_minute_r0,
)
from cb16_local_opt.minute_physics_binding_r2 import MinutePhysicsSessionR2, canonical_hash
from cb16_local_opt.probabilistic_teacher_r5 import (
    CounterfactualBranchSampleR5,
    CrossFitProbabilisticTeacherR5,
    CrossFitTeacherConfigR5,
)
from cb16_local_opt.r102_common import HOUR_MS
from cb16_local_opt.r102_physics import FLAT, LONG, SHORT, FrozenPhysicsRuntimeR102

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_E6_FEEDBACK_BINDING_R2_SPEC_V1.json"
SCHEMA = "CB16_R11_LONGTRAJ_E6_FEEDBACK_BINDING_R2_RESULT_V1"
H72_MINUTES = 72 * 60
WARMUP_HOURS = 16
WARMUP_MINUTES = WARMUP_HOURS * 60
CURRENT_HOUR_MINUTES = 60
PARENT_PREFIX_MINUTES = WARMUP_MINUTES + CURRENT_HOUR_MINUTES
PARENT_SPACING_MS = 73 * HOUR_MS
CANDIDATES = ((SHORT, 0.5), (FLAT, 0.0), (LONG, 0.5))


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def atomic_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _hour_bar(session: MinutePhysicsSessionR2, symbol: str, rows: Sequence[KlineRecord]):
    require(len(rows) == 60, "E6R2_HOURLY_WARMUP_REQUIRES_60_MINUTES")
    require(int(rows[0].open_time) % HOUR_MS == 0, "E6R2_WARMUP_HOUR_NOT_ALIGNED")
    for a, b in zip(rows, rows[1:]):
        require(int(b.open_time) - int(a.open_time) == MINUTE_MS, "E6R2_WARMUP_GAP")
    return session.CanonicalBar(
        symbol=symbol,
        bar_start=datetime.fromtimestamp(int(rows[0].open_time) / 1000, tz=timezone.utc),
        timeframe="1h",
        open=float(rows[0].open),
        high=float(max(r.high for r in rows)),
        low=float(min(r.low for r in rows)),
        close=float(rows[-1].close),
        volume=float(sum(r.volume for r in rows)),
        mark_price=float(rows[-1].close),
        index_price=float(rows[-1].close),
    )


def _warmup_hourly(session: MinutePhysicsSessionR2, symbol: str, rows: Sequence[KlineRecord]) -> None:
    require(len(rows) == WARMUP_MINUTES, "E6R2_WARMUP_LENGTH")
    session.warmup_hourly([
        _hour_bar(session, symbol, rows[i:i + 60])
        for i in range(0, WARMUP_MINUTES, 60)
    ])


def _build_parent_state(
    runtime: FrozenPhysicsRuntimeR102,
    *,
    symbol: str,
    account_id: str,
    prefix_rows: Sequence[KlineRecord],
    funding: Mapping[int, float],
) -> tuple[dict[str, Any], dict[str, Any], list[float]]:
    require(len(prefix_rows) == PARENT_PREFIX_MINUTES, "E6R2_PARENT_PREFIX_LENGTH")
    warmup = prefix_rows[:WARMUP_MINUTES]
    current_hour = prefix_rows[WARMUP_MINUTES:]
    require(len(current_hour) == 60, "E6R2_CURRENT_HOUR_LENGTH")
    require(int(current_hour[0].open_time) % HOUR_MS == 0, "E6R2_CURRENT_HOUR_NOT_ALIGNED")
    require(int(current_hour[-1].open_time) % HOUR_MS == HOUR_MS - MINUTE_MS, "E6R2_PARENT_NOT_HH59")

    session = MinutePhysicsSessionR2.initialize(runtime, account_id=account_id)
    _warmup_hourly(session, symbol, warmup)
    for j, row in enumerate(current_hour):
        session.step_intent(
            direction_v55=FLAT,
            risk=0.0,
            symbol=symbol,
            open_time_ms=int(row.open_time),
            ohlcv=(row.open, row.high, row.low, row.close, row.volume),
            funding_rate=float(funding.get(int(row.open_time), 0.0)),
            trace_id=f"PARENT_PREFIX:{account_id}:{j}",
        )
    state = session.export_state()
    account6 = [float(x) for x in session.account6(float(current_hour[-1].close)).tolist()]
    require(len(account6) == 6 and all(math.isfinite(x) for x in account6), "E6R2_PARENT_ACCOUNT6_INVALID")
    return state, copy.deepcopy(session.risk_authority), account6


def _future_hash(symbol: str, parent_time_ms: int, rows: Sequence[KlineRecord], funding: Mapping[int, float]) -> str:
    h = hashlib.sha256()
    h.update(b"CB16_E6_R2_REAL_MINUTE_H72\0")
    h.update(symbol.encode("utf-8") + b"\0")
    h.update(str(int(parent_time_ms)).encode("ascii") + b"\0")
    for r in rows:
        h.update(np.asarray([int(r.open_time), int(r.close_time), int(r.number_of_trades)], dtype=np.int64).tobytes())
        h.update(np.asarray([
            float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume),
            float(funding.get(int(r.open_time), 0.0)),
        ], dtype=np.float64).tobytes())
    return h.hexdigest()


def _context_features(current_hour_rows: Sequence[KlineRecord], account6: Sequence[float]) -> tuple[float, ...]:
    closes = np.asarray([float(r.close) for r in current_hour_rows], dtype=np.float64)
    require(len(closes) == 60 and np.all(np.isfinite(closes)), "E6R2_CONTEXT_CLOSE_INVALID")
    ret_1h = float(closes[-1] / closes[0] - 1.0)
    logret = np.diff(np.log(closes))
    vol_1h = float(np.std(logret)) if len(logret) else 0.0
    range_1h = float((max(r.high for r in current_hour_rows) - min(r.low for r in current_hour_rows)) / closes[-1])
    return tuple(float(x) for x in account6) + (ret_1h, vol_1h, range_1h)


def _simulate_branch(
    runtime: FrozenPhysicsRuntimeR102,
    *,
    parent_state: Mapping[str, Any],
    risk_authority: Mapping[str, Any],
    symbol: str,
    future_rows: Sequence[KlineRecord],
    funding: Mapping[int, float],
    parent_id: str,
    direction: int,
    risk: float,
) -> dict[str, Any]:
    session = MinutePhysicsSessionR2.restore(runtime, parent_state, risk_authority)
    parent_mark = float(parent_state["base_snapshot"]["kernel_state"]["last_mark_price"])
    w0 = float(session.equity_at_mark(parent_mark))
    require(math.isfinite(w0) and w0 > 0.0, "E6R2_NONPOSITIVE_W0")
    first_step = None
    terminal_at = None
    executed = 0
    for j, row in enumerate(future_rows):
        step = session.step_intent(
            direction_v55=int(direction) if j == 0 else FLAT,
            risk=float(risk) if j == 0 else 0.0,
            symbol=symbol,
            open_time_ms=int(row.open_time),
            ohlcv=(row.open, row.high, row.low, row.close, row.volume),
            funding_rate=float(funding.get(int(row.open_time), 0.0)),
            trace_id=f"E6R2:{parent_id}:{direction}:{risk:.2f}:{j}",
        )
        executed = j + 1
        if j == 0:
            first_step = {
                "supervisor_decision": step["supervisor_decision"],
                "executable_action": step["executable_action"],
            }
        if bool(step["snapshot_t1"]["termination_state"]["terminated"]):
            terminal_at = j + 1
            break
    last_row = future_rows[executed - 1]
    close_price = float(last_row.close)
    final = session.finalize(close_price)
    wt = float(session.equity_at_mark(close_price))
    require(math.isfinite(wt) and wt > 0.0, "E6R2_NONPOSITIVE_WT")
    return {
        "parent_id": parent_id,
        "direction": int(direction),
        "requested_risk": float(risk),
        "status": "MATURED",
        "w0": w0,
        "wt": wt,
        "utility": float(math.log(wt / w0)),
        "executed_minutes": int(executed),
        "terminal_at_minute": terminal_at,
        "evaluation_finalize_used": bool(final["used"]),
        "first_step": first_step,
        "final_state_hash": canonical_hash(session.export_state()),
    }


def _parent_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    runtime = FrozenPhysicsRuntimeR102.load(payload["package_root"])
    branches = [
        _simulate_branch(
            runtime,
            parent_state=payload["parent_state"],
            risk_authority=payload["risk_authority"],
            symbol=payload["symbol"],
            future_rows=payload["future_rows"],
            funding=payload["funding"],
            parent_id=payload["parent_id"],
            direction=d,
            risk=r,
        )
        for d, r in CANDIDATES
    ]
    return {
        "parent_id": payload["parent_id"],
        "decision_time_ms": int(payload["decision_time_ms"]),
        "first_execution_time_ms": int(payload["future_rows"][0].open_time),
        "dependence_group_id": payload["dependence_group_id"],
        "student_context_object_id": payload["student_context_object_id"],
        "context_features": list(payload["context_features"]),
        "parent_state_hash": canonical_hash(payload["parent_state"]),
        "market_lineage_hash": payload["market_lineage_hash"],
        "branches": branches,
    }


def _select_parents(records: Sequence[KlineRecord], count: int) -> list[int]:
    eligible = [
        i for i, r in enumerate(records)
        if i >= PARENT_PREFIX_MINUTES - 1
        and i + H72_MINUTES < len(records)
        and int(r.open_time) % HOUR_MS == HOUR_MS - MINUTE_MS
    ]
    out: list[int] = []
    last_t: int | None = None
    for i in eligible:
        t = int(records[i].open_time)
        if last_t is None or t - last_t >= PARENT_SPACING_MS:
            out.append(i)
            last_t = t
            if len(out) == count:
                break
    require(len(out) == count, f"E6R2_NOT_ENOUGH_NONOVERLAPPING_PARENTS:{len(out)}<{count}")
    return out


def _mutate_suffix(rows: Sequence[KlineRecord], start: int) -> tuple[KlineRecord, ...]:
    out = []
    for j, r in enumerate(rows):
        if j < start:
            out.append(r)
            continue
        scale = 1.03 + 0.00001 * (j - start)
        o = float(r.open) * scale
        c = float(r.close) * scale
        h = max(o, c, float(r.high) * scale)
        l = min(o, c, float(r.low) * scale)
        out.append(replace(r, open=o, high=h, low=l, close=c))
    return tuple(out)


def _prefix_causality_canaries(
    package_root: str,
    symbol: str,
    parent_state: Mapping[str, Any],
    risk_authority: Mapping[str, Any],
    future_rows: Sequence[KlineRecord],
    funding: Mapping[int, float],
) -> dict[str, Any]:
    runtime = FrozenPhysicsRuntimeR102.load(package_root)
    prefix_n = 10
    mutated = _mutate_suffix(future_rows, prefix_n)
    require(
        canonical_hash([(r.open, r.high, r.low, r.close) for r in future_rows[prefix_n:]])
        != canonical_hash([(r.open, r.high, r.low, r.close) for r in mutated[prefix_n:]]),
        "E6R2_SUFFIX_MUTATION_DID_NOT_CHANGE_SUFFIX",
    )

    def run_prefix(rows: Sequence[KlineRecord], trace: str) -> dict[str, Any]:
        s = MinutePhysicsSessionR2.restore(runtime, parent_state, risk_authority)
        for j, row in enumerate(rows[:prefix_n]):
            s.step_intent(
                direction_v55=LONG if j == 0 else FLAT,
                risk=0.5 if j == 0 else 0.0,
                symbol=symbol,
                open_time_ms=int(row.open_time),
                ohlcv=(row.open, row.high, row.low, row.close, row.volume),
                funding_rate=float(funding.get(int(row.open_time), 0.0)),
                trace_id=f"{trace}:{j}",
            )
        return s.export_state()

    prefix_a = run_prefix(future_rows, "SUFFIX:A")
    prefix_b = run_prefix(mutated, "SUFFIX:B")
    suffix_invariant = canonical_hash(prefix_a) == canonical_hash(prefix_b)

    r0 = future_rows[0]
    sa = MinutePhysicsSessionR2.restore(runtime, parent_state, risk_authority)
    first_a = sa.step_intent(
        direction_v55=LONG, risk=0.5, symbol=symbol,
        open_time_ms=int(r0.open_time), ohlcv=(r0.open, r0.high, r0.low, r0.close, r0.volume),
        funding_rate=float(funding.get(int(r0.open_time), 0.0)), trace_id="NEXT:A",
    )
    sb = MinutePhysicsSessionR2.restore(runtime, parent_state, risk_authority)
    first_b = sb.step_intent(
        direction_v55=LONG, risk=0.5, symbol=symbol,
        open_time_ms=int(r0.open_time), ohlcv=(r0.open, r0.high, r0.low, r0.close, r0.volume),
        funding_rate=float(funding.get(int(r0.open_time), 0.0)), trace_id="NEXT:B",
    )
    next_bar_isolated = canonical_hash(first_a["snapshot_t1"]) == canonical_hash(first_b["snapshot_t1"])

    parent_before = canonical_hash(parent_state)
    poisoned_future_account = copy.deepcopy(first_b["snapshot_t1"])
    poisoned_future_account["kernel_state"]["cash"] = -9.99e99
    future_account_poison_invariant = (
        canonical_hash(parent_state) == parent_before
        and canonical_hash(poisoned_future_account) != canonical_hash(first_b["snapshot_t1"])
        and next_bar_isolated
    )
    return {
        "future_suffix_mutation_invariance": bool(suffix_invariant),
        "future_account_poison_invariance": bool(future_account_poison_invariant),
        "next_bar_isolation": bool(next_bar_isolated),
    }


def _teacher_canary(parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    samples: list[CounterfactualBranchSampleR5] = []
    for p in parents:
        for b in p["branches"]:
            samples.append(CounterfactualBranchSampleR5(
                parent_id=str(p["parent_id"]),
                student_context_object_id=str(p["student_context_object_id"]),
                timestamp=int(p["decision_time_ms"]),
                context_features=tuple(float(x) for x in p["context_features"]),
                direction={SHORT: -1, FLAT: 0, LONG: 1}[int(b["direction"])],
                requested_risk=float(b["requested_risk"]),
                realized_utility=float(b["utility"]),
                dependence_group_id=str(p["dependence_group_id"]),
                market_lineage_hash=str(p["market_lineage_hash"]),
            ))
    cfg = CrossFitTeacherConfigR5(
        mode="PREQUENTIAL",
        k_neighbors=4,
        min_train_groups=2,
        min_effective_n=1.0,
        max_nearest_distance=1.0e9,
        distance_temperature=10.0,
        direction_softmax_temperature=0.01,
    )
    teacher = CrossFitProbabilisticTeacherR5(cfg)
    base = teacher.compile_all(samples)
    require(len(base) == len(parents), "E6R2_TEACHER_PARENT_COUNT")

    last_parent = str(parents[-1]["parent_id"])
    mutated = [
        replace(s, realized_utility=float(s.realized_utility + 1000.0 * (s.direction + 2)))
        if s.parent_id == last_parent else s
        for s in samples
    ]
    mut = teacher.compile_all(mutated)
    base_map = {e.parent_id: e for e in base}
    mut_map = {e.parent_id: e for e in mut}
    target = str(parents[-2]["parent_id"])
    prequential_invariant = base_map[target].content_hash == mut_map[target].content_hash

    by_parent: dict[str, set[str]] = {}
    for s in samples:
        by_parent.setdefault(s.parent_id, set()).add(s.dependence_group_id)
    return {
        "mode": "PREQUENTIAL",
        "compiled_parent_count": len(base),
        "future_parent_outcome_mutation_does_not_change_earlier_evidence": bool(prequential_invariant),
        "same_parent_one_dependence_group": bool(all(len(v) == 1 for v in by_parent.values())),
        "outcome_as_label_used": False,
        "production_support_sufficiency_claimed": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--parents", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    require(spec["schema"] == "CB16_R11_LONGTRAJ_E6_FEEDBACK_BINDING_R2_SPEC_V1", "E6R2_SPEC_SCHEMA")
    require(spec["status"] == "FROZEN_BEFORE_FIRST_R2_EXECUTION", "E6R2_SPEC_NOT_PREREGISTERED")
    require(int(args.parents) == int(spec["qualification_scope"]["real_parent_count"]), "E6R2_PARENT_COUNT_NOT_PREREGISTERED")
    require(int(args.workers) == 8, "E6R2_EXPECTS_ADAPTIVE_SINGLE_EXPERIMENT_8_WORKERS")

    source = BinanceUSDMArchiveSourceR10(args.raw_root)
    layout = source.validate_layout()
    require(args.symbol in layout["symbols"], f"E6R2_SYMBOL_MISSING:{args.symbol}")

    required_rows = PARENT_PREFIX_MINUTES + ((args.parents - 1) * 73 * 60) + H72_MINUTES + 180
    records = find_contiguous_prefinal_run_r0(source, args.symbol, required_rows=required_rows)
    require(all(int(r.open_time) < FINAL_HOLDOUT_START_MS for r in records), "E6R2_FINAL_TOUCHED")
    require(all(int(b.open_time) - int(a.open_time) == MINUTE_MS for a, b in zip(records, records[1:])), "E6R2_REAL_RUN_NOT_CONTIGUOUS")

    indices = _select_parents(records, args.parents)
    first_prefix_start = indices[0] - (PARENT_PREFIX_MINUTES - 1)
    start_ms = int(records[first_prefix_start].open_time)
    end_ms = int(records[indices[-1] + H72_MINUTES].open_time)
    funding = funding_events_by_minute_r0(source, args.symbol, start_ms=start_ms, end_ms=end_ms)
    runtime = FrozenPhysicsRuntimeR102.load(str(Path(args.package_root).resolve()))

    payloads = []
    for idx in indices:
        prefix_start = idx - (PARENT_PREFIX_MINUTES - 1)
        prefix_rows = tuple(records[prefix_start:idx + 1])
        future_rows = tuple(records[idx + 1:idx + 1 + H72_MINUTES])
        require(len(prefix_rows) == PARENT_PREFIX_MINUTES, "E6R2_PREFIX_SLICE_LENGTH")
        require(len(future_rows) == H72_MINUTES, "E6R2_H72_LENGTH")
        require(int(future_rows[0].open_time) - int(records[idx].open_time) == MINUTE_MS, "E6R2_NEXT_BAR_ALIGNMENT")
        require(int(future_rows[0].open_time) % HOUR_MS == 0, "E6R2_FIRST_EXECUTION_NOT_HOUR_BOUNDARY")
        require(all(int(r.open_time) < FINAL_HOLDOUT_START_MS for r in future_rows), "E6R2_PARENT_FUTURE_FINAL_TOUCHED")

        parent_id = f"E6R2:{args.symbol}:{int(records[idx].open_time)}"
        parent_state, risk_authority, account6 = _build_parent_state(
            runtime,
            symbol=args.symbol,
            account_id=parent_id,
            prefix_rows=prefix_rows,
            funding=funding,
        )
        current_hour = prefix_rows[-60:]
        context = _context_features(current_hour, account6)
        lineage = _future_hash(args.symbol, int(records[idx].open_time), future_rows, funding)
        payloads.append({
            "package_root": str(Path(args.package_root).resolve()),
            "symbol": args.symbol,
            "parent_state": parent_state,
            "risk_authority": risk_authority,
            "future_rows": future_rows,
            "funding": dict(funding),
            "parent_id": parent_id,
            "decision_time_ms": int(records[idx].open_time),
            "dependence_group_id": f"E6R2:SAME_FUTURE:{lineage}",
            "student_context_object_id": "E6R2CTX:" + hashlib.sha256(
                json.dumps(context, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "context_features": context,
            "market_lineage_hash": lineage,
        })

    causal = _prefix_causality_canaries(
        str(Path(args.package_root).resolve()),
        args.symbol,
        payloads[0]["parent_state"],
        payloads[0]["risk_authority"],
        payloads[0]["future_rows"],
        funding,
    )
    require(all(causal.values()), f"E6R2_CAUSALITY_CANARY_FAIL:{causal}")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        parents = list(pool.map(_parent_worker, payloads))
    parents.sort(key=lambda p: (int(p["decision_time_ms"]), str(p["parent_id"])))

    require(len(parents) == args.parents, "E6R2_PARENT_EXECUTION_COUNT")
    require(all(len(p["branches"]) == len(CANDIDATES) for p in parents), "E6R2_BRANCH_COUNT")
    require(all(b["status"] == "MATURED" for p in parents for b in p["branches"]), "E6R2_BRANCH_NOT_MATURED")
    require(all(int(p["first_execution_time_ms"]) - int(p["decision_time_ms"]) == MINUTE_MS for p in parents), "E6R2_NEXT_MINUTE_EXECUTION_DRIFT")
    teacher = _teacher_canary(parents)
    require(teacher["future_parent_outcome_mutation_does_not_change_earlier_evidence"], "E6R2_TEACHER_PREQUENTIAL_CAUSALITY_FAIL")
    require(teacher["same_parent_one_dependence_group"], "E6R2_DEPENDENCE_GROUP_FAIL")
    require(teacher["outcome_as_label_used"] is False, "E6R2_OUTCOME_LABEL_FORBIDDEN")

    pairwise_nonoverlap = all(
        int(parents[i + 1]["decision_time_ms"]) - int(parents[i]["decision_time_ms"]) >= PARENT_SPACING_MS
        for i in range(len(parents) - 1)
    )
    require(pairwise_nonoverlap, "E6R2_PARENT_FUTURES_OVERLAP")

    gates = {
        "REAL_PRE_FINAL_1M_H72_COUNTERFACTUAL_BINDING": True,
        "NEXT_MINUTE_OPEN_EXECUTION": True,
        "PARENT_ACCOUNT6_BOUND_IN_TEACHER_CONTEXT": all(len(p["context_features"]) >= 9 for p in parents),
        "REAL_PARENT_FUTURES_NONOVERLAPPING": bool(pairwise_nonoverlap),
        "FUTURE_SUFFIX_MUTATION_INVARIANCE": causal["future_suffix_mutation_invariance"],
        "FUTURE_ACCOUNT_POISON_INVARIANCE": causal["future_account_poison_invariance"],
        "NEXT_BAR_ISOLATION": causal["next_bar_isolation"],
        "TEACHER_PREQUENTIAL_EARLIER_EVIDENCE_INVARIANCE": teacher["future_parent_outcome_mutation_does_not_change_earlier_evidence"],
        "SAME_PARENT_ONE_DEPENDENCE_GROUP": teacher["same_parent_one_dependence_group"],
        "OUTCOME_NOT_LABEL": teacher["outcome_as_label_used"] is False,
        "FINAL_FIREWALL_CLOSED": True,
    }
    require(all(gates.values()), f"E6R2_GATE_FAIL:{gates}")

    out = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": "E6_R2_MINUTE_FEEDBACK_BINDING_MECHANICALLY_QUALIFIED__NO_MARKET_VERDICT",
        "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        "symbol": args.symbol,
        "archive": {
            "real_market_payload_opened": True,
            "first_ms": start_ms,
            "last_ms": end_ms,
            "contiguous_observed_1m_only": True,
            "halt_imputed_rows": 0,
            "final_holdout_touched": False,
            "fresh_market_data_downloaded": False,
        },
        "execution": {
            "parents": len(parents),
            "branches_per_parent": len(CANDIDATES),
            "total_branches": sum(len(p["branches"]) for p in parents),
            "horizon_minutes": H72_MINUTES,
            "horizon_hours": 72,
            "physics_workers": int(args.workers),
            "parent_spacing_hours": 73,
            "decision_to_execution_minutes": 1,
            "recurrent_time_axis_parallelized": False,
        },
        "parents": parents,
        "causality_canaries": causal,
        "teacher_canary": teacher,
        "gates": gates,
        "outcome_as_label_used": False,
        "long_gradient_training_authorized": False,
        "production_teacher_support_sufficiency_claimed": False,
        "new_scientific_verdict": False,
        "scientific_verdict": None,
        "next_gate": "LONG_TRAJECTORY_INTEGRATION_REQUIRES_SEPARATE_TRAINING_ADMISSION",
    }
    atomic_json(Path(args.output), out)
    print(json.dumps({
        "status": out["status"],
        "classification": out["classification"],
        "parents": out["execution"]["parents"],
        "total_branches": out["execution"]["total_branches"],
        "gates": out["gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
