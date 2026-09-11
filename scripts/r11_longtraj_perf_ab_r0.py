#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Sequence

import numpy as np
import torch

from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.r102_common import FORBIDDEN_FINAL_START_MS, H72, HOUR_MS
from cb16_local_opt.r102_market import load_anchor_frames
from cb16_local_opt.r102_physics import (
    CANDIDATES_R102,
    FrozenPhysicsRuntimeR102,
    build_parent_scenarios,
    simulate_h72_branch,
)
from scripts import r11_science_g0_historical_r1 as g0r1

SCHEMA = "CB16_R11_LONGTRAJ_PERF_AB_R0_V1"
QUALIFIED_INFRA_BASE = "8e70212230a001751b720332fc1a25fef5436211"
FINAL_START_MS = FORBIDDEN_FINAL_START_MS
_GLOBAL: dict[str, Any] = {}


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _set_single_thread_runtime() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _load_market_identity(g0_root: Path, symbol: str) -> tuple[dict[str, Any], Any]:
    lineage, g0_identity = g0r1.verify_g0_authority(g0_root)
    market_cache = MarketRuntimeCacheR11(g0_root)
    market = market_cache.get(symbol)
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == symbol)
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(g0r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "PERF_AB_ANCHOR_SHA_DRIFT")
    frames = load_anchor_frames(symbol, anchor_path)
    return {
        "lineage": lineage,
        "g0_identity": g0_identity,
        "anchor_path": str(anchor_path),
        "anchor_sha256": str(per_asset["anchors_sha256"]),
    }, (market_cache, market, frames)


def select_decision_times(g0_root: Path, symbol: str, frames_count: int) -> tuple[list[int], dict[str, Any]]:
    require(frames_count >= 8, "PERF_AB_FRAMES_BELOW_8")
    identity, (_cache, _market, frames) = _load_market_identity(g0_root, symbol)
    selected = g0r1.select_candidate_frames(
        frames,
        train_target=int(frames_count),
        validation_target=8,
        candidate_factor=3,
    )
    times = [int(frame.decision_time_ms) for split, frame in selected if split == "TRAIN"]
    require(len(times) >= frames_count, f"PERF_AB_TRAIN_FRAME_SHORTFALL:{len(times)}")
    times = times[:frames_count]
    require(len(set(times)) == len(times), "PERF_AB_DUPLICATE_DECISION_TIMES")
    require(all(t + (H72 - 1) * HOUR_MS < FINAL_START_MS for t in times), "PERF_AB_FINAL_BOUNDARY_VIOLATION")
    return times, identity


def _worker_init(package_root: str, g0_root: str, symbol: str) -> None:
    _set_single_thread_runtime()
    package = Path(package_root)
    g0 = Path(g0_root)
    physics = FrozenPhysicsRuntimeR102.load(package)
    market_cache = MarketRuntimeCacheR11(g0)
    market = market_cache.get(symbol)
    _GLOBAL.clear()
    _GLOBAL.update(
        {
            "physics": physics,
            "market_cache": market_cache,
            "market": market,
            "symbol": symbol,
        }
    )


def _branch_record(branch: dict[str, Any]) -> dict[str, Any]:
    utility = branch.get("utility")
    return {
        "status": str(branch.get("status")),
        "utility_hex": None if utility is None else float(utility).hex(),
        "termination_reason": None if branch.get("termination_reason") is None else str(branch.get("termination_reason")),
    }


def _eval_decision_time(t: int) -> dict[str, Any]:
    physics = _GLOBAL["physics"]
    market = _GLOBAL["market"]
    symbol = str(_GLOBAL["symbol"])
    scenarios = build_parent_scenarios(
        physics,
        symbol=symbol,
        decision_time_ms=int(t),
        hourly_ts=market.open_time_ms,
        hourly_ohlcv=market.ohlcv,
        funding=market.funding_rate,
        prehistory_hours=96,
    )
    rows: list[dict[str, Any]] = []
    eligible_scenarios = 0
    for scenario in sorted(scenarios, key=lambda x: str(x["scenario"])):
        if not bool(scenario["eligible_for_economic_evidence"]):
            continue
        eligible_scenarios += 1
        parent = {
            "parent_id": f"PERF_AB:{symbol}:{t}:{scenario['scenario']}",
            "account_id": scenario["account_id"],
            "snapshot": scenario["snapshot"],
            "risk_authority": scenario["risk_authority"],
            "current_mark": float(scenario["current_mark"]),
        }
        for direction_v55, requested_risk in CANDIDATES_R102:
            branch = simulate_h72_branch(
                physics,
                parent=parent,
                symbol=symbol,
                decision_time_ms=int(t),
                candidate_direction_v55=int(direction_v55),
                candidate_risk=float(requested_risk),
                hourly_ts=market.open_time_ms,
                hourly_ohlcv=market.ohlcv,
                funding=market.funding_rate,
            )
            rec = {
                "scenario": str(scenario["scenario"]),
                "direction_v55": int(direction_v55),
                "requested_risk_hex": float(requested_risk).hex(),
                **_branch_record(branch),
            }
            rows.append(rec)
    require(rows, f"PERF_AB_NO_ELIGIBLE_BRANCHES:{t}")
    matured = sum(1 for x in rows if x["status"] == "MATURED")
    return {
        "decision_time_ms": int(t),
        "eligible_scenarios": int(eligible_scenarios),
        "branches": int(len(rows)),
        "matured": int(matured),
        "digest": digest_obj(rows),
    }


def run_batch(
    *,
    workers: int,
    times: Sequence[int],
    package_root: Path,
    g0_root: Path,
    symbol: str,
) -> dict[str, Any]:
    require(workers >= 1, "PERF_AB_WORKERS_INVALID")
    start = time.perf_counter()
    if workers == 1:
        _worker_init(str(package_root), str(g0_root), symbol)
        rows = [_eval_decision_time(int(t)) for t in times]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(str(package_root), str(g0_root), symbol),
        ) as pool:
            rows = list(pool.map(_eval_decision_time, [int(t) for t in times], chunksize=1))
    wall = float(time.perf_counter() - start)
    rows = sorted(rows, key=lambda x: int(x["decision_time_ms"]))
    branches = sum(int(x["branches"]) for x in rows)
    matured = sum(int(x["matured"]) for x in rows)
    require(branches > 0 and wall > 0.0, "PERF_AB_ZERO_WORK")
    payload = [{"decision_time_ms": x["decision_time_ms"], "digest": x["digest"]} for x in rows]
    return {
        "workers": int(workers),
        "thread_policy": "ONE_THREAD_PER_PROCESS",
        "startup_included": True,
        "decision_times": int(len(rows)),
        "branches": int(branches),
        "matured": int(matured),
        "wall_seconds": wall,
        "decision_times_per_second": float(len(rows) / wall),
        "branches_per_second": float(branches / wall),
        "result_digest": digest_obj(payload),
        "per_time": rows,
    }


def run_workers_mode(args: argparse.Namespace) -> dict[str, Any]:
    counts = [int(x) for x in args.worker_counts.split(",") if x.strip()]
    require(counts == [1, 2, 4, 8], f"PERF_AB_WORKER_MATRIX_DRIFT:{counts}")
    times, identity = select_decision_times(args.g0_root, args.symbol, args.frames)
    results: list[dict[str, Any]] = []
    oracle_digest: str | None = None
    for count in counts:
        row = run_batch(
            workers=count,
            times=times,
            package_root=args.package_root,
            g0_root=args.g0_root,
            symbol=args.symbol,
        )
        if count == 1:
            oracle_digest = str(row["result_digest"])
            row["exact_parity_with_serial"] = True
        else:
            row["exact_parity_with_serial"] = str(row["result_digest"]) == oracle_digest
            require(bool(row["exact_parity_with_serial"]), f"PERF_AB_PHYSICS_PARITY_FAIL:{count}")
        results.append(row)
    serial = float(results[0]["branches_per_second"])
    for row in results:
        row["speedup_vs_1_worker"] = float(row["branches_per_second"] / serial)
    return {
        "schema": SCHEMA,
        "mode": "A_PHYSICS_WORKER_SCALING",
        "status": "PASS",
        "qualified_infra_base_sha": QUALIFIED_INFRA_BASE,
        "symbol": args.symbol,
        "frames": int(args.frames),
        "worker_matrix": counts,
        "serial_oracle_digest": oracle_digest,
        "results": results,
        "g0_identity": identity["g0_identity"],
        "anchor_sha256": identity["anchor_sha256"],
        "final_holdout_touched": False,
        "fresh_market_data_downloaded": False,
        "scientific_verdict": None,
        "classification": "PERFORMANCE_OVERLAY_ONLY__NO_SCIENCE_AUTHORITY",
    }


def run_child_mode(args: argparse.Namespace) -> int:
    times, identity = select_decision_times(args.g0_root, args.symbol, args.frames)
    result = run_batch(
        workers=int(args.workers),
        times=times,
        package_root=args.package_root,
        g0_root=args.g0_root,
        symbol=args.symbol,
    )
    out = {
        "schema": SCHEMA,
        "mode": "B_CHILD_EXPERIMENT",
        "status": "PASS",
        "experiment_id": str(args.experiment_id),
        "qualified_infra_base_sha": QUALIFIED_INFRA_BASE,
        "symbol": args.symbol,
        "frames": int(args.frames),
        "result": result,
        "anchor_sha256": identity["anchor_sha256"],
        "final_holdout_touched": False,
        "fresh_market_data_downloaded": False,
        "scientific_verdict": None,
        "classification": "PERFORMANCE_OVERLAY_ONLY__NO_SCIENCE_AUTHORITY",
    }
    write_json(args.output, out)
    return 0


def _run_experiment_group(
    *,
    budgets: Sequence[int],
    args: argparse.Namespace,
    group_index: int,
    oracle_digest: str,
) -> dict[str, Any]:
    require(sum(int(x) for x in budgets) <= 8, f"PERF_AB_CPU_BUDGET_OVER_8:{budgets}")
    group_dir = args.output.parent / f"b_group_{group_index}"
    group_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[subprocess.Popen[str], Path, int, str]] = []
    start = time.perf_counter()
    for i, workers in enumerate(budgets):
        out = group_dir / f"experiment_{i}.json"
        exp_id = f"B{len(budgets)}_{i}_W{workers}"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode", "child",
            "--g0-root", str(args.g0_root),
            "--package-root", str(args.package_root),
            "--symbol", args.symbol,
            "--frames", str(args.frames),
            "--workers", str(int(workers)),
            "--experiment-id", exp_id,
            "--output", str(out),
        ]
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        procs.append((subprocess.Popen(cmd, text=True, env=env), out, int(workers), exp_id))
    children: list[dict[str, Any]] = []
    for proc, out, workers, exp_id in procs:
        rc = proc.wait()
        require(rc == 0, f"PERF_AB_CHILD_FAILED:{exp_id}:RC={rc}")
        require(out.exists(), f"PERF_AB_CHILD_OUTPUT_MISSING:{exp_id}")
        child = json.loads(out.read_text(encoding="utf-8"))
        require(child["status"] == "PASS", f"PERF_AB_CHILD_NOT_PASS:{exp_id}")
        digest = str(child["result"]["result_digest"])
        require(digest == oracle_digest, f"PERF_AB_CHILD_PARITY_FAIL:{exp_id}")
        children.append(
            {
                "experiment_id": exp_id,
                "workers": workers,
                "wall_seconds": float(child["result"]["wall_seconds"]),
                "branches": int(child["result"]["branches"]),
                "branches_per_second": float(child["result"]["branches_per_second"]),
                "result_digest": digest,
                "exact_parity_with_serial": True,
            }
        )
    wall = float(time.perf_counter() - start)
    total_branches = sum(x["branches"] for x in children)
    return {
        "concurrent_experiments": int(len(budgets)),
        "worker_budgets": [int(x) for x in budgets],
        "total_worker_budget": int(sum(budgets)),
        "wall_seconds": wall,
        "aggregate_branches": int(total_branches),
        "aggregate_branches_per_second": float(total_branches / wall),
        "all_exact_parity_with_serial": True,
        "children": children,
    }


def run_experiments_mode(args: argparse.Namespace) -> dict[str, Any]:
    times, identity = select_decision_times(args.g0_root, args.symbol, args.frames)
    oracle = run_batch(
        workers=1,
        times=times,
        package_root=args.package_root,
        g0_root=args.g0_root,
        symbol=args.symbol,
    )
    oracle_digest = str(oracle["result_digest"])
    groups = ([8], [4, 4], [3, 3, 2])
    results = [
        _run_experiment_group(
            budgets=budgets,
            args=args,
            group_index=i + 1,
            oracle_digest=oracle_digest,
        )
        for i, budgets in enumerate(groups)
    ]
    base = float(results[0]["aggregate_branches_per_second"])
    for row in results:
        row["aggregate_speedup_vs_single_experiment"] = float(row["aggregate_branches_per_second"] / base)
    return {
        "schema": SCHEMA,
        "mode": "B_CONCURRENT_EXPERIMENT_SCALING",
        "status": "PASS",
        "qualified_infra_base_sha": QUALIFIED_INFRA_BASE,
        "symbol": args.symbol,
        "frames": int(args.frames),
        "serial_oracle": {
            "workers": 1,
            "result_digest": oracle_digest,
            "wall_seconds": oracle["wall_seconds"],
            "branches_per_second": oracle["branches_per_second"],
        },
        "experiment_matrix": [
            {"concurrent_experiments": 1, "worker_budgets": [8]},
            {"concurrent_experiments": 2, "worker_budgets": [4, 4]},
            {"concurrent_experiments": 3, "worker_budgets": [3, 3, 2]},
        ],
        "results": results,
        "g0_identity": identity["g0_identity"],
        "anchor_sha256": identity["anchor_sha256"],
        "mutable_state_policy": "PROCESS_ISOLATED__READ_ONLY_MARKET_AND_FROZEN_PHYSICS_INPUTS",
        "final_holdout_touched": False,
        "fresh_market_data_downloaded": False,
        "scientific_verdict": None,
        "classification": "PERFORMANCE_OVERLAY_ONLY__NO_SCIENCE_AUTHORITY",
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("workers", "experiments", "child"), required=True)
    ap.add_argument("--g0-root", type=Path, default=Path(os.environ.get("CB16_G0_ROOT", "/cb16/g0")))
    ap.add_argument("--package-root", type=Path, default=Path(os.environ.get("CB16_PACKAGE_ROOT", "/cb16/package")))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--worker-counts", default="1,2,4,8")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--experiment-id", default="CHILD")
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    require(args.symbol == "BTCUSDT", "PERF_AB_SCOPE_BTC_ONLY")
    require(args.frames >= 8, "PERF_AB_FRAMES_BELOW_8")
    if args.mode == "child":
        return run_child_mode(args)
    if args.mode == "workers":
        result = run_workers_mode(args)
    else:
        result = run_experiments_mode(args)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
