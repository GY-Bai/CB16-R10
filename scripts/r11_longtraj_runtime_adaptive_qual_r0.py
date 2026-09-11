#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cb16_local_opt.r11_adaptive_runtime_policy import (
    GLOBAL_PHYSICS_WORKER_BUDGET,
    MAX_CONCURRENT_EXPERIMENTS,
    POLICY_SCHEMA,
    plan_ready_independent_batch,
)
from scripts import r11_longtraj_perf_ab_r0 as perf

SCHEMA = "CB16_R11_LONGTRAJ_RUNTIME_ADAPTIVE_QUAL_R0_V1"
QUALIFIED_INFRA_BASE = "8e70212230a001751b720332fc1a25fef5436211"
PERFORMANCE_PARENT = "ae67e14a24c395c94b70209359b321e6dcc14aaa"


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-root", type=Path, required=True)
    ap.add_argument("--package-root", type=Path, required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    require(args.frames >= 8, "ADAPTIVE_QUAL_FRAMES_BELOW_8")

    idle_plan = plan_ready_independent_batch([])
    single_plan = plan_ready_independent_batch(["Q_SINGLE"])
    pair_plan = plan_ready_independent_batch(["Q_PAIR_A", "Q_PAIR_B"])
    backlog_plan = plan_ready_independent_batch(["Q_A", "Q_B", "Q_C"])

    require(idle_plan.mode == "IDLE", "ADAPTIVE_QUAL_IDLE_POLICY_DRIFT")
    require([x.physics_workers for x in single_plan.allocations] == [8], "ADAPTIVE_QUAL_SINGLE_POLICY_DRIFT")
    require([x.physics_workers for x in pair_plan.allocations] == [4, 4], "ADAPTIVE_QUAL_PAIR_POLICY_DRIFT")
    require(backlog_plan.concurrent_experiments == 2, "ADAPTIVE_QUAL_THREE_WAY_SCHEDULED")
    require(backlog_plan.queued_experiment_ids == ("Q_C",), "ADAPTIVE_QUAL_QUEUE_ORDER_DRIFT")
    require(single_plan.total_physics_workers <= GLOBAL_PHYSICS_WORKER_BUDGET, "ADAPTIVE_QUAL_SINGLE_BUDGET_OVER_8")
    require(pair_plan.total_physics_workers <= GLOBAL_PHYSICS_WORKER_BUDGET, "ADAPTIVE_QUAL_PAIR_BUDGET_OVER_8")
    require(pair_plan.concurrent_experiments <= MAX_CONCURRENT_EXPERIMENTS, "ADAPTIVE_QUAL_CONCURRENCY_OVER_2")

    times, identity = perf.select_decision_times(args.g0_root, args.symbol, args.frames)
    oracle = perf.run_batch(
        workers=1,
        times=times,
        package_root=args.package_root,
        g0_root=args.g0_root,
        symbol=args.symbol,
    )
    oracle_digest = str(oracle["result_digest"])

    single = perf.run_batch(
        workers=single_plan.allocations[0].physics_workers,
        times=times,
        package_root=args.package_root,
        g0_root=args.g0_root,
        symbol=args.symbol,
    )
    single_exact = str(single["result_digest"]) == oracle_digest
    require(single_exact, "ADAPTIVE_QUAL_SINGLE_PARITY_FAIL")

    group_args = argparse.Namespace(
        output=args.output,
        g0_root=args.g0_root,
        package_root=args.package_root,
        symbol=args.symbol,
        frames=args.frames,
    )
    pair = perf._run_experiment_group(
        budgets=[x.physics_workers for x in pair_plan.allocations],
        args=group_args,
        group_index=2,
        oracle_digest=oracle_digest,
    )
    require(pair["all_exact_parity_with_serial"] is True, "ADAPTIVE_QUAL_PAIR_PARITY_FAIL")
    require(pair["total_worker_budget"] == 8, "ADAPTIVE_QUAL_PAIR_WORKER_BUDGET_DRIFT")
    require(pair["concurrent_experiments"] == 2, "ADAPTIVE_QUAL_PAIR_CONCURRENCY_DRIFT")

    out = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": "RUNTIME_SCHEDULING_QUALIFICATION_ONLY__NO_SCIENCE_AUTHORITY",
        "policy_schema": POLICY_SCHEMA,
        "qualified_infra_base_sha": QUALIFIED_INFRA_BASE,
        "performance_overlay_parent_sha": PERFORMANCE_PARENT,
        "symbol": args.symbol,
        "frames": int(args.frames),
        "policy_checks": {
            "idle_mode": idle_plan.mode,
            "single_worker_budgets": [x.physics_workers for x in single_plan.allocations],
            "pair_worker_budgets": [x.physics_workers for x in pair_plan.allocations],
            "backlog_three_concurrent_experiments": backlog_plan.concurrent_experiments,
            "backlog_three_queued": list(backlog_plan.queued_experiment_ids),
            "max_concurrent_experiments": MAX_CONCURRENT_EXPERIMENTS,
            "global_physics_worker_budget": GLOBAL_PHYSICS_WORKER_BUDGET,
            "scheduler_infers_independence": False,
        },
        "serial_oracle": {
            "workers": 1,
            "wall_seconds": float(oracle["wall_seconds"]),
            "branches_per_second": float(oracle["branches_per_second"]),
            "result_digest": oracle_digest,
        },
        "single_ready_real_workload": {
            "workers": 8,
            "wall_seconds": float(single["wall_seconds"]),
            "branches_per_second": float(single["branches_per_second"]),
            "result_digest": str(single["result_digest"]),
            "exact_parity_with_serial": single_exact,
        },
        "two_ready_real_workload": pair,
        "g0_identity": identity["g0_identity"],
        "anchor_sha256": identity["anchor_sha256"],
        "final_holdout_touched": False,
        "fresh_market_data_downloaded": False,
        "new_scientific_verdict": False,
        "scientific_verdict": None,
        "next_gate": "FREEZE_RUNTIME_POLICY_IF_THIS_QUALIFICATION_PASSES",
    }
    write_json(args.output, out)
    digest_path = args.output.with_suffix(args.output.suffix + ".sha256")
    digest_path.write_text(sha256_file(args.output) + "  " + args.output.name + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
