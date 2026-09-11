from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from cb16_local_opt.full_minute_historical_replay_r0 import ReplayCentralBrain16M

ROOT = Path(__file__).resolve().parents[1]
PHYS_ROOT = ROOT / "authority/account_physics_r0/CB16_ACCOUNT_PHYSICS_STATE_V1_R0"
RUNTIME = PHYS_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME))

import account_physics_runtime_r0 as apr  # noqa: E402
from v55.kernel import CanonicalBar  # noqa: E402


def load_contract():
    return json.loads((PHYS_ROOT / "ACCOUNT_PHYSICS_CONTRACT_V1.json").read_text())


def bars(n: int):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    out = []
    px = 100.0
    for i in range(n):
        # deterministic, non-degenerate but mild path
        nxt = px * math.exp(0.0003 * math.sin(i / 13.0) + 0.0001)
        lo = min(px, nxt) * 0.999
        hi = max(px, nxt) * 1.001
        out.append(CanonicalBar(
            symbol="VECTORCANARY",
            bar_start=start + timedelta(minutes=i),
            timeframe="1m",
            open=px, high=hi, low=lo, close=nxt, volume=1000.0,
            mark_price=nxt, index_price=nxt,
        ))
        px = nxt
    return out


def action_for(account_index: int, step: int):
    if step < 24:
        return {"decision": 1}
    if step == 24:
        return {"decision": 2 if account_index % 2 == 0 else 0, "risk_multiplier": 0.25 + 0.05 * (account_index % 4)}
    return {"decision": 1}


def run_reference(batch: int, nsteps: int, contract):
    snapshots = [apr.initialize_snapshot(contract, risk_budget_remaining=1.0 - 0.001*i, risk_budget_capacity=1.0) for i in range(batch)]
    bs = bars(nsteps)
    hashes_by_step = []
    t0 = time.perf_counter()
    for step, bar in enumerate(bs):
        step_hashes = []
        for i in range(batch):
            r = apr.step_account(
                snapshots[i], action_for(i, step),
                {"bar": bar.to_dict(), "funding_rate": 0.0}, contract,
            )
            snapshots[i] = r["snapshot_t1"]
            step_hashes.append(apr.sha256_obj(snapshots[i]))
        hashes_by_step.append(step_hashes)
    elapsed = time.perf_counter() - t0
    return snapshots, hashes_by_step, elapsed


def run_persistent_pool(batch: int, nsteps: int, contract):
    supports = []
    kernels = []
    for i in range(batch):
        snap = apr.initialize_snapshot(contract, risk_budget_remaining=1.0 - 0.001*i, risk_budget_capacity=1.0)
        supports.append(copy.deepcopy(snap["observation_support_state"]))
        kernels.append(apr.restore_kernel(snap, contract))
    bs = bars(nsteps)
    hashes_by_step = []
    final_snaps = [None] * batch
    t0 = time.perf_counter()
    for step, bar in enumerate(bs):
        step_hashes = []
        for i, kernel in enumerate(kernels):
            kernel.step(bar, action_for(i, step), 0.0)
            snap = apr.snapshot_kernel(kernel, contract, supports[i])
            # Account observation remains available at every decision boundary.
            apr.project_observation(snap, float(bar.mark_price), contract)
            final_snaps[i] = snap
            step_hashes.append(apr.sha256_obj(snap))
        hashes_by_step.append(step_hashes)
    elapsed = time.perf_counter() - t0
    return final_snaps, hashes_by_step, elapsed


def physics_audit(batch_sizes=(1, 8, 32), nsteps=160):
    contract = load_contract()
    rows = []
    for b in batch_sizes:
        ref_snap, ref_hash, ref_t = run_reference(b, nsteps, contract)
        pool_snap, pool_hash, pool_t = run_persistent_pool(b, nsteps, contract)
        exact = ref_hash == pool_hash and [apr.sha256_obj(x) for x in ref_snap] == [apr.sha256_obj(x) for x in pool_snap]
        rows.append({
            "batch": b,
            "steps": nsteps,
            "account_steps": b*nsteps,
            "exact_every_step_hash_parity": exact,
            "reference_restore_step_snapshot_seconds": ref_t,
            "persistent_kernel_pool_seconds": pool_t,
            "reference_account_steps_per_second": (b*nsteps)/ref_t,
            "persistent_account_steps_per_second": (b*nsteps)/pool_t,
            "persistent_speedup": ref_t/pool_t,
        })
        if not exact:
            raise RuntimeError(f"PERSISTENT_POOL_PARITY_FAIL:B={b}")
    return rows


def brain_audit(batch_sizes=(1, 32, 256), iterations=80):
    if not torch.cuda.is_available():
        return {"status": "HARDWARE_LIMIT_NO_CUDA", "rows": []}
    device = torch.device("cuda")
    model = ReplayCentralBrain16M().to(device).eval()
    rows = []
    torch.manual_seed(20260911)
    with torch.inference_mode():
        for b in batch_sizes:
            op = torch.randn(b, 48, device=device, dtype=torch.float32)
            med = torch.randn(b, 48, device=device, dtype=torch.float32)
            acc = torch.randn(b, 6, device=device, dtype=torch.float32)
            for _ in range(10): model(op, med, acc)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iterations): model(op, med, acc)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            rows.append({
                "batch": b,
                "iterations": iterations,
                "samples": b*iterations,
                "seconds": elapsed,
                "samples_per_second": (b*iterations)/elapsed,
                "dtype": "FP32",
            })
    return {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "rows": rows,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    physics = physics_audit()
    brain = brain_audit()
    result = {
        "schema": "CB16_R11_LONGTRAJ_E4_VECTORIZATION_AUDIT_R1_RESULT_V1",
        "status": "PASS" if brain["status"] == "PASS" else "HARDWARE_LIMIT",
        "time_axis": "SEQUENTIAL_CAUSAL_SCAN_REQUIRED",
        "current_production_physics": "SCALAR_REFERENCE_STEP_ACCOUNT",
        "candidate_environment_parallelism": "PERSISTENT_KERNEL_POOL_ACROSS_INDEPENDENT_ACCOUNTS_ASSETS",
        "production_tensor_physics_claim": False,
        "physics_parity": physics,
        "brain_batching": brain,
        "interpretation_boundary": "A speedup from persistent independent kernels removes restore overhead but is not tensorized time. A future tensor Physics may only replace the scalar authority after broader exact parity qualification.",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
