from __future__ import annotations

"""Probe-only Thread-D scaling harness.

This file intentionally does NOT widen the active CC_FAST_R0 runtime contract.
It reuses the exact collector/account execution logic while relaxing only the
candidate-count guardrails so 128-account and 8/10-worker scaling can be measured.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time

from cb16_local_opt import cc_fast_account_workers_r0 as workers_mod
from cb16_local_opt import cc_fast_collector_r0 as collector_mod
from cb16_local_opt.cc_fast_benchmark_runner_r0 import run as benchmark_run


ALLOWED_PROBE_WORKERS = {2, 4, 6, 8, 10}


class ProbePersistentAccountWorkerPool(workers_mod.PersistentAccountWorkerPool):
    def __init__(
        self,
        *,
        worker_count,
        account_ids,
        market_descriptor,
        initial_equity,
        kernel_config=workers_mod.AccountKernelConfig(),
        startup_timeout_s=10.0,
    ):
        if worker_count not in ALLOWED_PROBE_WORKERS:
            raise ValueError("PROBE_WORKER_COUNT_INVALID")
        if not account_ids or len(set(account_ids)) != len(account_ids):
            raise ValueError("ACCOUNT_IDS_INVALID")
        self.worker_count = worker_count
        self.account_ids = tuple(account_ids)
        self._ctx = mp.get_context("spawn")
        self._in = [self._ctx.Queue(maxsize=4) for _ in range(worker_count)]
        self._out = self._ctx.Queue(maxsize=max(8, worker_count * 4))
        shards = [[] for _ in range(worker_count)]
        self._owner = {}
        for i, account_id in enumerate(self.account_ids):
            wid = i % worker_count
            shards[wid].append(account_id)
            self._owner[account_id] = wid
        self._processes = []
        for wid in range(worker_count):
            p = self._ctx.Process(
                target=workers_mod._worker_service,
                args=(
                    wid,
                    tuple(shards[wid]),
                    market_descriptor,
                    float(initial_equity),
                    self._in[wid],
                    self._out,
                    kernel_config,
                ),
                daemon=True,
            )
            p.start()
            self._processes.append(p)
        self.statuses = {}
        deadline = time.monotonic() + startup_timeout_s
        while len(self.statuses) < worker_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close(force=True)
                raise RuntimeError("ACCOUNT_WORKER_STARTUP_TIMEOUT")
            try:
                kind, payload = self._out.get(timeout=remaining)
            except queue.Empty as exc:
                self.close(force=True)
                raise RuntimeError("ACCOUNT_WORKER_STARTUP_TIMEOUT") from exc
            if kind != "STATUS":
                self.close(force=True)
                raise RuntimeError("ACCOUNT_WORKER_BAD_STARTUP_MESSAGE")
            self.statuses[payload.worker_id] = payload
        if any(x.cuda_initialized for x in self.statuses.values()):
            self.close(force=True)
            raise RuntimeError("ACCOUNT_WORKER_CUDA_INITIALIZED")


def _probe_collector_validate(self):
    if self.worker_count not in ALLOWED_PROBE_WORKERS:
        raise ValueError("PROBE_COLLECTOR_WORKER_COUNT_INVALID")
    if self.queue_max_bytes <= 0 or self.queue_max_age_s <= 0 or self.writer_chunk_facts <= 0:
        raise ValueError("COLLECTOR_CONFIG_INVALID")
    if self.max_steps is not None and self.max_steps <= 1:
        raise ValueError("COLLECTOR_MAX_STEPS_INVALID")
    return self


def run_probe(*, output: str, workers: int, accounts: int, market_steps: int, chunk_facts: int, policy_batch: int) -> dict:
    if accounts <= 0 or accounts > 128:
        raise ValueError("PROBE_ACCOUNT_COUNT_INVALID")
    original_pool = collector_mod.PersistentAccountWorkerPool
    original_validate = collector_mod.CollectorConfig.validate
    try:
        collector_mod.PersistentAccountWorkerPool = ProbePersistentAccountWorkerPool
        collector_mod.CollectorConfig.validate = _probe_collector_validate
        payload = benchmark_run(
            worker_count=workers,
            accounts=accounts,
            market_steps=market_steps,
            chunk_facts=chunk_facts,
            policy_batch=policy_batch,
        )
    finally:
        collector_mod.PersistentAccountWorkerPool = original_pool
        collector_mod.CollectorConfig.validate = original_validate
    payload["probe_only"] = True
    payload["active_runtime_contract_modified"] = False
    payload["probe_parameter_extension"] = {
        "allowed_workers": sorted(ALLOWED_PROBE_WORKERS),
        "max_accounts": 128,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "workers": workers,
        "accounts": accounts,
        "transitions_per_s": payload["collector_report"]["compliant_transitions_per_s"],
        "verdict": payload["semantic"]["verdict"],
    }, sort_keys=True))
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--accounts", type=int, required=True)
    p.add_argument("--market-steps", type=int, default=384)
    p.add_argument("--chunk-facts", type=int, default=128)
    p.add_argument("--policy-batch", type=int, default=128)
    a = p.parse_args()
    run_probe(
        output=a.output,
        workers=a.workers,
        accounts=a.accounts,
        market_steps=a.market_steps,
        chunk_facts=a.chunk_facts,
        policy_batch=a.policy_batch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
