#!/usr/bin/env python3
from __future__ import annotations

"""Lifecycle-safe entrypoint for the R11 Stage-2 integrated burst.

The Task-A H72 engine intentionally uses Linux ``fork`` so immutable, preloaded market
arrays are shared copy-on-write across workers.  Forking after the parent has started
threads or initialized CUDA is not a valid long-lived integration topology.  This
entrypoint therefore creates and exercises the persistent H72 pool first, while the
process is single-threaded and before any CUDA context exists, then hands that exact
pool to the existing Stage-2 integration driver.

Only WHEN the engineering worker processes are created changes.  Physics, market
payloads, trace ordering, evidence identity, training arithmetic, gradient ownership,
and Champion/Challenger semantics remain owned by the existing R11 engines.
"""

import json
import math
import os
import time
from pathlib import Path

from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.r102_evidence_cache import load_parent_physics_states, load_teacher_samples
from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102
from cb16_local_opt.trace_process_runtime_r11 import ForkProcessTraceRuntimeR11
from scripts.run_r11_stage2_task_a_trace_process_benchmark import _choose_items, _trace_hash


def main() -> int:
    prefork_started = time.monotonic()
    package_root = Path(os.environ["CB16_R11_BURST_PACKAGE_ROOT"]).resolve()
    legacy_root = Path(os.environ["CB16_R11_BURST_LEGACY_R104_ROOT"]).resolve()
    trace_workers = int(os.environ.get("CB16_R11_BURST_TRACE_WORKERS", "8"))
    configured_warmup = int(os.environ.get("CB16_R11_BURST_WARMUP_SECONDS", "30"))

    manifest_path = legacy_root / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("stride_hours", -1)) != 256:
        raise RuntimeError("R11_STAGE2_PREFORK_EXPECTED_R104_STRIDE256")
    if bool(manifest.get("final_holdout_2025_09_accessed", False)):
        raise RuntimeError("R11_STAGE2_PREFORK_REFUSES_OPENED_HOLDOUT")

    parents, _samples = load_teacher_samples(manifest["parents_file"], manifest["branches_file"])
    parent_states = load_parent_physics_states(manifest["parent_states_file"])
    physics = FrozenPhysicsRuntimeR102.load(package_root)
    trace_items = _choose_items(parents, parent_states, 96)
    symbols = sorted({x.symbol for x in trace_items})

    process_cache = MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    process_cache.preload(symbols)
    process_cache.assert_read_only()
    prefork_runtime = ForkProcessTraceRuntimeR11(
        physics=physics,
        market_cache=process_cache,
        max_workers=trace_workers,
    )

    # First run creates all fork workers.  Second run proves that the same executor is
    # reusable and completion order cannot alter the canonical H72 receipt identity.
    first_rows = prefork_runtime.run(trace_items)
    executor_after_first = prefork_runtime.executor_identity
    second_rows = prefork_runtime.run(list(reversed(trace_items)))
    executor_after_second = prefork_runtime.executor_identity
    if executor_after_first is None or executor_after_first != executor_after_second:
        raise RuntimeError("R11_STAGE2_PREFORK_POOL_NOT_PERSISTENT")
    if _trace_hash(first_rows) != _trace_hash(second_rows):
        raise RuntimeError("R11_STAGE2_PREFORK_TRACE_IDENTITY_DRIFT")
    process_cache.assert_read_only()

    # Import the CUDA-capable integration driver only after the fork pool is alive.
    # Importing torch itself is harmless here; the target initializes CUDA later.
    import scripts.run_r11_stage2_integration_burst as target

    real_cache_type = target.MarketRuntimeCacheR11
    cache_calls = 0

    def cache_factory(root):
        nonlocal cache_calls
        cache_calls += 1
        if cache_calls == 2:
            requested_root = Path(root).resolve()
            expected_root = (legacy_root / "evidence_cache").resolve()
            if requested_root != expected_root:
                raise RuntimeError(
                    f"R11_STAGE2_PREFORK_CACHE_ROOT_DRIFT:{requested_root}:{expected_root}"
                )
            return process_cache
        return real_cache_type(root)

    def trace_runtime_factory(*, physics, market_cache, max_workers):
        if int(max_workers) != trace_workers:
            raise RuntimeError(
                f"R11_STAGE2_PREFORK_WORKER_COUNT_DRIFT:{max_workers}:{trace_workers}"
            )
        if market_cache is not process_cache:
            raise RuntimeError("R11_STAGE2_PREFORK_CACHE_INSTANCE_DRIFT")
        if prefork_runtime.executor_identity is None:
            raise RuntimeError("R11_STAGE2_PREFORK_EXECUTOR_LOST")
        return prefork_runtime

    target.MarketRuntimeCacheR11 = cache_factory
    target.ForkProcessTraceRuntimeR11 = trace_runtime_factory

    # The supervisor measures from subprocess launch.  Charge prefork setup against the
    # configured warmup rather than silently shifting the measured interval later.
    prefork_elapsed = time.monotonic() - prefork_started
    remaining_warmup = max(0, int(math.ceil(configured_warmup - prefork_elapsed)))
    os.environ["CB16_R11_BURST_WARMUP_SECONDS"] = str(remaining_warmup)
    os.environ["CB16_R11_TRACE_POOL_LIFECYCLE"] = "PREFORKED_BEFORE_THREADS_AND_CUDA"

    try:
        return int(target.main())
    finally:
        # The target normally owns shutdown.  These calls are idempotent and keep early
        # target failures from leaving worker processes or mmap/cache handles alive.
        try:
            prefork_runtime.close()
        finally:
            process_cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
