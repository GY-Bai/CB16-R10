from __future__ import annotations

"""Narrow startup hook for the Stage-2 integration burst only.

Python imports ``sitecustomize`` before executing the requested script.  The R11 H72
process runtime deliberately uses Linux fork + copy-on-write market arrays; therefore
the long-lived worker pool must exist before the integration process starts threads or
initializes CUDA.  The failed 34171916851 burst forked only after both had happened and
later lost a child with ``BrokenProcessPool``.

This hook is inert for every other Python entrypoint.  It changes process lifecycle only;
all scientific identity and correctness checks remain in the normal Stage-2 driver.
"""

import json
import math
import os
import sys
import time
from pathlib import Path


_ENTRYPOINT = "run_r11_stage2_integration_burst.py"


def _is_stage2_burst_entrypoint() -> bool:
    if not sys.argv:
        return False
    try:
        return Path(sys.argv[0]).name == _ENTRYPOINT
    except (TypeError, ValueError):
        return False


def _install_stage2_prefork() -> None:
    required = (
        "CB16_R11_BURST_PACKAGE_ROOT",
        "CB16_R11_BURST_LEGACY_R104_ROOT",
        "CB16_R11_BURST_TRACE_WORKERS",
    )
    if any(not os.environ.get(name) for name in required):
        raise RuntimeError("R11_STAGE2_PREFORK_ENV_INCOMPLETE")

    prefork_started = time.monotonic()
    package_root = Path(os.environ["CB16_R11_BURST_PACKAGE_ROOT"]).resolve()
    legacy_root = Path(os.environ["CB16_R11_BURST_LEGACY_R104_ROOT"]).resolve()
    trace_workers = int(os.environ["CB16_R11_BURST_TRACE_WORKERS"])
    configured_warmup = int(os.environ.get("CB16_R11_BURST_WARMUP_SECONDS", "30"))

    # These imports match the already-qualified Task-A fork microbenchmark path.  They
    # do not initialize CUDA.  The process pool is forced live before the Stage-2 driver
    # imports/uses its CUDA training stack or starts telemetry/storage/orchestrator threads.
    import cb16_local_opt.market_runtime_cache_r11 as market_mod
    import cb16_local_opt.trace_process_runtime_r11 as process_mod
    from cb16_local_opt.r102_evidence_cache import load_parent_physics_states, load_teacher_samples
    from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102
    from scripts.run_r11_stage2_task_a_trace_process_benchmark import _choose_items, _trace_hash

    real_cache_type = market_mod.MarketRuntimeCacheR11
    real_trace_type = process_mod.ForkProcessTraceRuntimeR11

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

    process_cache = real_cache_type(legacy_root / "evidence_cache")
    process_cache.preload(symbols)
    process_cache.assert_read_only()
    prefork_runtime = real_trace_type(
        physics=physics,
        market_cache=process_cache,
        max_workers=trace_workers,
    )

    # The first run physically creates the fork workers.  A second completion-order
    # perturbation proves that the exact same executor is reusable and receipt identity
    # does not depend on scheduling order before CUDA/threaded overlap begins.
    first_rows = prefork_runtime.run(trace_items)
    executor_after_first = prefork_runtime.executor_identity
    second_rows = prefork_runtime.run(list(reversed(trace_items)))
    executor_after_second = prefork_runtime.executor_identity
    if executor_after_first is None or executor_after_first != executor_after_second:
        raise RuntimeError("R11_STAGE2_PREFORK_POOL_NOT_PERSISTENT")
    if _trace_hash(first_rows) != _trace_hash(second_rows):
        raise RuntimeError("R11_STAGE2_PREFORK_TRACE_IDENTITY_DRIFT")
    process_cache.assert_read_only()

    cache_calls = 0

    def cache_factory(root):
        nonlocal cache_calls
        cache_calls += 1
        # The Stage-2 driver creates exactly two market caches in its H72 identity block:
        # serial reference first, persistent process cache second.  Reuse the pre-forked
        # cache only for the second construction so no duplicate process-cache payload is
        # introduced into the 16-GiB integration memory envelope.
        if cache_calls == 2:
            requested_root = Path(root).resolve()
            expected_root = (legacy_root / "evidence_cache").resolve()
            if requested_root != expected_root:
                raise RuntimeError(
                    f"R11_STAGE2_PREFORK_CACHE_ROOT_DRIFT:{requested_root}:{expected_root}"
                )
            return process_cache
        return real_cache_type(root)

    def trace_factory(*, physics, market_cache, max_workers):
        if int(max_workers) != trace_workers:
            raise RuntimeError(
                f"R11_STAGE2_PREFORK_WORKER_COUNT_DRIFT:{max_workers}:{trace_workers}"
            )
        if market_cache is not process_cache:
            raise RuntimeError("R11_STAGE2_PREFORK_CACHE_INSTANCE_DRIFT")
        if prefork_runtime.executor_identity is None:
            raise RuntimeError("R11_STAGE2_PREFORK_EXECUTOR_LOST")
        return prefork_runtime

    # Patch the module exports before the driver executes its ``from ... import ...``
    # statements.  Every non-Stage-2 process sees the original classes unchanged.
    market_mod.MarketRuntimeCacheR11 = cache_factory
    process_mod.ForkProcessTraceRuntimeR11 = trace_factory

    # The qualification supervisor measures from subprocess launch.  Charge prefork work
    # to the existing warmup budget so the driver's measured interval remains aligned with
    # the supervisor rather than silently sliding several seconds later.
    prefork_elapsed = time.monotonic() - prefork_started
    remaining_warmup = max(0, int(math.ceil(configured_warmup - prefork_elapsed)))
    os.environ["CB16_R11_BURST_WARMUP_SECONDS"] = str(remaining_warmup)
    os.environ["CB16_R11_TRACE_POOL_LIFECYCLE"] = "PREFORKED_BEFORE_THREADS_AND_CUDA"
    print(
        "CB16_R11_STAGE2_TRACE_POOL_PREFORK=PASS "
        f"workers={trace_workers} symbols={len(symbols)} "
        f"executor={executor_after_second} prefork_seconds={prefork_elapsed:.6f} "
        f"remaining_warmup_seconds={remaining_warmup}",
        file=sys.stderr,
        flush=True,
    )


if _is_stage2_burst_entrypoint():
    _install_stage2_prefork()
