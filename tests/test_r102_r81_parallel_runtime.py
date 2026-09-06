from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.r102_common import HOUR_MS
from cb16_local_opt.r102_evidence_cache import ParentContextR102
from cb16_local_opt.r102_learning import compile_teacher_evidence
from cb16_local_opt.r102_parallel_runtime import (
    H72ParentGroupJobR102,
    ram_action_r82,
    run_counterfactual_h72_farm_r102,
)
from cb16_local_opt.r102_physics import CANDIDATES_R102, FrozenPhysicsRuntimeR102, build_parent_scenarios, simulate_h72_branch
from cb16_local_opt.r102_runtime_authority import (
    EXPECTED_RUNTIME_AUTHORITY_CONTENT_HASH,
    EXPECTED_RUNTIME_AUTHORITY_FILE_SHA256,
    EXPECTED_RUNTIME_PROFILE_HASH,
    load_r102_runtime_parallelism,
)


def _market_fixture():
    start = 1609459200000
    ts = np.arange(200, dtype=np.int64) * HOUR_MS + start
    base = 30000 * (1 + 0.0002 * np.arange(200)) + 100 * np.sin(np.arange(200) / 7)
    o = base
    c = base * (1 + 0.0001 * np.sin(np.arange(200) / 3))
    h = np.maximum(o, c) * 1.001
    l = np.minimum(o, c) * 0.999
    v = np.full(200, 100.0)
    bars = np.stack([o, h, l, c, v], 1).astype(np.float32)
    funding = np.zeros(200, dtype=np.float64)
    funding[::8] = 0.0001
    return ts, bars, funding


def _teacher_fixture():
    parents: dict[str, ParentContextR102] = {}
    samples: list[CounterfactualBranchSampleR5] = []
    for i in range(50):
        split = "TRAIN" if i < 40 else "VALIDATION"
        t = 1_600_000_000_000 + i * 512 * HOUR_MS
        op = tuple(math.sin(i / 9 + j * 0.01) * 0.1 for j in range(48))
        med = tuple(math.cos(i / 11 + j * 0.01) * 0.1 for j in range(48))
        acc = (0.0, 0.0, 1.0, 0.01 * (i % 4), 1.0, 0.0)
        p = ParentContextR102(
            parent_id=f"P{i}", dependence_group_id=f"G{i}", symbol="BTCUSDT",
            decision_time_ms=t, split=split, scenario="CLEAN",
            operator48=op, medium48=med, account6=acc, ordered4h30=tuple([0.0] * 30),
            current_mark=30000.0, snapshot_sha256=f"snap{i}",
            eligible_for_economic_evidence=True, market_lineage_hash=f"m{i}",
        )
        parents[p.parent_id] = p
        for d, r in [(0, 0.0)] + [(d, r) for d in (-1, 1) for r in (0.25, 0.5, 0.75, 1.0)]:
            u = 0.003 * math.sin(i / 7) * d * r - 0.0002 * r * r
            samples.append(CounterfactualBranchSampleR5(
                p.parent_id, p.student_context_object_id, t, p.student_features,
                d, r, u, p.dependence_group_id, p.market_lineage_hash,
            ))
    return parents, samples


def test_r81_baseline_is_preserved_while_r82_active_overlay_is_six_workers():
    r = load_r102_runtime_parallelism(ROOT, live_environment_check=False)
    assert r.runtime_authority_content_hash == EXPECTED_RUNTIME_AUTHORITY_CONTENT_HASH
    assert r.runtime_profile_hash == EXPECTED_RUNTIME_PROFILE_HASH
    assert EXPECTED_RUNTIME_AUTHORITY_FILE_SHA256 == "f50757da882cee0f6def11ec9c6e38e1065f415032f7d1405b249beb997e92df"
    limits = json.loads((ROOT / "authority/R8_1_qualification/R8_1_RUNTIME_LIMITS_FROZEN.json").read_text())
    assert limits["h72_worker_scaling"]["selected_workers"] == 2
    assert r.performance_overlay == "R8_2_6W_RAM_ADAPTIVE"
    assert r.h72_workers == 6
    assert r.h72_threads_per_worker == 1
    assert r.h72_max_in_flight == 8
    assert r.h72_minimum_workers == 1
    assert r.teacher_workers == 6
    assert r.teacher_threads_per_worker == 1
    assert r.experience_shards == 4
    assert r.gpu_qualified_train_batch_ceiling == 8192
    assert r.cache_hit_first is True
    assert r.runtime_scheduling_identity_is_not_scientific_cache_identity is True
    r10_batch = 512
    assert r10_batch <= r.gpu_qualified_train_batch_ceiling
    assert r10_batch == 512
    assert r.single_cuda_owner is True
    assert r.do_not_run_h72_and_teacher_at_full_concurrency is True


def test_h72_active_six_worker_results_equal_serial_exactly():
    r = load_r102_runtime_parallelism(ROOT, live_environment_check=False)
    ts, bars, funding = _market_fixture()
    runtime = FrozenPhysicsRuntimeR102.load(ROOT)
    t = int(ts[96])
    parent = build_parent_scenarios(
        runtime, symbol="BTCUSDT", decision_time_ms=t,
        hourly_ts=ts, hourly_ohlcv=bars, funding=funding, prehistory_hours=96,
    )[0]

    serial = []
    for direction, risk in CANDIDATES_R102:
        serial.append((int(direction), float(risk), simulate_h72_branch(
            runtime, parent=parent, symbol="BTCUSDT", decision_time_ms=t,
            candidate_direction_v55=int(direction), candidate_risk=float(risk),
            hourly_ts=ts, hourly_ohlcv=bars, funding=funding,
        )))

    parallel = run_counterfactual_h72_farm_r102(
        package_root=ROOT, symbol="BTCUSDT", hourly_ts=ts, hourly_ohlcv=bars, funding=funding,
        jobs=[H72ParentGroupJobR102(ordinal=0, parent_id="P0", parent=parent, decision_time_ms=t)],
        workers=r.h72_workers, threads_per_worker=r.h72_threads_per_worker,
        max_in_flight=r.h72_max_in_flight,
    )
    assert len(parallel) == 1
    _, pid, branch_results = parallel[0]
    assert pid == "P0"
    assert branch_results == serial


def test_teacher_active_six_worker_content_hashes_equal_serial():
    r = load_r102_runtime_parallelism(ROOT, live_environment_check=False)
    parents, samples = _teacher_fixture()
    serial_train, serial_val = compile_teacher_evidence(samples, parents, workers=1, threads_per_worker=1, max_in_flight=1)
    par_train, par_val = compile_teacher_evidence(
        samples, parents,
        workers=r.teacher_workers,
        threads_per_worker=r.teacher_threads_per_worker,
        max_in_flight=r.h72_max_in_flight,
    )
    assert [x.content_hash for x in par_train] == [x.content_hash for x in serial_train]
    assert [x.content_hash for x in par_val] == [x.content_hash for x in serial_val]


def test_ram_monitor_never_requests_worker_removal_below_one_effective_worker():
    assert ram_action_r82(0.50, 6) == "NONE"
    assert ram_action_r82(0.85, 6) == "RETIRE_ONE"
    assert ram_action_r82(0.92, 6) == "KILL_ONE"
    assert ram_action_r82(0.99, 1) == "PAUSE"
