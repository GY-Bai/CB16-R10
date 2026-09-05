from __future__ import annotations

"""R8.1-qualified bounded CPU parallelism for R10 real-historical H72 work.

The main process remains the single CUDA owner.  Spawned workers perform only
Frozen Physics CPU replay.  Results are reordered by submission ordinal so
worker scheduling cannot change evidence identity.
"""

import concurrent.futures as cf
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .r102_physics import CANDIDATES_R102, FrozenPhysicsRuntimeR102, simulate_h72_branch

_H72_RUNTIME: FrozenPhysicsRuntimeR102 | None = None
_H72_SYMBOL: str | None = None
_H72_TS: np.ndarray | None = None
_H72_BARS: np.ndarray | None = None
_H72_FUNDING: np.ndarray | None = None


def _set_worker_threads(threads: int) -> None:
    n = str(int(threads))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n
    # CUDA remains owned by the main process.  Merely importing torch in the
    # spawn bootstrap is harmless; replay workers never call torch.cuda.


def _h72_worker_init(
    package_root: str,
    symbol: str,
    hourly_ts: np.ndarray,
    hourly_ohlcv: np.ndarray,
    funding: np.ndarray,
    threads: int,
) -> None:
    global _H72_RUNTIME, _H72_SYMBOL, _H72_TS, _H72_BARS, _H72_FUNDING
    _set_worker_threads(threads)
    _H72_RUNTIME = FrozenPhysicsRuntimeR102.load(package_root)
    _H72_SYMBOL = str(symbol)
    _H72_TS = np.asarray(hourly_ts)
    _H72_BARS = np.asarray(hourly_ohlcv)
    _H72_FUNDING = np.asarray(funding)


def _require_worker_state() -> tuple[FrozenPhysicsRuntimeR102, str, np.ndarray, np.ndarray, np.ndarray]:
    if _H72_RUNTIME is None or _H72_SYMBOL is None or _H72_TS is None or _H72_BARS is None or _H72_FUNDING is None:
        raise RuntimeError("R102_H72_WORKER_NOT_INITIALIZED")
    return _H72_RUNTIME, _H72_SYMBOL, _H72_TS, _H72_BARS, _H72_FUNDING


@dataclass(frozen=True)
class H72ParentGroupJobR102:
    ordinal: int
    parent_id: str
    parent: Mapping[str, Any]
    decision_time_ms: int


@dataclass(frozen=True)
class H72ActionJobR102:
    ordinal: int
    parent_id: str
    parent: Mapping[str, Any]
    decision_time_ms: int
    candidate_direction_v55: int
    candidate_risk: float


def _counterfactual_worker(job: H72ParentGroupJobR102) -> tuple[int, str, list[tuple[int, float, dict[str, Any]]]]:
    runtime, symbol, hts, bars, funding = _require_worker_state()
    out: list[tuple[int, float, dict[str, Any]]] = []
    for direction, risk in CANDIDATES_R102:
        result = simulate_h72_branch(
            runtime,
            parent=job.parent,
            symbol=symbol,
            decision_time_ms=int(job.decision_time_ms),
            candidate_direction_v55=int(direction),
            candidate_risk=float(risk),
            hourly_ts=hts,
            hourly_ohlcv=bars,
            funding=funding,
        )
        out.append((int(direction), float(risk), result))
    return job.ordinal, job.parent_id, out


def _single_action_worker(job: H72ActionJobR102) -> tuple[int, str, dict[str, Any]]:
    runtime, symbol, hts, bars, funding = _require_worker_state()
    result = simulate_h72_branch(
        runtime,
        parent=job.parent,
        symbol=symbol,
        decision_time_ms=int(job.decision_time_ms),
        candidate_direction_v55=int(job.candidate_direction_v55),
        candidate_risk=float(job.candidate_risk),
        hourly_ts=hts,
        hourly_ohlcv=bars,
        funding=funding,
    )
    return job.ordinal, job.parent_id, result


def _bounded_process_map(
    worker_fn,
    jobs: Sequence[Any],
    *,
    package_root: str | Path,
    symbol: str,
    hourly_ts: np.ndarray,
    hourly_ohlcv: np.ndarray,
    funding: np.ndarray,
    workers: int,
    threads_per_worker: int,
    max_in_flight: int,
) -> list[Any]:
    if not jobs:
        return []
    workers = int(workers)
    threads_per_worker = int(threads_per_worker)
    max_in_flight = int(max_in_flight)
    if workers <= 0 or threads_per_worker <= 0:
        raise ValueError("R102_INVALID_H72_WORKER_CONFIG")
    if max_in_flight < workers:
        raise ValueError("R102_H72_MAX_IN_FLIGHT_LT_WORKERS")

    # Serial mode remains available for deterministic regression comparison.
    if workers == 1:
        _h72_worker_init(
            str(Path(package_root).resolve()), symbol, hourly_ts, hourly_ohlcv, funding, threads_per_worker
        )
        return [worker_fn(j) for j in jobs]

    ctx = mp.get_context("spawn")
    pending: dict[cf.Future, int] = {}
    results: dict[int, Any] = {}
    with cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_h72_worker_init,
        initargs=(
            str(Path(package_root).resolve()), str(symbol), np.asarray(hourly_ts),
            np.asarray(hourly_ohlcv), np.asarray(funding), threads_per_worker,
        ),
    ) as ex:
        next_i = 0
        while next_i < len(jobs) or pending:
            while next_i < len(jobs) and len(pending) < max_in_flight:
                f = ex.submit(worker_fn, jobs[next_i])
                pending[f] = next_i
                next_i += 1
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in done:
                idx = pending.pop(f)
                results[idx] = f.result()
    return [results[i] for i in range(len(jobs))]


def run_counterfactual_h72_farm_r102(
    *,
    package_root: str | Path,
    symbol: str,
    hourly_ts: np.ndarray,
    hourly_ohlcv: np.ndarray,
    funding: np.ndarray,
    jobs: Sequence[H72ParentGroupJobR102],
    workers: int,
    threads_per_worker: int,
    max_in_flight: int,
) -> list[tuple[int, str, list[tuple[int, float, dict[str, Any]]]]]:
    return _bounded_process_map(
        _counterfactual_worker,
        jobs,
        package_root=package_root,
        symbol=symbol,
        hourly_ts=hourly_ts,
        hourly_ohlcv=hourly_ohlcv,
        funding=funding,
        workers=workers,
        threads_per_worker=threads_per_worker,
        max_in_flight=max_in_flight,
    )


def run_single_action_h72_farm_r102(
    *,
    package_root: str | Path,
    symbol: str,
    hourly_ts: np.ndarray,
    hourly_ohlcv: np.ndarray,
    funding: np.ndarray,
    jobs: Sequence[H72ActionJobR102],
    workers: int,
    threads_per_worker: int,
    max_in_flight: int,
) -> list[tuple[int, str, dict[str, Any]]]:
    return _bounded_process_map(
        _single_action_worker,
        jobs,
        package_root=package_root,
        symbol=symbol,
        hourly_ts=hourly_ts,
        hourly_ohlcv=hourly_ohlcv,
        funding=funding,
        workers=workers,
        threads_per_worker=threads_per_worker,
        max_in_flight=max_in_flight,
    )
