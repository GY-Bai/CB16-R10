from __future__ import annotations

"""
Spawn-safe CPU trajectory farm.

R3 deliberately separates:
  policy inference / action production
from:
  authoritative batched account replay.

For the GTX1060 node, the intended high-throughput layout is:
  one GPU process (or the main process) -> batched actions
  multiple CPU replay workers          -> vectorized Physics
  persistent Experience Lake           -> trajectory/evidence artifacts

The replay workers never initialize CUDA.  This avoids CUDA poison-fork and keeps
the GTX1060 dedicated to Trader inference/training.
"""

import concurrent.futures as cf
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .vectorized_physics import (
    AccountBatchState,
    MarketBar,
    VectorPhysicsConfig,
    VectorizedPhysics,
)


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class MemmapArraySpec:
    path: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def open(self, mode: str = "r") -> np.ndarray:
        arr = np.load(self.path, mmap_mode=mode, allow_pickle=False)
        if tuple(arr.shape) != self.shape or str(arr.dtype) != self.dtype:
            raise RuntimeError(f"MEMMAP_SPEC_MISMATCH:{self.path}")
        return arr


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


class MemmapBundleWriter:
    """Materialize reusable NumPy arrays on SSD for zero-full-copy worker access."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, name: str, array: np.ndarray) -> MemmapArraySpec:
        arr = np.asarray(array)
        target = self.root / f"{name}.npy"
        tmp = self.root / f".{name}.{os.getpid()}.partial.npy"
        np.save(tmp, arr, allow_pickle=False)
        os.replace(tmp, target)
        return MemmapArraySpec(
            path=str(target),
            dtype=str(arr.dtype),
            shape=tuple(int(x) for x in arr.shape),
            sha256=sha256_file(target),
        )


@dataclass(frozen=True)
class TrajectoryReplayJob:
    job_id: str
    policy_generation: int
    policy_weight_hash: str
    market_lineage_hash: str
    account_lineage_hash: str
    config: VectorPhysicsConfig
    start_row: int
    stop_row: int
    initial_accounts: int
    timestamps: MemmapArraySpec
    opens: MemmapArraySpec
    highs: MemmapArraySpec
    lows: MemmapArraySpec
    closes: MemmapArraySpec
    volumes: MemmapArraySpec
    directions: MemmapArraySpec      # [T,N] int8
    requested_risks: MemmapArraySpec # [T,N] float32/64
    executable_directions: MemmapArraySpec | None = None
    executable_risks: MemmapArraySpec | None = None

    @property
    def identity_hash(self) -> str:
        return canonical_hash(self)

    def validate(self) -> None:
        if not (0 <= self.start_row < self.stop_row):
            raise ValueError("invalid replay row range")
        if self.initial_accounts <= 0:
            raise ValueError("initial_accounts must be positive")
        t = self.stop_row - self.start_row
        if self.directions.shape[0] < self.stop_row or self.requested_risks.shape[0] < self.stop_row:
            raise ValueError("action schedule too short")
        if self.directions.shape[1] != self.initial_accounts:
            raise ValueError("direction account dimension mismatch")
        if self.requested_risks.shape[1] != self.initial_accounts:
            raise ValueError("risk account dimension mismatch")
        for spec in (self.timestamps, self.opens, self.highs, self.lows, self.closes, self.volumes):
            if spec.shape[0] < self.stop_row:
                raise ValueError("market array too short")
        if self.executable_directions is not None and self.executable_directions.shape != self.directions.shape:
            raise ValueError("executable direction shape mismatch")
        if self.executable_risks is not None and self.executable_risks.shape != self.requested_risks.shape:
            raise ValueError("executable risk shape mismatch")


@dataclass(frozen=True)
class TrajectoryReplayResult:
    job_id: str
    job_identity_hash: str
    policy_generation: int
    policy_weight_hash: str
    market_lineage_hash: str
    account_lineage_hash: str
    physics_version: str
    physics_config_hash: str
    rows_processed: int
    accounts: int
    starting_equity_sum: float
    ending_equity_sum: float
    total_log_equity_reward: float
    total_fee: float
    total_turnover: float
    liquidations: int
    final_equity: tuple[float, ...]
    final_position_qty: tuple[float, ...]
    step_reward_sum_by_account: tuple[float, ...]
    completed_at_unix: float = 0.0

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def _configure_worker(cpu_threads: int) -> None:
    """Initializer run in each spawned worker."""
    cpu_threads = max(1, int(cpu_threads))
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    # Importing torch is intentionally not required here. If it is already installed/imported,
    # cap its CPU threads without touching CUDA.
    try:
        import torch
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass


def _worker_replay(job: TrajectoryReplayJob) -> TrajectoryReplayResult:
    job.validate()

    ts = job.timestamps.open()
    op = job.opens.open()
    hi = job.highs.open()
    lo = job.lows.open()
    cl = job.closes.open()
    vol = job.volumes.open()
    req_d = job.directions.open()
    req_r = job.requested_risks.open()
    exe_d = job.executable_directions.open() if job.executable_directions else req_d
    exe_r = job.executable_risks.open() if job.executable_risks else req_r

    engine = VectorizedPhysics(job.config)
    state = AccountBatchState.empty(job.initial_accounts, job.config)
    start_eq = float(state.balance.sum())
    reward_by_account = np.zeros(job.initial_accounts, dtype=np.float64)
    total_fee = 0.0
    total_turnover = 0.0

    last_ts = None
    for row in range(job.start_row, job.stop_row):
        cur_ts = int(ts[row])
        if last_ts is not None and cur_ts <= last_ts:
            raise RuntimeError(f"REPLAY_CHRONOLOGY_VIOLATION:{job.job_id}:{row}")
        last_ts = cur_ts
        bar = MarketBar(
            open=float(op[row]),
            high=float(hi[row]),
            low=float(lo[row]),
            close=float(cl[row]),
            funding_rate=0.0,
        )
        rd = np.asarray(req_d[row], dtype=np.int8)
        rr = np.asarray(req_r[row], dtype=np.float64)
        ed = np.asarray(exe_d[row], dtype=np.int8)
        er = np.asarray(exe_r[row], dtype=np.float64)

        # External Supervisor is allowed to clamp/reject but not increase risk.
        if np.any(er - rr > 1e-12):
            raise RuntimeError("SUPERVISOR_RISK_INCREASE_IN_REPLAY")
        if np.any((ed == 0) & (np.abs(er) > 1e-12)):
            raise RuntimeError("EXECUTABLE_FLAT_NONZERO_RISK")

        rec = engine.step(
            state,
            bar,
            executable_direction=ed,
            executable_risk=er,
            requested_direction=rd,
            dependence_group_count=job.initial_accounts,
        )
        reward_by_account += rec.log_equity_reward
        total_fee += float(rec.fee.sum())
        total_turnover += float(rec.turnover_notional.sum())

    final_eq = state.balance + np.where(
        state.position_qty != 0,
        state.position_qty * (float(cl[job.stop_row - 1]) - state.entry_price),
        0.0,
    )

    return TrajectoryReplayResult(
        job_id=job.job_id,
        job_identity_hash=job.identity_hash,
        policy_generation=job.policy_generation,
        policy_weight_hash=job.policy_weight_hash,
        market_lineage_hash=job.market_lineage_hash,
        account_lineage_hash=job.account_lineage_hash,
        physics_version="CB16_LOCAL_VECTOR_PHYSICS_R2",
        physics_config_hash=job.config.config_hash,
        rows_processed=job.stop_row - job.start_row,
        accounts=job.initial_accounts,
        starting_equity_sum=start_eq,
        ending_equity_sum=float(final_eq.sum()),
        total_log_equity_reward=float(reward_by_account.sum()),
        total_fee=total_fee,
        total_turnover=total_turnover,
        liquidations=int(state.liquidation_count.sum()),
        final_equity=tuple(float(x) for x in final_eq),
        final_position_qty=tuple(float(x) for x in state.position_qty),
        step_reward_sum_by_account=tuple(float(x) for x in reward_by_account),
        completed_at_unix=0.0,  # deterministic status-driving result
    )


@dataclass(frozen=True)
class TrajectoryFarmConfig:
    workers: int = 4
    cpu_threads_per_worker: int = 1
    start_method: str = "spawn"
    max_in_flight: int = 8

    def validate(self) -> None:
        if self.workers <= 0 or self.cpu_threads_per_worker <= 0:
            raise ValueError("worker/thread count must be positive")
        if self.start_method not in {"spawn", "forkserver"}:
            raise ValueError("use spawn or forkserver; fork is deliberately disallowed")
        if self.max_in_flight < self.workers:
            raise ValueError("max_in_flight should be >= workers")


class SpawnTrajectoryFarm:
    """Process-isolated replay farm with deterministic result ordering."""

    def __init__(self, config: TrajectoryFarmConfig):
        config.validate()
        self.config = config

    def run(self, jobs: Sequence[TrajectoryReplayJob]) -> list[TrajectoryReplayResult]:
        if not jobs:
            return []
        ids = [j.job_id for j in jobs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate job_id")

        ctx = mp.get_context(self.config.start_method)
        pending: dict[cf.Future, int] = {}
        results: dict[int, TrajectoryReplayResult] = {}

        with cf.ProcessPoolExecutor(
            max_workers=self.config.workers,
            mp_context=ctx,
            initializer=_configure_worker,
            initargs=(self.config.cpu_threads_per_worker,),
        ) as ex:
            next_submit = 0
            while next_submit < len(jobs) or pending:
                while (
                    next_submit < len(jobs)
                    and len(pending) < self.config.max_in_flight
                ):
                    job = jobs[next_submit]
                    fut = ex.submit(_worker_replay, job)
                    pending[fut] = next_submit
                    next_submit += 1

                done, _ = cf.wait(
                    pending,
                    return_when=cf.FIRST_COMPLETED,
                )
                for fut in done:
                    idx = pending.pop(fut)
                    res = fut.result()
                    if res.job_id != jobs[idx].job_id:
                        raise RuntimeError("TRAJECTORY_FARM_RESULT_ID_MISMATCH")
                    results[idx] = res

        return [results[i] for i in range(len(jobs))]


def partition_row_ranges(
    *,
    start: int,
    stop: int,
    chunk_rows: int,
) -> list[tuple[int, int]]:
    if not (0 <= start < stop) or chunk_rows <= 0:
        raise ValueError("invalid partition parameters")
    out = []
    x = start
    while x < stop:
        y = min(stop, x + chunk_rows)
        out.append((x, y))
        x = y
    return out
