
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
import queue
import time
from typing import Any, Iterable, Mapping, Sequence

from .cc_fast_account_kernel_r0 import AccountKernelConfig, step_account_scalar, target_quantity_from_risk
from .cc_fast_account_state_r0 import AccountStateSoA
from .cc_fast_market_cache_r0 import SharedMarketDescriptor, SharedMarketView
from .cc_fast_policy_broker_r0 import worker_cuda_detector
from .cc_fast_wire_r0 import FastEnvironmentTransition, semantic_sha256

WORKER_SCHEMA = "CB16_R11_CC_FAST_ACCOUNT_WORKERS_V1"


@dataclass(frozen=True)
class AccountWorkerCommand:
    ordinal: int
    account_lineage_id: str
    decision_index: int
    step: int
    nominal_direction: str
    nominal_target_risk: float
    policy_decision_ref: str
    permission_status: str = "ALLOW"
    permission_reason: str = "D_SYNTHETIC_PERFORMANCE_FIXTURE"


@dataclass(frozen=True)
class AccountWorkerResult:
    ordinal: int
    worker_id: int
    worker_pid: int
    account_lineage_id: str
    decision_index: int
    transition: FastEnvironmentTransition
    snapshot: dict[str, float | int | bool]


@dataclass(frozen=True)
class WorkerStatus:
    worker_id: int
    pid: int
    cuda_initialized: bool
    numeric_threads: Mapping[str, str]
    owned_accounts: tuple[str, ...]


def _set_worker_thread_limits() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"


def _worker_service(
    worker_id: int,
    account_ids: tuple[str, ...],
    market_descriptor: SharedMarketDescriptor,
    initial_equity: float,
    in_q,
    out_q,
    kernel_config: AccountKernelConfig,
) -> None:
    _set_worker_thread_limits()
    if worker_cuda_detector():
        raise RuntimeError("WORKER_CUDA_INITIALIZATION_FORBIDDEN")
    with SharedMarketView(market_descriptor) as market_view:
        market = market_view.array
        if market.ndim != 2 or market.shape[1] != 2:
            raise RuntimeError("WORKER_MARKET_LAYOUT_INVALID")
        state = AccountStateSoA.allocate(account_ids, initial_equity=initial_equity)
        index = {account_id: i for i, account_id in enumerate(account_ids)}
        out_q.put(("STATUS", WorkerStatus(
            worker_id=worker_id,
            pid=os.getpid(),
            cuda_initialized=worker_cuda_detector(),
            numeric_threads={
                k: os.environ.get(k, "") for k in (
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"
                )
            },
            owned_accounts=account_ids,
        )))
        while True:
            batch = in_q.get()
            if batch is None:
                out_q.put(("CLOSED", worker_id))
                return
            if batch == "__CRASH__":
                os._exit(91)
            results: list[AccountWorkerResult] = []
            for command in batch:
                if command.account_lineage_id not in index:
                    raise RuntimeError("ACCOUNT_OWNERSHIP_VIOLATION")
                i = index[command.account_lineage_id]
                if int(state.decision_index[i]) != command.decision_index:
                    raise RuntimeError("ACCOUNT_DECISION_ORDER_VIOLATION")
                step = command.step
                if step < 0 or step + 1 >= market.shape[0]:
                    raise RuntimeError("ACCOUNT_MARKET_STEP_INVALID")
                price_before = float(market[step, 0])
                price_after = float(market[step + 1, 0])
                funding_rate = float(market[step + 1, 1])
                if command.nominal_direction == "FLAT":
                    target_quantity = 0.0
                else:
                    target_quantity = target_quantity_from_risk(
                        command.nominal_direction,
                        command.nominal_target_risk,
                        float(state.equity[i]),
                        price_before,
                    )
                result = step_account_scalar(
                    state, i,
                    account_lineage_id=command.account_lineage_id,
                    environment_time_before_ns=step,
                    environment_time_after_ns=step + 1,
                    mark_price_before=price_before,
                    mark_price_after=price_after,
                    target_quantity=target_quantity,
                    funding_rate=funding_rate,
                    policy_decision_ref=command.policy_decision_ref,
                    permitted_target_direction=command.nominal_direction,
                    permitted_target_risk=command.nominal_target_risk,
                    permission_status=command.permission_status,
                    permission_reason=command.permission_reason,
                    boundary_type="NONE",
                    config=kernel_config,
                )
                results.append(AccountWorkerResult(
                    ordinal=command.ordinal,
                    worker_id=worker_id,
                    worker_pid=os.getpid(),
                    account_lineage_id=command.account_lineage_id,
                    decision_index=command.decision_index,
                    transition=result.transition,
                    snapshot=state.snapshot(i),
                ))
            out_q.put(("RESULTS", results))


class PersistentAccountWorkerPool:
    """
    Persistent spawned process shards.

    Each logical account is assigned to exactly one worker for the pool lifetime.
    The worker owns authoritative mutable account state; the main process only keeps
    compact returned snapshots for observation assembly. Market state is read-only
    shared memory and is not copied per account/worker.
    """

    def __init__(
        self,
        *,
        worker_count: int,
        account_ids: Sequence[str],
        market_descriptor: SharedMarketDescriptor,
        initial_equity: float,
        kernel_config: AccountKernelConfig = AccountKernelConfig(),
        startup_timeout_s: float = 10.0,
    ):
        if worker_count not in {2, 4, 6}:
            raise ValueError("WORKER_COUNT_OUTSIDE_CC_CANDIDATE_SET")
        if not account_ids or len(set(account_ids)) != len(account_ids):
            raise ValueError("ACCOUNT_IDS_INVALID")
        self.worker_count = worker_count
        self.account_ids = tuple(account_ids)
        self._ctx = mp.get_context("spawn")
        self._in = [self._ctx.Queue(maxsize=4) for _ in range(worker_count)]
        self._out = self._ctx.Queue(maxsize=max(8, worker_count * 4))
        shards: list[list[str]] = [[] for _ in range(worker_count)]
        self._owner: dict[str, int] = {}
        for i, account_id in enumerate(self.account_ids):
            wid = i % worker_count
            shards[wid].append(account_id)
            self._owner[account_id] = wid
        self._processes = []
        for wid in range(worker_count):
            p = self._ctx.Process(
                target=_worker_service,
                args=(wid, tuple(shards[wid]), market_descriptor, float(initial_equity), self._in[wid], self._out, kernel_config),
                daemon=True,
            )
            p.start()
            self._processes.append(p)
        self.statuses: dict[int, WorkerStatus] = {}
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

    def owner_for(self, account_id: str) -> int:
        return self._owner[account_id]

    def execute(self, commands: Sequence[AccountWorkerCommand], *, timeout_s: float = 20.0) -> list[AccountWorkerResult]:
        if not commands:
            return []
        if len({(c.account_lineage_id, c.decision_index) for c in commands}) != len(commands):
            raise RuntimeError("DUPLICATE_ACCOUNT_DECISION_TASK")
        batches: dict[int, list[AccountWorkerCommand]] = {}
        for c in commands:
            try:
                wid = self._owner[c.account_lineage_id]
            except KeyError as exc:
                raise RuntimeError("UNKNOWN_ACCOUNT") from exc
            batches.setdefault(wid, []).append(c)
        for wid, batch in batches.items():
            if not self._processes[wid].is_alive():
                raise RuntimeError("ACCOUNT_WORKER_CRASHED")
            self._in[wid].put(batch, timeout=timeout_s)
        result: list[AccountWorkerResult] = []
        expected_batches = len(batches)
        deadline = time.monotonic() + timeout_s
        for _ in range(expected_batches):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("ACCOUNT_WORKER_RESULT_TIMEOUT")
            try:
                kind, payload = self._out.get(timeout=remaining)
            except queue.Empty as exc:
                dead = [p.pid for p in self._processes if not p.is_alive()]
                if dead:
                    raise RuntimeError("ACCOUNT_WORKER_CRASHED") from exc
                raise RuntimeError("ACCOUNT_WORKER_RESULT_TIMEOUT") from exc
            if kind != "RESULTS":
                raise RuntimeError("ACCOUNT_WORKER_UNEXPECTED_MESSAGE")
            result.extend(payload)
        result.sort(key=lambda x: x.ordinal)
        if [x.ordinal for x in result] != sorted(c.ordinal for c in commands):
            raise RuntimeError("ACCOUNT_WORKER_RESULT_IDENTITY_MISMATCH")
        return result

    def terminate_worker(self, worker_id: int) -> None:
        self._processes[worker_id].terminate()
        self._processes[worker_id].join(timeout=2)

    def close(self, *, force: bool = False) -> None:
        for wid, p in enumerate(self._processes):
            if p.is_alive() and not force:
                try:
                    self._in[wid].put(None, timeout=0.2)
                except Exception:
                    pass
        if not force:
            deadline = time.monotonic() + 5.0
            for p in self._processes:
                p.join(timeout=max(0.0, deadline - time.monotonic()))
        for p in self._processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
        for q in self._in:
            try:
                q.close()
            except Exception:
                pass
        try:
            self._out.close()
        except Exception:
            pass

    def __enter__(self) -> "PersistentAccountWorkerPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
