from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

POLICY_SCHEMA = "CB16_R11_LONGTRAJ_ADAPTIVE_RUNTIME_POLICY_R0_V1"
GLOBAL_PHYSICS_WORKER_BUDGET = 8
MAX_CONCURRENT_EXPERIMENTS = 2
THREADS_PER_PHYSICS_PROCESS = 1


@dataclass(frozen=True)
class ExperimentAllocation:
    experiment_id: str
    physics_workers: int


@dataclass(frozen=True)
class RuntimeBatchPlan:
    schema: str
    mode: str
    allocations: tuple[ExperimentAllocation, ...]
    queued_experiment_ids: tuple[str, ...]
    total_physics_workers: int
    threads_per_physics_process: int

    @property
    def concurrent_experiments(self) -> int:
        return len(self.allocations)


def _normalize_ready_ids(ready_independent_experiment_ids: Sequence[str]) -> tuple[str, ...]:
    ids = tuple(str(x).strip() for x in ready_independent_experiment_ids)
    if any(not x for x in ids):
        raise ValueError("ADAPTIVE_RUNTIME_EMPTY_EXPERIMENT_ID")
    if len(set(ids)) != len(ids):
        raise ValueError("ADAPTIVE_RUNTIME_DUPLICATE_EXPERIMENT_ID")
    return ids


def plan_ready_independent_batch(
    ready_independent_experiment_ids: Sequence[str],
) -> RuntimeBatchPlan:
    """Plan one execution wave from an already-authorized independent ready queue.

    The caller owns dependency/independence authority. This function MUST NOT infer
    scientific independence. It only maps a queue already declared mutually
    independent and runnable onto the qualified Shanxi runtime topology.

    Policy:
      * 0 ready experiments -> idle
      * 1 ready experiment  -> 1 x 8 Physics processes
      * >=2 ready experiments -> 2 x 4 Physics processes, remainder stays queued
      * never schedule >2 concurrent experiments
      * never exceed 8 total Physics processes
      * every Physics process remains single-threaded
    """
    ids = _normalize_ready_ids(ready_independent_experiment_ids)

    if not ids:
        plan = RuntimeBatchPlan(
            schema=POLICY_SCHEMA,
            mode="IDLE",
            allocations=(),
            queued_experiment_ids=(),
            total_physics_workers=0,
            threads_per_physics_process=THREADS_PER_PHYSICS_PROCESS,
        )
    elif len(ids) == 1:
        plan = RuntimeBatchPlan(
            schema=POLICY_SCHEMA,
            mode="LATENCY_FIRST_SINGLE",
            allocations=(ExperimentAllocation(ids[0], 8),),
            queued_experiment_ids=(),
            total_physics_workers=8,
            threads_per_physics_process=THREADS_PER_PHYSICS_PROCESS,
        )
    else:
        plan = RuntimeBatchPlan(
            schema=POLICY_SCHEMA,
            mode="THROUGHPUT_FIRST_PAIR",
            allocations=(
                ExperimentAllocation(ids[0], 4),
                ExperimentAllocation(ids[1], 4),
            ),
            queued_experiment_ids=ids[2:],
            total_physics_workers=8,
            threads_per_physics_process=THREADS_PER_PHYSICS_PROCESS,
        )

    if plan.concurrent_experiments > MAX_CONCURRENT_EXPERIMENTS:
        raise RuntimeError("ADAPTIVE_RUNTIME_CONCURRENCY_OVER_2")
    if plan.total_physics_workers > GLOBAL_PHYSICS_WORKER_BUDGET:
        raise RuntimeError("ADAPTIVE_RUNTIME_PHYSICS_BUDGET_OVER_8")
    if any(x.physics_workers < 1 for x in plan.allocations):
        raise RuntimeError("ADAPTIVE_RUNTIME_INVALID_WORKER_ALLOCATION")
    if sum(x.physics_workers for x in plan.allocations) != plan.total_physics_workers:
        raise RuntimeError("ADAPTIVE_RUNTIME_WORKER_SUM_MISMATCH")
    return plan
