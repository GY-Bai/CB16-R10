from __future__ import annotations

import pytest

from cb16_local_opt.r11_adaptive_runtime_policy import (
    GLOBAL_PHYSICS_WORKER_BUDGET,
    MAX_CONCURRENT_EXPERIMENTS,
    THREADS_PER_PHYSICS_PROCESS,
    plan_ready_independent_batch,
)


def test_idle_plan() -> None:
    plan = plan_ready_independent_batch([])
    assert plan.mode == "IDLE"
    assert plan.allocations == ()
    assert plan.queued_experiment_ids == ()
    assert plan.total_physics_workers == 0


def test_single_ready_uses_all_eight_workers() -> None:
    plan = plan_ready_independent_batch(["A"])
    assert plan.mode == "LATENCY_FIRST_SINGLE"
    assert [(x.experiment_id, x.physics_workers) for x in plan.allocations] == [("A", 8)]
    assert plan.queued_experiment_ids == ()
    assert plan.total_physics_workers == GLOBAL_PHYSICS_WORKER_BUDGET == 8
    assert plan.concurrent_experiments == 1
    assert plan.threads_per_physics_process == THREADS_PER_PHYSICS_PROCESS == 1


def test_two_ready_split_four_four() -> None:
    plan = plan_ready_independent_batch(["A", "B"])
    assert plan.mode == "THROUGHPUT_FIRST_PAIR"
    assert [(x.experiment_id, x.physics_workers) for x in plan.allocations] == [("A", 4), ("B", 4)]
    assert plan.queued_experiment_ids == ()
    assert plan.total_physics_workers == 8
    assert plan.concurrent_experiments == MAX_CONCURRENT_EXPERIMENTS == 2


def test_three_or_more_never_schedule_three_way() -> None:
    plan = plan_ready_independent_batch(["A", "B", "C", "D"])
    assert [(x.experiment_id, x.physics_workers) for x in plan.allocations] == [("A", 4), ("B", 4)]
    assert plan.queued_experiment_ids == ("C", "D")
    assert plan.concurrent_experiments == 2
    assert plan.total_physics_workers == 8


def test_fifo_order_is_preserved() -> None:
    plan = plan_ready_independent_batch(["E3", "E1", "E2"])
    assert [x.experiment_id for x in plan.allocations] == ["E3", "E1"]
    assert plan.queued_experiment_ids == ("E2",)


def test_duplicate_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="ADAPTIVE_RUNTIME_DUPLICATE_EXPERIMENT_ID"):
        plan_ready_independent_batch(["A", "A"])


def test_empty_id_fails_closed() -> None:
    with pytest.raises(ValueError, match="ADAPTIVE_RUNTIME_EMPTY_EXPERIMENT_ID"):
        plan_ready_independent_batch(["A", " "])
