from pathlib import Path

from cb16_local_opt.r102_parallel_runtime import ram_action_r82
from cb16_local_opt.r102_runtime_authority import load_r102_runtime_parallelism


def test_r83_runtime_overlay_starts_eight_workers_and_preserves_cache_identity():
    rp = load_r102_runtime_parallelism(Path(__file__).resolve().parents[1], live_environment_check=False)
    assert rp.performance_overlay == "R8_3_8W_RAM_ADAPTIVE"
    assert rp.h72_workers == 8
    assert rp.h72_threads_per_worker == 1
    assert rp.h72_max_in_flight >= 8
    assert rp.h72_minimum_workers == 1
    assert rp.ram_backpressure_high == 0.85
    assert rp.ram_hard_stop == 0.92
    assert rp.cache_hit_first is True
    assert rp.runtime_scheduling_identity_is_not_scientific_cache_identity is True
    assert rp.teacher_workers == 8
    assert rp.teacher_threads_per_worker == 1


def test_ram_monitor_retires_then_kills_one_worker_at_thresholds():
    assert ram_action_r82(0.50, 8, high=0.85, hard=0.92) == "NONE"
    assert ram_action_r82(0.85, 8, high=0.85, hard=0.92) == "RETIRE_ONE"
    assert ram_action_r82(0.91, 4, high=0.85, hard=0.92) == "RETIRE_ONE"
    assert ram_action_r82(0.92, 3, high=0.85, hard=0.92) == "KILL_ONE"
    assert ram_action_r82(0.99, 1, high=0.85, hard=0.92) == "PAUSE"


def test_runtime_worker_change_is_not_a_scientific_knob():
    rp = load_r102_runtime_parallelism(Path(__file__).resolve().parents[1], live_environment_check=False).as_dict()
    assert rp["scientific_semantics_changed"] is False
    assert rp["runtime_scheduling_identity_is_not_scientific_cache_identity"] is True
    assert rp["cache_hit_first"] is True
    assert rp["gpu_qualified_train_batch_ceiling"] == 8192
