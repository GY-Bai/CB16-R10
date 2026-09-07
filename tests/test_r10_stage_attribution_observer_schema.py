from __future__ import annotations

from cb16_diagnostics.stage_attribution import build_stage_attribution


def _observer_sample(ts: float, *, generation: int = 59, stage: str = "SNAPSHOT_OR_CHALLENGER_TRAINING"):
    return {
        "schema": "CB16_R10_RUNTIME_SAMPLE_R0",
        "wall_time_unix": float(ts),
        "host": {
            "cpu_busy_pct": 30.0,
            "iowait_pct": 2.0,
        },
        "process_tree": {
            "cpu_pct_one_core_100": 120.0,
            "rss_bytes": 123456,
            "read_bytes": 1000,
            "write_bytes": 2000,
            "read_bytes_per_s": 1024.0,
            "write_bytes_per_s": 2048.0,
        },
        "gpu": {
            "available": True,
            "gpus": [
                {
                    "index": 0,
                    "name": "GTX 1060",
                    "gpu_util_pct": 10.0,
                    "memory_used_mib": 512.0,
                    "power_w": 40.0,
                }
            ],
        },
        "generation": {
            "active_generation": generation,
            "inferred_stage": stage,
        },
        "final_holdout_2025_09_accessed": False,
    }


def test_current_runtime_observer_schema_is_consumed_without_discard():
    rows = [_observer_sample(1000 + 5 * i) for i in range(20)]
    result = build_stage_attribution(rows)

    assert result["status"] == "PASS"
    assert result["input_samples_total"] == 20
    assert result["sample_count"] == 20
    assert result["discarded_sample_count"] == 0
    assert result["generation_count_observed"] == 1

    stage = result["stages"][0]
    assert stage["generation"] == 59
    assert stage["stage"] == "SNAPSHOT_OR_CHALLENGER_TRAINING"
    assert stage["mean_host_cpu_percent"] == 30.0
    assert stage["mean_process_cpu_percent"] == 120.0
    assert stage["mean_iowait_percent"] == 2.0
    assert stage["mean_gpu_util_percent"] == 10.0
    assert stage["peak_gpu_mem_used_bytes"] == 512.0 * (1 << 20)
    assert stage["mean_gpu_power_watts"] == 40.0
    assert stage["read_bytes_per_second"] == 1024.0
    assert stage["write_bytes_per_second"] == 2048.0


def test_missing_timestamp_is_explicitly_counted_not_silently_lost():
    rows = [_observer_sample(1000), _observer_sample(1005)]
    rows.append({"schema": "CB16_R10_RUNTIME_SAMPLE_R0", "generation": {"active_generation": 59}})
    result = build_stage_attribution(rows)

    assert result["input_samples_total"] == 3
    assert result["sample_count"] == 2
    assert result["discarded_sample_count"] == 1
    assert result["discarded_no_timestamp"] == 1


def test_zero_valid_samples_is_not_reported_as_pass():
    result = build_stage_attribution([
        {"schema": "CB16_R10_RUNTIME_SAMPLE_R0", "generation": {"active_generation": 59}}
    ])
    assert result["status"] == "NO_VALID_SAMPLES"
    assert result["sample_count"] == 0
    assert result["discarded_sample_count"] == 1
