from __future__ import annotations

from cb16_diagnostics.stage_attribution import build_stage_attribution


def _sample(ts, generation, stage, *, cpu=20.0, host_cpu=30.0, iowait=1.0, gpu=10.0, read=0, write=0):
    return {
        "timestamp_unix": float(ts),
        "current_generation": generation,
        "inferred_stage": stage,
        "process_tree": {
            "cpu_percent": float(cpu),
            "rss_bytes": 1000,
            "read_bytes": int(read),
            "write_bytes": int(write),
        },
        "host": {
            "cpu_percent": float(host_cpu),
            "iowait_percent": float(iowait),
        },
        "gpu": {
            "utilization_gpu_percent": float(gpu),
            "memory_used_bytes": 2000,
            "power_watts": 50.0,
        },
    }


def test_short_window_is_not_overclaimed():
    rows = [
        _sample(i * 5, 59, "SNAPSHOT_OR_CHALLENGER_TRAINING", host_cpu=20, gpu=5)
        for i in range(4)
    ]
    result = build_stage_attribution(rows)
    stage = result["stages"][0]
    assert stage["bottleneck"]["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert stage["bottleneck"]["confidence"] == "LOW"
    assert result["safety"]["writes_to_canonical_run_root"] is False
    assert result["safety"]["scientific_semantics_changed"] is False
    assert result["safety"]["final_holdout_2025_09_accessed"] is False


def test_wait_bound_requires_longer_evidence():
    rows = [
        _sample(i * 5, 60, "SNAPSHOT_OR_CHALLENGER_TRAINING", host_cpu=25, gpu=10)
        for i in range(20)
    ]
    result = build_stage_attribution(rows)
    stage = result["stages"][0]
    assert stage["observed_seconds"] == 95.0
    assert stage["bottleneck"]["verdict"] == "SERIAL_BARRIER_OR_WAIT_BOUND"
    assert stage["bottleneck"]["confidence"] == "HIGH"


def test_gpu_bound_and_generation_rollup():
    rows = []
    for i in range(8):
        rows.append(_sample(i * 5, 61, "CHALLENGER_TRAINING", host_cpu=45, gpu=90, read=i * 1000, write=i * 2000))
    for i in range(8, 15):
        rows.append(_sample(i * 5, 61, "TOURNAMENT_OR_PERSIST", host_cpu=15, gpu=5, read=i * 1000, write=i * 2000))
    result = build_stage_attribution(rows)
    training = next(x for x in result["stages"] if x["stage"] == "CHALLENGER_TRAINING")
    assert training["bottleneck"]["verdict"] == "GPU_COMPUTE_BOUND"
    assert result["generation_count_observed"] == 1
    assert result["generations"][0]["generation"] == 61
    assert result["generations"][0]["dominant_observed_stage"] in {"CHALLENGER_TRAINING", "TOURNAMENT_OR_PERSIST"}
