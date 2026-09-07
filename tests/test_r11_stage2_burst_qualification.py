from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_r11_stage2_burst_qualification import (
    DEFAULT_THRESHOLDS,
    EXACT_INTEGRATION_BRANCH,
    TELEMETRY_SCHEMA,
    evaluate_thresholds,
    load_trigger,
    summarize_runtime,
)


def _trigger() -> dict:
    return {
        "schema": "CB16_R11_STAGE2_BURST_TRIGGER_V1",
        "armed": True,
        "qualification_branch": EXACT_INTEGRATION_BRANCH,
        "warmup_seconds": 30,
        "measured_seconds": 420,
        "hard_timeout_minutes": 15,
        "teacher_workers": 8,
        "trace_workers": 8,
        "queue_depth": 8,
        "buffer_mib": 64,
        "workload_argv": ["{python}", "scripts/run_r11_stage2_integration_burst.py"],
        "forbidden": {
            "fresh_market_data_download": True,
            "final_holdout_open": True,
            "oci_compute": True,
            "daemon_mode": True,
            "long_endurance": True,
        },
    }


def test_trigger_exact_branch_and_duration(tmp_path: Path) -> None:
    p = tmp_path / "trigger.json"
    p.write_text(json.dumps(_trigger()))
    x = load_trigger(p)
    assert x["qualification_branch"] == EXACT_INTEGRATION_BRANCH
    assert x["measured_seconds"] == 420


def test_trigger_rejects_wildcard_branch(tmp_path: Path) -> None:
    x = _trigger()
    x["qualification_branch"] = "ai/r11-stage2-*"
    p = tmp_path / "trigger.json"
    p.write_text(json.dumps(x))
    with pytest.raises(RuntimeError, match="BRANCH_ALLOWLIST_DRIFT"):
        load_trigger(p)


def test_trigger_rejects_unapproved_entrypoint(tmp_path: Path) -> None:
    x = _trigger()
    x["workload_argv"] = ["{python}", "scripts/anything_else.py"]
    p = tmp_path / "trigger.json"
    p.write_text(json.dumps(x))
    with pytest.raises(RuntimeError, match="ENTRYPOINT_NOT_ALLOWLISTED"):
        load_trigger(p)


def test_runtime_counter_rates_and_stalls() -> None:
    rows = [
        {
            "schema": TELEMETRY_SCHEMA,
            "monotonic_seconds": 100.0,
            "counters": {
                "generations_committed": 10,
                "teacher_evidence": 100,
                "traces": 20,
                "training_steps": 30,
                "fp32_training_examples": 3000,
                "barrier_block_seconds": 5.0,
                "fsync_stall_seconds": 1.0,
                "ssd_metadata_ops": 1000,
                "hdd_read_bytes": 10000,
                "hdd_write_bytes": 20000,
            },
            "gauges": {
                "teacher_worker_utilization_pct": 80.0,
                "trace_worker_utilization_pct": 70.0,
                "fsync_latency_p95_ms": 12.0,
                "pipeline_queue_depth": 4.0,
            },
        },
        {
            "schema": TELEMETRY_SCHEMA,
            "monotonic_seconds": 160.0,
            "counters": {
                "generations_committed": 12,
                "teacher_evidence": 220,
                "traces": 80,
                "training_steps": 150,
                "fp32_training_examples": 15000,
                "barrier_block_seconds": 8.0,
                "fsync_stall_seconds": 2.0,
                "ssd_metadata_ops": 1600,
                "hdd_read_bytes": 70000,
                "hdd_write_bytes": 140000,
            },
            "gauges": {
                "teacher_worker_utilization_pct": 90.0,
                "trace_worker_utilization_pct": 80.0,
                "fsync_latency_p95_ms": 20.0,
                "pipeline_queue_depth": 6.0,
            },
        },
    ]
    r = summarize_runtime(rows, 60.0)
    assert r["generations_per_min"] == pytest.approx(2.0)
    assert r["teacher_evidence_per_sec"] == pytest.approx(2.0)
    assert r["traces_per_sec"] == pytest.approx(1.0)
    assert r["training_steps_per_sec"] == pytest.approx(2.0)
    assert r["fp32_training_examples_per_sec"] == pytest.approx(200.0)
    assert r["barrier_block_ratio"] == pytest.approx(0.05)
    assert r["fsync_stall_ratio"] == pytest.approx(1.0 / 60.0)
    assert r["teacher_worker_utilization_avg_pct"] == pytest.approx(85.0)
    assert r["trace_worker_utilization_avg_pct"] == pytest.approx(75.0)


def test_acceptance_thresholds_and_90pct_baseline() -> None:
    host = {
        "cpu": {"avg_utilization_pct": 75.0},
        "gpu": {"avg_utilization_pct": 70.0, "idle_ratio": 0.10},
        "pipeline": {"sampled_cpu_gpu_overlap_pct": 60.0},
        "memory": {
            "available_ram_min_bytes": 4 * 1024**3,
            "swap_out_pages": 0,
            "major_faults_process_tree": 0,
        },
        "io": {"hdd": {"avg_queue_depth": 0.5}},
    }
    runtime = {
        "generations_per_min": 2.0,
        "teacher_evidence_per_sec": 100.0,
        "traces_per_sec": 10.0,
        "training_steps_per_sec": 20.0,
        "fp32_training_examples_per_sec": 1000.0,
        "barrier_block_ratio": 0.02,
        "fsync_stall_ratio": 0.01,
    }
    assert evaluate_thresholds(host, runtime, DEFAULT_THRESHOLDS, None) == []
    baseline = {k: runtime[k] for k in (
        "generations_per_min",
        "teacher_evidence_per_sec",
        "traces_per_sec",
        "training_steps_per_sec",
        "fp32_training_examples_per_sec",
    )}
    runtime["traces_per_sec"] = 8.0
    failures = evaluate_thresholds(host, runtime, DEFAULT_THRESHOLDS, baseline)
    assert any("traces_per_sec_baseline_ratio" in x for x in failures)
