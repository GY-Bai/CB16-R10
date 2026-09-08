from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "stage3_e1_endurance_r11", ROOT / "cb16_local_opt" / "stage3_e1_endurance_r11.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
assert spec.loader is not None
spec.loader.exec_module(m)


def sample(t, *, cycles=0, traces=0, steps=0, examples=0, rss=1000, vram=2000,
           threads=10, procs=8, fds=20, q=0, major=0, swap=0, fsync=1.0, barrier=1.0):
    return m.ResourceSample(
        monotonic_seconds=float(t),
        counters=m.CounterSnapshot(
            cycles=cycles, traces=traces, optimizer_steps=steps, examples=examples,
            io_requests=cycles, io_bytes_logical=cycles * 100, worker_deaths=0, restart_count=0,
        ),
        rss_bytes=rss, vram_bytes=vram, thread_count=threads, process_count=procs,
        fd_count=fds, cpu_utilization_pct=50.0, gpu_utilization_pct=20.0,
        major_faults=major, swap_in_bytes=swap, swap_out_bytes=swap,
        queue_depth=q, queue_capacity=8, hdd_queue_depth=0,
        ssd_read_bytes=0, ssd_write_bytes=0, hdd_read_bytes=0, hdd_write_bytes=0,
        fsync_latency_ms=fsync, barrier_latency_ms=barrier,
        trace_processes_alive=8, storage_writer_alive=True,
    )


def stable_samples(n=10):
    return [sample(i * 5, cycles=i, traces=i * 96, steps=i * 4, examples=i * 1024,
                   rss=1000 + i, vram=2000 + i) for i in range(n)]


def test_stage2_runtime_defaults_are_frozen():
    m.E1Config(duration_seconds=180).validate()
    try:
        m.E1Config(duration_seconds=180, trace_workers=7).validate()
    except ValueError as exc:
        assert "R11_E1_STAGE2_DEFAULT_DRIFT" in str(exc)
    else:
        raise AssertionError("trace worker drift accepted")


def test_stable_rolling_windows_pass():
    r = m.analyze_endurance(stable_samples(), m.E1Config(duration_seconds=180))
    assert r["verdict"] == "PASS", r
    assert r["rolling_window_samples"] == 3
    assert r["throughput"]["traces_per_sec"]["median"] > 0


def test_throughput_collapse_fails():
    rows = []
    traces = steps = examples = cycles = 0
    for i, inc in enumerate([100,100,100,100,100,100,20,20,20,20,20,20]):
        traces += inc; steps += max(1, inc // 20); examples += max(1, inc) * 8; cycles += 1
        rows.append(sample(i * 5, cycles=cycles, traces=traces, steps=steps, examples=examples))
    r = m.analyze_endurance(rows, m.E1Config(duration_seconds=180))
    assert m.E1Failure.THROUGHPUT_COLLAPSE.value in r["failures"], r


def test_resource_growth_is_two_part_slope_and_absolute_gate():
    cfg = m.E1Config(duration_seconds=180, rss_growth_limit_bytes=100,
                     rss_slope_limit_bytes_per_min=100)
    rows = [m.ResourceSample(**{**s.__dict__, "rss_bytes": 1000 + i * 1000})
            for i, s in enumerate(stable_samples())]
    r = m.analyze_endurance(rows, cfg)
    assert m.E1Failure.RSS_UNBOUNDED_GROWTH.value in r["failures"]


def test_dead_process_and_queue_runaway_fail_closed():
    rows = stable_samples()
    rows[-1] = m.ResourceSample(**{**rows[-1].__dict__, "trace_processes_alive": 7, "queue_depth": 8})
    rows[-2] = m.ResourceSample(**{**rows[-2].__dict__, "queue_depth": 8})
    rows[-3] = m.ResourceSample(**{**rows[-3].__dict__, "queue_depth": 8})
    r = m.analyze_endurance(rows, m.E1Config(duration_seconds=180))
    assert m.E1Failure.PROCESS_DEATH.value in r["failures"]
    assert m.E1Failure.QUEUE_RUNAWAY.value in r["failures"]


def test_dual_verdict_requires_correctness_and_endurance():
    checks = {k: True for k in (
        "semantic_freeze_pass", "teacher_identity_unchanged", "h72_identity_unchanged",
        "fp32_amp_false", "gradient_authority_unchanged", "frozen_authority_unchanged",
        "final_holdout_untouched", "replay_created_zero",
    )}
    report = m.make_report(config=m.E1Config(duration_seconds=180), samples=stable_samples(),
                           correctness_checks=checks, identities={"x":"y"})
    assert report["status"] == "PASS"
    assert report["correctness_identity"]["verdict"] == "PASS"
    assert report["runtime_endurance"]["verdict"] == "PASS"
    assert report["readiness"]["ready_for_30m"] is True
    bad = dict(checks); bad["replay_created_zero"] = False
    report = m.make_report(config=m.E1Config(duration_seconds=180), samples=stable_samples(),
                           correctness_checks=bad, identities={})
    assert report["status"] == "FAIL"
    assert m.E1Failure.REPLAY_EVIDENCE_VIOLATION.value in report["correctness_identity"]["failures"]


def test_report_explicitly_disclaims_scientific_learning():
    checks = {k: True for k in (
        "semantic_freeze_pass", "teacher_identity_unchanged", "h72_identity_unchanged",
        "fp32_amp_false", "gradient_authority_unchanged", "frozen_authority_unchanged",
        "final_holdout_untouched", "replay_created_zero",
    )}
    r = m.make_report(config=m.E1Config(duration_seconds=180), samples=stable_samples(),
                      correctness_checks=checks, identities={})
    assert r["correctness_identity"]["new_scientific_verdict"] is False
    assert "NOT_SCIENTIFIC_EVIDENCE" in r["correctness_identity"]["repeated_replay_role"]
    assert r["runtime_lifecycle"]["tournament_or_lineage_exercised"] is False


def test_workflow_is_separate_and_uses_exact_runner():
    wf = (ROOT / ".github/workflows/cb16-r11-stage3-e1-data-plane-soak.yml").read_text()
    assert "cb16-r11-stage2-burst-qualification.yml" not in wf
    assert "runs-on: [self-hosted, shanxi, cb16-wss-qualification]" in wf
    assert "cb16-r10-canonical" not in wf
    assert "runs-on: [self-hosted, shanxi]" not in wf
    assert "smoke_3m" in wf and "soak_30m" in wf
    assert "2h" not in wf and "6h" not in wf
    assert "OMP_NUM_THREADS: '1'" in wf
    assert "MKL_NUM_THREADS: '1'" in wf
    assert "OPENBLAS_NUM_THREADS: '1'" in wf


def test_driver_constructs_long_lived_objects_outside_measured_loop():
    src = (ROOT / "scripts/run_r11_stage3_e1_data_plane_soak.py").read_text()
    loop = src.index("while time.monotonic() < deadline:")
    for needle in ("ForkProcessTraceRuntimeR11(", "PersistentTeacherReplayR11(",
                   "TrainingRuntimeR11(", "IOThroughputRuntimeR11("):
        assert src.index(needle) < loop
        assert needle not in src[loop:]
    assert "decide_tournament" not in src
    assert "release_next_generation" not in src
    assert "train_challenger(" not in src
