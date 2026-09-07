from __future__ import annotations

"""Read-only sidecar runtime observer for CB16 R10 campaigns.

This module never writes into the canonical campaign root. It samples host/process/GPU
telemetry and observes generation artifact transitions from a separate diagnostics root.
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

SCHEMA_SAMPLE = "CB16_R10_RUNTIME_SAMPLE_R0"
SCHEMA_SUMMARY = "CB16_R10_RUNTIME_DIAGNOSTIC_SUMMARY_R0"
SCHEMA_EVENT = "CB16_R10_RUNTIME_ARTIFACT_EVENT_R0"


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _jsonl_append(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, allow_nan=False) + "\n")


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _read_proc_stat_cpu() -> tuple[int, int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    vals = [int(x) for x in fields[1:]]
    total = sum(vals)
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    iowait = vals[4] if len(vals) > 4 else 0
    return total, idle, iowait


def _read_per_cpu() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
            continue
        if not fields[0][3:].isdigit():
            continue
        vals = [int(x) for x in fields[1:]]
        total = sum(vals)
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        out[fields[0]] = (total, idle)
    return out


def _read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        parts = v.strip().split()
        if parts and parts[0].isdigit():
            out[k] = int(parts[0]) * 1024
    return out


def _proc_tail(pid: int) -> list[str] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    pos = text.rfind(")")
    if pos < 0:
        return None
    return text[pos + 2 :].split()


def _proc_status(pid: int) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return out
    for line in lines:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        parts = v.strip().split()
        if parts and parts[0].isdigit():
            mult = 1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1
            out[k] = int(parts[0]) * mult
    return out


def _proc_io(pid: int) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        lines = Path(f"/proc/{pid}/io").read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return out
    for line in lines:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.isdigit():
            out[k] = int(v)
    return out


def _process_tree(root_pid: int) -> list[int]:
    ppid_map: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        tail = _proc_tail(pid)
        if not tail or len(tail) < 2:
            continue
        try:
            ppid = int(tail[1])
        except ValueError:
            continue
        ppid_map.setdefault(ppid, []).append(pid)
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(ppid_map.get(pid, ()))
    return sorted(seen)


def _tree_metrics(root_pid: int | None) -> dict[str, Any]:
    if root_pid is None:
        return {
            "root_pid": None, "processes": 0, "cpu_ticks": 0, "rss_bytes": 0,
            "vm_bytes": 0, "threads": 0, "read_bytes": 0, "write_bytes": 0,
        }
    pids = _process_tree(root_pid)
    ticks = rss = vm = threads = rb = wb = 0
    alive = 0
    for pid in pids:
        tail = _proc_tail(pid)
        if not tail or len(tail) < 13:
            continue
        alive += 1
        try:
            ticks += int(tail[11]) + int(tail[12])
        except (ValueError, IndexError):
            pass
        st = _proc_status(pid)
        rss += st.get("VmRSS", 0)
        vm += st.get("VmSize", 0)
        threads += st.get("Threads", 0)
        io = _proc_io(pid)
        rb += io.get("read_bytes", 0)
        wb += io.get("write_bytes", 0)
    return {
        "root_pid": root_pid, "processes": alive, "cpu_ticks": ticks, "rss_bytes": rss,
        "vm_bytes": vm, "threads": threads, "read_bytes": rb, "write_bytes": wb,
    }


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode("utf-8", "replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ""


def detect_r104_pid() -> int | None:
    candidates: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        if "run_r104_long_research.py" in _cmdline(pid):
            candidates.append(pid)
    return min(candidates) if candidates else None


def _gpu_sample() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"available": False, "gpus": []}
    if cp.returncode != 0:
        return {"available": False, "gpus": [], "error": cp.stderr.strip()[:200]}
    rows = []
    for line in cp.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 8:
            continue
        def num(x: str) -> float | None:
            try:
                return float(x)
            except ValueError:
                return None
        rows.append({
            "index": int(parts[0]), "name": parts[1], "gpu_util_pct": num(parts[2]),
            "memory_util_pct": num(parts[3]), "memory_used_mib": num(parts[4]),
            "memory_total_mib": num(parts[5]), "power_w": num(parts[6]), "temperature_c": num(parts[7]),
        })
    return {"available": bool(rows), "gpus": rows}


def _generation_number(path: Path) -> int | None:
    name = path.name
    if len(name) >= 2 and name[0] == "G" and name[1:].isdigit():
        return int(name[1:])
    return None


def _generation_state(run_root: Path) -> dict[str, Any]:
    root = run_root / "generations"
    rows: list[tuple[int, Path]] = []
    if root.is_dir():
        for p in root.iterdir():
            if not p.is_dir():
                continue
            n = _generation_number(p)
            if n is not None:
                rows.append((n, p))
    rows.sort()
    completed = [n for n, p in rows if (p / "GENERATION_RESULT.json").is_file()]
    active_rows = [(n, p) for n, p in rows if not (p / "GENERATION_RESULT.json").is_file()]
    if not active_rows:
        return {
            "completed_generations": len(completed),
            "highest_completed_generation": max(completed) if completed else None,
            "active_generation": None,
            "inferred_stage": "IDLE_OR_COMPLETE",
        }
    g, gd = active_rows[-1]
    train_receipt = gd / f"CHALLENGER_TRAINING_RECEIPT_G{g}.json"
    if (gd / "champion_after.pt").is_file():
        stage = "FINAL_GENERATION_PERSISTENCE"
    elif (gd / "challenger.pt").is_file():
        stage = "ADJUDICATION_OR_CHAMPION_PERSISTENCE"
    elif train_receipt.is_file() or (gd / f"CHALLENGER_TRAINED_RECOVERY_G{g}.pt").is_file():
        stage = "VALIDATION_OR_TOURNAMENT"
    elif (gd / "ON_POLICY_REAL_TRACE_RECEIPT.json").is_file():
        stage = "SNAPSHOT_OR_CHALLENGER_TRAINING"
    else:
        stage = "ON_POLICY_TRACE_OR_PREP"
    return {
        "completed_generations": len(completed),
        "highest_completed_generation": max(completed) if completed else None,
        "active_generation": g,
        "inferred_stage": stage,
    }


def _artifact_snapshot(run_root: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    gr = run_root / "generations"
    if not gr.is_dir():
        return out
    names = {"ON_POLICY_REAL_TRACE_RECEIPT.json", "GENERATION_RESULT.json", "challenger.pt", "champion_after.pt"}
    for gd in gr.glob("G*"):
        if not gd.is_dir():
            continue
        g = _generation_number(gd)
        if g is None:
            continue
        candidates = [p for p in gd.iterdir() if p.is_file() and (
            p.name in names
            or p.name == f"CHALLENGER_TRAINING_RECEIPT_G{g}.json"
            or p.name == f"SNAPSHOT_CONSUMPTION_G{g}.json"
        )]
        for p in candidates:
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            out[str(p.relative_to(run_root))] = (st.st_mtime_ns, st.st_size)
    return out


def _bottleneck(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"verdict": "INSUFFICIENT_SAMPLES"}
    tail = samples[-min(len(samples), 60):]
    host_cpu = [float(x["host"]["cpu_busy_pct"]) for x in tail if x["host"].get("cpu_busy_pct") is not None]
    iowait = [float(x["host"]["iowait_pct"]) for x in tail if x["host"].get("iowait_pct") is not None]
    proc_cpu = [float(x["process_tree"]["cpu_pct_one_core_100"]) for x in tail if x["process_tree"].get("cpu_pct_one_core_100") is not None]
    gpu = []
    for x in tail:
        for row in x.get("gpu", {}).get("gpus", []):
            if row.get("gpu_util_pct") is not None:
                gpu.append(float(row["gpu_util_pct"]))
    mem_avail = [int(x["host"]["mem_available_bytes"]) for x in tail if x["host"].get("mem_available_bytes") is not None]
    mem_total = [int(x["host"]["mem_total_bytes"]) for x in tail if x["host"].get("mem_total_bytes") is not None]

    m_cpu = median(host_cpu) if host_cpu else None
    m_iow = median(iowait) if iowait else None
    m_proc = median(proc_cpu) if proc_cpu else None
    m_gpu = median(gpu) if gpu else None
    avail_frac = (median(mem_avail) / median(mem_total)) if mem_avail and mem_total and median(mem_total) > 0 else None

    if avail_frac is not None and avail_frac < 0.08:
        verdict = "MEMORY_PRESSURE"
    elif m_iow is not None and m_iow >= 15.0:
        verdict = "IO_WAIT_BOUND"
    elif m_gpu is not None and m_gpu >= 75.0:
        verdict = "GPU_BUSY"
    elif m_cpu is not None and m_cpu >= 75.0 and (m_gpu is None or m_gpu < 45.0):
        verdict = "CPU_BOUND_OR_CPU_FEED_BOUND"
    elif m_cpu is not None and m_cpu < 45.0 and (m_gpu is None or m_gpu < 45.0):
        verdict = "SERIAL_BARRIER_OR_WAIT_BOUND"
    else:
        verdict = "MIXED_OR_NO_CLEAR_BOTTLENECK"
    return {
        "verdict": verdict,
        "median_host_cpu_busy_pct": m_cpu,
        "median_process_tree_cpu_pct_one_core_100": m_proc,
        "median_iowait_pct": m_iow,
        "median_gpu_util_pct": m_gpu,
        "median_mem_available_fraction": avail_frac,
        "interpretation": "DIAGNOSTIC_ONLY_NOT_PIPELINE_PASS_DRIVER",
    }


@dataclass
class RuntimeObserver:
    run_root: Path
    out_dir: Path
    interval_s: float = 5.0
    pid: int | None = None

    def __post_init__(self) -> None:
        self.run_root = self.run_root.resolve()
        self.out_dir = self.out_dir.resolve()
        if _inside(self.out_dir, self.run_root):
            raise ValueError("DIAGNOSTIC_OUTPUT_MUST_BE_OUTSIDE_CANONICAL_RUN_ROOT")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, duration_s: float | None = None, stop_when_complete: bool = False) -> dict[str, Any]:
        samples_path = self.out_dir / "runtime_samples.jsonl"
        events_path = self.out_dir / "artifact_events.jsonl"
        summary_path = self.out_dir / "RUNTIME_DIAGNOSTIC_SUMMARY.json"
        started_wall = time.time()
        started_mono = time.monotonic()
        baseline_artifacts = _artifact_snapshot(self.run_root)
        known = dict(baseline_artifacts)
        local_samples: list[dict[str, Any]] = []
        previous_sys: tuple[int, int, int] | None = None
        previous_per_cpu: dict[str, tuple[int, int]] | None = None
        previous_tree: dict[str, Any] | None = None
        previous_mono: float | None = None
        samples_written = 0
        clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

        while True:
            now_wall = time.time()
            now_mono = time.monotonic()
            if self.pid is None or not Path(f"/proc/{self.pid}").exists():
                self.pid = detect_r104_pid()
            sys_now = _read_proc_stat_cpu()
            per_cpu_now = _read_per_cpu()
            tree = _tree_metrics(self.pid)
            host_cpu_pct = iowait_pct = None
            proc_cpu_pct = read_bps = write_bps = None
            if previous_sys is not None:
                dticks = max(1, sys_now[0] - previous_sys[0])
                didle = max(0, sys_now[1] - previous_sys[1])
                diow = max(0, sys_now[2] - previous_sys[2])
                host_cpu_pct = max(0.0, min(100.0, 100.0 * (dticks - didle) / dticks))
                iowait_pct = max(0.0, min(100.0, 100.0 * diow / dticks))
            if previous_tree is not None and previous_mono is not None:
                dt = max(1e-6, now_mono - previous_mono)
                proc_cpu_pct = max(0.0, 100.0 * (tree["cpu_ticks"] - previous_tree["cpu_ticks"]) / clk_tck / dt)
                read_bps = max(0.0, (tree["read_bytes"] - previous_tree["read_bytes"]) / dt)
                write_bps = max(0.0, (tree["write_bytes"] - previous_tree["write_bytes"]) / dt)

            per_core_busy: dict[str, float] = {}
            if previous_per_cpu is not None:
                for cpu, (total_now, idle_now) in per_cpu_now.items():
                    prev = previous_per_cpu.get(cpu)
                    if prev is None:
                        continue
                    dticks = max(1, total_now - prev[0])
                    didle = max(0, idle_now - prev[1])
                    per_core_busy[cpu] = max(0.0, min(100.0, 100.0 * (dticks - didle) / dticks))

            mem = _read_meminfo()
            gen = _generation_state(self.run_root)
            sample = {
                "schema": SCHEMA_SAMPLE,
                "wall_time_unix": now_wall,
                "elapsed_s": now_mono - started_mono,
                "host": {
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                    "cpu_busy_pct": host_cpu_pct,
                    "per_core_busy_pct": per_core_busy,
                    "iowait_pct": iowait_pct,
                    "mem_total_bytes": mem.get("MemTotal"),
                    "mem_available_bytes": mem.get("MemAvailable"),
                    "swap_total_bytes": mem.get("SwapTotal"),
                    "swap_free_bytes": mem.get("SwapFree"),
                },
                "process_tree": {
                    **tree,
                    "cpu_pct_one_core_100": proc_cpu_pct,
                    "read_bytes_per_s": read_bps,
                    "write_bytes_per_s": write_bps,
                },
                "gpu": _gpu_sample(),
                "generation": gen,
                "final_holdout_2025_09_accessed": False,
            }
            _jsonl_append(samples_path, sample)
            samples_written += 1
            local_samples.append(sample)
            if len(local_samples) > 600:
                local_samples = local_samples[-600:]

            current = _artifact_snapshot(self.run_root)
            for rel, meta in sorted(current.items()):
                if rel not in known or known[rel] != meta:
                    _jsonl_append(events_path, {
                        "schema": SCHEMA_EVENT, "observed_wall_time_unix": now_wall,
                        "elapsed_s": now_mono - started_mono, "relative_path": rel,
                        "mtime_ns": meta[0], "size_bytes": meta[1],
                        "preexisting_at_observer_start": False,
                    })
            known = current

            summary = {
                "schema": SCHEMA_SUMMARY,
                "status": "RUNNING",
                "run_root": str(self.run_root),
                "diagnostic_out": str(self.out_dir),
                "observer_started_unix": started_wall,
                "samples_written": samples_written,
                "preexisting_artifact_count": len(baseline_artifacts),
                "current_generation_state": gen,
                "current_pid": self.pid,
                "bottleneck": _bottleneck(local_samples),
                "scientific_semantics_changed": False,
                "writes_to_canonical_run_root": False,
                "final_holdout_2025_09_accessed": False,
            }
            _atomic_json(summary_path, summary)

            complete = gen["active_generation"] is None and gen["completed_generations"] >= 100
            if stop_when_complete and complete:
                break
            if duration_s is not None and now_mono - started_mono >= duration_s:
                break
            previous_sys, previous_per_cpu, previous_tree, previous_mono = sys_now, per_cpu_now, tree, now_mono
            time.sleep(max(0.2, float(self.interval_s)))

        summary["status"] = "COMPLETE"
        summary["observer_completed_unix"] = time.time()
        summary["bottleneck"] = _bottleneck(local_samples)
        _atomic_json(summary_path, summary)
        return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only CB16 R10 runtime sidecar profiler")
    ap.add_argument("--run-root", default="/data/cb16_hdd/cb16_runtime/R10_4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--stop-when-complete", action="store_true")
    a = ap.parse_args()
    obs = RuntimeObserver(Path(a.run_root), Path(a.out), interval_s=a.interval, pid=a.pid)
    result = obs.run(duration_s=a.duration, stop_when_complete=a.stop_when_complete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
