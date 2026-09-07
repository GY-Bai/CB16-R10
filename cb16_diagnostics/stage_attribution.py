from __future__ import annotations

"""Aggregate read-only CB16 R10 runtime samples into stage/generation attribution.

Consumes diagnostics sidecar JSONL only. It never writes into the canonical
campaign root and never opens FINAL holdout data.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

SCHEMA = "CB16_R10_STAGE_ATTRIBUTION_R0"
SAFETY = {
    "writes_to_canonical_run_root": False,
    "scientific_semantics_changed": False,
    "final_holdout_2025_09_accessed": False,
    "status_driving": False,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"INVALID_RUNTIME_JSONL:{path}:{i}:{exc}") from exc
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _num(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def _get(row: dict[str, Any], key: str) -> Any:
    cur: Any = row
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _first_num(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        v = _num(_get(row, key))
        if v is not None:
            return v
    return None


def _first_str(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        cur = _get(row, key)
        if cur is not None:
            return str(cur)
    return None


def _generation(row: dict[str, Any]) -> int | None:
    # Current RuntimeObserver R0 schema first, legacy/synthetic aliases second.
    for key in (
        "generation.active_generation",
        "current_generation_state.active_generation",
        "current_generation_state.generation",
        "current_generation",
    ):
        v = _first_num(row, key)
        if v is not None:
            return int(v)
    top = row.get("generation")
    v = _num(top)
    return int(v) if v is not None else None


def _stage(row: dict[str, Any]) -> str:
    return (
        _first_str(
            row,
            "generation.inferred_stage",
            "current_generation_state.inferred_stage",
            "inferred_stage",
            "stage",
        )
        or "UNKNOWN"
    )


def _timestamp(row: dict[str, Any]) -> float | None:
    return _first_num(
        row,
        "wall_time_unix",
        "timestamp_unix",
        "ts_unix",
        "sample_time_unix",
        "timestamp",
    )


def _gpu_metrics(row: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    # Legacy/synthetic scalar layout.
    util = _first_num(
        row,
        "gpu.utilization_gpu_percent",
        "gpu.gpu_util_percent",
        "gpu_util_percent",
    )
    mem = _first_num(row, "gpu.memory_used_bytes", "gpu_mem_used_bytes")
    power = _first_num(row, "gpu.power_watts", "gpu_power_watts")
    if util is not None or mem is not None or power is not None:
        return util, mem, power

    # Current RuntimeObserver R0 emits gpu.gpus[] in nvidia-smi units.
    gpu = row.get("gpu")
    rows = gpu.get("gpus") if isinstance(gpu, dict) else None
    if not isinstance(rows, list):
        return None, None, None

    utils: list[float] = []
    mem_mib: list[float] = []
    powers: list[float] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        v = _num(item.get("gpu_util_pct"))
        if v is not None:
            utils.append(v)
        v = _num(item.get("memory_used_mib"))
        if v is not None:
            mem_mib.append(v)
        v = _num(item.get("power_w"))
        if v is not None:
            powers.append(v)

    # For utilization, max is the useful "is any accelerator saturated?" signal.
    # Memory/power are aggregated across devices.
    return (
        max(utils) if utils else None,
        sum(mem_mib) * (1 << 20) if mem_mib else None,
        sum(powers) if powers else None,
    )


def _metrics(row: dict[str, Any]) -> dict[str, float | None]:
    gpu_util, gpu_mem, gpu_power = _gpu_metrics(row)
    return {
        "cpu_percent": _first_num(
            row,
            "process_tree.cpu_pct_one_core_100",
            "process_tree.cpu_percent",
            "process_cpu_percent",
            "cpu_percent",
        ),
        "host_cpu_percent": _first_num(
            row,
            "host.cpu_busy_pct",
            "host.cpu_percent",
            "host_cpu_percent",
        ),
        "iowait_percent": _first_num(
            row,
            "host.iowait_pct",
            "host.iowait_percent",
            "iowait_percent",
        ),
        "rss_bytes": _first_num(
            row,
            "process_tree.rss_bytes",
            "process_rss_bytes",
            "rss_bytes",
        ),
        "read_bytes": _first_num(
            row,
            "process_tree.read_bytes",
            "process_read_bytes",
            "read_bytes",
        ),
        "write_bytes": _first_num(
            row,
            "process_tree.write_bytes",
            "process_write_bytes",
            "write_bytes",
        ),
        "read_bytes_per_second_direct": _first_num(
            row,
            "process_tree.read_bytes_per_s",
            "process_tree.read_bytes_per_second",
            "read_bytes_per_second",
        ),
        "write_bytes_per_second_direct": _first_num(
            row,
            "process_tree.write_bytes_per_s",
            "process_tree.write_bytes_per_second",
            "write_bytes_per_second",
        ),
        "gpu_util_percent": gpu_util,
        "gpu_mem_used_bytes": gpu_mem,
        "gpu_power_watts": gpu_power,
    }


def _avg(values: Iterable[float | None]) -> float | None:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return mean(xs) if xs else None


def _max(values: Iterable[float | None]) -> float | None:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return max(xs) if xs else None


def _delta(first: float | None, last: float | None) -> float | None:
    if first is None or last is None:
        return None
    # Process-tree membership can change and reset aggregate counters. In that
    # case do not fabricate negative throughput; direct rates are preferred.
    d = float(last) - float(first)
    return d if d >= 0.0 else None


def _classify_stage(stage: dict[str, Any]) -> dict[str, Any]:
    n = int(stage.get("samples", 0))
    duration = float(stage.get("observed_seconds", 0.0))
    gpu = stage.get("mean_gpu_util_percent")
    cpu = stage.get("mean_host_cpu_percent")
    iowait = stage.get("mean_iowait_percent")
    read_rate = stage.get("read_bytes_per_second")
    write_rate = stage.get("write_bytes_per_second")

    evidence: list[str] = []
    verdict = "INSUFFICIENT_EVIDENCE"
    confidence = "LOW"
    if n >= 6 and duration >= 20.0:
        confidence = "MEDIUM" if n < 20 or duration < 90 else "HIGH"
        if iowait is not None and iowait >= 15.0:
            verdict = "DISK_IO_WAIT_BOUND"
            evidence.append(f"mean_iowait_percent={iowait:.2f}")
        elif gpu is not None and gpu >= 75.0:
            verdict = "GPU_COMPUTE_BOUND"
            evidence.append(f"mean_gpu_util_percent={gpu:.2f}")
        elif cpu is not None and cpu >= 80.0:
            verdict = "CPU_COMPUTE_BOUND"
            evidence.append(f"mean_host_cpu_percent={cpu:.2f}")
        elif gpu is not None and gpu < 25.0 and cpu is not None and cpu < 55.0:
            verdict = "SERIAL_BARRIER_OR_WAIT_BOUND"
            evidence.extend(
                [
                    f"mean_gpu_util_percent={gpu:.2f}",
                    f"mean_host_cpu_percent={cpu:.2f}",
                ]
            )
        elif gpu is not None and gpu < 40.0 and cpu is not None and cpu >= 55.0:
            verdict = "CPU_OR_INPUT_PIPELINE_LIMITING_GPU"
            evidence.extend(
                [
                    f"mean_gpu_util_percent={gpu:.2f}",
                    f"mean_host_cpu_percent={cpu:.2f}",
                ]
            )
        elif (read_rate or 0.0) + (write_rate or 0.0) > 100 * (1 << 20):
            verdict = "HIGH_STORAGE_THROUGHPUT"
            evidence.append("aggregate_io_rate_gt_100MiB_s")
        else:
            verdict = "NO_CLEAR_SINGLE_RESOURCE_BOTTLENECK"
    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence": evidence,
        "samples": n,
        "observed_seconds": duration,
    }


def build_stage_attribution(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[tuple[float, int | None, str, dict[str, float | None], str]] = []
    schema_counts: Counter[str] = Counter()
    discarded_no_timestamp = 0

    for row in samples:
        schema_counts[str(row.get("schema") or "UNSPECIFIED")] += 1
        ts = _timestamp(row)
        if ts is None:
            discarded_no_timestamp += 1
            continue
        rows.append((ts, _generation(row), _stage(row), _metrics(row), str(row.get("schema") or "UNSPECIFIED")))
    rows.sort(key=lambda x: x[0])

    buckets: dict[tuple[int | None, str], list[tuple[float, dict[str, float | None]]]] = defaultdict(list)
    for ts, gen, stage, metrics, _ in rows:
        buckets[(gen, stage)].append((ts, metrics))

    stages: list[dict[str, Any]] = []
    for (gen, stage), group in sorted(
        buckets.items(),
        key=lambda kv: ((kv[0][0] if kv[0][0] is not None else -1), kv[1][0][0]),
    ):
        first_ts = group[0][0]
        last_ts = group[-1][0]
        observed_seconds = max(0.0, last_ts - first_ts)
        metrics = [x[1] for x in group]

        # Prefer RuntimeObserver's interval-normalized rates. Fall back to
        # cumulative counter delta for old/synthetic schemas.
        read_rate = _avg(m["read_bytes_per_second_direct"] for m in metrics)
        write_rate = _avg(m["write_bytes_per_second_direct"] for m in metrics)
        read_delta = _delta(metrics[0]["read_bytes"], metrics[-1]["read_bytes"])
        write_delta = _delta(metrics[0]["write_bytes"], metrics[-1]["write_bytes"])
        denom = observed_seconds if observed_seconds > 0 else None
        if read_rate is None and read_delta is not None and denom:
            read_rate = read_delta / denom
        if write_rate is None and write_delta is not None and denom:
            write_rate = write_delta / denom

        item: dict[str, Any] = {
            "generation": gen,
            "stage": stage,
            "samples": len(group),
            "first_timestamp_unix": first_ts,
            "last_timestamp_unix": last_ts,
            "observed_seconds": observed_seconds,
            "mean_process_cpu_percent": _avg(m["cpu_percent"] for m in metrics),
            "mean_host_cpu_percent": _avg(m["host_cpu_percent"] for m in metrics),
            "mean_iowait_percent": _avg(m["iowait_percent"] for m in metrics),
            "peak_rss_bytes": _max(m["rss_bytes"] for m in metrics),
            "mean_gpu_util_percent": _avg(m["gpu_util_percent"] for m in metrics),
            "peak_gpu_mem_used_bytes": _max(m["gpu_mem_used_bytes"] for m in metrics),
            "mean_gpu_power_watts": _avg(m["gpu_power_watts"] for m in metrics),
            "read_bytes_delta": read_delta,
            "write_bytes_delta": write_delta,
            "read_bytes_per_second": read_rate,
            "write_bytes_per_second": write_rate,
        }
        item["bottleneck"] = _classify_stage(item)
        stages.append(item)

    generation_summary: dict[int, dict[str, Any]] = {}
    for item in stages:
        gen = item["generation"]
        if gen is None:
            continue
        g = generation_summary.setdefault(
            int(gen),
            {
                "generation": int(gen),
                "observed_seconds": 0.0,
                "samples": 0,
                "stages": [],
                "dominant_observed_stage": None,
            },
        )
        g["observed_seconds"] += float(item["observed_seconds"])
        g["samples"] += int(item["samples"])
        g["stages"].append(
            {
                "stage": item["stage"],
                "observed_seconds": item["observed_seconds"],
                "bottleneck": item["bottleneck"],
            }
        )
    for g in generation_summary.values():
        if g["stages"]:
            g["dominant_observed_stage"] = max(
                g["stages"], key=lambda x: float(x["observed_seconds"])
            )["stage"]

    valid = bool(rows)
    return {
        "schema": SCHEMA,
        "status": "PASS" if valid else "NO_VALID_SAMPLES",
        "safety": dict(SAFETY),
        "input_samples_total": len(samples),
        "sample_count": len(rows),
        "discarded_sample_count": len(samples) - len(rows),
        "discarded_no_timestamp": discarded_no_timestamp,
        "input_schema_counts": dict(sorted(schema_counts.items())),
        "stage_bucket_count": len(stages),
        "generation_count_observed": len(generation_summary),
        "stages": stages,
        "generations": [generation_summary[k] for k in sorted(generation_summary)],
        "interpretation": "OBSERVATIONAL_STAGE_ATTRIBUTION_ONLY__NOT_SCIENTIFIC_VERDICT",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    samples = _read_jsonl(Path(args.runtime_jsonl))
    result = build_stage_attribution(samples)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "input_samples_total": result["input_samples_total"],
                "sample_count": result["sample_count"],
                "discarded_sample_count": result["discarded_sample_count"],
                "stage_bucket_count": result["stage_bucket_count"],
                "generation_count_observed": result["generation_count_observed"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
