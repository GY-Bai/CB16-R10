#!/usr/bin/env python3
"""R4 bounded S1-shaped write-path measurement harness.

Runs the accepted S1 smoke workload unchanged through an instrumentation
``sitecustomize`` shim, aggregates per-surface write/fsync/SQLite metrics, and
records physical-device telemetry. It never modifies the S1 runtime or the
scientific manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT_DIR = REPO_ROOT / "ci" / "r4_write_path_instrument"
UPDATE_JOURNAL_RE = re.compile(r"/updates/([0-9a-f]{64})\.json$")

SURFACE_CONSUMERS = {
    "observation_store": "durable observation fact store",
    "sqlite_index": "observation index / SQLite WAL",
    "replay_materialization": "durable replay materializer",
    "update_journal": "exactly-once learner update journal",
    "checkpoint_store": "parent/child checkpoint store",
    "generation_continuity": "generation-switch continuity receipts",
    "provenance": "R4 provenance exporter/auditor",
    "artifact_staging": "qualification artifact staging",
    "joint_batch": "joint action batch materialization",
    "learner": "learner/optimizer runtime",
    "other": "unclassified runtime writes",
}
DURABILITY_REQUIREMENTS = {
    "observation_store": "fsync-before-success",
    "sqlite_index": "SQLite WAL/commit durable boundary",
    "replay_materialization": "durable replay provenance",
    "update_journal": "three-phase durable journal",
    "checkpoint_store": "durable child checkpoint bytes",
    "generation_continuity": "durable generation-switch receipt",
    "provenance": "artifact-exported provenance",
    "artifact_staging": "exported artifact staging",
    "joint_batch": "in-memory/durable batch provenance",
    "learner": "child checkpoint commit",
    "other": "unclassified",
}
REQUIRED_SURFACES = (
    "observation_store",
    "sqlite_index",
    "replay_materialization",
    "update_journal",
    "checkpoint_store",
    "generation_continuity",
    "provenance",
    "artifact_staging",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _diskstats() -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    path = Path("/proc/diskstats")
    if not path.exists():
        return stats
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 14:
            continue
        name = fields[2]
        stats[name] = {
            "reads": int(fields[3]),
            "sectors_read": int(fields[5]),
            "ms_reading": int(fields[6]),
            "writes": int(fields[7]),
            "sectors_written": int(fields[9]),
            "ms_writing": int(fields[10]),
        }
    return stats


def _mount_for(path: Path) -> dict[str, Any]:
    target = str(path.resolve())
    best: tuple[int, str, str, str] = (-1, "", "", "")
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 10:
            continue
        mount_point = fields[4]
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) > best[0]:
                separator = fields.index("-") if "-" in fields else len(fields)
                fstype = fields[separator + 1] if separator + 1 < len(fields) else "UNKNOWN"
                source = fields[separator + 2] if separator + 2 < len(fields) else "UNKNOWN"
                best = (len(mount_point), mount_point, fstype, source)
    mount_point, fstype, source = best[1], best[2], best[3]
    major_minor = None
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[4] == mount_point:
            major_minor = fields[2]
            break
    rotational = None
    if major_minor:
        try:
            rotational = Path(f"/sys/dev/block/{major_minor}/queue/rotational").read_text(encoding="utf-8").strip() == "1"
        except OSError:
            rotational = None
    return {"mount_point": mount_point, "fstype": fstype, "source": source, "major_minor": major_minor, "rotational": rotational}


def _aggregate(metrics_dir: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], set[str]]:
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    unique_paths: set[str] = set()
    for path in sorted(metrics_dir.glob("pid-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for category, ops in payload.get("summary", {}).items():
            for op, values in ops.items():
                bucket = summary.setdefault(category, {}).setdefault(
                    op,
                    {"count": 0, "bytes": 0, "duration_ns": 0, "extra": {}},
                )
                bucket["count"] += int(values.get("count", 0))
                bucket["bytes"] += int(values.get("bytes", 0))
                bucket["duration_ns"] += int(values.get("duration_ns", 0))
                for name, value in (values.get("extra") or {}).items():
                    bucket["extra"][name] = bucket["extra"].get(name, 0) + int(value)
        unique_paths.update(payload.get("unique_paths", []))
    return summary, unique_paths


def _count_ops(summary: dict[str, dict[str, dict[str, Any]]], category: str) -> dict[str, Any]:
    ops = summary.get(category, {})
    return {
        "write_calls": int(ops.get("write", {}).get("count", 0)) + int(ops.get("os_write", {}).get("count", 0)),
        "bytes_written": int(ops.get("write", {}).get("bytes", 0)) + int(ops.get("os_write", {}).get("bytes", 0)),
        "flush_calls": int(ops.get("flush", {}).get("count", 0)),
        "fsync_calls": int(ops.get("fsync", {}).get("count", 0)),
        "fsync_duration_ns": int(ops.get("fsync", {}).get("duration_ns", 0)),
        "replace_calls": int(ops.get("replace", {}).get("count", 0)) + int(ops.get("rename", {}).get("count", 0)),
        "sqlite_commit_calls": int(ops.get("sqlite_commit", {}).get("count", 0)),
        "sqlite_commit_duration_ns": int(ops.get("sqlite_commit", {}).get("duration_ns", 0)),
        "sqlite_locked_errors": int(ops.get("sqlite_locked_error", {}).get("count", 0)),
        "sqlite_wal_pragma_calls": int(ops.get("sqlite_wal_pragma", {}).get("count", 0)),
        "open_write_calls": int(ops.get("os_open_write", {}).get("count", 0)),
        "mkstemp_calls": int(ops.get("mkstemp", {}).get("count", 0)),
    }


def _classify_access(stats: dict[str, Any]) -> str:
    if stats["write_calls"] <= 0:
        return "NO_WRITES_OBSERVED"
    average = stats["bytes_written"] / stats["write_calls"]
    if average <= 65536 and stats["fsync_calls"] >= stats["write_calls"]:
        return "RANDOM_OR_SMALL_FSYNC_HEAVY"
    if average >= 1048576:
        return "SEQUENTIAL_LARGE"
    return "MIXED"


def _surface_report(category: str, summary: dict[str, dict[str, dict[str, Any]]], updates: int, mount: dict[str, Any]) -> dict[str, Any]:
    stats = _count_ops(summary, category)
    durable_syncs = stats["fsync_calls"] + stats["sqlite_commit_calls"]
    return {
        "surface": category,
        "consumer": SURFACE_CONSUMERS.get(category, "unclassified"),
        "bytes_written": stats["bytes_written"],
        "write_calls": stats["write_calls"],
        "bytes_per_update": (stats["bytes_written"] / updates) if updates else None,
        "writes_per_update": (stats["write_calls"] / updates) if updates else None,
        "durable_syncs": durable_syncs,
        "durable_sync_frequency_per_update": (durable_syncs / updates) if updates else None,
        "fsync_calls": stats["fsync_calls"],
        "fsync_duration_ms": stats["fsync_duration_ns"] / 1_000_000,
        "sqlite_commit_calls": stats["sqlite_commit_calls"],
        "sqlite_transaction_wait_ms_per_commit": (
            stats["sqlite_commit_duration_ns"] / stats["sqlite_commit_calls"] / 1_000_000
            if stats["sqlite_commit_calls"]
            else None
        ),
        "sqlite_locked_errors": stats["sqlite_locked_errors"],
        "wal_or_checkpoint_activity": stats["sqlite_wal_pragma_calls"],
        "random_versus_sequential": _classify_access(stats),
        "physical_device": mount,
        "durability_requirement": DURABILITY_REQUIREMENTS.get(category, "unclassified"),
    }


def measurement_status(exit_code: int, committed_updates: int, missing_surfaces: list[str]) -> str:
    if exit_code != 0 or committed_updates <= 0 or missing_surfaces:
        return "EVIDENCE_INSUFFICIENT"
    return "PASS"


def _find_committed_updates(output_root: Path, unique_paths: set[str]) -> tuple[int, str]:
    for candidate in (
        output_root / "provenance" / "artifact_only_update_trace_audit.json",
        output_root / "ARTIFACT_ONLY_UPDATE_TRACE_AUDIT.json",
    ):
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                for key in ("traced_update_count", "committed_update_count", "update_count"):
                    if isinstance(payload.get(key), int) and payload[key] > 0:
                        return int(payload[key]), f"{candidate}:{key}"
            except (OSError, json.JSONDecodeError):
                pass
    observed_ids = {match.group(1) for item in unique_paths for match in [UPDATE_JOURNAL_RE.search(item)] if match}
    if observed_ids:
        return len(observed_ids), "observed_update_journal_ids"
    for candidate in (output_root / "SMOKE_RESULT.json", output_root / "S1_RESULT.json"):
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                for key in ("artifact_only_update_trace_count", "committed_updates_count"):
                    if isinstance(payload.get(key), int) and payload[key] > 0:
                        return int(payload[key]), f"{candidate}:{key}"
            except (OSError, json.JSONDecodeError):
                pass
    return 0, "UNRESOLVED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--fast-hot-root", type=Path, default=Path("/cb16/fast_hot"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--keep-run-root", action="store_true")
    args = parser.parse_args(argv)

    run_root = args.fast_hot_root / "r4" / args.run_id
    scratch = run_root / "scratch"
    output = run_root / "output"
    metrics = run_root / "instrument" / "metrics"
    for path in (scratch, output, metrics):
        path.mkdir(parents=True, exist_ok=True)

    disk_before = _diskstats()
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(item for item in (str(INSTRUMENT_DIR), str(args.s1_root), existing_pythonpath) if item)
    env["CB16_R4_METRICS_DIR"] = str(metrics)
    env["CB16_R4_MONITORED_ROOTS"] = os.pathsep.join((str(run_root), str(scratch), str(output)))
    command = [
        sys.executable,
        str(args.s1_root / "scripts" / "run_r11_post_cc_s1_learnability.py"),
        "--repo-root",
        str(args.s1_root),
        "--mode",
        "smoke",
        "--output-root",
        str(output),
        "--scratch-root",
        str(scratch),
        "--workers",
        str(args.workers),
    ]
    started = time.time()
    process = subprocess.run(command, cwd=str(args.s1_root), env=env, text=True, capture_output=True, check=False)
    wall_seconds = time.time() - started
    (run_root / "s1_smoke_stdout.json").write_text(process.stdout, encoding="utf-8")
    (run_root / "s1_smoke_stderr.txt").write_text(process.stderr, encoding="utf-8")
    disk_after = _diskstats()

    summary, unique_paths = _aggregate(metrics)
    updates, updates_source = _find_committed_updates(output, unique_paths)
    mount = _mount_for(args.fast_hot_root)

    surfaces = [
        _surface_report(category, summary, updates, mount)
        for category in sorted(set(REQUIRED_SURFACES) | set(summary.keys()))
    ]
    def _has_activity(entry: dict[str, Any]) -> bool:
        return any(
            entry.get(name, 0)
            for name in (
                "bytes_written",
                "write_calls",
                "fsync_calls",
                "durable_syncs",
                "sqlite_commit_calls",
                "open_write_calls",
                "replace_calls",
            )
        )

    observed_surfaces = [entry["surface"] for entry in surfaces if _has_activity(entry)]
    missing_surfaces = [surface for surface in REQUIRED_SURFACES if surface not in observed_surfaces]

    device_io: dict[str, Any] = {}
    for name in ("sda", "sdb"):
        before = disk_before.get(name)
        after = disk_after.get(name)
        if before and after:
            device_io[name] = {
                "sectors_read_delta": after["sectors_read"] - before["sectors_read"],
                "sectors_written_delta": after["sectors_written"] - before["sectors_written"],
                "ms_reading_delta": after["ms_reading"] - before["ms_reading"],
                "ms_writing_delta": after["ms_writing"] - before["ms_writing"],
                "rotational": True if name == "sda" else (False if name == "sdb" else None),
            }

    status = measurement_status(process.returncode, updates, missing_surfaces)
    paths_by_category: dict[str, list[str]] = {}
    for item in unique_paths:
        category, _, observed_path = item.partition("|")
        paths_by_category.setdefault(category, []).append(observed_path)
    metric_process_count = len(list(metrics.glob("pid-*.json")))
    report = {
        "schema": "CB16_R21_RC2_R4_WRITE_PATH_INVENTORY_V1",
        "task_id": "R4",
        "status": status,
        "measured_at_utc": _now(),
        "s1_runtime_root": str(args.s1_root),
        "fast_hot_root": str(args.fast_hot_root),
        "run_root": str(run_root),
        "s1_exit_code": process.returncode,
        "wall_seconds": wall_seconds,
        "workers": args.workers,
        "committed_updates": updates,
        "committed_updates_source": updates_source,
        "metric_process_count": metric_process_count,
        "observed_paths_by_category": {category: sorted(paths)[:8] for category, paths in sorted(paths_by_category.items())},
        "observed_surfaces": observed_surfaces,
        "missing_required_surfaces": missing_surfaces,
        "physical_device": mount,
        "device_io_deltas": device_io,
        "surfaces": surfaces,
        "s1_stdout_path": str(run_root / "s1_smoke_stdout.json"),
        "s1_stderr_path": str(run_root / "s1_smoke_stderr.txt"),
        "scientific_manifest_changed": False,
        "s1_runtime_changed": False,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    (args.json_out.parent / "R4_S1_STDOUT.json").write_text(process.stdout, encoding="utf-8")
    (args.json_out.parent / "R4_S1_STDERR.txt").write_text(process.stderr, encoding="utf-8")
    args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not args.keep_run_root:
        shutil.rmtree(run_root, ignore_errors=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
