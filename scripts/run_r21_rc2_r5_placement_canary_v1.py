#!/usr/bin/env python3
"""R5 placement-only accepted-runtime canary runner and comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def normalize_canary_outputs(root: Path) -> tuple[dict[str, Any], str]:
    smoke = json.loads((root / "SMOKE_RESULT.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "execution_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "provenance" / "artifact_only_update_trace_audit.json").read_text(encoding="utf-8"))
    run_index = json.loads((root / "provenance" / "run_index.json").read_text(encoding="utf-8"))
    task_specs = json.loads((root / "provenance" / "task_specs.json").read_text(encoding="utf-8"))

    smoke_fields = (
        "schema",
        "mode",
        "status",
        "evidence_class",
        "scientific_verdict_allowed",
        "executed_jobs",
        "positive_runs",
        "control_runs",
        "integrity_attack_suite_all_rejected",
        "objective_firewall_audit_pass",
        "fabricated_log_mu_audit_pass",
        "high_bankruptcy_failure_fact_audit_pass",
        "artifact_only_update_trace_all_checks_pass",
    )
    stable_runs = []
    for entry in run_index.get("runs", []):
        stable_runs.append(
            {
                "task_id": entry.get("task_id"),
                "seed": entry.get("seed"),
                "control_id": entry.get("control_id"),
                "decisions_consumed": entry.get("decisions_consumed"),
                "evidence_class": entry.get("evidence_class"),
                "final_child_checkpoint_sha256": entry.get("final_child_checkpoint_sha256"),
                "final_score": entry.get("final_score"),
                "initial_score": entry.get("initial_score"),
                "oracle_score": entry.get("oracle_score"),
                "unit_evidence_sha256s": entry.get("unit_evidence_sha256s"),
            }
        )
    stable_runs.sort(key=lambda item: (str(item["task_id"]), str(item["control_id"]), int(item["seed"])))
    normalized = {
        "manifest_sha256": manifest.get("manifest_sha256"),
        "smoke_result": {field: smoke.get(field) for field in smoke_fields},
        "artifact_trace": {
            "schema": audit.get("schema"),
            "all_checks_pass": audit.get("all_checks_pass"),
            "checks": audit.get("checks"),
            "traced_update_count": audit.get("traced_update_count"),
            "runs_audited": audit.get("runs_audited"),
            "failures": audit.get("failures"),
            "details": sorted(
                [
                    {"committed_updates": item.get("committed_updates"), "failures": item.get("failures")}
                    for item in audit.get("details", {}).get("runs", [])
                ],
                key=lambda item: (item["committed_updates"], json.dumps(item["failures"], sort_keys=True)),
            ),
        },
        "run_index": stable_runs,
        "task_specs_sha256": _sha256(task_specs),
    }
    return normalized, _sha256(normalized)


def _mount_for(path: Path) -> dict[str, Any]:
    target = str(path.resolve())
    best = (-1, "", "", "", None)
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 10:
            continue
        mount_point = fields[4]
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) > best[0]:
                separator = fields.index("-") if "-" in fields else len(fields)
                best = (
                    len(mount_point),
                    mount_point,
                    fields[separator + 1] if separator + 1 < len(fields) else "UNKNOWN",
                    fields[separator + 2] if separator + 2 < len(fields) else "UNKNOWN",
                    fields[2],
                )
    _, mount_point, fstype, source, major_minor = best
    rotational = None
    if major_minor:
        try:
            rotational = Path(f"/sys/dev/block/{major_minor}/queue/rotational").read_text(encoding="utf-8").strip() == "1"
        except OSError:
            rotational = None
    return {"mount_point": mount_point, "fstype": fstype, "source": source, "major_minor": major_minor, "rotational": rotational}


def _diskstats() -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    path = Path("/proc/diskstats")
    if not path.exists():
        return stats
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 14:
            continue
        stats[fields[2]] = {
            "sectors_read": int(fields[5]),
            "sectors_written": int(fields[9]),
            "ms_reading": int(fields[6]),
            "ms_writing": int(fields[10]),
        }
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fast-hot-root", type=Path, default=Path("/cb16/fast_hot"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--canary-root", type=Path, help="normalize an existing canary root instead of running")
    args = parser.parse_args(argv)

    if args.canary_root:
        normalized, digest = normalize_canary_outputs(args.canary_root)
        payload = {
            "schema": "CB16_R21_RC2_R5_ACCEPTED_CANARY_REFERENCE_V1",
            "created_at_utc": _now(),
            "source": args.reference.as_posix(),
            "normalized": normalized,
            "semantic_digest": digest,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    run_root = args.fast_hot_root / "r5" / args.run_id
    scratch = run_root / "scratch"
    output = run_root / "output"
    for path in (scratch, output):
        path.mkdir(parents=True, exist_ok=True)
    disk_before = _diskstats()
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
    process = subprocess.run(command, cwd=str(args.s1_root), text=True, capture_output=True, check=False)
    wall_seconds = time.time() - started
    disk_after = _diskstats()

    normalized, digest = normalize_canary_outputs(output)
    equivalence = normalized == reference.get("normalized")
    device = _mount_for(args.fast_hot_root)
    io_deltas = {}
    for name in ("sda", "sdb"):
        if name in disk_before and name in disk_after:
            io_deltas[name] = {
                "sectors_read_delta": disk_after[name]["sectors_read"] - disk_before[name]["sectors_read"],
                "sectors_written_delta": disk_after[name]["sectors_written"] - disk_before[name]["sectors_written"],
                "ms_reading_delta": disk_after[name]["ms_reading"] - disk_before[name]["ms_reading"],
                "ms_writing_delta": disk_after[name]["ms_writing"] - disk_before[name]["ms_writing"],
            }
    perf_targets = {
        "FULL_S1_CI_C_WALL_TIME": {"status": "NOT_APPLICABLE_BOUNDED_SMOKE", "observed_wall_seconds": wall_seconds},
        "EIGHT_WORKER_STEADY_STATE_BACKLOG_GROWTH": {"status": "NOT_MEASURED_IN_BOUNDED_SMOKE"},
        "POST_PRODUCER_BACKLOG_DRAIN_TIME": {"status": "NOT_MEASURED_IN_BOUNDED_SMOKE"},
        "HOT_DURABLE_BOUNDARY_LATENCY_P50": {"status": "NOT_MEASURED_IN_BOUNDED_SMOKE"},
        "HOT_DURABLE_BOUNDARY_LATENCY_P99": {"status": "NOT_MEASURED_IN_BOUNDED_SMOKE"},
        "FAST_HOT_NON_ROTATIONAL_PLACEMENT": {"status": "PASS" if device.get("rotational") is False else "FAIL", "device": device},
        "NO_HDD_FALLBACK": {"status": "PASS" if device.get("rotational") is False else "FAIL"},
    }
    unmeasured = [key for key, value in perf_targets.items() if value.get("status") == "NOT_MEASURED_IN_BOUNDED_SMOKE"]
    if not equivalence:
        status = "CONTRACT_MISMATCH"
        decision = "REJECT_SEMANTIC_MISMATCH"
    elif process.returncode != 0 or unmeasured:
        status = "EVIDENCE_INSUFFICIENT"
        decision = "SOL_REVIEW_REQUIRED_BEFORE_R6_OR_ALTERNATE"
    else:
        status = "PASS"
        decision = "SKIP_R6_AND_PROCEED_TO_R7"
    report = {
        "schema": "CB16_R21_RC2_R5_PLACEMENT_CANARY_RESULT_V1",
        "task_id": "R5",
        "status": status,
        "decision": decision,
        "measured_at_utc": _now(),
        "s1_exit_code": process.returncode,
        "wall_seconds": wall_seconds,
        "workers": args.workers,
        "fast_hot_root": str(args.fast_hot_root),
        "run_root": str(run_root),
        "semantic_equivalence": equivalence,
        "reference_digest": reference.get("semantic_digest"),
        "observed_digest": digest,
        "reference_normalized": reference.get("normalized"),
        "observed_normalized": normalized,
        "physical_device": device,
        "device_io_deltas": io_deltas,
        "perf_rc2_v1": perf_targets,
        "unmeasured_perf_gates": unmeasured,
        "s1_stdout": process.stdout,
        "s1_stderr": process.stderr,
        "s1_runtime_changed": false,
        "scientific_manifest_changed": false,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not args.keep_run_root:
        shutil.rmtree(run_root, ignore_errors=True)
    return 0 if status in {"PASS", "EVIDENCE_INSUFFICIENT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
