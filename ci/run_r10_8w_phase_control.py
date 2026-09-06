#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pwd
import subprocess
import sys
from pathlib import Path

SOURCE_SHA = "f056ae6a0722e3e92d71793024a6e6d3fe9af003"
R102_ROOT = Path("/home/bgy/cb16_ssd/runtime/R10_2/qualification_r8_3_8w")
R103_ROOT = Path("/home/bgy/cb16_ssd/runtime/R10_3/qualification_r8_3_8w")
R104_ROOT = Path("/data/cb16_hdd/cb16_runtime/R10_4")
CANON_R102 = Path("/home/bgy/cb16_ssd/runtime/R10_2/FINAL_RESULT_R102.json")
CANON_R103 = Path("/home/bgy/cb16_ssd/runtime/R10_3/FINAL_RESULT_R102.json")


def run(cmd: list[str], *, cwd: Path) -> None:
    print("EXEC", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def compare_identity(candidate_path: Path, authority_path: Path) -> None:
    candidate = load_json(candidate_path)
    authority = load_json(authority_path)
    for key in ("final_champion_semantic_sha256", "attempts_completed", "promotions", "rejections"):
        if candidate.get(key) != authority.get(key):
            raise RuntimeError(
                f"RESULT_IDENTITY_MISMATCH:{key}:{candidate.get(key)}!={authority.get(key)}"
            )
    if not str(candidate.get("final_status", "")).endswith("PASS"):
        raise RuntimeError(f"CANDIDATE_NOT_PASS:{candidate.get('final_status')}")
    if not str(authority.get("final_status", "")).endswith("PASS"):
        raise RuntimeError(f"AUTHORITY_NOT_PASS:{authority.get('final_status')}")
    print("SCIENTIFIC_RESULT_IDENTITY_EQUIVALENCE=PASS", flush=True)
    print("final_champion_semantic_sha256=", candidate.get("final_champion_semantic_sha256"), flush=True)


def ensure_qualification_authority(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "QUALIFICATION_AUTHORITY_R8_3_8W.json"
    expected = {
        "schema": "CB16_R8_3_8W_QUALIFICATION_AUTHORITY_V1",
        "source_sha": SOURCE_SHA,
        "purpose": "EXPLICIT_CONTROLLED_8W_FULL_QUALIFICATION",
        "cache_policy": "CONTROLLED_REBUILD_ONLY_IN_DEDICATED_QUALIFICATION_ROOTS; CANONICAL_R104_CACHE_HIT_FIRST",
        "scientific_semantics_changed": False,
        "final_holdout_2025_09_accessed": False,
    }
    if path.exists():
        current = load_json(path)
        for key, value in expected.items():
            if current.get(key) != value:
                raise RuntimeError(f"QUALIFICATION_AUTHORITY_DRIFT:{key}")
        print("QUALIFICATION_AUTHORITY_REUSE=PASS", flush=True)
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
        print("QUALIFICATION_AUTHORITY_CREATED=PASS", flush=True)


def active_r10_processes() -> list[str]:
    p = subprocess.run(["ps", "-eo", "pid=,comm=,args="], check=True, capture_output=True, text=True)
    out = []
    for line in p.stdout.splitlines():
        if "scripts/run_r102_pipeline.py" in line or "scripts/run_r103_expansion.py" in line or "scripts/run_r104_long_research.py" in line:
            out.append(line.strip())
    return out


def provision_python(source_root: Path, phase: str, evidence_base: Path) -> Path:
    scripts = source_root / "provision" / "scripts"
    sys.path.insert(0, str(scripts))
    import provision_python as pp  # type: ignore

    result = pp.provision(phase)
    (evidence_base / "python_provision.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    py = Path(result["python"]["venv"]) / "bin" / "python"
    if not py.is_file() or not os.access(py, os.X_OK):
        raise RuntimeError(f"PROVISIONED_PYTHON_NOT_EXECUTABLE:{py}")
    print("PROVISION_PROFILE=", phase, flush=True)
    print("PROVISION_CACHE_HIT=", result["python"].get("cache_hit"), flush=True)
    print("PROVISION_INSTALLER=", result["python"].get("installer"), flush=True)
    return py


def live_runtime_check(py: Path, source_root: Path) -> None:
    code = r'''
import json, torch
from cb16_local_opt.r102_runtime_authority import load_r102_runtime_parallelism
assert torch.cuda.is_available(), "CUDA_UNAVAILABLE"
r = load_r102_runtime_parallelism('.', live_environment_check=True)
assert r.performance_overlay == 'R8_3_8W_RAM_ADAPTIVE'
assert r.h72_workers == 8
assert r.teacher_workers == 8
assert r.h72_threads_per_worker == 1
assert r.teacher_threads_per_worker == 1
assert r.h72_max_in_flight == 8
assert r.cache_hit_first is True
assert r.runtime_scheduling_identity_is_not_scientific_cache_identity is True
print('CUDA_PYTHON_ENV=PASS', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print('R8_3_LIVE_RUNTIME_AUTHORITY=PASS')
print(json.dumps({'overlay':r.performance_overlay,'h72_workers':r.h72_workers,'teacher_workers':r.teacher_workers,'threads':r.h72_threads_per_worker,'max_in_flight':r.h72_max_in_flight,'cache_hit_first':r.cache_hit_first},sort_keys=True))
'''
    subprocess.run([str(py), "-c", code], cwd=source_root, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["r102", "r103", "r104"])
    ap.add_argument("--source-root", required=True, type=Path)
    ap.add_argument("--source-sha", required=True)
    ap.add_argument("--evidence-base", required=True, type=Path)
    args = ap.parse_args()

    source_root = args.source_root.resolve()
    evidence_base = args.evidence_base.resolve()
    evidence_base.mkdir(parents=True, exist_ok=True)

    if args.source_sha != SOURCE_SHA:
        raise RuntimeError(f"SOURCE_AUTHORITY_DRIFT:{args.source_sha}")
    user = pwd.getpwuid(os.geteuid()).pw_name
    print("WHOAMI=", user, flush=True)
    if user != "cb16-ci":
        raise RuntimeError(f"CANONICAL_RUNNER_USER_MISMATCH:{user}")
    active = active_r10_processes()
    if active:
        raise RuntimeError("ACTIVE_R10_SCIENTIFIC_PROCESS_DETECTED:" + " | ".join(active))

    py = provision_python(source_root, args.phase, evidence_base)
    (evidence_base / "python_bin.txt").write_text(str(py) + "\n")
    live_runtime_check(py, source_root)

    if args.phase == "r102":
        ensure_qualification_authority(R102_ROOT)
        print("CACHE_POLICY=EXPLICIT_CONTROLLED_8W_QUALIFICATION_REBUILD", flush=True)
        run([str(py), "scripts/run_r102_pipeline.py", "--run-root", str(R102_ROOT)], cwd=source_root)
        compare_identity(R102_ROOT / "FINAL_RESULT_R102.json", CANON_R102)
    elif args.phase == "r103":
        compare_identity(R102_ROOT / "FINAL_RESULT_R102.json", CANON_R102)
        ensure_qualification_authority(R103_ROOT)
        print("CACHE_POLICY=EXPLICIT_CONTROLLED_8W_QUALIFICATION_REBUILD", flush=True)
        run([
            str(py), "scripts/run_r103_expansion.py",
            "--r102-root", str(R102_ROOT),
            "--run-root", str(R103_ROOT),
        ], cwd=source_root)
        compare_identity(R103_ROOT / "FINAL_RESULT_R102.json", CANON_R103)
    else:
        compare_identity(R103_ROOT / "FINAL_RESULT_R102.json", CANON_R103)
        print("CACHE_POLICY=CANONICAL_R104_CACHE_HIT_FIRST_RECOVERY", flush=True)
        run([
            str(py), "scripts/run_r104_long_research.py",
            "--r103-root", str(R103_ROOT),
            "--run-root", str(R104_ROOT),
        ], cwd=source_root)

    print("PHASE_CONTROL=PASS", flush=True)
    print("FINAL_HOLDOUT_2025_09_ACCESSED=NO", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
