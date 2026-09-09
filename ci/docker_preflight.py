#!/usr/bin/env python3
"""Fail-closed preflight for the CB16 R11 Docker self-hosted runner.

This checks only execution-container topology and runtime capability. It does not
open scientific payloads, start training, or create evidence.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import sys
from pathlib import Path

FORBIDDEN_HOST_PATHS = (
    "/home/bgy",
    "/data/cb16_hdd",
    "/data/cb16_ci",
    "/cb16_r11_runtime",
)

DEPLOYED_PROVISION_ENV = Path("/run/secrets/provision.env")
LEGACY_PROVISION_ALIAS = Path("/run/secrets/cb16-provision.env")

DEFAULTS = {
    "CB16_RAW_ROOT": "/cb16/raw",
    "CB16_G0_ROOT": "/cb16/g0",
    "CB16_R104_ROOT": "/cb16/runtime/r104",
    "CB16_VERIFIED_VENV": "/cb16/venv",
    "CB16_UV_CACHE_DIR": "/cb16/uv-cache",
    "CB16_CI_WORKER_ROOT": "/cb16/worker",
    "CB16_PROVISION_ENV": str(DEPLOYED_PROVISION_ENV),
    "CB16_PACKAGE_ROOT": "/cb16/package",
    "CB16_PARENT_R101_ROOT": "/cb16/parent-r101",
    "CB16_PARENT_G0": "/cb16/parents/r10_1.pt",
}


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def check_read_only_directory(path: Path) -> tuple[bool, str]:
    if not path.is_dir():
        return False, "MISSING_DIRECTORY"
    probe = path / f".cb16_docker_preflight_write_probe_{os.getpid()}"
    try:
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        if exc.errno in {errno.EROFS, errno.EACCES, errno.EPERM}:
            return True, errno.errorcode.get(exc.errno, str(exc.errno))
        return False, f"UNEXPECTED_WRITE_ERROR:{exc.errno}"
    else:
        os.close(fd)
        try:
            probe.unlink()
        finally:
            return False, "WRITE_SUCCEEDED"


def check_writable_directory(path: Path) -> tuple[bool, str]:
    if not path.is_dir():
        return False, "MISSING_DIRECTORY"
    probe = path / f".cb16_docker_preflight_write_probe_{os.getpid()}"
    try:
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, b"docker-preflight\n")
        os.fsync(fd)
        os.close(fd)
        probe.unlink()
        return True, "WRITE_UNLINK_PASS"
    except OSError as exc:
        try:
            if probe.exists():
                probe.unlink()
        except OSError:
            pass
        return False, f"WRITE_FAILED:{exc.errno}"


def check_no_host_paths(env_map: dict[str, str]) -> tuple[bool, list[str]]:
    hits: list[str] = []
    for key, value in sorted(env_map.items()):
        if not isinstance(value, str):
            continue
        for prefix in FORBIDDEN_HOST_PATHS:
            if prefix in value:
                hits.append(f"env:{key}:{prefix}")
    try:
        mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        mountinfo = ""
    for prefix in FORBIDDEN_HOST_PATHS:
        if prefix in mountinfo:
            hits.append(f"mountinfo:{prefix}")
    return not hits, hits


def torch_cuda_report(venv: Path) -> tuple[bool, dict]:
    py = venv / "bin" / "python"
    if not py.is_file() or not os.access(py, os.X_OK):
        return False, {"error": "VERIFIED_VENV_PYTHON_NOT_EXECUTABLE", "python": str(py)}
    code = (
        "import json,torch;"
        "print(json.dumps({"
        "'torch_version':torch.__version__,"
        "'torch_cuda_version':torch.version.cuda,"
        "'cuda_available':bool(torch.cuda.is_available()),"
        "'device_count':int(torch.cuda.device_count()),"
        "'device0':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        "},sort_keys=True))"
    )
    proc = run([str(py), "-c", code])
    if proc.returncode != 0:
        return False, {
            "error": "TORCH_CUDA_CANARY_FAILED",
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
        }
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return False, {"error": "TORCH_CUDA_CANARY_BAD_JSON", "stdout_tail": proc.stdout[-2000:]}
    return bool(payload.get("cuda_available") and payload.get("device_count", 0) >= 1), payload


def resolve_provision_env(requested: str) -> tuple[Path, str]:
    path = Path(requested)
    if path.is_file():
        return path, "REQUESTED_PATH"
    if path == LEGACY_PROVISION_ALIAS and DEPLOYED_PROVISION_ENV.is_file():
        return DEPLOYED_PROVISION_ENV, "LEGACY_ALIAS_TO_DEPLOYED_VOLUME_FILE"
    return path, "UNRESOLVED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, required=True)
    args = ap.parse_args()

    resolved = {key: os.environ.get(key, default) for key, default in DEFAULTS.items()}
    failures: list[str] = []

    provision_env, provision_resolution = resolve_provision_env(resolved["CB16_PROVISION_ENV"])
    resolved["CB16_PROVISION_ENV"] = str(provision_env)

    for key, value in resolved.items():
        if key == "CB16_PROVISION_ENV":
            continue
        if not value.startswith("/cb16/"):
            failures.append(f"NON_ABSTRACT_CB16_PATH:{key}:{value}")

    no_host_paths, host_path_hits = check_no_host_paths({**os.environ, **resolved})
    if not no_host_paths:
        failures.append("HOST_BUSINESS_PATH_VISIBLE")

    raw = Path(resolved["CB16_RAW_ROOT"])
    raw_ro, raw_ro_detail = check_read_only_directory(raw)
    if not raw_ro:
        failures.append(f"RAW_ROOT_NOT_READ_ONLY:{raw_ro_detail}")

    g0 = Path(resolved["CB16_G0_ROOT"])
    g0_rw, g0_rw_detail = check_writable_directory(g0)
    if not g0_rw:
        failures.append(f"G0_ROOT_NOT_WRITABLE:{g0_rw_detail}")

    r104 = Path(resolved["CB16_R104_ROOT"])
    if not r104.is_dir():
        failures.append("R104_ROOT_MISSING")

    if not provision_env.is_file():
        failures.append("PROVISION_ENV_MISSING")

    nvidia = run(["nvidia-smi", "-L"])
    nvidia_ok = nvidia.returncode == 0 and bool(nvidia.stdout.strip())
    if not nvidia_ok:
        failures.append("NVIDIA_SMI_UNAVAILABLE")

    torch_ok, torch_report = torch_cuda_report(Path(resolved["CB16_VERIFIED_VENV"]))
    if not torch_ok:
        failures.append("TORCH_CUDA_UNAVAILABLE")

    result = {
        "schema": "CB16_R11_DOCKER_RUNNER_PREFLIGHT_V1",
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "paths": resolved,
        "provision_env_resolution": provision_resolution,
        "host_business_paths_visible": not no_host_paths,
        "host_path_hits": host_path_hits,
        "raw_root_read_only": raw_ro,
        "raw_root_probe": raw_ro_detail,
        "g0_root_writable": g0_rw,
        "g0_root_probe": g0_rw_detail,
        "r104_root_present": r104.is_dir(),
        "provision_env_present": provision_env.is_file(),
        "nvidia_smi_pass": nvidia_ok,
        "nvidia_smi": nvidia.stdout.strip().splitlines() if nvidia_ok else [],
        "torch_cuda": torch_report,
        "failures": failures,
        "scientific_payload_opened": False,
        "training_started": False,
        "final_holdout_payload_opened": False,
        "new_evidence_created": False,
        "new_scientific_verdict": False,
    }
    atomic_json(args.json_out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 78


if __name__ == "__main__":
    sys.exit(main())
