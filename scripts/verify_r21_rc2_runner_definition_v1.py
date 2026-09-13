#!/usr/bin/env python3
"""R21 RC2 R1 runner definition verifier.

This verifier is read-only. It never creates, starts, stops, removes or
reconfigures a container. Three modes are supported:

* ``--static``       validate the repo definition against frozen live facts.
* ``--self-check``   run inside the live Shanxi runner and compare the current
                     execution surface with the repo definition.
* ``--inspect-json`` compare the repo definition with a ``docker inspect`` JSON
                     captured by an authorized read-only host session.

A failure returns exit code 1 and a machine-readable result. The tool does not
claim that a disposable runner has been launched; R1 forbids host changes and
that missing launch is recorded as EXECUTION_BLOCKED in the launch spec.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import pwd
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "infra" / "shanxi_runner" / "runner_launch_spec_v1.json"
BUILD_CONTEXT_PATH = REPO_ROOT / "infra" / "shanxi_runner" / "build_context_v1.json"
DOCKERFILE_PATH = REPO_ROOT / "infra" / "shanxi_runner" / "Dockerfile"

EXPECTED_SCHEMA = "CB16_SHANXI_RUNNER_LAUNCH_SPEC_V1"
EXPECTED_MOUNT_COUNT = 13
EXPECTED_SHM_BYTES = 67108864
EXPECTED_UV_VERSION = "0.12.6"
EXPECTED_UV_SHA256 = "d381f11517c66523211b0876552ff7dea5c1b4b0f13800571b35225761302fba"
EXPECTED_UV_SIZE_BYTES = 51102256
REQUIRED_ENV_NAMES = (
    "CB16_RAW_ROOT",
    "CB16_G0_ROOT",
    "CB16_R104_ROOT",
    "CB16_VERIFIED_VENV",
    "CB16_UV_CACHE_DIR",
    "CB16_CI_WORKER_ROOT",
    "CB16_PACKAGE_ROOT",
    "CB16_PARENT_R101_ROOT",
    "CB16_PARENT_G0",
    "CB16_PROVISION_ENV",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "HOME",
    "PATH",
    "LANG",
    "PYTHON_VERSION",
    "PYTHON_SHA256",
    "PYTHONUNBUFFERED",
)
SELF_CHECK_ENV_NAMES = REQUIRED_ENV_NAMES


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _env_map(spec: dict[str, Any]) -> dict[str, str]:
    return {entry["name"]: entry["value"] for entry in spec["environment"]}


def _mount_map(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["container_path"]: entry for entry in spec["mounts"]}


def _check(checks: list[dict[str, Any]], name: str, condition: bool, detail: str = "") -> None:
    checks.append({"check": name, "ok": bool(condition), "detail": detail})


def static_check(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    spec = _load_json(repo_root / SPEC_PATH.relative_to(REPO_ROOT))
    build_context = _load_json(repo_root / BUILD_CONTEXT_PATH.relative_to(REPO_ROOT))
    dockerfile_text = (repo_root / DOCKERFILE_PATH.relative_to(REPO_ROOT)).read_text(encoding="utf-8")
    checks: list[dict[str, Any]] = []

    _check(checks, "spec_schema", spec.get("schema") == EXPECTED_SCHEMA, str(spec.get("schema")))
    _check(
        checks,
        "spec_pending_review_status",
        spec.get("status") == "DS_PROPOSED_PENDING_SOL_REVIEW",
        str(spec.get("status")),
    )

    run = spec["run"]
    _check(checks, "run_name", run.get("name") == "cb16-runner-r11", str(run.get("name")))
    _check(checks, "run_restart_policy", run.get("restart_policy") == "unless-stopped", str(run.get("restart_policy")))
    _check(checks, "run_network_mode", run.get("network_mode") == "cb16-net", str(run.get("network_mode")))
    _check(checks, "run_gpu_request", run.get("gpu_request") == "all", str(run.get("gpu_request")))
    _check(checks, "run_shm_size", run.get("shm_size_bytes") == EXPECTED_SHM_BYTES, str(run.get("shm_size_bytes")))
    _check(checks, "run_user", run.get("user") == "1001:1001", str(run.get("user")))
    _check(
        checks,
        "run_working_dir",
        run.get("working_dir") == "/home/cb16-runner/actions-runner",
        str(run.get("working_dir")),
    )
    _check(checks, "run_entrypoint", run.get("entrypoint") is None, str(run.get("entrypoint")))
    _check(checks, "run_command", run.get("command") == ["./run.sh"], str(run.get("command")))
    _check(checks, "run_privileged", run.get("privileged") is False, str(run.get("privileged")))
    _check(checks, "run_memory_limit", run.get("memory_limit_bytes") == 0, str(run.get("memory_limit_bytes")))
    _check(checks, "run_nano_cpus", run.get("nano_cpus") == 0, str(run.get("nano_cpus")))

    mounts = _mount_map(spec)
    _check(checks, "mount_count", len(spec["mounts"]) == EXPECTED_MOUNT_COUNT, str(len(spec["mounts"])))
    expected_mounts = {
        "/cb16/raw": (False, "cb16-raw"),
        "/cb16/g0": (True, "cb16-g0"),
        "/cb16/runtime/r104": (False, "cb16-runtime-r104"),
        "/cb16/venv": (True, "cb16-venv"),
        "/cb16/uv-cache": (True, "cb16-uv-cache"),
        "/cb16/worker": (True, "cb16-worker"),
        "/cb16/package": (False, "cb16-package"),
        "/cb16/parent-r101": (False, "cb16-parent-r101"),
        "/cb16/parents": (False, "cb16-parent-g0"),
        "/cb16/r2-authority": (False, "cb16-r2-authority"),
        "/cb16/r2-native": (True, "cb16-r2-native"),
        "/run/secrets": (False, "cb16-config"),
        "/home/cb16-runner/actions-runner": (True, "cb16-runner-docker"),
    }
    _check(checks, "mount_destinations", set(mounts) == set(expected_mounts), str(sorted(mounts)))
    for container_path, (mount_rw, volume) in expected_mounts.items():
        entry = mounts.get(container_path)
        _check(checks, f"mount_present:{container_path}", entry is not None, "")
        if entry is not None:
            _check(
                checks,
                f"mount_flags:{container_path}",
                entry.get("mount_rw") is mount_rw and entry.get("volume") == volume,
                json.dumps(entry, sort_keys=True),
            )

    env = _env_map(spec)
    for name in REQUIRED_ENV_NAMES:
        _check(checks, f"env_present:{name}", name in env, "")
    _check(checks, "env_cb16_provision_env_value", env.get("CB16_PROVISION_ENV") == "/run/secrets/cb16-provision.env", str(env.get("CB16_PROVISION_ENV")))
    _check(checks, "uid", spec["uid_gid_contract"].get("uid") == 1001, str(spec["uid_gid_contract"].get("uid")))
    _check(checks, "gid", spec["uid_gid_contract"].get("gid") == 1001, str(spec["uid_gid_contract"].get("gid")))

    uv_entries = build_context.get("required_build_context_files", [])
    _check(checks, "uv_entry_count", len(uv_entries) == 1, str(len(uv_entries)))
    if uv_entries:
        uv = uv_entries[0]
        _check(checks, "uv_version", uv.get("version") == EXPECTED_UV_VERSION, str(uv.get("version")))
        _check(checks, "uv_sha256", uv.get("sha256") == EXPECTED_UV_SHA256, str(uv.get("sha256")))
        _check(checks, "uv_size_bytes", uv.get("size_bytes") == EXPECTED_UV_SIZE_BYTES, str(uv.get("size_bytes")))

    _check(checks, "dockerfile_from", "FROM python:3.10-slim-bookworm" in dockerfile_text, "")
    _check(checks, "dockerfile_copy_uv", "COPY uv /usr/local/bin/uv" in dockerfile_text, "")
    _check(checks, "dockerfile_user", "USER cb16-runner" in dockerfile_text, "")
    _check(checks, "dockerfile_uid_gid", "--gid 1001 cb16-runner" in dockerfile_text and "-u 1001 -g 1001" in dockerfile_text, "")

    divergences = {entry["divergence_id"]: entry for entry in spec.get("divergences", [])}
    _check(checks, "divergence_r1_div_001", "R1-DIV-001" in divergences, "")
    if "R1-DIV-001" in divergences:
        div = divergences["R1-DIV-001"]
        _check(checks, "divergence_r1_div_001_classification", div.get("classification") == "CONTRACT_MISMATCH", str(div.get("classification")))
        _check(checks, "divergence_r1_div_001_no_auto_fix", div.get("auto_fix_applied") is False, str(div.get("auto_fix_applied")))
        _check(checks, "divergence_r1_div_001_host_unchanged", div.get("host_changed") is False, str(div.get("host_changed")))

    reproduction = spec.get("reproduction_status", {})
    launch = reproduction.get("disposable_runner_launch", {})
    _check(checks, "disposable_launch_marked_blocked", launch.get("status") == "EXECUTION_BLOCKED", str(launch.get("status")))
    _check(checks, "disposable_launch_host_changes_false", launch.get("host_changes_applied") is False, str(launch.get("host_changes_applied")))
    _check(checks, "bit_identical_not_claimed", reproduction.get("bit_identical_image_rebuild") == "NOT_CLAIMED", "")
    return checks


def _read_os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values


def _mount_options_by_destination() -> dict[str, set[str]]:
    options: dict[str, set[str]] = {}
    path = Path("/proc/self/mountinfo")
    if not path.exists():
        return options
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        destination = fields[4]
        options[destination] = set(fields[5].split(","))
    return options


def _parse_environ_bytes(raw: bytes) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        text = item.decode("utf-8", errors="replace")
        if "=" in text:
            name, value = text.split("=", 1)
            values[name] = value
    return values


def _read_pid1_env() -> dict[str, str] | None:
    """Read the container PID 1 environment (the docker-run launch env).

    A workflow-level ``env:`` block can override ``os.environ`` for the current
    job. Comparing against ``/proc/1/environ`` instead keeps the self-check
    bound to the container launch contract rather than to GitHub Actions
    workflow overrides.
    """
    path = Path("/proc/1/environ")
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return _parse_environ_bytes(raw)


def self_check(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    spec = _load_json(repo_root / SPEC_PATH.relative_to(REPO_ROOT))
    checks: list[dict[str, Any]] = []
    env = _env_map(spec)
    uid_contract = spec["uid_gid_contract"]

    _check(checks, "self_uid", os.getuid() == uid_contract["uid"], str(os.getuid()))
    _check(checks, "self_gid", os.getgid() == uid_contract["gid"], str(os.getgid()))
    try:
        account = pwd.getpwuid(os.getuid())
        _check(checks, "self_user_name", account.pw_name == uid_contract["container_user"], account.pw_name)
    except KeyError as exc:  # pragma: no cover - defensive
        _check(checks, "self_user_name", False, str(exc))

    os_release = _read_os_release()
    _check(checks, "self_os_id", os_release.get("ID") == "debian", str(os_release.get("ID")))
    _check(checks, "self_os_version", os_release.get("VERSION_ID") == "12", str(os_release.get("VERSION_ID")))
    _check(checks, "self_python_version", platform.python_version() == "3.10.21", platform.python_version())

    shm = os.statvfs("/dev/shm")
    shm_size = shm.f_frsize * shm.f_blocks
    _check(checks, "self_shm_size_bytes", shm_size == EXPECTED_SHM_BYTES, str(shm_size))

    pid_one_cwd = Path("/proc/1/cwd")
    if pid_one_cwd.exists():
        _check(checks, "self_pid1_working_dir", os.readlink(pid_one_cwd) == spec["run"]["working_dir"], os.readlink(pid_one_cwd))
    else:  # pragma: no cover - defensive
        _check(checks, "self_pid1_working_dir", False, "missing /proc/1/cwd")

    pid1_env = _read_pid1_env()
    _check(checks, "self_pid1_env_available", pid1_env is not None, "/proc/1/environ")
    live_env = pid1_env if pid1_env is not None else os.environ
    env_source = "pid1" if pid1_env is not None else "process_env"
    for name in SELF_CHECK_ENV_NAMES:
        _check(
            checks,
            f"self_env:{name}",
            live_env.get(name) == env.get(name),
            f"source={env_source} live={live_env.get(name)!r}",
        )

    live_provision_target = Path(env["CB16_PROVISION_ENV"])
    _check(
        checks,
        "self_divergence_r1_div_001_target_missing",
        not live_provision_target.exists(),
        f"{live_provision_target} exists={live_provision_target.exists()}",
    )
    _check(
        checks,
        "self_divergence_r1_div_001_secrets_copy_present",
        Path("/run/secrets/provision.env").exists(),
        "/run/secrets/provision.env",
    )

    mount_options = _mount_options_by_destination()
    for entry in spec["mounts"]:
        container_path = entry["container_path"]
        _check(checks, f"self_mount_present:{container_path}", container_path in mount_options, "")
        if container_path in mount_options:
            is_read_only = "ro" in mount_options[container_path]
            _check(
                checks,
                f"self_mount_ro_flag:{container_path}",
                is_read_only is (not entry["mount_rw"]),
                f"ro={is_read_only} spec_mount_rw={entry['mount_rw']}",
            )

    gpu = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
    gpu_text = (gpu.stdout or "").strip()
    _check(checks, "self_gpu_visible", gpu.returncode == 0 and "GTX 1060" in gpu_text, gpu_text or gpu.stderr.strip())
    return checks


def _normalise_image_id(value: str) -> str:
    return value if value.startswith("sha256:") else f"sha256:{value}"


def inspect_check(inspect_path: Path, repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    spec = _load_json(repo_root / SPEC_PATH.relative_to(REPO_ROOT))
    inspect_obj = _load_json(inspect_path)
    if isinstance(inspect_obj, list):
        inspect_obj = inspect_obj[0]
    checks: list[dict[str, Any]] = []
    cfg = inspect_obj.get("Config", {})
    hc = inspect_obj.get("HostConfig", {})
    run = spec["run"]

    _check(checks, "inspect_name", inspect_obj.get("Name") == f"/{run['name']}", str(inspect_obj.get("Name")))
    _check(checks, "inspect_image", cfg.get("Image") == spec["image"]["reference"], str(cfg.get("Image")))
    _check(
        checks,
        "inspect_image_id",
        _normalise_image_id(inspect_obj.get("Image", "")) == spec["image"]["live_image_id"],
        str(inspect_obj.get("Image")),
    )
    _check(checks, "inspect_user", cfg.get("User") == run["user"], str(cfg.get("User")))
    _check(checks, "inspect_working_dir", cfg.get("WorkingDir") == run["working_dir"], str(cfg.get("WorkingDir")))
    _check(checks, "inspect_cmd", cfg.get("Cmd") == run["command"], str(cfg.get("Cmd")))
    _check(checks, "inspect_entrypoint", cfg.get("Entrypoint") is run["entrypoint"], str(cfg.get("Entrypoint")))
    _check(checks, "inspect_network", hc.get("NetworkMode") == run["network_mode"], str(hc.get("NetworkMode")))
    _check(checks, "inspect_shm", hc.get("ShmSize") == run["shm_size_bytes"], str(hc.get("ShmSize")))
    _check(checks, "inspect_restart", hc.get("RestartPolicy", {}).get("Name") == run["restart_policy"], str(hc.get("RestartPolicy")))
    _check(checks, "inspect_privileged", hc.get("Privileged") is run["privileged"], str(hc.get("Privileged")))
    _check(checks, "inspect_memory", hc.get("Memory") == run["memory_limit_bytes"], str(hc.get("Memory")))
    _check(checks, "inspect_nano_cpus", hc.get("NanoCpus") == run["nano_cpus"], str(hc.get("NanoCpus")))

    spec_mounts = _mount_map(spec)
    live_mounts = {entry["Destination"]: entry for entry in inspect_obj.get("Mounts", [])}
    _check(checks, "inspect_mount_count", len(live_mounts) == EXPECTED_MOUNT_COUNT, str(len(live_mounts)))
    _check(checks, "inspect_mount_destinations", set(live_mounts) == set(spec_mounts), str(sorted(live_mounts)))
    for container_path, entry in spec_mounts.items():
        live = live_mounts.get(container_path)
        _check(checks, f"inspect_mount_present:{container_path}", live is not None, "")
        if live is not None:
            _check(
                checks,
                f"inspect_mount_rw:{container_path}",
                bool(live.get("RW")) is entry["mount_rw"],
                f"live_rw={live.get('RW')} spec_mount_rw={entry['mount_rw']}",
            )

    live_env = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in cfg.get("Env", []) if "=" in item}
    spec_env = _env_map(spec)
    for name, value in spec_env.items():
        _check(checks, f"inspect_env:{name}", live_env.get(name) == value, f"live={live_env.get(name)!r} spec={value!r}")
    return checks


def _result(mode: str, checks: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    failed = [entry for entry in checks if not entry["ok"]]
    result: dict[str, Any] = {
        "schema": "CB16_R21_RC2_R1_RUNNER_DEFINITION_VERIFICATION_V1",
        "mode": mode,
        "status": "PASS" if not failed else "FAIL",
        "checked_at_utc": _now_utc(),
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "failed_checks": failed,
    }
    if extra:
        result.update(extra)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--inspect-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    modes = [args.static, args.self_check, args.inspect_json is not None]
    if sum(1 for mode in modes if mode) > 1:
        parser.error("choose only one of --static, --self-check or --inspect-json")
    if args.inspect_json is not None:
        checks = inspect_check(args.inspect_json, args.repo_root)
        mode = "inspect-json"
    elif args.self_check:
        checks = self_check(args.repo_root)
        mode = "self-check"
    else:
        checks = static_check(args.repo_root)
        mode = "static"

    result = _result(mode, checks)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
