#!/usr/bin/env python3
"""R21 RC2 R3 Recovery runner gate verifier.

Runs read-only checks plus a bounded create/fsync/unlink write probe inside the
new R21 Recovery runner. It does not start training, read scientific payloads,
or mutate any protected input.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import pwd
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_SPEC_PATH = REPO_ROOT / "infra" / "shanxi_runner" / "runner_launch_spec_r21_v2.json"
PROFILE_ID = "R21_RC2_FAST_HOT_RUNNER_V2"
EXPECTED_SHM_BYTES = 2147483648
FAST_HOT_PATH = Path("/cb16/fast_hot")
PROTECTED_RO_PATHS = (
    "/cb16/raw",
    "/cb16/runtime/r104",
    "/cb16/package",
    "/cb16/parent-r101",
    "/cb16/parents",
    "/cb16/r2-authority",
    "/run/secrets",
)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _check(checks: list[dict[str, Any]], name: str, condition: bool, detail: str = "") -> None:
    checks.append({"check": name, "ok": bool(condition), "detail": detail})


def _mountinfo() -> list[list[str]]:
    path = Path("/proc/self/mountinfo")
    if not path.exists():
        return []
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mount_entry(mounts: list[list[str]], destination: str) -> list[str] | None:
    for fields in mounts:
        if len(fields) > 5 and fields[4] == destination:
            return fields
    return None


def _mount_is_read_only(fields: list[str] | None) -> bool | None:
    if fields is None or len(fields) < 6:
        return None
    return "ro" in set(fields[5].split(","))


def _rotational_for_mount(fields: list[str] | None) -> tuple[bool | None, str]:
    candidates: list[Path] = []
    if fields is not None and len(fields) > 2 and ":" in fields[2]:
        major, minor = fields[2].split(":", 1)
        candidates.append(Path(f"/sys/dev/block/{major}:{minor}/queue/rotational"))
    candidates.append(Path("/sys/block/sdb/queue/rotational"))
    for candidate in candidates:
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        return value == "1", str(candidate)
    return None, "UNREADABLE"


def verify(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    spec = _load_json(repo_root / PROFILE_SPEC_PATH.relative_to(REPO_ROOT))
    checks: list[dict[str, Any]] = []
    env = {entry["name"]: entry["value"] for entry in spec["environment"]}

    _check(checks, "profile_schema", spec.get("schema") == "CB16_SHANXI_RUNNER_LAUNCH_SPEC_R21_V2", str(spec.get("schema")))
    _check(checks, "profile_id_env", os.environ.get("CB16_RUNNER_PROFILE_ID") == PROFILE_ID, str(os.environ.get("CB16_RUNNER_PROFILE_ID")))
    _check(checks, "profile_id_spec", spec.get("profile_id") == PROFILE_ID, str(spec.get("profile_id")))
    _check(checks, "hostname", platform.node() == spec["container"]["hostname"], platform.node())
    _check(checks, "uid", os.getuid() == 1001, str(os.getuid()))
    _check(checks, "gid", os.getgid() == 1001, str(os.getgid()))
    try:
        _check(checks, "user_name", pwd.getpwuid(os.getuid()).pw_name == "cb16-runner", pwd.getpwuid(os.getuid()).pw_name)
    except KeyError as exc:  # pragma: no cover - defensive
        _check(checks, "user_name", False, str(exc))

    os_release = {}
    if Path("/etc/os-release").exists():
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip().strip('"')
    _check(checks, "os_id", os_release.get("ID") == "debian", str(os_release.get("ID")))
    _check(checks, "os_version", os_release.get("VERSION_ID") == "12", str(os_release.get("VERSION_ID")))
    _check(checks, "python_version", platform.python_version() == "3.10.21", platform.python_version())

    shm_size = os.statvfs("/dev/shm").f_frsize * os.statvfs("/dev/shm").f_blocks
    _check(checks, "shm_size_2gib", shm_size == EXPECTED_SHM_BYTES, str(shm_size))

    for name, value in env.items():
        _check(checks, f"env:{name}", os.environ.get(name) == value, f"live={os.environ.get(name)!r}")

    provision_env = Path(os.environ.get("CB16_PROVISION_ENV", "/nonexistent"))
    _check(checks, "provision_env_exists", provision_env.is_file(), str(provision_env))
    _check(checks, "provision_env_readable", os.access(provision_env, os.R_OK), str(provision_env))

    _check(checks, "fast_hot_dir", FAST_HOT_PATH.is_dir(), str(FAST_HOT_PATH))
    if FAST_HOT_PATH.is_dir():
        info = FAST_HOT_PATH.stat()
        _check(checks, "fast_hot_owner_uid", info.st_uid == 1001, str(info.st_uid))
        _check(checks, "fast_hot_owner_gid", info.st_gid == 1001, str(info.st_gid))
        _check(checks, "fast_hot_mode", stat.S_IMODE(info.st_mode) == 0o750, oct(stat.S_IMODE(info.st_mode)))

        probe = FAST_HOT_PATH / f".cb16_r3_write_probe_{os.getpid()}"
        try:
            fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(fd, b"r21-rc2-r3\n")
            os.fsync(fd)
            os.close(fd)
            probe.unlink()
            _check(checks, "fast_hot_write_probe", True, "WRITE_FSYNC_UNLINK_PASS")
        except OSError as exc:
            try:
                if probe.exists():
                    probe.unlink()
            except OSError:
                pass
            _check(checks, "fast_hot_write_probe", False, f"{type(exc).__name__}:{exc}")

        profile_file = FAST_HOT_PATH / "PROFILE_IDENTITY.json"
        guard_file = FAST_HOT_PATH / "CAPACITY_GUARD.json"
        try:
            profile_payload = json.loads(profile_file.read_text(encoding="utf-8"))
            _check(checks, "fast_hot_profile_file", profile_payload.get("profile_id") == PROFILE_ID, json.dumps(profile_payload, sort_keys=True))
        except Exception as exc:
            _check(checks, "fast_hot_profile_file", False, f"{type(exc).__name__}:{exc}")
        try:
            guard_payload = json.loads(guard_file.read_text(encoding="utf-8"))
            expected_quota = spec["fast_hot"]["quota_bytes"]
            expected_reserve = spec["fast_hot"]["reserve_floor_bytes"]
            _check(
                checks,
                "fast_hot_capacity_guard",
                guard_payload.get("quota_bytes") == expected_quota and guard_payload.get("reserve_floor_bytes") == expected_reserve,
                json.dumps(guard_payload, sort_keys=True),
            )
        except Exception as exc:
            _check(checks, "fast_hot_capacity_guard", False, f"{type(exc).__name__}:{exc}")

    mounts = _mountinfo()
    for path in PROTECTED_RO_PATHS:
        entry = _mount_entry(mounts, path)
        _check(checks, f"protected_ro:{path}", _mount_is_read_only(entry) is True, f"entry={entry}")
    fast_hot_entry = _mount_entry(mounts, str(FAST_HOT_PATH))
    _check(checks, "fast_hot_mount_rw", _mount_is_read_only(fast_hot_entry) is False, f"entry={fast_hot_entry}")
    rotational, source = _rotational_for_mount(fast_hot_entry)
    _check(checks, "fast_hot_non_rotational", rotational is False, f"source={source} rotational={rotational}")

    gpu = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
    gpu_text = (gpu.stdout or "").strip()
    _check(checks, "gpu_visible", gpu.returncode == 0 and "GTX 1060" in gpu_text, gpu_text or gpu.stderr.strip())
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    checks = verify(args.repo_root)
    failed = [entry for entry in checks if not entry["ok"]]
    result = {
        "schema": "CB16_R21_RC2_R3_RECOVERY_RUNNER_GATE_V1",
        "profile_id": PROFILE_ID,
        "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PASS" if not failed else "FAIL",
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "failed_checks": failed,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
