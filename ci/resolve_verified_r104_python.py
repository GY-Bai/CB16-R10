#!/usr/bin/env python3
"""Resolve the already-qualified canonical R10.4 Python environment without network access.

This helper intentionally does not install or upgrade packages.  The native R2
qualification must reuse the exact R10.4 environment that already ran the frozen
canonical campaign.  If that environment cannot be verified, the helper records
only sanitized host mirror/proxy-presence evidence and fails closed so a separate
host repair can be performed without weakening scientific/runtime identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

EXPECTED_ENV_HASH = "b6e3e3c287f5f4e8ab0cb1b80a7af8aba0803a0f66a09d4501e7051a33edf7ba"
EXPECTED_PACKAGES_SHA256 = "edd56213452ec52f2d0464126be2e84bf5f456f049b167530d20e4e6359256d6"
EXPECTED_PYTHON_VERSION = "3.10.12"
EXPECTED_VENV = Path("/data/cb16_ci/venvs") / EXPECTED_ENV_HASH

PIP_ROUTE_KEYS = {
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "PIP_TRUSTED_HOST",
    "PIP_FIND_LINKS",
    "PIP_NO_INDEX",
}
UV_ROUTE_KEYS = {
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "UV_FIND_LINKS",
}
PROXY_ROUTE_KEYS = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
}


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _safe_env_keys(path: Path) -> tuple[bool, bool, set[str]]:
    if not path.exists():
        return False, False, set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return True, False, set()
    keys: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return True, True, keys


def sanitized_host_route_summary() -> dict:
    worker_root = Path(os.environ.get("CB16_CI_WORKER_ROOT", "/data/cb16_ci"))
    provision_paths = [Path("/etc/cb16-ci/provision.env"), worker_root / "provision.env"]
    union: set[str] = set()
    present_count = 0
    readable_count = 0
    for path in provision_paths:
        present, readable, keys = _safe_env_keys(path)
        present_count += int(present)
        readable_count += int(readable)
        union.update(keys)

    process_keys = {k for k in os.environ if k in PIP_ROUTE_KEYS | UV_ROUTE_KEYS | PROXY_ROUTE_KEYS}
    home = Path.home()
    standard_pip = [Path("/etc/pip.conf"), home / ".pip/pip.conf", home / ".config/pip/pip.conf"]
    standard_uv = [home / ".config/uv/uv.toml", home / ".uv/uv.toml"]

    return {
        "provision_env_present": present_count > 0,
        "provision_env_readable": readable_count > 0,
        "pip_route_key_present": bool(union & PIP_ROUTE_KEYS),
        "uv_route_key_present": bool(union & UV_ROUTE_KEYS),
        "proxy_route_key_present": bool(union & PROXY_ROUTE_KEYS),
        "process_route_key_present": bool(process_keys),
        "standard_pip_config_present": any(p.exists() for p in standard_pip),
        "standard_uv_config_present": any(p.exists() for p in standard_uv),
        "values_redacted": True,
    }


def _run(py: Path, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(py), "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def resolved_package_hash(py: Path) -> str:
    code = (
        "import importlib.metadata as m,json;"
        "x=sorted((d.metadata.get('Name','').lower(),d.version) for d in m.distributions());"
        "print(json.dumps(x,separators=(',',':')))"
    )
    p = _run(py, code)
    if p.returncode != 0:
        raise RuntimeError("PACKAGE_INVENTORY_FAILED")
    return hashlib.sha256(p.stdout.strip().encode("utf-8")).hexdigest()


def verify_canonical_venv() -> tuple[bool, list[str], dict]:
    reasons: list[str] = []
    py = EXPECTED_VENV / "bin/python"
    ready_path = EXPECTED_VENV / "READY.json"
    observed: dict = {
        "venv": str(EXPECTED_VENV),
        "ready_present": ready_path.is_file(),
        "python_executable": py.is_file() and os.access(py, os.X_OK),
    }

    if not ready_path.is_file():
        reasons.append("READY_JSON_MISSING")
        return False, reasons, observed
    if not observed["python_executable"]:
        reasons.append("PYTHON_NOT_EXECUTABLE")
        return False, reasons, observed

    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        reasons.append("READY_JSON_UNREADABLE")
        return False, reasons, observed

    ready_env = ready.get("environment_sha256") or ready.get("environment_hash")
    ready_pkg = ready.get("resolved_packages_sha256")
    observed["ready_environment_hash"] = ready_env
    observed["ready_packages_sha256"] = ready_pkg
    if ready_env != EXPECTED_ENV_HASH:
        reasons.append("READY_ENVIRONMENT_HASH_MISMATCH")
    if ready_pkg != EXPECTED_PACKAGES_SHA256:
        reasons.append("READY_PACKAGE_HASH_MISMATCH")

    version = _run(py, "import platform; print(platform.python_version())")
    observed["python_version"] = version.stdout.strip() if version.returncode == 0 else None
    if version.returncode != 0 or observed["python_version"] != EXPECTED_PYTHON_VERSION:
        reasons.append("PYTHON_VERSION_MISMATCH")

    try:
        inventory_hash = resolved_package_hash(py)
    except RuntimeError as exc:
        inventory_hash = None
        reasons.append(str(exc))
    observed["resolved_packages_sha256"] = inventory_hash
    if inventory_hash != EXPECTED_PACKAGES_SHA256:
        reasons.append("PACKAGE_INVENTORY_HASH_MISMATCH")

    canary = _run(
        py,
        "import torch,numpy,pandas,lightgbm,pytest;"
        "assert torch.cuda.is_available(),'CUDA_UNAVAILABLE';"
        "print(torch.__version__);print(torch.version.cuda);print(torch.cuda.get_device_name(0))",
    )
    observed["import_cuda_canary_pass"] = canary.returncode == 0
    if canary.returncode != 0:
        reasons.append("IMPORT_OR_CUDA_CANARY_FAILED")

    return not reasons, reasons, observed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    route_summary = sanitized_host_route_summary()
    ok, reasons, observed = verify_canonical_venv()
    result = {
        "schema": "CB16_R104_VERIFIED_PYTHON_REUSE_V1",
        "status": "READY" if ok else "FAIL_CLOSED",
        "mode": "VERIFIED_CANONICAL_R104_VENV_REUSE" if ok else "HOST_REPAIR_REQUIRED_NO_NETWORK_FALLBACK",
        "environment_id": "r104",
        "expected": {
            "environment_hash": EXPECTED_ENV_HASH,
            "resolved_packages_sha256": EXPECTED_PACKAGES_SHA256,
            "python_version": EXPECTED_PYTHON_VERSION,
            "venv": str(EXPECTED_VENV),
        },
        "python": {
            "status": "READY" if ok else "INVALID",
            "cache_hit": ok,
            "venv": str(EXPECTED_VENV),
            "installer": "verified_existing_uv_environment" if ok else None,
            "python_version": observed.get("python_version"),
            "resolved_packages_sha256": observed.get("resolved_packages_sha256"),
            "public_index_fallback_used": False,
            "public_direct_fallback_used": False,
            "network_install_attempted": False,
        },
        "observed": observed,
        "host_route_summary": route_summary,
        "failure_reasons": reasons,
        "scientific_semantics_changed": False,
        "final_holdout_2025_09_accessed": False,
    }
    atomic_write_json(args.out, result)

    if not ok:
        print("R104_VERIFIED_VENV_REUSE=FAIL_CLOSED")
        print("HOST_ROUTE_SUMMARY=" + json.dumps(route_summary, sort_keys=True))
        print("FAILURE_REASONS=" + ",".join(reasons))
        return 78

    print("R104_VERIFIED_VENV_REUSE=PASS")
    print("ENVIRONMENT_HASH=" + EXPECTED_ENV_HASH)
    print("PACKAGES_SHA256=" + EXPECTED_PACKAGES_SHA256)
    print("PYTHON_VERSION=" + EXPECTED_PYTHON_VERSION)
    print("NETWORK_INSTALL_ATTEMPTED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
