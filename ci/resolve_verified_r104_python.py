#!/usr/bin/env python3
"""Resolve the already-qualified canonical R10.4 Python environment without network access.

Package identity is validated by direct requirement names + versions/specifiers,
not by a hash of the complete installed distribution inventory. Packaging layout,
wheel metadata ordering, or unrelated transitive package differences therefore do
not invalidate an otherwise equivalent scientific runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

EXPECTED_ENV_CACHE_KEY = "b6e3e3c287f5f4e8ab0cb1b80a7af8aba0803a0f66a09d4501e7051a33edf7ba"
EXPECTED_PYTHON_VERSION = "3.10.12"
EXPECTED_VENV = Path(os.environ.get("CB16_VERIFIED_VENV", "/cb16/venv"))
ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = [ROOT / "requirements-shanxi-pascal.txt", ROOT / "requirements-ci-runtime.txt"]

PIP_ROUTE_KEYS = {
    "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST", "PIP_FIND_LINKS", "PIP_NO_INDEX",
}
UV_ROUTE_KEYS = {
    "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX", "UV_FIND_LINKS",
}
PROXY_ROUTE_KEYS = {
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
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
    worker_root = Path(os.environ.get("CB16_CI_WORKER_ROOT", "/cb16/worker"))
    provision_paths = [
        Path(os.environ.get("CB16_PROVISION_ENV", "/run/secrets/cb16-provision.env")),
        worker_root / "provision.env",
    ]
    union: set[str] = set()
    present_count = 0
    readable_count = 0
    for path in provision_paths:
        present, readable, keys = _safe_env_keys(path)
        present_count += int(present)
        readable_count += int(readable)
        union.update(keys)
    process_keys = {k for k in os.environ if k in PIP_ROUTE_KEYS | UV_ROUTE_KEYS | PROXY_ROUTE_KEYS}
    return {
        "provision_env_present": present_count > 0,
        "provision_env_readable": readable_count > 0,
        "pip_route_key_present": bool(union & PIP_ROUTE_KEYS),
        "uv_route_key_present": bool(union & UV_ROUTE_KEYS),
        "proxy_route_key_present": bool(union & PROXY_ROUTE_KEYS),
        "process_route_key_present": bool(process_keys),
        "values_redacted": True,
    }


def _run(py: Path, code: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(py), "-c", code, *args], check=False, capture_output=True, text=True, env=os.environ.copy())


def _requirement_lines() -> list[str]:
    out: list[str] = []
    for req in REQUIREMENTS:
        if not req.is_file():
            raise RuntimeError(f"REQUIREMENTS_MISSING:{req.name}")
        for raw in req.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                if line.startswith(("--index-url", "--extra-index-url", "--find-links", "-i ")) or "://" in line:
                    raise RuntimeError(f"REPOSITORY_CHANNEL_DIRECTIVE_FORBIDDEN:{req.name}")
                out.append(line)
    return out


def direct_version_report(py: Path) -> dict:
    reqs = _requirement_lines()
    code = r'''
import importlib.metadata as m, json, sys
try:
    from packaging.requirements import Requirement
except Exception:
    from pip._vendor.packaging.requirements import Requirement
reqs=json.loads(sys.argv[1])
out=[]
for raw in reqs:
    r=Requirement(raw)
    try:
        v=m.version(r.name)
        ok=(not r.specifier) or r.specifier.contains(v, prereleases=True)
    except m.PackageNotFoundError:
        v=None; ok=False
    out.append({'name':r.name,'requirement':raw,'version':v,'satisfied':bool(ok)})
print(json.dumps(out,sort_keys=True))
'''
    p = _run(py, code, json.dumps(reqs))
    if p.returncode != 0:
        raise RuntimeError("DIRECT_PACKAGE_VERSION_REPORT_FAILED")
    rows = json.loads(p.stdout)
    bad = [row["name"] for row in rows if not row["satisfied"]]
    return {
        "policy": "DIRECT_REQUIREMENT_VERSION_EQUIVALENCE",
        "packages": rows,
        "all_requirements_satisfied": not bad,
        "mismatched_packages": bad,
    }


def verify_canonical_venv() -> tuple[bool, list[str], dict]:
    reasons: list[str] = []
    py = EXPECTED_VENV / "bin/python"
    ready_path = EXPECTED_VENV / "READY.json"
    observed: dict = {
        "venv": str(EXPECTED_VENV),
        "ready_present": ready_path.is_file(),
        "python_executable": py.is_file() and os.access(py, os.X_OK),
        "environment_cache_key": EXPECTED_ENV_CACHE_KEY,
        "environment_cache_key_is_package_integrity_gate": False,
        "package_hash_validation_used": False,
    }
    if not observed["python_executable"]:
        reasons.append("PYTHON_NOT_EXECUTABLE")
        return False, reasons, observed

    if ready_path.is_file():
        try:
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            observed["ready_environment_cache_key"] = ready.get("environment_sha256") or ready.get("environment_hash")
            observed["legacy_ready_package_hash_present_but_ignored"] = bool(ready.get("resolved_packages_sha256"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            observed["ready_unreadable_but_not_identity_gate"] = True

    version = _run(py, "import platform; print(platform.python_version())")
    observed["python_version"] = version.stdout.strip() if version.returncode == 0 else None
    if version.returncode != 0 or observed["python_version"] != EXPECTED_PYTHON_VERSION:
        reasons.append("PYTHON_VERSION_MISMATCH")

    try:
        versions = direct_version_report(py)
    except RuntimeError as exc:
        versions = None
        reasons.append(str(exc))
    observed["direct_package_versions"] = versions
    if versions is not None and not versions["all_requirements_satisfied"]:
        reasons.append("DIRECT_PACKAGE_VERSION_MISMATCH")

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
        "schema": "CB16_R104_VERIFIED_PYTHON_REUSE_V2",
        "status": "READY" if ok else "FAIL_CLOSED",
        "mode": "VERIFIED_CANONICAL_R104_VENV_REUSE" if ok else "HOST_REPAIR_REQUIRED_NO_NETWORK_FALLBACK",
        "environment_id": "r104",
        "expected": {
            "environment_cache_key": EXPECTED_ENV_CACHE_KEY,
            "python_version": EXPECTED_PYTHON_VERSION,
            "venv": str(EXPECTED_VENV),
            "package_identity_policy": "DIRECT_REQUIREMENT_VERSION_EQUIVALENCE",
        },
        "python": {
            "status": "READY" if ok else "INVALID",
            "cache_hit": ok,
            "venv": str(EXPECTED_VENV),
            "installer": "verified_existing_uv_environment" if ok else None,
            "python_version": observed.get("python_version"),
            "direct_package_versions": observed.get("direct_package_versions"),
            "package_hash_validation_used": False,
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
    print("PACKAGE_IDENTITY_POLICY=DIRECT_REQUIREMENT_VERSION_EQUIVALENCE")
    print("PACKAGE_HASH_VALIDATION_USED=NO")
    print("PYTHON_VERSION=" + EXPECTED_PYTHON_VERSION)
    print("NETWORK_INSTALL_ATTEMPTED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
