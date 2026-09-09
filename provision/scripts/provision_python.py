#!/usr/bin/env python3
"""Create or reuse a CB16 Python environment from host-owned UV routing.

Repository code defines package/version requirements only. It must not select a
package mirror, public index, wheel URL, or find-links location. Those transport
choices belong to the host environment and are injected into the Docker runner
through CB16_PROVISION_ENV plus the local GOST routing policy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from provision_common import VENV_ROOT, atomic_write_json, ensure_dirs, repo_root
from resolve_environment import resolve

_ROUTE_KEYS = (
    "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST", "PIP_FIND_LINKS",
    "PIP_NO_INDEX", "PIP_NO_CACHE_DIR", "PIP_CONFIG_FILE",
    "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX", "UV_FIND_LINKS",
    "UV_NO_INDEX", "UV_NO_CACHE",
    "HF_ENDPOINT", "HF_TOKEN",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
)

_FORBIDDEN_REQUIREMENT_PREFIXES = (
    "--index-url", "--extra-index-url", "--find-links", "-i ",
)


def load_provision_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in (
        Path(os.environ.get("CB16_PROVISION_ENV", "/run/secrets/cb16-provision.env")),
        Path(os.environ.get("CB16_CI_WORKER_ROOT", "/cb16/worker")) / "provision.env",
    ):
        if not path.exists():
            continue
        try:
            lines = path.read_text().splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def _install_env(prov_env: dict[str, str]) -> dict[str, str]:
    """Inherit host routing exactly; do not invent or rewrite package channels."""
    env = os.environ.copy()
    for key in _ROUTE_KEYS:
        if prov_env.get(key):
            env[key] = prov_env[key]
    return env


def _requirement_lines(reqs: list[Path]) -> list[str]:
    out: list[str] = []
    for req in reqs:
        for raw in req.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                out.append(line)
    return out


def _forbidden_requirement_channels(reqs: list[Path]) -> list[str]:
    hits: list[str] = []
    for req in reqs:
        for lineno, raw in enumerate(req.read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lower = line.lower()
            if lower.startswith(_FORBIDDEN_REQUIREMENT_PREFIXES) or "://" in line:
                hits.append(f"{req.name}:{lineno}")
    return hits


def _validate_host_env_policy(python_cfg: dict, reqs: list[Path], install_env: dict[str, str]) -> dict:
    policy = str(python_cfg.get("index_policy", "INHERIT")).upper()
    installer_policy = str(python_cfg.get("installer_policy", "UV_PREFERRED")).upper()
    forbidden = _forbidden_requirement_channels(reqs)
    if forbidden:
        raise RuntimeError("PYTHON_REQUIREMENT_CHANNEL_FORBIDDEN:" + ",".join(forbidden))

    if policy == "HOST_ENV_REQUIRED":
        if not (install_env.get("UV_INDEX_URL") or install_env.get("UV_DEFAULT_INDEX")):
            raise RuntimeError("PYTHON_HOST_ENV_REQUIRED_UV_PRIMARY_ROUTE_MISSING")
        needs_accel = any("+cu" in line.lower() for line in _requirement_lines(reqs))
        if needs_accel and not (install_env.get("UV_EXTRA_INDEX_URL") or install_env.get("UV_FIND_LINKS")):
            raise RuntimeError("PYTHON_HOST_ENV_REQUIRED_UV_ACCELERATOR_ROUTE_MISSING")
    elif policy not in {"INHERIT", "HOST_ENV_REQUIRED"}:
        raise RuntimeError(f"PYTHON_INDEX_POLICY_UNSUPPORTED:{policy}")

    if installer_policy not in {"UV_REQUIRED", "UV_PREFERRED"}:
        raise RuntimeError(f"PYTHON_INSTALLER_POLICY_UNSUPPORTED:{installer_policy}")

    return {
        "index_policy": policy,
        "installer_policy": installer_policy,
        "uv_primary_route_present": bool(install_env.get("UV_INDEX_URL") or install_env.get("UV_DEFAULT_INDEX")),
        "uv_extra_route_present": bool(install_env.get("UV_EXTRA_INDEX_URL")),
        "uv_find_links_present": bool(install_env.get("UV_FIND_LINKS")),
        "proxy_present": bool(install_env.get("HTTPS_PROXY") or install_env.get("https_proxy")),
        "no_proxy_present": bool(install_env.get("NO_PROXY") or install_env.get("no_proxy")),
        "repository_channel_directive_count": len(forbidden),
        "routing_owner": "HOST_ENVIRONMENT_AND_LOCAL_PROXY",
        "values_redacted": True,
    }


def _uv_install_command(venv_python: Path, reqs: list[Path]) -> list[str]:
    """Build the UV command with zero repository-selected channel arguments."""
    cmd = ["uv", "pip", "install", "--python", str(venv_python), "--index-strategy", "unsafe-best-match"]
    for req in reqs:
        cmd += ["-r", str(req)]
    return cmd


def _assert_no_channel_args(cmd: list[str]) -> None:
    forbidden = {"--index-url", "--extra-index-url", "--find-links", "-i"}
    if any(token in forbidden or "://" in token for token in cmd):
        raise RuntimeError("PYTHON_INSTALL_COMMAND_CHANNEL_FORBIDDEN")


def _classify_failure(text: str) -> str:
    s = text.lower()
    if "no space left" in s or "disk quota" in s:
        return "DISK_SPACE"
    if any(x in s for x in ("401", "403", "407", "unauthorized", "forbidden", "proxy authentication required")):
        return "INDEX_AUTH"
    if any(x in s for x in ("timed out", "timeout", "connection", "name resolution", "dns", "ssl", "certificate", "failed to fetch", "network", "proxy")):
        return "NETWORK_OR_INDEX"
    if any(x in s for x in ("no matching distribution", "requires-python", "requires python", "not a supported wheel", "unsupported python")):
        return "NO_COMPATIBLE_DISTRIBUTION"
    if any(x in s for x in ("resolution impossible", "conflicting dependencies", "no solution found", "cannot install")):
        return "RESOLUTION_CONFLICT"
    if any(x in s for x in ("failed building wheel", "build backend", "subprocess-exited-with-error", "cmake", "compiler")):
        return "BUILD_FAILURE"
    return "UNKNOWN"


def _run_capture(cmd: list[str], *, env: dict[str, str] | None = None) -> tuple[bool, str, str]:
    try:
        p = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", ""
    except OSError:
        return False, "OS_ERROR", ""
    text = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode == 0:
        return True, "PASS", text
    return False, _classify_failure(text), text


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split("."))


def _python_satisfies(spec: str) -> bool:
    current = tuple(sys.version_info[:3])
    for raw in filter(None, (x.strip() for x in spec.split(","))):
        m = re.fullmatch(r"(>=|<=|>|<|==)\s*([0-9]+(?:\.[0-9]+){0,2})", raw)
        if not m:
            raise RuntimeError("PYTHON_REQUIRES_SPEC_UNSUPPORTED")
        op, version = m.groups()
        wanted = _version_tuple(version)
        lhs = current[: len(wanted)]
        ok = {">=": lhs >= wanted, "<=": lhs <= wanted, ">": lhs > wanted, "<": lhs < wanted, "==": lhs == wanted}[op]
        if not ok:
            return False
    return True


def _direct_version_report(venv_python: Path, reqs: list[Path]) -> dict:
    requirements = _requirement_lines(reqs)
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
    p = subprocess.run([str(venv_python), "-c", code, json.dumps(requirements)], check=False, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("DIRECT_PACKAGE_VERSION_REPORT_FAILED")
    rows = json.loads(p.stdout)
    if not all(row["satisfied"] for row in rows):
        bad = [row["name"] for row in rows if not row["satisfied"]]
        raise RuntimeError("DIRECT_PACKAGE_VERSION_MISMATCH:" + ",".join(bad))
    return {
        "policy": "DIRECT_REQUIREMENT_VERSION_EQUIVALENCE",
        "packages": rows,
        "all_requirements_satisfied": True,
    }


def canary_imports(venv_python: Path, imports: list[str]) -> None:
    if not imports:
        return
    code = "\n".join(f"import {imp.split('[')[0].split(':')[0]}" for imp in imports)
    ok, cls, _ = _run_capture([str(venv_python), "-c", code])
    if not ok:
        raise RuntimeError(f"PYTHON_IMPORT_CANARY_FAILED_{cls}")


def provision(profile: str) -> dict:
    resolved = resolve(profile)
    manifest = resolved["manifest"]
    python_cfg = manifest.get("python") or {}
    required_python = python_cfg.get("requires")
    if required_python and not _python_satisfies(required_python):
        raise RuntimeError(f"PYTHON_VERSION_UNSUPPORTED_{sys.version_info.major}_{sys.version_info.minor}")

    reqs = [repo_root() / r for r in manifest.get("requirements", [])]
    for req in reqs:
        if not req.is_file():
            raise RuntimeError(f"PYTHON_REQUIREMENTS_FILE_MISSING_{req.name.replace('.', '_').upper()}")

    install_env = _install_env(load_provision_env())
    route_summary = _validate_host_env_policy(python_cfg, reqs, install_env)
    installer_policy = str(python_cfg.get("installer_policy", "UV_PREFERRED")).upper()
    if installer_policy == "UV_REQUIRED" and not shutil.which("uv"):
        raise RuntimeError("PYTHON_UV_REQUIRED_BUT_NOT_INSTALLED")

    env_hash = resolved["environment_sha256"]  # cache/profile key only; not a package-binary identity gate
    venv_dir = VENV_ROOT / env_hash
    ready_path = venv_dir / "READY.json"
    venv_python = venv_dir / "bin/python"

    if ready_path.exists() and venv_python.is_file():
        try:
            ready = json.loads(ready_path.read_text())
            canary_imports(venv_python, manifest.get("imports", []))
            versions = _direct_version_report(venv_python, reqs)
            return {
                "environment_id": resolved["environment_id"],
                "environment_hash": env_hash,
                "python": {
                    "status": "READY", "cache_hit": True, "venv": str(venv_dir),
                    "installer": ready.get("installer", "uv_cached"),
                    "python_version": ready.get("python_version"),
                    "direct_package_versions": versions,
                    "package_hash_validation_used": False,
                    "public_index_fallback_used": False,
                    "public_direct_fallback_used": False,
                    "install_route_summary": route_summary,
                },
            }
        except Exception:
            shutil.rmtree(venv_dir, ignore_errors=True)

    ensure_dirs()
    shutil.rmtree(venv_dir, ignore_errors=True)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)

    ok, cls, _ = _run_capture(["uv", "venv", str(venv_dir)], env=install_env)
    if not ok:
        raise RuntimeError(f"PYTHON_UV_VENV_FAILED_{cls}")
    venv_python = venv_dir / "bin/python"

    cmd = _uv_install_command(venv_python, reqs)
    _assert_no_channel_args(cmd)
    ok, cls, _ = _run_capture(cmd, env=install_env)
    if not ok:
        raise RuntimeError(f"PYTHON_REQUIREMENTS_INSTALL_FAILED_UV_{cls}")
    ok, cls, _ = _run_capture(["uv", "pip", "check", "--python", str(venv_python)], env=install_env)
    if not ok:
        raise RuntimeError(f"PYTHON_DEPENDENCY_CHECK_FAILED_UV_{cls}")

    canary_imports(venv_python, manifest.get("imports", []))
    versions = _direct_version_report(venv_python, reqs)
    pyver = subprocess.run([str(venv_python), "-c", "import platform;print(platform.python_version())"], check=True, capture_output=True, text=True).stdout.strip()
    ready = {
        "schema": "CB16_PYTHON_ENV_READY_V2",
        "environment_id": resolved["environment_id"],
        "environment_hash": env_hash,
        "environment_sha256": env_hash,
        "environment_hash_role": "PROFILE_CACHE_KEY_ONLY",
        "installer": "uv",
        "python_version": pyver,
        "package_identity_policy": "DIRECT_REQUIREMENT_VERSION_EQUIVALENCE",
        "direct_package_versions": versions,
        "package_hash_validation_used": False,
        "public_index_fallback_used": False,
        "public_direct_fallback_used": False,
        "install_route_summary": route_summary,
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    atomic_write_json(ready_path, ready)
    return {
        "environment_id": resolved["environment_id"],
        "environment_hash": env_hash,
        "python": {
            "status": "READY", "cache_hit": False, "venv": str(venv_dir),
            "installer": "uv", "python_version": pyver,
            "direct_package_versions": versions,
            "package_hash_validation_used": False,
            "public_index_fallback_used": False,
            "public_direct_fallback_used": False,
            "install_route_summary": route_summary,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = provision(args.profile)
    if args.out:
        atomic_write_json(Path(args.out), result)
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
