#!/usr/bin/env python3
"""Create or reuse a CB16 Python environment for a resolved environment manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
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

_PACKAGE_ROUTE_KEYS = (
    "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS",
    "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX", "UV_FIND_LINKS",
)


def load_provision_env() -> dict[str, str]:
    """Load host-local provision routing without making it a repository secret.

    `/etc/cb16-ci/provision.env` is the canonical Shanxi binding. A worker-root
    override remains supported. Unreadable optional files are skipped here;
    profiles that require a host mirror fail closed later with an explicit route
    policy error instead of a raw PermissionError.
    """
    out: dict[str, str] = {}
    for path in (
        Path("/etc/cb16-ci/provision.env"),
        Path(os.environ.get("CB16_CI_WORKER_ROOT", "/data/cb16_ci")) / "provision.env",
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


def _bridge_installer_routes(env: dict[str, str]) -> dict[str, str]:
    """Make host mirror bindings usable by both uv and pip."""
    env = env.copy()
    uv_primary = env.get("UV_INDEX_URL") or env.get("UV_DEFAULT_INDEX")
    pip_primary = env.get("PIP_INDEX_URL")
    if pip_primary and not uv_primary:
        env["UV_INDEX_URL"] = pip_primary
    elif uv_primary and not pip_primary:
        env["PIP_INDEX_URL"] = uv_primary

    for pip_key, uv_key in (
        ("PIP_EXTRA_INDEX_URL", "UV_EXTRA_INDEX_URL"),
        ("PIP_FIND_LINKS", "UV_FIND_LINKS"),
    ):
        if env.get(pip_key) and not env.get(uv_key):
            env[uv_key] = env[pip_key]
        elif env.get(uv_key) and not env.get(pip_key):
            env[pip_key] = env[uv_key]
    return env


def _configured_package_mirror_hosts(env: dict[str, str]) -> list[str]:
    hosts: set[str] = set()
    for key in _PACKAGE_ROUTE_KEYS:
        for raw in (env.get(key) or "").split():
            try:
                u = urllib.parse.urlsplit(raw)
            except ValueError:
                continue
            if u.scheme in {"http", "https"} and u.hostname:
                hosts.add(u.hostname.lower())
    return sorted(hosts)


def _merge_no_proxy(existing: str, hosts: list[str]) -> str:
    current: list[str] = []
    seen: set[str] = set()
    for raw in existing.replace(" ", ",").split(","):
        token = raw.strip()
        if token and token not in seen:
            current.append(token)
            seen.add(token)
    for host in hosts:
        if host not in seen:
            current.append(host)
            seen.add(host)
    return ",".join(current)


def _apply_mirror_proxy_policy(env: dict[str, str], python_cfg: dict) -> dict[str, str]:
    """Route only host-declared package mirrors directly when requested.

    The runner-wide proxy remains active for GitHub/Azure/other external traffic.
    Only hostnames already present in the host-local package mirror bindings are
    added to NO_PROXY. Repository requirement files cannot add hosts because R10.4
    forbids embedded index directives.
    """
    env = env.copy()
    policy = str(python_cfg.get("mirror_proxy_policy", "INHERIT")).upper()
    if policy == "INHERIT":
        return env
    if policy != "DIRECT_FOR_HOST_MIRRORS":
        raise RuntimeError(f"PYTHON_MIRROR_PROXY_POLICY_UNSUPPORTED:{policy}")
    hosts = _configured_package_mirror_hosts(env)
    if not hosts:
        raise RuntimeError("PYTHON_DIRECT_HOST_MIRROR_POLICY_HAS_NO_CONFIGURED_HOSTS")
    existing = env.get("NO_PROXY") or env.get("no_proxy") or ""
    merged = _merge_no_proxy(existing, hosts)
    env["NO_PROXY"] = merged
    env["no_proxy"] = merged
    return env


def _install_env(prov_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for key in _ROUTE_KEYS:
        if prov_env.get(key):
            env[key] = prov_env[key]
    return _bridge_installer_routes(env)


def _route_summary(env: dict[str, str], *, python_cfg: dict, embedded_indexes: list[str]) -> dict:
    index_policy = str(python_cfg.get("index_policy", "INHERIT")).upper()
    mirror_proxy_policy = str(python_cfg.get("mirror_proxy_policy", "INHERIT")).upper()
    hosts = _configured_package_mirror_hosts(env)
    no_proxy_tokens = {x.strip().lower() for x in (env.get("NO_PROXY") or env.get("no_proxy") or "").replace(" ", ",").split(",") if x.strip()}
    bypassed = sum(1 for h in hosts if h in no_proxy_tokens)
    return {
        "index_policy": index_policy,
        "mirror_proxy_policy": mirror_proxy_policy,
        "pip_primary_index_present": bool(env.get("PIP_INDEX_URL")),
        "uv_primary_index_present": bool(env.get("UV_INDEX_URL") or env.get("UV_DEFAULT_INDEX")),
        "pip_extra_index_present": bool(env.get("PIP_EXTRA_INDEX_URL")),
        "uv_extra_index_present": bool(env.get("UV_EXTRA_INDEX_URL")),
        "pip_find_links_present": bool(env.get("PIP_FIND_LINKS")),
        "uv_find_links_present": bool(env.get("UV_FIND_LINKS")),
        "proxy_present": bool(env.get("HTTPS_PROXY") or env.get("https_proxy")),
        "configured_package_mirror_host_count": len(hosts),
        "package_mirror_hosts_in_no_proxy_count": bypassed,
        "all_package_mirror_hosts_bypass_proxy": bool(hosts) and bypassed == len(hosts),
        "embedded_index_directive_count": len(embedded_indexes),
        "values_redacted": True,
    }


def _embedded_index_directives(reqs: list[Path]) -> list[str]:
    hits: list[str] = []
    for req in reqs:
        for lineno, raw in enumerate(req.read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if re.match(r"^(?:--index-url|--extra-index-url|-i)(?:\s|=)", line):
                hits.append(f"{req.name}:{lineno}")
    return hits


def _validate_index_policy(python_cfg: dict, reqs: list[Path], install_env: dict[str, str]) -> dict:
    policy = str(python_cfg.get("index_policy", "INHERIT")).upper()
    proxy_policy = str(python_cfg.get("mirror_proxy_policy", "INHERIT")).upper()
    allow_embedded = bool(python_cfg.get("allow_embedded_index_directives", True))
    accelerator_route_required = bool(python_cfg.get("accelerator_wheel_route_required", False))
    embedded = _embedded_index_directives(reqs)

    if not allow_embedded and embedded:
        raise RuntimeError("PYTHON_REQUIREMENTS_EMBEDDED_INDEX_FORBIDDEN:" + ",".join(embedded))

    if policy == "HOST_MIRROR_REQUIRED":
        primary = bool(install_env.get("PIP_INDEX_URL") or install_env.get("UV_INDEX_URL") or install_env.get("UV_DEFAULT_INDEX"))
        if not primary:
            raise RuntimeError("PYTHON_HOST_MIRROR_REQUIRED_PRIMARY_INDEX_MISSING")
        if accelerator_route_required:
            accelerator = bool(
                install_env.get("PIP_EXTRA_INDEX_URL") or install_env.get("UV_EXTRA_INDEX_URL")
                or install_env.get("PIP_FIND_LINKS") or install_env.get("UV_FIND_LINKS")
            )
            if not accelerator:
                raise RuntimeError("PYTHON_HOST_MIRROR_REQUIRED_ACCELERATOR_ROUTE_MISSING")
    elif policy not in {"INHERIT", "PUBLIC_ALLOWED"}:
        raise RuntimeError(f"PYTHON_INDEX_POLICY_UNSUPPORTED:{policy}")

    if proxy_policy not in {"INHERIT", "DIRECT_FOR_HOST_MIRRORS"}:
        raise RuntimeError(f"PYTHON_MIRROR_PROXY_POLICY_UNSUPPORTED:{proxy_policy}")
    if proxy_policy == "DIRECT_FOR_HOST_MIRRORS":
        hosts = _configured_package_mirror_hosts(install_env)
        no_proxy_tokens = {x.strip().lower() for x in (install_env.get("NO_PROXY") or install_env.get("no_proxy") or "").replace(" ", ",").split(",") if x.strip()}
        if not hosts or any(h not in no_proxy_tokens for h in hosts):
            raise RuntimeError("PYTHON_DIRECT_HOST_MIRROR_NO_PROXY_BINDING_INCOMPLETE")

    return _route_summary(install_env, python_cfg=python_cfg, embedded_indexes=embedded)


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


def _run_capture(cmd: list[str], *, env: dict[str, str] | None = None) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except OSError:
        return False, "OS_ERROR"
    if p.returncode == 0:
        return True, "PASS"
    return False, _classify_failure((p.stdout or "") + "\n" + (p.stderr or ""))


def _run_checked(cmd: list[str], error_code: str, **kwargs) -> None:
    try:
        subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{error_code}:exit={e.returncode}") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{error_code}:timeout") from None
    except OSError as e:
        raise RuntimeError(f"{error_code}:{type(e).__name__}") from None


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


def _public_pypi_env(base_env: dict[str, str], *, direct: bool = False) -> dict[str, str]:
    env = base_env.copy()
    for key in (
        "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST", "PIP_NO_INDEX", "PIP_FIND_LINKS",
        "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX", "UV_FIND_LINKS", "UV_NO_INDEX",
    ):
        env.pop(key, None)
    env["PIP_INDEX_URL"] = "https://pypi.org/simple"
    env["UV_INDEX_URL"] = "https://pypi.org/simple"
    env["PIP_CONFIG_FILE"] = os.devnull
    if direct:
        for key in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        ):
            env.pop(key, None)
    return env


def _create_stdlib_venv(venv_dir: Path) -> Path:
    shutil.rmtree(venv_dir, ignore_errors=True)
    _run_checked([sys.executable, "-m", "venv", str(venv_dir)], "PYTHON_VENV_CREATE_FAILED")
    py = venv_dir / "bin" / "python"
    ok, _ = _run_capture([str(py), "-m", "pip", "--version"])
    if not ok:
        _run_checked([str(py), "-m", "ensurepip", "--upgrade"], "PYTHON_ENSUREPIP_FAILED")
    return py


def _resolved_package_hash(venv_python: Path) -> str:
    code = (
        "import importlib.metadata as m,json;"
        "x=sorted((d.metadata.get('Name','').lower(),d.version) for d in m.distributions());"
        "print(json.dumps(x,separators=(',',':')))"
    )
    p = subprocess.run([str(venv_python), "-c", code], check=True, capture_output=True, text=True)
    return hashlib.sha256(p.stdout.strip().encode("utf-8")).hexdigest()


def canary_imports(venv_python: Path, imports: list[str]) -> None:
    if not imports:
        return
    code = "\n".join(f"import {imp.split('[')[0].split(':')[0]}" for imp in imports)
    ok, cls = _run_capture([str(venv_python), "-c", code])
    if not ok:
        raise RuntimeError(f"PYTHON_IMPORT_CANARY_FAILED_{cls}")


def _pip_install(venv_python: Path, reqs: list[Path], env: dict[str, str]) -> tuple[bool, str]:
    cmd = [str(venv_python), "-m", "pip", "install"]
    for r in reqs:
        cmd += ["-r", str(r)]
    return _run_capture(cmd, env=env)


def provision(profile: str) -> dict:
    resolved = resolve(profile)
    manifest = resolved["manifest"]
    python_cfg = manifest.get("python") or {}
    required_python = python_cfg.get("requires")
    allow_public_fallback = bool(python_cfg.get("allow_public_index_fallback", False))
    if required_python and not _python_satisfies(required_python):
        raise RuntimeError(f"PYTHON_VERSION_UNSUPPORTED_{sys.version_info.major}_{sys.version_info.minor}")

    env_hash = resolved["environment_sha256"]
    venv_dir = VENV_ROOT / env_hash
    ready_path = venv_dir / "READY.json"
    venv_python = venv_dir / "bin/python"
    if ready_path.exists() and venv_python.is_file():
        ready = json.loads(ready_path.read_text())
        if ready.get("environment_sha256") == env_hash or ready.get("environment_hash") == env_hash:
            package_hash = ready.get("resolved_packages_sha256") or _resolved_package_hash(venv_python)
            return {
                "environment_id": resolved["environment_id"],
                "environment_hash": env_hash,
                "python": {
                    "status": "READY", "cache_hit": True, "venv": str(venv_dir),
                    "installer": ready.get("installer", "cached"),
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "resolved_packages_sha256": package_hash,
                    "public_index_fallback_used": bool(ready.get("public_index_fallback_used", False)),
                    "public_direct_fallback_used": bool(ready.get("public_direct_fallback_used", False)),
                    "install_route_summary": ready.get("install_route_summary"),
                },
            }
        shutil.rmtree(venv_dir, ignore_errors=True)

    reqs = [repo_root() / r for r in manifest.get("requirements", [])]
    for r in reqs:
        if not r.is_file():
            raise RuntimeError(f"PYTHON_REQUIREMENTS_FILE_MISSING_{r.name.replace('.', '_').upper()}")

    prov_env = load_provision_env()
    install_env = _install_env(prov_env)
    install_env = _apply_mirror_proxy_policy(install_env, python_cfg)
    route_summary = _validate_index_policy(python_cfg, reqs, install_env)

    ensure_dirs()
    shutil.rmtree(venv_dir, ignore_errors=True)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)

    installer = "stdlib-venv"
    uv_failure: str | None = None
    public_fallback_used = False
    public_direct_fallback_used = False
    if shutil.which("uv"):
        ok, cls = _run_capture(["uv", "venv", str(venv_dir)], env=install_env)
        if ok:
            venv_python = venv_dir / "bin/python"
            installer = "uv"
            if reqs:
                cmd = ["uv", "pip", "install", "--python", str(venv_python), "--index-strategy", "unsafe-best-match"]
                primary = install_env.get("UV_INDEX_URL") or install_env.get("UV_DEFAULT_INDEX")
                if primary:
                    cmd += ["--index-url", primary]
                for extra in (install_env.get("UV_EXTRA_INDEX_URL") or "").split():
                    cmd += ["--extra-index-url", extra]
                for fl in (install_env.get("UV_FIND_LINKS") or "").split():
                    cmd += ["--find-links", fl]
                for r in reqs:
                    cmd += ["-r", str(r)]
                ok, cls = _run_capture(cmd, env=install_env)
                if not ok:
                    uv_failure = cls
            if uv_failure is None:
                ok, cls = _run_capture(["uv", "pip", "check", "--python", str(venv_python)], env=install_env)
                if not ok:
                    uv_failure = f"CHECK_{cls}"
        else:
            uv_failure = f"VENV_{cls}"

    if not shutil.which("uv") or uv_failure is not None:
        venv_python = _create_stdlib_venv(venv_dir)
        installer = "pip" if uv_failure is None else "pip_fallback"
        if reqs:
            ok, pip_failure = _pip_install(venv_python, reqs, install_env)
            if not ok and pip_failure in {"INDEX_AUTH", "NETWORK_OR_INDEX"} and allow_public_fallback:
                venv_python = _create_stdlib_venv(venv_dir)
                public_env = _public_pypi_env(install_env, direct=False)
                ok, public_failure = _pip_install(venv_python, reqs, public_env)
                if not ok and public_failure in {"INDEX_AUTH", "NETWORK_OR_INDEX"}:
                    venv_python = _create_stdlib_venv(venv_dir)
                    direct_env = _public_pypi_env(install_env, direct=True)
                    ok, direct_failure = _pip_install(venv_python, reqs, direct_env)
                    if not ok:
                        raise RuntimeError(f"PYTHON_REQUIREMENTS_DIRECT_PUBLIC_FALLBACK_FAILED_PIP_{direct_failure}")
                    install_env = direct_env
                    installer = "pip_public_direct_fallback"
                    public_fallback_used = True
                    public_direct_fallback_used = True
                elif not ok:
                    uv_part = f"UV_{uv_failure}_" if uv_failure else ""
                    raise RuntimeError(f"PYTHON_REQUIREMENTS_PUBLIC_FALLBACK_FAILED_{uv_part}PIP_{public_failure}")
                else:
                    install_env = public_env
                    installer = "pip_public_fallback"
                    public_fallback_used = True
            elif not ok:
                uv_part = f"UV_{uv_failure}_" if uv_failure else ""
                raise RuntimeError(f"PYTHON_REQUIREMENTS_INSTALL_FAILED_{uv_part}PIP_{pip_failure}")
        ok, pip_check = _run_capture([str(venv_python), "-m", "pip", "check"], env=install_env)
        if not ok:
            raise RuntimeError(f"PYTHON_DEPENDENCY_CHECK_FAILED_PIP_{pip_check}")

    canary_imports(venv_python, manifest.get("imports", []))
    package_hash = _resolved_package_hash(venv_python)
    ready = {
        "schema": "CB16_PYTHON_ENV_READY_V1",
        "environment_id": resolved["environment_id"],
        "environment_hash": env_hash,
        "environment_sha256": env_hash,
        "installer": installer,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "resolved_packages_sha256": package_hash,
        "public_index_fallback_used": public_fallback_used,
        "public_direct_fallback_used": public_direct_fallback_used,
        "install_route_summary": route_summary,
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    atomic_write_json(ready_path, ready)
    return {
        "environment_id": resolved["environment_id"],
        "environment_hash": env_hash,
        "python": {
            "status": "READY", "cache_hit": False, "venv": str(venv_dir),
            "installer": installer, "python_version": ready["python_version"],
            "resolved_packages_sha256": package_hash,
            "public_index_fallback_used": public_fallback_used,
            "public_direct_fallback_used": public_direct_fallback_used,
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
