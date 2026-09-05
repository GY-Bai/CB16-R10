#!/usr/bin/env python3
"""Create or reuse a CB16 Python environment for a resolved environment manifest."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from provision_common import (
    VENV_ROOT,
    atomic_write_json,
    ensure_dirs,
    load_json,
    repo_root,
)
from resolve_environment import resolve


def load_provision_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in (Path("/etc/cb16-ci/provision.env"), Path(os.environ.get("CB16_CI_WORKER_ROOT", "/data/cb16_ci")) / "provision.env"):
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k] = v
    return out


def canary_imports(venv_python: Path, imports: list[str]) -> None:
    if not imports:
        return
    code = "\n".join(f"import {imp.split('[')[0].split(':')[0]}" for imp in imports)
    subprocess.run([str(venv_python), "-c", code], check=True, capture_output=True)


def provision(profile: str) -> dict:
    resolved = resolve(profile)
    manifest = resolved["manifest"]
    env_hash = resolved["environment_sha256"]
    venv_dir = VENV_ROOT / env_hash
    ready_path = venv_dir / "READY.json"
    if ready_path.exists():
        ready = json.loads(ready_path.read_text())
        if ready.get("environment_sha256") == env_hash or ready.get("environment_hash") == env_hash:
            return {
                "environment_id": resolved["environment_id"],
                "environment_hash": env_hash,
                "python": {"status": "READY", "cache_hit": True, "venv": str(venv_dir)},
            }
        import shutil
        shutil.rmtree(venv_dir, ignore_errors=True)
    ensure_dirs()
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    venv_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("uv"):
        subprocess.run(["uv", "venv", str(venv_dir)], check=True)
        venv_python = venv_dir / "bin" / "python"
    else:
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        venv_python = venv_dir / "bin" / "python"
    prov_env = load_provision_env()
    # Install requirements from repo-relative paths.
    reqs = [repo_root() / r for r in manifest.get("requirements", [])]
    if reqs:
        if shutil.which("uv"):
            cmd = ["uv", "pip", "install", "--python", str(venv_python)]
            for r in reqs:
                cmd += ["-r", str(r)]
        else:
            cmd = [str(venv_python), "-m", "pip", "install"]
            for r in reqs:
                cmd += ["-r", str(r)]
        env = os.environ.copy()
        if prov_env.get("PIP_INDEX_URL"):
            env["PIP_INDEX_URL"] = prov_env["PIP_INDEX_URL"]
        if prov_env.get("PIP_EXTRA_INDEX_URL"):
            env["PIP_EXTRA_INDEX_URL"] = prov_env["PIP_EXTRA_INDEX_URL"]
        subprocess.run(cmd, check=True, env=env, capture_output=True)
    if shutil.which("uv"):
        subprocess.run(["uv", "pip", "check", "--python", str(venv_python)], check=True, capture_output=True)
    else:
        subprocess.run([str(venv_python), "-m", "pip", "check"], check=True, capture_output=True)
    canary_imports(venv_python, manifest.get("imports", []))
    atomic_write_json(
        ready_path,
        {
            "schema": "CB16_PYTHON_ENV_READY_V1",
            "environment_id": resolved["environment_id"],
            "environment_hash": env_hash,
            "environment_sha256": env_hash,
            "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        },
    )
    return {
        "environment_id": resolved["environment_id"],
        "environment_hash": env_hash,
        "python": {"status": "READY", "cache_hit": False, "venv": str(venv_dir)},
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
