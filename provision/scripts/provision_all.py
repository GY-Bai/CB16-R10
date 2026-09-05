#!/usr/bin/env python3
"""CB16 Provisioner V1 orchestration entrypoint."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import os
import subprocess
from pathlib import Path

from provision_common import JOB_ROOT, atomic_write_json, ensure_dirs
from provision_python import load_provision_env, provision
from provision_assets import provision_assets
from resolve_environment import resolve

SYSTEM_CHECKS = {
    "git": ["git", "--version"],
    "zstd": ["zstd", "--version"],
    "nvidia_gpu": ["nvidia-smi"],
    "cuda": ["nvidia-smi"],
    "ffmpeg": ["ffmpeg", "-version"],
}


def check_system_capabilities(manifest: dict) -> dict:
    results = {}
    ok = True
    for cap in manifest.get("system_capabilities", []):
        cmd = SYSTEM_CHECKS.get(cap)
        if not cmd:
            results[cap] = "UNKNOWN_CHECK"
            ok = False
            continue
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            results[cap] = "PASS"
        except Exception:
            results[cap] = "MISSING"
            ok = False
    return {"status": "PASS" if ok else "FAIL", "checks": results}


def write_resolved_env(job_root: Path, resolved: dict, python_result: dict, asset_result: dict) -> Path:
    env_path = job_root / "resolved.env"
    lines = []
    if python_result["python"].get("venv"):
        lines.append(f"CB16_VENV={python_result['python']['venv']}")
        lines.append(f"CI_PYTHON={Path(python_result['python']['venv']) / 'bin' / 'python'}")
    for a in asset_result.get("assets", []):
        if a.get("local_path"):
            key = "CB16_ASSET_" + a["asset_id"].upper().replace("-", "_")
            lines.append(f"{key}={a['local_path']}")
    lines.append(f"CB16_ENVIRONMENT_SHA256={resolved['environment_sha256']}")
    env_path.write_text("\n".join(lines) + "\n")
    return env_path


def _error_code(exc: Exception, default: str) -> str:
    text = str(exc)
    if text:
        token = text.split(":", 1)[0].strip()
        if token and token.replace("_", "").isalnum() and token.upper() == token:
            return token
    return default


def _write_public_summary(
    *,
    resolved: dict,
    profile: str,
    status: str,
    python_result: dict | None = None,
    asset_result: dict | None = None,
    failure_stage: str | None = None,
    error_code: str | None = None,
    error_type: str | None = None,
) -> None:
    ci_out = os.environ.get("CI_OUT")
    if not ci_out:
        return
    out = Path(ci_out)
    out.mkdir(parents=True, exist_ok=True)

    py = {"status": "NOT_RUN", "cache_hit": None}
    if isinstance(python_result, dict):
        source = python_result.get("python", {})
        py = {
            "status": source.get("status"),
            "cache_hit": source.get("cache_hit"),
            "installer": source.get("installer"),
            "python_version": source.get("python_version"),
            "resolved_packages_sha256": source.get("resolved_packages_sha256"),
            "public_index_fallback_used": source.get("public_index_fallback_used"),
        }

    assets = []
    if isinstance(asset_result, dict):
        for a in asset_result.get("assets", []):
            assets.append(
                {
                    "asset_id": a.get("asset_id"),
                    "status": a.get("status"),
                    "provider": a.get("provider"),
                    "cache_hit": a.get("cache_hit"),
                    "integrity_mode": a.get("integrity_mode"),
                    "asset_manifest_sha256": a.get("asset_manifest_sha256"),
                    "sha256": a.get("sha256"),
                }
            )

    summary = {
        "schema": "CB16_PROVISION_SUMMARY_V1",
        "commit_sha": os.environ.get("CI_COMMIT_SHA", "unknown"),
        "ci_profile": profile,
        "environment_id": resolved["environment_id"],
        "environment_sha256": resolved["environment_sha256"],
        "runtime_identity": resolved.get("runtime_identity"),
        "status": status,
        "python": py,
        "assets": assets,
    }
    if failure_stage:
        summary["failure_stage"] = failure_stage
    if error_code:
        summary["error_code"] = error_code
    if error_type:
        summary["error_type"] = error_type
    atomic_write_json(out / "provisioning_summary.json", summary)


def _fail(
    *,
    job_root: Path,
    resolved: dict,
    profile: str,
    system: dict,
    failure_stage: str,
    error_code: str,
    error_type: str,
    python_result: dict | None = None,
    asset_result: dict | None = None,
) -> int:
    report = {
        "schema": "CB16_PROVISION_RESULT_V1",
        "status": "PROVISION_FAILED",
        "environment_id": resolved["environment_id"],
        "environment_hash": resolved["environment_sha256"],
        "runtime_identity": resolved.get("runtime_identity"),
        "failure_stage": failure_stage,
        "error_code": error_code,
        "error_type": error_type,
        "system": system,
    }
    if python_result is not None:
        report["python"] = python_result
    if asset_result is not None:
        report["assets"] = asset_result
    atomic_write_json(job_root / "provisioning.json", report)
    _write_public_summary(
        resolved=resolved,
        profile=profile,
        status="PROVISION_FAILED",
        python_result=python_result,
        asset_result=asset_result,
        failure_stage=failure_stage,
        error_code=error_code,
        error_type=error_type,
    )
    print(json.dumps(report, indent=2))
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--job-id", default="manual")
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    ensure_dirs()
    job_root = JOB_ROOT / args.job_id
    job_root.mkdir(parents=True, exist_ok=True)

    resolved = resolve(args.profile)
    system = check_system_capabilities(resolved["manifest"])
    if system["status"] != "PASS":
        return _fail(
            job_root=job_root,
            resolved=resolved,
            profile=args.profile,
            system=system,
            failure_stage="system_capabilities",
            error_code="SYSTEM_CAPABILITY_MISSING",
            error_type="CapabilityCheckFailure",
        )

    try:
        python_result = provision(args.profile)
    except Exception as exc:
        return _fail(
            job_root=job_root,
            resolved=resolved,
            profile=args.profile,
            system=system,
            failure_stage="python",
            error_code=_error_code(exc, "PYTHON_PROVISION_FAILED"),
            error_type=type(exc).__name__,
        )

    asset_env = os.environ.copy()
    asset_env.update(load_provision_env())
    try:
        asset_result = provision_assets(args.profile, asset_env)
    except Exception as exc:
        return _fail(
            job_root=job_root,
            resolved=resolved,
            profile=args.profile,
            system=system,
            failure_stage="assets",
            error_code=_error_code(exc, "ASSET_PROVISION_FAILED"),
            error_type=type(exc).__name__,
            python_result=python_result,
        )

    if asset_result["status"] != "PASS":
        return _fail(
            job_root=job_root,
            resolved=resolved,
            profile=args.profile,
            system=system,
            failure_stage="assets",
            error_code="ASSET_RESULT_NOT_READY",
            error_type="AssetProvisionFailure",
            python_result=python_result,
            asset_result=asset_result,
        )

    env_path = write_resolved_env(job_root, resolved, python_result, asset_result)
    report = {
        "schema": "CB16_PROVISION_RESULT_V1",
        "status": "READY",
        "environment_id": resolved["environment_id"],
        "environment_hash": resolved["environment_sha256"],
        "runtime_identity": resolved.get("runtime_identity"),
        "python": python_result,
        "assets": asset_result,
        "system": system,
        "resolved_env": str(env_path),
    }
    atomic_write_json(job_root / "provisioning.json", report)
    atomic_write_json(job_root / "resolved_environment.json", report)
    _write_public_summary(
        resolved=resolved,
        profile=args.profile,
        status="READY",
        python_result=python_result,
        asset_result=asset_result,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
