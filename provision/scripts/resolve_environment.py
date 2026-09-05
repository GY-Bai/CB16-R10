#!/usr/bin/env python3
"""Resolve a CB16 environment manifest for a CI profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from provision_common import PROVISION_ROOT, atomic_write_json, load_json, repo_root


def environment_path(profile: str) -> Path:
    return PROVISION_ROOT / "environments" / f"{profile}.json"


def compute_environment_hash(manifest: dict, repo: Path) -> str:
    h = hashlib.sha256()
    runtime_identity = {
        "python_implementation": platform.python_implementation(),
        "python_major_minor": [sys.version_info.major, sys.version_info.minor],
        "sys_platform": sys.platform,
        "machine": platform.machine().lower(),
    }
    h.update(json.dumps(runtime_identity, sort_keys=True, separators=(",", ":")).encode())
    h.update(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    for req in manifest.get("requirements", []):
        p = repo / req
        if p.exists():
            h.update(req.encode("utf-8"))
            h.update(b"\0")
            h.update(p.read_bytes())
    return h.hexdigest()


def resolve(profile: str) -> dict:
    p = environment_path(profile)
    if not p.exists():
        raise RuntimeError(f"UNKNOWN_ENVIRONMENT_PROFILE:{profile}")
    manifest = load_json(p)
    if manifest.get("schema") != "CB16_ENVIRONMENT_V1":
        raise RuntimeError(f"BAD_ENVIRONMENT_SCHEMA:{profile}")
    env_hash = compute_environment_hash(manifest, repo_root())
    return {
        "profile": profile,
        "environment_id": manifest.get("id", profile),
        "environment_sha256": env_hash,
        "manifest": manifest,
        "runtime_identity": {
            "python_implementation": platform.python_implementation(),
            "python_major_minor": [sys.version_info.major, sys.version_info.minor],
            "sys_platform": sys.platform,
            "machine": platform.machine().lower(),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    resolved = resolve(args.profile)
    if args.out:
        atomic_write_json(Path(args.out), resolved)
    else:
        print(json.dumps(resolved, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
