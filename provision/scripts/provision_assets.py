#!/usr/bin/env python3
"""Provision external CB16 assets declared in provision/assets."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import fcntl
import json
import os
import shutil
import sys
from pathlib import Path

from provision_common import (
    ASSET_REGISTRY_DIR,
    LOCK_ROOT,
    PROVISION_ROOT,
    atomic_write_json,
    ensure_dirs,
    load_json,
    load_registry,
    repo_root,
    save_registry,
)
from provision.providers import PROVIDERS, Candidate, ProviderError
from resolve_environment import resolve

ASSET_DIR = PROVISION_ROOT / "assets"


def find_asset_manifest(asset_id: str) -> dict:
    candidates = list(ASSET_DIR.glob("*.json")) + list(ASSET_DIR.glob("examples/*.json"))
    for p in candidates:
        manifest = load_json(p)
        if manifest.get("asset_id") == asset_id:
            return manifest
    raise RuntimeError(f"UNKNOWN_ASSET_ID:{asset_id}")


def all_asset_manifests() -> list[dict]:
    out = []
    for p in sorted(ASSET_DIR.glob("*.json")):
        out.append(load_json(p))
    for p in sorted((ASSET_DIR / "examples").glob("*.json")):
        out.append(load_json(p))
    return out


def acquire_lock(asset_id: str):
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_ROOT / "assets" / f"{asset_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def release_lock(fh) -> None:
    try:
        fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()


def _storage_base(manifest: dict) -> Path:
    root = Path(os.environ.get("CB16_ASSET_STORAGE_ROOT", "/data/cb16_ci/assets"))
    logical = manifest.get("storage", {}).get("logical_path", "")
    return root / logical


def provision_asset(manifest: dict, env: dict[str, str]) -> dict:
    asset_id = manifest["asset_id"]
    if not env:
        env = os.environ.copy()
    lock = acquire_lock(asset_id)
    try:
        registry = load_registry()
        entry = registry.get(asset_id)
        if entry and entry.get("status") == "READY":
            return {
                "asset_id": asset_id,
                "status": "READY",
                "cache_hit": True,
                "sha256": entry.get("sha256"),
                "local_path": entry.get("local_path"),
            }

        dest = _storage_base(manifest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        last_errors = []
        for source in manifest.get("sources", []):
            for provider in PROVIDERS:
                if not provider.can_handle(source):
                    continue
                try:
                    candidate = provider.resolve(source, env)
                    if candidate is None:
                        continue
                    # For local existing, just register resolved path.
                    if provider.name == "local" and candidate.local_path is not None:
                        local = candidate.local_path
                        # If logical destination is a directory and local is a directory, register it.
                        if local.is_dir():
                            registry[asset_id] = {
                                "asset_id": asset_id,
                                "local_path": str(local),
                                "status": "READY",
                                "cache_hit": True,
                                "sha256": None,
                            }
                            save_registry(registry)
                            return {
                                "asset_id": asset_id,
                                "status": "READY",
                                "cache_hit": True,
                                "local_path": str(local),
                            }
                        # File local: copy to destination and verify.
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        tmp = dest.with_suffix(dest.suffix + ".partial")
                        shutil.copy2(local, tmp)
                        provider.verify(tmp, manifest)
                        tmp.replace(dest)
                        break
                    provider.download(candidate, dest, env)
                    provider.verify(dest, manifest)
                    break
                except Exception as e:
                    last_errors.append(f"{provider.name}:{type(e).__name__}:{e}")
                    continue
            else:
                continue
            break
        else:
            raise ProviderError(
                "ALL_PROVIDERS_FAILED:" + ";".join(last_errors[-5:]) if last_errors else "NO_SOURCES"
            )

        sha = None
        import hashlib
        if dest.is_file():
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        if manifest.get("integrity_mode") == "DISCOVERY":
            # Discovery assets are never automatically frozen authority.
            discovery = {
                "schema": "CB16_DISCOVERED_ASSET_V1",
                "asset_id": asset_id,
                "observed_sha256": sha,
                "status": "AVAILABLE_FOR_DISCOVERY",
                "provider": "mixed",
            }
            atomic_write_json(ASSET_REGISTRY_DIR / f"DISCOVERED_{asset_id}.json", discovery)
        registry[asset_id] = {
            "asset_id": asset_id,
            "local_path": str(dest),
            "status": "READY",
            "cache_hit": False,
            "sha256": sha,
        }
        save_registry(registry)
        return {
            "asset_id": asset_id,
            "status": "READY",
            "cache_hit": False,
            "sha256": sha,
            "local_path": str(dest),
        }
    finally:
        release_lock(lock)


def provision_assets(profile: str, env: dict[str, str] | None = None) -> dict:
    resolved = resolve(profile)
    asset_ids = resolved["manifest"].get("assets", [])
    if env is None:
        env = os.environ.copy()
    results = []
    for asset_id in asset_ids:
        manifest = find_asset_manifest(asset_id)
        results.append(provision_asset(manifest, env))
    return {
        "profile": profile,
        "assets": results,
        "status": "PASS" if all(r["status"] == "READY" for r in results) else "FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--asset-id", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.asset_id:
        manifest = find_asset_manifest(args.asset_id)
        result = provision_asset(manifest, os.environ.copy())
    else:
        result = provision_assets(args.profile, os.environ.copy())
    if args.out:
        atomic_write_json(Path(args.out), result)
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
