#!/usr/bin/env python3
"""Provision external CB16 assets declared in provision/assets."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from provision_common import (
    ASSET_REGISTRY_DIR,
    LOCK_ROOT,
    PROVISION_ROOT,
    atomic_write_json,
    ensure_dirs,
    load_json,
    load_registry,
    save_registry,
)
from provision.providers import PROVIDERS, Provider, ProviderError
from provision.providers.base import integrity_target, sha256_file
from resolve_environment import resolve

ASSET_DIR = PROVISION_ROOT / "assets"


def find_asset_manifest(asset_id: str) -> dict:
    candidates = list(ASSET_DIR.glob("*.json")) + list(ASSET_DIR.glob("examples/*.json"))
    for p in candidates:
        manifest = load_json(p)
        if manifest.get("asset_id") == asset_id:
            return manifest
    raise RuntimeError(f"UNKNOWN_ASSET_ID:{asset_id}")


def manifest_identity(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _verified_sha(destination: Path, manifest: dict) -> str | None:
    integrity = manifest.get("integrity") or {}
    if not integrity.get("sha256"):
        return None
    return sha256_file(integrity_target(destination, manifest))


def _verify_existing(path: Path, manifest: dict) -> tuple[bool, str | None]:
    try:
        Provider().verify(path, manifest)
        return True, _verified_sha(path, manifest)
    except Exception:
        return False, None


def _ready_result(
    *, asset_id: str, manifest: dict, path: Path, cache_hit: bool, provider: str, sha256: str | None
) -> dict:
    return {
        "asset_id": asset_id,
        "status": "READY",
        "cache_hit": cache_hit,
        "provider": provider,
        "integrity_mode": manifest.get("integrity_mode"),
        "asset_manifest_sha256": manifest_identity(manifest),
        "sha256": sha256,
        "local_path": str(path),
    }


def _invalidate_managed_integrity_target(cached: Path, dest: Path, manifest: dict) -> None:
    """Remove only a corrupt managed target, never an externally registered local asset."""
    try:
        if cached.resolve() != dest.resolve():
            return
        target = integrity_target(dest, manifest)
        if target.is_file():
            target.unlink()
    except Exception:
        # Re-acquisition/verification remains fail-closed even if cleanup itself cannot run.
        pass


def provision_asset(manifest: dict, env: dict[str, str]) -> dict:
    asset_id = manifest["asset_id"]
    if not env:
        env = os.environ.copy()
    lock = acquire_lock(asset_id)
    try:
        dest = _storage_base(manifest)
        registry = load_registry()
        entry = registry.get(asset_id)
        if entry and entry.get("status") == "READY" and entry.get("local_path"):
            cached = Path(entry["local_path"])
            ok, verified_sha = _verify_existing(cached, manifest)
            if ok:
                return _ready_result(
                    asset_id=asset_id,
                    manifest=manifest,
                    path=cached,
                    cache_hit=True,
                    provider=str(entry.get("provider") or "registry"),
                    sha256=verified_sha or entry.get("sha256"),
                )
            _invalidate_managed_integrity_target(cached, dest, manifest)
            registry.pop(asset_id, None)
            save_registry(registry)

        dest.parent.mkdir(parents=True, exist_ok=True)
        last_errors: list[str] = []
        winning_provider: str | None = None
        winning_path: Path | None = None

        for source in manifest.get("sources", []):
            handled = False
            for provider in PROVIDERS:
                if not provider.can_handle(source):
                    continue
                handled = True
                try:
                    candidate = provider.resolve(source, env)
                    if candidate is None:
                        continue

                    if provider.name == "local" and candidate.local_path is not None:
                        local = candidate.local_path
                        provider.verify(local, manifest)
                        winning_provider = provider.name
                        winning_path = local
                        break

                    provider.download(candidate, dest, env)
                    provider.verify(dest, manifest)
                    winning_provider = provider.name
                    winning_path = dest
                    break
                except Exception as exc:
                    text = str(exc)
                    code = text.split(":", 1)[0] if text else type(exc).__name__
                    last_errors.append(f"{provider.name}:{code}")
                    continue
            if winning_path is not None:
                break
            if not handled:
                last_errors.append(f"source:{source.get('type','unknown')}:NO_PROVIDER")

        if winning_path is None or winning_provider is None:
            detail = ";".join(last_errors[-8:]) if last_errors else "NO_SOURCES"
            raise ProviderError("ALL_PROVIDERS_FAILED:" + detail)

        verified_sha = _verified_sha(winning_path, manifest)
        if manifest.get("integrity_mode") == "DISCOVERY":
            discovery = {
                "schema": "CB16_DISCOVERED_ASSET_V1",
                "asset_id": asset_id,
                "asset_manifest_sha256": manifest_identity(manifest),
                "observed_sha256": verified_sha,
                "status": "AVAILABLE_FOR_DISCOVERY",
                "provider": winning_provider,
            }
            atomic_write_json(ASSET_REGISTRY_DIR / f"DISCOVERED_{asset_id}.json", discovery)

        registry = load_registry()
        registry[asset_id] = {
            "asset_id": asset_id,
            "local_path": str(winning_path),
            "status": "READY",
            "cache_hit": False,
            "provider": winning_provider,
            "integrity_mode": manifest.get("integrity_mode"),
            "asset_manifest_sha256": manifest_identity(manifest),
            "sha256": verified_sha,
        }
        save_registry(registry)
        return _ready_result(
            asset_id=asset_id,
            manifest=manifest,
            path=winning_path,
            cache_hit=False,
            provider=winning_provider,
            sha256=verified_sha,
        )
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
