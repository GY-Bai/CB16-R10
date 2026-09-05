#!/usr/bin/env python3
"""Verify registered CB16 assets against their declared integrity."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import hashlib
import json
import sys
from pathlib import Path

from provision_common import ASSET_REGISTRY_DIR, load_registry
from provision_assets import all_asset_manifests


def verify() -> dict:
    registry = load_registry()
    results = []
    for manifest in all_asset_manifests():
        asset_id = manifest["asset_id"]
        entry = registry.get(asset_id)
        if not entry:
            results.append({"asset_id": asset_id, "status": "NOT_REGISTERED"})
            continue
        local = Path(entry.get("local_path", ""))
        integrity = manifest.get("integrity") or {}
        sha = integrity.get("sha256")
        if not local.exists():
            results.append({"asset_id": asset_id, "status": "MISSING"})
            continue
        if local.is_file() and sha:
            actual = hashlib.sha256(local.read_bytes()).hexdigest()
            ok = actual == sha
            results.append({"asset_id": asset_id, "status": "PASS" if ok else "FAIL", "sha256": actual})
        else:
            results.append({"asset_id": asset_id, "status": "PASS", "note": "directory/discovery"})
    return {"schema": "CB16_ASSET_VERIFY_V1", "status": "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL", "results": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = verify()
    if args.out:
        from provision_common import atomic_write_json
        atomic_write_json(Path(args.out), result)
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
