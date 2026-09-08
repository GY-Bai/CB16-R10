#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("CB16_R104_ROOT", "/data/cb16_hdd/cb16_runtime/R10_4"))
OUT = Path(os.environ.get("CB16_G0_R104_AUTHORITY_OUT", "/tmp/cb16_r11_g0_r104_authority.json"))
MAX_JSON_BYTES = 1_000_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_small_json(path: Path) -> Any:
    st = path.stat()
    if st.st_size > MAX_JSON_BYTES:
        raise RuntimeError(f"JSON_TOO_LARGE:{path}:{st.st_size}")
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_projection(obj: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            scalar_projection(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        if len(obj) <= 16 and all(isinstance(x, (str, int, float, bool)) or x is None for x in obj):
            out[prefix] = obj
        else:
            out[prefix + ".__len__"] = len(obj)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out[prefix] = obj
    return out


def describe(path: Path, *, parse_json: bool = False, hash_file: bool = True) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    rec: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    if hash_file and path.is_file():
        rec["sha256"] = sha256_file(path)
    if parse_json and path.is_file():
        obj = load_small_json(path)
        rec["top_level_keys"] = sorted(obj.keys()) if isinstance(obj, dict) else []
        rec["scalars"] = scalar_projection(obj)
    return rec


def main() -> int:
    if not ROOT.is_dir():
        raise SystemExit(f"R104_ROOT_NOT_FOUND:{ROOT}")
    g00 = ROOT / "generations" / "G00"
    g66 = ROOT / "generations" / "G66"
    g67 = ROOT / "generations" / "G67"
    if not g00.is_dir() or not g66.is_dir():
        raise SystemExit("R104_G00_OR_G66_MISSING")

    g66_files = sorted(p.name for p in g66.iterdir())
    json_candidates = sorted(p for p in g66.iterdir() if p.is_file() and p.suffix.lower() in {".json", ".jsonl"})

    result: dict[str, Any] = {
        "schema": "CB16_R11_G0_R104_AUTHORITY_INSPECTION_V1",
        "status": "INSPECTED_READ_ONLY",
        "root": str(ROOT),
        "g00_generation_result": describe(g00 / "GENERATION_RESULT.json", parse_json=True),
        "g66_generation_result": describe(g66 / "GENERATION_RESULT.json", parse_json=True),
        "g66_snapshot_consumption": describe(g66 / "SNAPSHOT_CONSUMPTION_G66.json", parse_json=True),
        "g66_champion_after": describe(g66 / "champion_after.pt", parse_json=False),
        "g66_files": g66_files,
        "g66_json_files": [describe(p, parse_json=True) for p in json_candidates],
        "g67": {
            "exists": g67.exists(),
            "is_dir": g67.is_dir(),
            "entries": sorted(p.name for p in g67.iterdir()) if g67.is_dir() else [],
        },
        "root_small_records": [],
        "safety": {
            "market_payload_opened": False,
            "market_data_downloaded": False,
            "training_started": False,
            "runtime_files_modified": False,
        },
    }

    for name in (
        "REAL_EVIDENCE_CACHE_MANIFEST_R102.json",
        "EXECUTION_BLOCKER_RECEIPT_R104.json",
        "CAMPAIGN_STATE.json",
        "FINAL_RESULT.json",
    ):
        p = ROOT / name
        if p.exists():
            result["root_small_records"].append(describe(p, parse_json=True))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "schema": result["schema"],
        "status": result["status"],
        "g66_files": g66_files,
        "g66_generation_result_scalars": result["g66_generation_result"].get("scalars", {}),
        "g66_snapshot_scalars": result["g66_snapshot_consumption"].get("scalars", {}),
        "g66_checkpoint_sha256": result["g66_champion_after"].get("sha256"),
        "g67_entries": result["g67"]["entries"],
        "safety": result["safety"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
