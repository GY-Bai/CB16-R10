#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("CB16_R104_ROOT", "/data/cb16_hdd/cb16_runtime/R10_4"))
OUT = Path(os.environ.get("CB16_G0_R104_AUTHORITY_OUT", "/tmp/cb16_r11_g0_r104_authority.json"))
DIAG_ROOT = Path(os.environ.get("CB16_CANONICAL_RUNNER_DIAG_ROOT", "/data/cb16_ci/github-runner-canonical/actions-runner/_diag"))
MAX_JSON_BYTES = 1_000_000
MAX_DIAG_FILE_BYTES = 64 * 1024 * 1024
MAX_DIAG_FILES = 160
MAX_DIAG_MATCHES = 200
DIAG_NEEDLES = (
    "34075942291",
    "101612700658",
    "f056ae6a0722e3e92d71793024a6e6d3fe9af003",
    "G66",
    "run_r104_long_research.py",
)


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


def iso_utc_from_ns(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def describe(path: Path, *, parse_json: bool = False, hash_file: bool = True) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    rec: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "mtime_utc": iso_utc_from_ns(int(st.st_mtime_ns)),
        "ctime_ns": int(st.st_ctime_ns),
        "ctime_utc": iso_utc_from_ns(int(st.st_ctime_ns)),
    }
    if hash_file and path.is_file():
        rec["sha256"] = sha256_file(path)
    if parse_json and path.is_file():
        obj = load_small_json(path)
        rec["top_level_keys"] = sorted(obj.keys()) if isinstance(obj, dict) else []
        rec["scalars"] = scalar_projection(obj)
    return rec


def generation_timeline(start: int = 58, end: int = 67) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in range(start, end + 1):
        gd = ROOT / "generations" / f"G{g:02d}"
        rec: dict[str, Any] = {
            "generation": g,
            "path": str(gd),
            "exists": gd.exists(),
            "is_dir": gd.is_dir(),
        }
        if not gd.is_dir():
            out.append(rec)
            continue
        entries = sorted(p for p in gd.iterdir())
        rec["entries"] = [p.name for p in entries]
        if entries:
            stats = [(p.name, int(p.stat().st_mtime_ns)) for p in entries]
            rec["earliest_entry_mtime_utc"] = iso_utc_from_ns(min(ns for _, ns in stats))
            rec["latest_entry_mtime_utc"] = iso_utc_from_ns(max(ns for _, ns in stats))
            rec["latest_entry"] = max(stats, key=lambda x: x[1])[0]
        for name, parse_json, hash_file in (
            ("GENERATION_RESULT.json", True, True),
            (f"SNAPSHOT_CONSUMPTION_G{g:02d}.json", True, True),
            ("ON_POLICY_REAL_TRACE_RECEIPT.json", True, True),
            ("champion_after.pt", False, True),
        ):
            p = gd / name
            if p.exists():
                rec[name] = describe(p, parse_json=parse_json, hash_file=hash_file)
        out.append(rec)
    return out


def bounded_diag_matches() -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": str(DIAG_ROOT),
        "exists": DIAG_ROOT.is_dir(),
        "needles": list(DIAG_NEEDLES),
        "files_considered": 0,
        "files_scanned": 0,
        "matches": [],
        "truncated": False,
    }
    if not DIAG_ROOT.is_dir():
        return result

    files = [p for p in DIAG_ROOT.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    result["files_considered"] = min(len(files), MAX_DIAG_FILES)
    for p in files[:MAX_DIAG_FILES]:
        try:
            st = p.stat()
            if st.st_size > MAX_DIAG_FILE_BYTES:
                continue
            result["files_scanned"] += 1
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if any(n in line for n in DIAG_NEEDLES):
                        result["matches"].append(
                            {
                                "file": str(p),
                                "line": lineno,
                                "text": line.rstrip("\n")[:2000],
                            }
                        )
                        if len(result["matches"]) >= MAX_DIAG_MATCHES:
                            result["truncated"] = True
                            return result
        except (OSError, UnicodeError) as exc:
            result.setdefault("scan_errors", []).append(f"{p}:{type(exc).__name__}")
    return result


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
        "schema": "CB16_R11_G0_R104_AUTHORITY_INSPECTION_V2",
        "status": "INSPECTED_READ_ONLY",
        "root": str(ROOT),
        "g00_generation_result": describe(g00 / "GENERATION_RESULT.json", parse_json=True),
        "g66_generation_result": describe(g66 / "GENERATION_RESULT.json", parse_json=True),
        "g66_snapshot_consumption": describe(g66 / "SNAPSHOT_CONSUMPTION_G66.json", parse_json=True),
        "g66_champion_after": describe(g66 / "champion_after.pt", parse_json=False),
        "g66_files": g66_files,
        "g66_json_files": [describe(p, parse_json=True) for p in json_candidates],
        "generation_timeline_g58_g67": generation_timeline(58, 67),
        "canonical_runner_diag": bounded_diag_matches(),
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
            "diag_files_modified": False,
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

    timeline_summary = []
    for x in result["generation_timeline_g58_g67"]:
        timeline_summary.append(
            {
                "generation": x["generation"],
                "exists": x["exists"],
                "earliest": x.get("earliest_entry_mtime_utc"),
                "latest": x.get("latest_entry_mtime_utc"),
                "latest_entry": x.get("latest_entry"),
                "generation_result_mtime": x.get("GENERATION_RESULT.json", {}).get("mtime_utc"),
                "checkpoint_mtime": x.get("champion_after.pt", {}).get("mtime_utc"),
                "checkpoint_sha256": x.get("champion_after.pt", {}).get("sha256"),
            }
        )
    summary = {
        "schema": result["schema"],
        "status": result["status"],
        "g66_generation_result_scalars": result["g66_generation_result"].get("scalars", {}),
        "g66_snapshot_scalars": result["g66_snapshot_consumption"].get("scalars", {}),
        "g66_checkpoint_sha256": result["g66_champion_after"].get("sha256"),
        "g67_entries": result["g67"]["entries"],
        "generation_timeline": timeline_summary,
        "diag_match_count": len(result["canonical_runner_diag"]["matches"]),
        "diag_matches": result["canonical_runner_diag"]["matches"][:40],
        "safety": result["safety"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
