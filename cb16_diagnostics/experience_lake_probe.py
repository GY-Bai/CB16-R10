from __future__ import annotations

"""Read-only Experience Lake timing probe for CB16 R10 diagnostics.

R0.1 distinguishes training EVIDENCE_PACKAGE objects from on-policy trace objects.
This matters under recovery because trace objects may predate the current attempt, while
generation training evidence and snapshot sealing can occur later. The probe never
mutates the Lake or canonical campaign root.
"""

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "CB16_R10_EXPERIENCE_LAKE_TIMING_PROBE_R0_1"
SAFETY = {
    "writes_to_canonical_run_root": False,
    "scientific_semantics_changed": False,
    "final_holdout_2025_09_accessed": False,
    "status_driving": False,
}
_SNAPSHOT_RE = re.compile(r"R102_G(\d+)_TRAINING_SNAPSHOT$")


def _mtime(path: Path) -> float | None:
    try:
        return float(path.stat().st_mtime)
    except FileNotFoundError:
        return None


def _ro_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return max(0.0, float(b) - float(a))


def _generation_artifacts(run_root: Path, generation: int) -> dict[str, float | None]:
    gd = run_root / "generations" / f"G{generation:02d}"
    return {
        "on_policy_receipt_mtime_unix": _mtime(gd / "ON_POLICY_REAL_TRACE_RECEIPT.json"),
        "snapshot_consumption_receipt_mtime_unix": _mtime(gd / f"SNAPSHOT_CONSUMPTION_G{generation}.json"),
        "training_receipt_mtime_unix": _mtime(gd / f"CHALLENGER_TRAINING_RECEIPT_G{generation}.json"),
        "trained_recovery_mtime_unix": _mtime(gd / f"CHALLENGER_TRAINED_RECOVERY_G{generation}.pt"),
        "challenger_mtime_unix": _mtime(gd / "challenger.pt"),
        "champion_after_mtime_unix": _mtime(gd / "champion_after.pt"),
        "generation_result_mtime_unix": _mtime(gd / "GENERATION_RESULT.json"),
    }


def _blank_type() -> dict[str, Any]:
    return {
        "object_count": 0,
        "bytes_raw": 0,
        "bytes_stored": 0,
        "first_object_created_at_unix": None,
        "last_object_created_at_unix": None,
        "shards": {},
    }


def _merge_stat(dst: dict[str, Any], *, shard: str, count: int, first: float | None,
                last: float | None, raw_b: int, stored_b: int) -> None:
    dst["object_count"] += int(count)
    dst["bytes_raw"] += int(raw_b)
    dst["bytes_stored"] += int(stored_b)
    if first is not None and (dst["first_object_created_at_unix"] is None or first < dst["first_object_created_at_unix"]):
        dst["first_object_created_at_unix"] = first
    if last is not None and (dst["last_object_created_at_unix"] is None or last > dst["last_object_created_at_unix"]):
        dst["last_object_created_at_unix"] = last
    dst["shards"][shard] = {
        "object_count": int(count),
        "first_object_created_at_unix": first,
        "last_object_created_at_unix": last,
        "bytes_raw": int(raw_b),
        "bytes_stored": int(stored_b),
    }


def probe_experience_lake(*, run_root: str | Path) -> dict[str, Any]:
    rr = Path(run_root).resolve()
    metadata = rr / "experience_lake" / "metadata"
    dbs = sorted(metadata.glob("experience_*.sqlite"))
    if not dbs:
        return {
            "schema": SCHEMA, "status": "NO_EXPERIENCE_LAKE_METADATA",
            "run_root": str(rr), "safety": dict(SAFETY), "shards_found": 0,
            "generations": [],
        }

    per_gen: dict[int, dict[str, Any]] = defaultdict(lambda: {
        "generation": None,
        "all_objects": _blank_type(),
        "object_types": {},
    })
    snapshots: dict[int, dict[str, Any]] = {}
    shard_errors: list[dict[str, Any]] = []

    for db in dbs:
        shard = db.stem
        try:
            conn = _ro_conn(db)
            rows = conn.execute(
                """
                SELECT generation,object_type,COUNT(*),MIN(created_at),MAX(created_at),
                       COALESCE(SUM(bytes_raw),0),COALESCE(SUM(bytes_stored),0)
                FROM objects
                GROUP BY generation,object_type
                ORDER BY generation,object_type
                """
            ).fetchall()
            snap_rows = conn.execute(
                """
                SELECT snapshot_id,parent_generation,object_count,created_at,content_hash
                FROM snapshots ORDER BY created_at
                """
            ).fetchall()
            conn.close()
        except sqlite3.Error as exc:
            shard_errors.append({"db": str(db), "error": f"{type(exc).__name__}:{exc}"})
            continue

        for gen, typ, count, first_ts, last_ts, raw_b, stored_b in rows:
            gen = int(gen)
            typ = str(typ)
            first = float(first_ts) if first_ts is not None else None
            last = float(last_ts) if last_ts is not None else None
            g = per_gen[gen]
            g["generation"] = gen
            t = g["object_types"].setdefault(typ, _blank_type())
            _merge_stat(t, shard=shard, count=count, first=first, last=last,
                        raw_b=raw_b, stored_b=stored_b)
            _merge_stat(g["all_objects"], shard=f"{shard}:{typ}", count=count, first=first,
                        last=last, raw_b=raw_b, stored_b=stored_b)

        for snapshot_id, parent_generation, object_count, created_at, content_hash in snap_rows:
            m = _SNAPSHOT_RE.search(str(snapshot_id))
            gen = int(m.group(1)) if m else int(parent_generation)
            row = {
                "snapshot_id": str(snapshot_id),
                "parent_generation": int(parent_generation),
                "object_count": int(object_count),
                "created_at_unix": float(created_at),
                "content_hash": str(content_hash),
                "metadata_shard": shard,
            }
            old = snapshots.get(gen)
            if old is not None and old["content_hash"] != row["content_hash"]:
                shard_errors.append({
                    "generation": gen,
                    "error": "SNAPSHOT_METADATA_CONFLICT_ACROSS_SHARDS",
                    "first": old, "second": row,
                })
            snapshots[gen] = row

    generations: list[dict[str, Any]] = []
    for gen in sorted(set(per_gen) | set(snapshots)):
        base = per_gen.get(gen, {"generation": gen, "all_objects": _blank_type(), "object_types": {}})
        all_obj = base["all_objects"]
        types = base["object_types"]
        evidence = types.get("EVIDENCE_PACKAGE", _blank_type())
        decisions = types.get("DECISION_EVENT", _blank_type())
        outcomes = types.get("OUTCOME_SAMPLE", _blank_type())
        snap = snapshots.get(gen)
        art = _generation_artifacts(rr, gen)
        sealed_ts = snap.get("created_at_unix") if snap else None

        all_span = _delta(all_obj["first_object_created_at_unix"], all_obj["last_object_created_at_unix"])
        ev_span = _delta(evidence["first_object_created_at_unix"], evidence["last_object_created_at_unix"])
        first_ev_to_seal = _delta(evidence["first_object_created_at_unix"], sealed_ts)
        on_policy_to_ev = _delta(art["on_policy_receipt_mtime_unix"], evidence["first_object_created_at_unix"])
        seal_to_training = _delta(sealed_ts, art["training_receipt_mtime_unix"])

        earliest_trace = min(
            [x for x in (
                decisions["first_object_created_at_unix"],
                outcomes["first_object_created_at_unix"],
            ) if x is not None],
            default=None,
        )
        cross_attempt_trace = bool(
            earliest_trace is not None
            and art["on_policy_receipt_mtime_unix"] is not None
            and earliest_trace < art["on_policy_receipt_mtime_unix"] - 60.0
        )
        cross_attempt_evidence = bool(
            evidence["first_object_created_at_unix"] is not None
            and art["on_policy_receipt_mtime_unix"] is not None
            and evidence["first_object_created_at_unix"] < art["on_policy_receipt_mtime_unix"] - 60.0
        )

        generations.append({
            "generation": gen,
            "snapshot": snap,
            "snapshot_sealed": snap is not None,
            "artifacts": art,
            "object_types": types,
            "all_object_count": all_obj["object_count"],
            "training_evidence_object_count": evidence["object_count"],
            "decision_event_object_count": decisions["object_count"],
            "outcome_sample_object_count": outcomes["object_count"],
            "snapshot_object_count_matches_training_evidence": (
                bool(snap) and int(snap["object_count"]) == int(evidence["object_count"])
            ),
            "historical_all_object_creation_span_seconds": all_span,
            "historical_span_crosses_prior_attempt_trace": cross_attempt_trace,
            "training_evidence_span_crosses_prior_attempt": cross_attempt_evidence,
            "training_evidence_insert_span_seconds": ev_span,
            "training_evidence_object_rate_per_second": (
                float(evidence["object_count"]) / ev_span
                if ev_span is not None and ev_span > 0 and evidence["object_count"] > 1 else None
            ),
            "training_evidence_stored_bytes_per_second": (
                float(evidence["bytes_stored"]) / ev_span
                if ev_span is not None and ev_span > 0 else None
            ),
            "on_policy_receipt_to_first_training_evidence_seconds": on_policy_to_ev,
            "first_training_evidence_to_snapshot_seal_seconds": first_ev_to_seal,
            "snapshot_seal_to_training_receipt_seconds": seal_to_training,
            "training_receipt_to_challenger_seconds": _delta(
                art["training_receipt_mtime_unix"], art["challenger_mtime_unix"]
            ),
            "challenger_to_generation_result_seconds": _delta(
                art["challenger_mtime_unix"], art["generation_result_mtime_unix"]
            ),
        })

    return {
        "schema": SCHEMA,
        "status": "PASS" if not shard_errors else "PASS_WITH_READ_WARNINGS",
        "run_root": str(rr),
        "experience_lake_root": str(rr / "experience_lake"),
        "shards_found": len(dbs),
        "shard_errors": shard_errors,
        "generation_count_observed": len(generations),
        "generations": generations,
        "safety": dict(SAFETY),
        "interpretation": (
            "READ_ONLY_DURABILITY_TIMING_PROBE__TYPE_AWARE__RECOVERY_AWARE__"
            "NOT_SCIENTIFIC_VERDICT"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only CB16 R10 Experience Lake timing probe")
    ap.add_argument("--run-root", default="/data/cb16_hdd/cb16_runtime/R10_4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = probe_experience_lake(run_root=args.run_root)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.relative_to(Path(args.run_root).resolve())
    except ValueError:
        pass
    else:
        raise ValueError("DIAGNOSTIC_OUTPUT_MUST_BE_OUTSIDE_CANONICAL_RUN_ROOT")
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "shards_found": result["shards_found"],
        "generation_count_observed": result["generation_count_observed"],
        **result["safety"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
