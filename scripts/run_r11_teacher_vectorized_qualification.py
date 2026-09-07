#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cb16_local_opt.r102_common import sha256_obj
from cb16_local_opt.r102_evidence_cache import load_teacher_samples
from cb16_local_opt.r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102
from cb16_local_opt.r102_teacher_incremental import (
    compile_teacher_evidence_incremental,
    teacher_authority_identity,
)
from cb16_local_opt.rearchitecture_authority_r11 import verify_static_semantic_contracts
from cb16_local_opt.teacher_vectorized_r11 import compile_teacher_evidence_vectorized_r11


NUMERIC_ATOL = 5e-10
MAX_REPORTED_MISMATCHES = 24


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _finite_diff(a: float, b: float) -> float:
    if math.isinf(a) or math.isinf(b):
        return 0.0 if a == b else float("inf")
    if math.isnan(a) or math.isnan(b):
        return float("inf")
    return abs(float(a) - float(b))


def _compare_evidence(old_rows, new_rows) -> dict[str, Any]:
    old = {e.parent_id: e for e in old_rows}
    new = {e.parent_id: e for e in new_rows}
    exact_mismatches: list[dict[str, Any]] = []
    numeric_mismatches: list[dict[str, Any]] = []
    max_abs = 0.0
    numeric_values_checked = 0

    if set(old) != set(new):
        exact_mismatches.append({
            "field": "parent_id_set",
            "old_only": sorted(set(old) - set(new))[:10],
            "new_only": sorted(set(new) - set(old))[:10],
        })

    def exact(parent_id: str, field: str, a: Any, b: Any) -> None:
        if a != b and len(exact_mismatches) < MAX_REPORTED_MISMATCHES:
            exact_mismatches.append({"parent_id": parent_id, "field": field, "old": a, "new": b})

    def numeric(parent_id: str, field: str, a: float, b: float) -> None:
        nonlocal max_abs, numeric_values_checked
        d = _finite_diff(float(a), float(b))
        numeric_values_checked += 1
        max_abs = max(max_abs, d)
        if d > NUMERIC_ATOL and len(numeric_mismatches) < MAX_REPORTED_MISMATCHES:
            numeric_mismatches.append({
                "parent_id": parent_id,
                "field": field,
                "old": float(a),
                "new": float(b),
                "abs_diff": d,
            })

    for parent_id in sorted(set(old) & set(new)):
        a = old[parent_id]
        b = new[parent_id]
        for field in (
            "evidence_id", "parent_id", "student_context_object_id",
            "target_dependence_group_id", "timestamp", "teacher_version",
            "teacher_protocol_hash", "train_dependence_group_hash",
        ):
            exact(parent_id, field, getattr(a, field), getattr(b, field))
        for field in ("status", "lane", "unique_train_dependence_groups", "reasons", "protocol_hash"):
            exact(parent_id, f"admission.{field}", getattr(a.admission, field), getattr(b.admission, field))
        numeric(parent_id, "admission.minimum_action_effective_dependence_n",
                a.admission.minimum_action_effective_dependence_n,
                b.admission.minimum_action_effective_dependence_n)
        numeric(parent_id, "admission.maximum_action_nearest_distance",
                a.admission.maximum_action_nearest_distance,
                b.admission.maximum_action_nearest_distance)

        exact(parent_id, "action_law_count", len(a.action_laws), len(b.action_laws))
        for i, (la, lb) in enumerate(zip(a.action_laws, b.action_laws)):
            prefix = f"law[{i}]"
            exact(parent_id, prefix + ".direction", la.direction, lb.direction)
            exact(parent_id, prefix + ".requested_risk", la.requested_risk, lb.requested_risk)
            exact(parent_id, prefix + ".quantile_levels", la.quantile_levels, lb.quantile_levels)
            exact(parent_id, prefix + ".unique_dependence_groups",
                  la.unique_dependence_groups, lb.unique_dependence_groups)
            exact(parent_id, prefix + ".support_dependence_group_hash",
                  la.support_dependence_group_hash, lb.support_dependence_group_hash)
            for field in (
                "mean_utility", "std_utility", "effective_dependence_n",
                "nearest_distance", "max_distance_used",
            ):
                numeric(parent_id, prefix + "." + field, getattr(la, field), getattr(lb, field))
            exact(parent_id, prefix + ".quantile_count", len(la.quantiles), len(lb.quantiles))
            for qi, (qa, qb) in enumerate(zip(la.quantiles, lb.quantiles)):
                numeric(parent_id, f"{prefix}.quantile[{qi}]", qa, qb)

        exact(parent_id, "direction_target_prob_count", len(a.direction_target_probs), len(b.direction_target_probs))
        for i, (pa, pb) in enumerate(zip(a.direction_target_probs, b.direction_target_probs)):
            numeric(parent_id, f"direction_target_probs[{i}]", pa, pb)
        numeric(parent_id, "requested_risk_target", a.requested_risk_target, b.requested_risk_target)
        exact(parent_id, "direction_weight", a.direction_weight, b.direction_weight)
        exact(parent_id, "sizing_weight", a.sizing_weight, b.sizing_weight)
        exact(
            parent_id,
            "direction_argmax",
            max(range(3), key=lambda i: a.direction_target_probs[i]),
            max(range(3), key=lambda i: b.direction_target_probs[i]),
        )

    return {
        "old_count": len(old),
        "new_count": len(new),
        "exact_mismatch_count_reported": len(exact_mismatches),
        "numeric_mismatch_count_reported": len(numeric_mismatches),
        "numeric_values_checked": numeric_values_checked,
        "max_abs_numeric_diff": max_abs,
        "numeric_atol": NUMERIC_ATOL,
        "exact_mismatches": exact_mismatches,
        "numeric_mismatches": numeric_mismatches,
        "pass": (
            len(old) == len(new)
            and not exact_mismatches
            and not numeric_mismatches
            and max_abs <= NUMERIC_ATOL
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="R11 vectorized Teacher full read-only oracle qualification")
    ap.add_argument("--legacy-r104-root", required=True)
    ap.add_argument("--teacher-cache-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-targets", type=int, default=64)
    args = ap.parse_args()

    static_guard = verify_static_semantic_contracts(ROOT)
    legacy_root = Path(args.legacy_r104_root).resolve()
    cache_root = Path(args.teacher_cache_root).resolve()
    out_path = Path(args.out).resolve()

    manifest_path = legacy_root / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"R11_R104_EVIDENCE_MANIFEST_MISSING:{manifest_path}")
    cache_manifest = json.loads(manifest_path.read_text())
    if int(cache_manifest.get("stride_hours", -1)) != 256:
        raise RuntimeError(f"R11_EXPECTED_R104_STRIDE_256:{cache_manifest.get('stride_hours')}")
    if bool(cache_manifest.get("final_holdout_2025_09_accessed", False)):
        raise RuntimeError("R11_REFUSES_CACHE_THAT_ACCESSED_FINAL_HOLDOUT")

    parents, samples = load_teacher_samples(cache_manifest["parents_file"], cache_manifest["branches_file"])

    oracle_identity = teacher_authority_identity(
        source_identity=cache_manifest,
        train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,
    )
    oracle_hash = sha256_obj(oracle_identity)
    oracle_manifest = cache_root / f"{oracle_hash}.manifest.json"
    oracle_payload = cache_root / f"{oracle_hash}.teacher.json.gz"
    if not oracle_manifest.is_file() or not oracle_payload.is_file():
        raise RuntimeError(
            "R11_LEGACY_TEACHER_ORACLE_NOT_PRECOMPILED__REFUSE_TO_RECOMPUTE_OLD_RUNTIME:"
            f"{oracle_hash}"
        )

    t0 = time.perf_counter()
    old_train, old_val, old_receipt = compile_teacher_evidence_incremental(
        samples=samples,
        parents=parents,
        source_identity=cache_manifest,
        cache_root=cache_root,
        train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,
        workers=1,
        threads_per_worker=1,
        max_in_flight=1,
    )
    oracle_load_seconds = time.perf_counter() - t0
    if old_receipt.get("mode") != "REUSED_VERIFIED_AUTHORITY":
        raise RuntimeError(f"R11_ORACLE_MUST_BE_WARM_REUSE:{old_receipt}")

    t1 = time.perf_counter()
    new_train, new_val, stats = compile_teacher_evidence_vectorized_r11(
        samples=samples,
        parents=parents,
        train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,
        block_targets=int(args.block_targets),
    )
    vectorized_seconds = time.perf_counter() - t1

    train_cmp = _compare_evidence(old_train, new_train)
    val_cmp = _compare_evidence(old_val, new_val)
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    passed = bool(train_cmp["pass"] and val_cmp["pass"])
    result = {
        "schema": "CB16_R11_VECTORIZED_TEACHER_QUALIFICATION_V1",
        "status": "PASS" if passed else "FAIL",
        "scientific_semantics_changed": False,
        "legacy_python_runtime_authority": False,
        "legacy_compiled_teacher_role": "READ_ONLY_QUALIFICATION_ORACLE",
        "static_semantic_guard": static_guard,
        "source": {
            "legacy_r104_root": str(legacy_root),
            "evidence_manifest": str(manifest_path),
            "stride_hours": int(cache_manifest["stride_hours"]),
            "parent_count": len(parents),
            "branch_sample_count": len(samples),
            "final_holdout_2025_09_accessed": False,
        },
        "oracle": {
            "authority_hash": oracle_hash,
            "mode": old_receipt["mode"],
            "load_seconds": oracle_load_seconds,
            "train_count": len(old_train),
            "validation_count": len(old_val),
        },
        "r11": {
            "engine": stats.engine,
            "vectorized_seconds": vectorized_seconds,
            "block_targets": int(args.block_targets),
            "stats": asdict(stats),
            "peak_rss_kib": peak_rss_kib,
            "train_count": len(new_train),
            "validation_count": len(new_val),
        },
        "equivalence": {
            "policy": "EXACT_DISCRETE_SEMANTICS_PLUS_TIGHT_FLOAT64_NUMERICAL_EQUIVALENCE",
            "train": train_cmp,
            "validation": val_cmp,
        },
        "forbidden_work": {
            "raw_1m_archive_read": False,
            "final_holdout_payload_read": False,
            "h72_physics_execution": False,
            "central_brain_training": False,
            "champion_or_challenger_mutation": False,
            "legacy_teacher_recomputation": False,
        },
    }
    _atomic_json(out_path, result)
    print(json.dumps({
        "status": result["status"],
        "oracle_hash": oracle_hash,
        "vectorized_seconds": vectorized_seconds,
        "train_pass": train_cmp["pass"],
        "validation_pass": val_cmp["pass"],
        "train_max_abs": train_cmp["max_abs_numeric_diff"],
        "validation_max_abs": val_cmp["max_abs_numeric_diff"],
        "stats": asdict(stats),
        "peak_rss_kib": peak_rss_kib,
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
