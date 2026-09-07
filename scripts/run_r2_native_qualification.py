#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cb16_local_opt.r102_common import ALL_SUPPORTED_SYMBOLS_R102
from cb16_local_opt.r2_campaign import run_campaign_r2

DEFAULT_PACKAGE_ROOT = "/home/bgy/m3-infra/CB16_SHANXI_R10_2_REAL_HISTORICAL_G0_LEARNING_V1"
DEFAULT_DATA_ROOT = "/data/cb16_hdd/binance_usdm_1m_funding_2020_2026"
DEFAULT_R103_ROOT = "/home/bgy/cb16_ssd/runtime/R10_3/qualification_r8_3_8w"
DEFAULT_LEGACY_R104_ROOT = "/data/cb16_hdd/cb16_runtime/R10_4"
DEFAULT_PARENT_R101 = "/home/bgy/m3-infra/CB16_SHANXI_FROZEN_BODY_G0_BRAIN_R10_1_THIN_V1"
DEFAULT_PARENT_G0 = "/home/bgy/cb16_ssd/runtime/R10_1/G0/central_brain_g0_r10_1.pt"
CACHE_MANIFEST = "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
PROFILE = "R2_NATIVE_R104_EQUIVALENCE_R0"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _is_inside(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _hard_safety(run_root: Path, metadata_root: Path, payload_root: Path, legacy_root: Path) -> None:
    for label, p in (("run_root", run_root), ("metadata_root", metadata_root), ("payload_root", payload_root)):
        if _is_inside(p, legacy_root):
            raise RuntimeError(f"R2_QUALIFICATION_REFUSES_CANONICAL_R104_PATH:{label}:{p}")
    if "2025_09" in str(run_root) or "2025_09" in str(metadata_root) or "2025_09" in str(payload_root):
        raise RuntimeError("R2_QUALIFICATION_FINAL_HOLDOUT_PATH_REFUSED")


def _source_file_state(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _prepare_read_only_legacy_cache_binding(
    *, legacy_root: Path, run_root: Path, expected_stride: int, expected_prehistory: int
) -> dict[str, Any]:
    source_cache = legacy_root / "evidence_cache"
    source_manifest = source_cache / CACHE_MANIFEST
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    manifest = json.loads(source_manifest.read_text())
    if manifest.get("schema") != "CB16_R10_2_REAL_EVIDENCE_CACHE_MANIFEST_V1":
        raise RuntimeError("R2_LEGACY_CACHE_SCHEMA_MISMATCH")
    if manifest.get("final_holdout_2025_09_accessed") is not False:
        raise RuntimeError("R2_LEGACY_CACHE_FINAL_HOLDOUT_BOUNDARY_FAIL")
    if int(manifest.get("stride_hours", -1)) != int(expected_stride):
        raise RuntimeError("R2_LEGACY_CACHE_STRIDE_MISMATCH")
    if int(manifest.get("prehistory_hours", -1)) != int(expected_prehistory):
        raise RuntimeError("R2_LEGACY_CACHE_PREHISTORY_MISMATCH")

    source_files = []
    for file_key, hash_key in (
        ("parents_file", "parents_sha256"),
        ("parent_states_file", "parent_states_sha256"),
        ("branches_file", "branches_sha256"),
    ):
        p = Path(manifest[file_key])
        if not p.is_file():
            raise FileNotFoundError(p)
        observed = _sha256_file(p)
        if observed != manifest[hash_key]:
            raise RuntimeError(f"R2_LEGACY_CACHE_HASH_MISMATCH:{file_key}")
        source_files.append({**_source_file_state(p), "sha256": observed, "manifest_key": file_key})

    target_cache = run_root / "evidence_cache"
    target_cache.mkdir(parents=True, exist_ok=True)
    target_manifest = target_cache / CACHE_MANIFEST
    if target_manifest.exists():
        old = json.loads(target_manifest.read_text())
        if old != manifest:
            raise RuntimeError("R2_EXTERNAL_CACHE_BINDING_CONFLICT")
    else:
        _atomic_json(target_manifest, manifest)

    # Reuse compact, already-frozen administrative results so the qualification
    # measures the native R2 path rather than redoing unrelated preflight/control work.
    copied_compact = []
    for name in ("TEN_SYMBOL_DATA_PREFLIGHT_R102.json", "F0_F1_F2_F3_CONTROLS_R102.json"):
        src = legacy_root / name
        dst = run_root / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            copied_compact.append(name)

    receipt = {
        "schema": "CB16_R2_EXTERNAL_EVIDENCE_CACHE_BINDING_V1",
        "mode": "READ_ONLY_LEGACY_CACHE_REFERENCE_VIA_COPIED_MANIFEST",
        "source_cache_root": str(source_cache),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256_file(source_manifest),
        "source_files": source_files,
        "target_manifest": str(target_manifest),
        "copied_compact_receipts": copied_compact,
        "source_payload_files_copied": False,
        "source_payload_files_modified": False,
        "scientific_semantics_changed": False,
        "final_holdout_2025_09_accessed": False,
    }
    _atomic_json(run_root / "R2_EXTERNAL_EVIDENCE_CACHE_BINDING.json", receipt)
    return receipt


def _get(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        cur = cur[part]
    return cur


def _compare_generation(r2: dict[str, Any], legacy: dict[str, Any], generation: int) -> dict[str, Any]:
    exact_paths = (
        "parent_champion_semantic_sha256",
        "challenger.semantic_sha256",
        "champion_after.semantic_sha256",
        "tournament.decision",
        "training.challenger_semantic_sha256",
        "training.optimizer_steps",
        "behavior_before.sha256",
        "behavior_after.sha256",
        "on_policy_real_trace.trace_count",
        "on_policy_real_trace.matured",
    )
    numeric_paths = (
        "validation_before.loss",
        "validation_before.direction_loss",
        "validation_before.sizing_loss",
        "training.validation_after.loss",
        "training.validation_after.direction_loss",
        "training.validation_after.sizing_loss",
        "training.parameter_l2_delta",
        "tournament.relative_improvement",
    )
    mismatches = []
    exact = {}
    numeric = {}
    for path in exact_paths:
        a, b = _get(r2, path), _get(legacy, path)
        ok = a == b
        exact[path] = {"r2": a, "legacy": b, "equal": ok}
        if not ok:
            mismatches.append(path)
    for path in numeric_paths:
        a, b = float(_get(r2, path)), float(_get(legacy, path))
        tol = max(1e-12, 1e-10 * max(abs(a), abs(b), 1.0))
        ok = abs(a - b) <= tol
        numeric[path] = {"r2": a, "legacy": b, "abs_diff": abs(a - b), "tolerance": tol, "equal": ok}
        if not ok:
            mismatches.append(path)
    return {
        "generation": int(generation),
        "pass": not mismatches,
        "mismatches": mismatches,
        "exact": exact,
        "numeric": numeric,
        "expected_identity_difference": {
            "training_snapshot_hash": {
                "r2": _get(r2, "training_snapshot.snapshot_hash"),
                "legacy": _get(legacy, "training_snapshot.snapshot_hash"),
                "must_match": False,
                "reason": "R2 snapshot authority intentionally references immutable evidence-set membership rather than legacy generation-specific ExperienceObjects",
            }
        },
    }


def _capture_generation_files(run_root: Path, attempts: int) -> dict[str, str]:
    out = {}
    for g in range(attempts):
        gd = run_root / "generations" / f"G{g:02d}"
        for name in (
            "GENERATION_RESULT.json",
            "ON_POLICY_REAL_TRACE_RECEIPT.json",
            f"CHALLENGER_TRAINING_RECEIPT_G{g}.json",
            "challenger.pt",
            "champion_after.pt",
        ):
            p = gd / name
            if not p.is_file():
                raise FileNotFoundError(p)
            out[str(p.relative_to(run_root))] = _sha256_file(p)
    return out


def _assert_source_cache_unchanged(binding: dict[str, Any]) -> bool:
    for item in binding["source_files"]:
        p = Path(item["path"])
        now = _source_file_state(p)
        if now["size"] != item["size"] or now["mtime_ns"] != item["mtime_ns"]:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="CB16 R2 native two-generation storage/scientific equivalence qualification")
    ap.add_argument("--package-root", default=DEFAULT_PACKAGE_ROOT)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--r103-root", default=DEFAULT_R103_ROOT)
    ap.add_argument("--legacy-r104-root", default=DEFAULT_LEGACY_R104_ROOT)
    ap.add_argument("--parent-r101-root", default=DEFAULT_PARENT_R101)
    ap.add_argument("--parent-g0", default=DEFAULT_PARENT_G0)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--metadata-root", required=True)
    ap.add_argument("--payload-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if not (1 <= a.attempts <= 2):
        raise RuntimeError("R2_NATIVE_QUALIFICATION_ATTEMPTS_MUST_BE_1_OR_2")

    run_root = Path(a.run_root).resolve()
    metadata_root = Path(a.metadata_root).resolve()
    payload_root = Path(a.payload_root).resolve()
    legacy_root = Path(a.legacy_r104_root).resolve()
    _hard_safety(run_root, metadata_root, payload_root, legacy_root)
    run_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    payload_root.mkdir(parents=True, exist_ok=True)

    r103 = Path(a.r103_root).resolve()
    prerequisite = r103 / "FINAL_RESULT_R102.json"
    start_checkpoint = r103 / "generations/G19/champion_after.pt"
    if not prerequisite.is_file() or not start_checkpoint.is_file():
        raise FileNotFoundError(f"R2_R103_PREREQUISITE_MISSING:{prerequisite}:{start_checkpoint}")
    prev = json.loads(prerequisite.read_text())
    if not str(prev.get("final_status", "")).endswith("PASS"):
        raise RuntimeError("R2_R103_PREREQUISITE_NOT_PASS")

    binding = _prepare_read_only_legacy_cache_binding(
        legacy_root=legacy_root, run_root=run_root, expected_stride=256, expected_prehistory=96
    )

    kwargs = dict(
        package_root=a.package_root,
        data_root=a.data_root,
        run_root=run_root,
        parent_r101_root=a.parent_r101_root,
        parent_g0=a.parent_g0,
        metadata_root=metadata_root,
        payload_roots=[payload_root],
        device=a.device,
        symbols=ALL_SUPPORTED_SYMBOLS_R102,
        attempts=a.attempts,
        stride_hours=256,
        prehistory_hours=96,
        epochs=12,
        batch_size=512,
        lr=3e-4,
        profile_name=PROFILE,
        prerequisite_result=prerequisite,
        start_checkpoint=start_checkpoint,
        codec="zstd",
    )

    started = time.perf_counter()
    first = run_campaign_r2(**kwargs)
    first_seconds = time.perf_counter() - started
    first_materialize = json.loads((run_root / "R2_TRAINING_EVIDENCE_MATERIALIZATION.json").read_text())
    _atomic_json(run_root / "R2_TRAINING_EVIDENCE_MATERIALIZATION_FIRST.json", first_materialize)
    _atomic_json(run_root / "FINAL_RESULT_R2_FIRST.json", first)

    generation_equivalence = []
    for g in range(a.attempts):
        r2g = json.loads((run_root / "generations" / f"G{g:02d}" / "GENERATION_RESULT.json").read_text())
        legacy_path = legacy_root / "generations" / f"G{g:02d}" / "GENERATION_RESULT.json"
        if not legacy_path.is_file():
            raise FileNotFoundError(legacy_path)
        legacyg = json.loads(legacy_path.read_text())
        generation_equivalence.append(_compare_generation(r2g, legacyg, g))

    files_before_replay = _capture_generation_files(run_root, a.attempts)
    final_path = run_root / "FINAL_RESULT_R2.json"
    final_path.unlink()
    replay_started = time.perf_counter()
    replay = run_campaign_r2(**kwargs)
    replay_seconds = time.perf_counter() - replay_started
    replay_materialize = json.loads((run_root / "R2_TRAINING_EVIDENCE_MATERIALIZATION.json").read_text())
    _atomic_json(run_root / "R2_TRAINING_EVIDENCE_MATERIALIZATION_REPLAY.json", replay_materialize)
    files_after_replay = _capture_generation_files(run_root, a.attempts)

    replay_files_unchanged = files_before_replay == files_after_replay
    source_cache_unchanged = _assert_source_cache_unchanged(binding)
    equivalence_pass = all(x["pass"] for x in generation_equivalence)
    first_pass = str(first.get("final_status", "")).endswith("PASS") and bool(first.get("mechanistic_pipeline_pass"))
    replay_pass = str(replay.get("final_status", "")).endswith("PASS") and bool(replay.get("mechanistic_pipeline_pass"))
    storage_pass = bool(first.get("storage_audit", {}).get("pass")) and bool(first.get("event_journal_audit", {}).get("pass"))
    one_lane = int(first.get("storage_stats", {}).get("payload_lanes", 0)) == 1
    replay_zero_new_payload = int(replay_materialize.get("created_payload_count", -1)) == 0
    same_final_champion = first.get("final_champion_semantic_sha256") == replay.get("final_champion_semantic_sha256")

    passed = all((
        first_pass,
        replay_pass,
        storage_pass,
        one_lane,
        equivalence_pass,
        replay_zero_new_payload,
        replay_files_unchanged,
        same_final_champion,
        source_cache_unchanged,
    ))

    result = {
        "schema": "CB16_R2_NATIVE_EQUIVALENCE_QUALIFICATION_V1",
        "status": "PASS" if passed else "FAIL",
        "profile": PROFILE,
        "attempts": a.attempts,
        "first_run_seconds": first_seconds,
        "replay_seconds": replay_seconds,
        "first_final_status": first.get("final_status"),
        "replay_final_status": replay.get("final_status"),
        "first_materialization": first_materialize,
        "replay_materialization": replay_materialize,
        "generation_equivalence": generation_equivalence,
        "replay_generation_files_unchanged": replay_files_unchanged,
        "same_final_champion_after_replay": same_final_champion,
        "source_cache_unchanged": source_cache_unchanged,
        "source_cache_binding": binding,
        "storage_stats": first.get("storage_stats"),
        "storage_audit": first.get("storage_audit"),
        "event_journal_audit": first.get("event_journal_audit"),
        "run_root": str(run_root),
        "metadata_root": str(metadata_root),
        "payload_root": str(payload_root),
        "canonical_r104_modified": False,
        "scientific_semantics_changed": False,
        "final_holdout_2025_09_accessed": False,
        "gate": {
            "first_campaign_pass": first_pass,
            "replay_campaign_pass": replay_pass,
            "storage_and_event_audits_pass": storage_pass,
            "single_hdd_payload_lane": one_lane,
            "legacy_scientific_equivalence_pass": equivalence_pass,
            "replay_created_zero_new_payloads": replay_zero_new_payload,
            "replay_generation_files_byte_identical": replay_files_unchanged,
            "source_legacy_cache_unchanged": source_cache_unchanged,
        },
    }
    _atomic_json(Path(a.out), result)
    print(json.dumps({
        "status": result["status"],
        "first_run_seconds": first_seconds,
        "replay_seconds": replay_seconds,
        "first_created_payload_count": first_materialize.get("created_payload_count"),
        "replay_created_payload_count": replay_materialize.get("created_payload_count"),
        "final_champion": first.get("final_champion_semantic_sha256"),
        "generation_equivalence": [x["pass"] for x in generation_equivalence],
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
