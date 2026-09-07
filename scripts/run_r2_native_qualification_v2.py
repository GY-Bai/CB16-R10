#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cb16_local_opt.r102_common import ALL_SUPPORTED_SYMBOLS_R102


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _file_state(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def _copy_frozen_adoption_receipt(*, legacy_r104_root: Path, run_root: Path) -> dict:
    src = legacy_r104_root / "PARENT_ADOPTION_RECEIPT_R102.json"
    if not src.is_file():
        raise FileNotFoundError(f"R2_FROZEN_ADOPTION_RECEIPT_MISSING:{src}")
    obj = json.loads(src.read_text())
    if obj.get("schema") != "CB16_R10_2_PARENT_ADOPTION_RECEIPT_V1":
        raise RuntimeError("R2_FROZEN_ADOPTION_RECEIPT_SCHEMA_MISMATCH")
    if obj.get("status") != "R10_1_PARENT_AND_FROZEN_AUTHORITIES_ADOPTED":
        raise RuntimeError(f"R2_FROZEN_ADOPTION_STATUS_INVALID:{obj.get('status')}")
    if obj.get("scientific_semantics_changed") is not False:
        raise RuntimeError("R2_FROZEN_ADOPTION_SEMANTICS_CHANGED")

    run_root.mkdir(parents=True, exist_ok=True)
    dst = run_root / "PARENT_ADOPTION_RECEIPT_R102.json"
    if dst.exists():
        existing = json.loads(dst.read_text())
        if existing != obj:
            raise RuntimeError("R2_FROZEN_ADOPTION_RECEIPT_CONFLICT")
    else:
        shutil.copy2(src, dst)
    return {
        "schema": "CB16_R2_FROZEN_ADOPTION_REUSE_V1",
        "source": str(src),
        "destination": str(dst),
        "parent_adoption_reexecuted": False,
        "package_root_modified": False,
        "scientific_semantics_changed": False,
        "final_holdout_2025_09_accessed": False,
    }


def _prepare_market_cache_authority_binding(*, legacy_r104_root: Path, run_root: Path) -> dict[str, Any]:
    source_root = legacy_r104_root / "evidence_cache" / "market_cache"
    if not source_root.is_dir():
        raise FileNotFoundError(f"R2_MARKET_CACHE_AUTHORITY_MISSING:{source_root}")

    source_files = []
    for symbol in ALL_SUPPORTED_SYMBOLS_R102:
        src = source_root / f"{symbol}.hourly_r102.npz"
        if not src.is_file():
            raise FileNotFoundError(f"R2_MARKET_CACHE_SYMBOL_MISSING:{symbol}:{src}")
        source_files.append({**_file_state(src), "symbol": str(symbol)})

    target_root = run_root / "evidence_cache" / "market_cache"
    target_root.parent.mkdir(parents=True, exist_ok=True)
    source_resolved = source_root.resolve(strict=True)

    if target_root.is_symlink():
        if target_root.resolve(strict=True) != source_resolved:
            raise RuntimeError(
                f"R2_MARKET_CACHE_BINDING_CONFLICT:{target_root}:{target_root.resolve(strict=True)}:{source_resolved}"
            )
    elif target_root.exists():
        raise RuntimeError(f"R2_MARKET_CACHE_TARGET_MUST_BE_SYMLINK:{target_root}")
    else:
        target_root.symlink_to(source_root, target_is_directory=True)

    if not target_root.is_symlink() or target_root.resolve(strict=True) != source_resolved:
        raise RuntimeError("R2_MARKET_CACHE_SYMLINK_BINDING_FAIL")

    for item in source_files:
        target_file = target_root / f"{item['symbol']}.hourly_r102.npz"
        if not target_file.is_file():
            raise FileNotFoundError(f"R2_MARKET_CACHE_BOUND_SYMBOL_UNREADABLE:{target_file}")
        if target_file.resolve(strict=True) != Path(item["path"]).resolve(strict=True):
            raise RuntimeError(f"R2_MARKET_CACHE_BOUND_SYMBOL_IDENTITY_FAIL:{item['symbol']}")

    receipt = {
        "schema": "CB16_R2_MARKET_CACHE_AUTHORITY_BINDING_V1",
        "mode": "SYMLINK_TO_FROZEN_R104_READ_ONLY_AUTHORITY",
        "source_market_cache_root": str(source_root),
        "target_market_cache_root": str(target_root),
        "market_cache_symbols": [str(x) for x in ALL_SUPPORTED_SYMBOLS_R102],
        "source_files": source_files,
        "market_cache_payload_files_copied": False,
        "source_payload_files_modified": False,
        "post_run_verified": False,
        "scientific_semantics_changed": False,
        "final_holdout_2025_09_accessed": False,
    }
    _atomic_json(run_root / "R2_MARKET_CACHE_AUTHORITY_BINDING.json", receipt)
    return receipt


def _market_cache_authority_unchanged(binding: dict[str, Any]) -> bool:
    source_root = Path(binding["source_market_cache_root"])
    target_root = Path(binding["target_market_cache_root"])
    try:
        if not target_root.is_symlink():
            return False
        if target_root.resolve(strict=True) != source_root.resolve(strict=True):
            return False
        for item in binding["source_files"]:
            src = Path(item["path"])
            now = _file_state(src)
            if now["size"] != item["size"] or now["mtime_ns"] != item["mtime_ns"]:
                return False
            target_file = target_root / f"{item['symbol']}.hourly_r102.npz"
            if not target_file.is_file():
                return False
            if target_file.resolve(strict=True) != src.resolve(strict=True):
                return False
    except (FileNotFoundError, OSError):
        return False
    return True


def _finalize_market_cache_receipts(
    *, run_root: Path, market_binding: dict[str, Any], unchanged: bool, require_external: bool
) -> dict[str, Any] | None:
    market_binding = dict(market_binding)
    market_binding["post_run_verified"] = True
    market_binding["source_files_unchanged"] = bool(unchanged)
    market_binding["source_payload_files_modified"] = not bool(unchanged)
    _atomic_json(run_root / "R2_MARKET_CACHE_AUTHORITY_BINDING.json", market_binding)

    external_path = run_root / "R2_EXTERNAL_EVIDENCE_CACHE_BINDING.json"
    if not external_path.is_file():
        if require_external:
            raise FileNotFoundError(f"R2_EXTERNAL_CACHE_BINDING_RECEIPT_MISSING:{external_path}")
        return None

    external = json.loads(external_path.read_text())
    external["schema"] = "CB16_R2_EXTERNAL_EVIDENCE_CACHE_BINDING_V2"
    external["mode"] = "READ_ONLY_LEGACY_CACHE_REFERENCE_VIA_COPIED_MANIFEST_AND_MARKET_CACHE_SYMLINK"
    external["source_market_cache_root"] = market_binding["source_market_cache_root"]
    external["target_market_cache_root"] = market_binding["target_market_cache_root"]
    external["market_cache_binding"] = market_binding["mode"]
    external["market_cache_symbols"] = market_binding["market_cache_symbols"]
    external["market_cache_files"] = market_binding["source_files"]
    external["market_cache_payload_files_copied"] = False
    external["market_cache_source_files_unchanged"] = bool(unchanged)
    external["source_payload_files_modified"] = (
        bool(external.get("source_payload_files_modified", False)) or not bool(unchanged)
    )
    external["scientific_semantics_changed"] = False
    external["final_holdout_2025_09_accessed"] = False
    _atomic_json(external_path, external)
    return external


def _prepare_persistent_teacher_authority_binding(
    *, metadata_root: Path, payload_root: Path, legacy_r104_root: Path,
    explicit_root: Path | None = None,
) -> dict[str, Any]:
    """Bind transient metadata to a persistent R2-only compiled Teacher authority."""
    metadata_root.mkdir(parents=True, exist_ok=True)
    if explicit_root is None:
        if len(payload_root.parents) < 2:
            raise RuntimeError("R2_TEACHER_CACHE_PAYLOAD_ROOT_TOO_SHALLOW")
        persistent_root = payload_root.parent.parent / "compiled_teacher_authority"
    else:
        persistent_root = explicit_root
    persistent_root = persistent_root.resolve()
    legacy_resolved = legacy_r104_root.resolve()
    if persistent_root == legacy_resolved or legacy_resolved in persistent_root.parents:
        raise RuntimeError("R2_TEACHER_CACHE_REFUSES_CANONICAL_R104_PATH")
    if "2025_09" in str(persistent_root):
        raise RuntimeError("R2_TEACHER_CACHE_FINAL_HOLDOUT_PATH_REFUSED")
    persistent_root.mkdir(parents=True, exist_ok=True)

    target = metadata_root / "compiled_teacher_authority"
    if target.is_symlink():
        if target.resolve(strict=True) != persistent_root.resolve(strict=True):
            raise RuntimeError("R2_TEACHER_CACHE_BINDING_CONFLICT")
    elif target.exists():
        raise RuntimeError("R2_TEACHER_CACHE_TARGET_MUST_BE_SYMLINK")
    else:
        target.symlink_to(persistent_root, target_is_directory=True)
    if not target.is_symlink() or target.resolve(strict=True) != persistent_root.resolve(strict=True):
        raise RuntimeError("R2_TEACHER_CACHE_SYMLINK_BINDING_FAIL")

    return {
        "schema": "CB16_R2_COMPILED_TEACHER_PERSISTENT_BINDING_V1",
        "mode": "TRANSIENT_METADATA_SYMLINK_TO_PERSISTENT_R2_HDD_AUTHORITY",
        "persistent_root": str(persistent_root),
        "metadata_binding": str(target),
        "canonical_r104_modified": False,
        "scientific_semantics_changed": False,
        "final_holdout_2025_09_accessed": False,
    }


def _manifest_names(root: Path) -> list[str]:
    return sorted(p.name for p in root.glob("*.manifest.json") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description="R2 native qualification with read-only frozen package authority")
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--r103-root", required=True)
    ap.add_argument("--legacy-r104-root", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--metadata-root", required=True)
    ap.add_argument("--payload-root", required=True)
    ap.add_argument("--teacher-cache-root")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    package_root = Path(args.package_root).resolve()
    r103_root = Path(args.r103_root).resolve()
    legacy_root = Path(args.legacy_r104_root).resolve()
    run_root = Path(args.run_root).resolve()
    metadata_root = Path(args.metadata_root).resolve()
    payload_root = Path(args.payload_root).resolve()
    out_path = Path(args.out).resolve()

    if not package_root.is_dir():
        raise FileNotFoundError(f"R2_PACKAGE_AUTHORITY_ALIAS_MISSING:{package_root}")
    if not r103_root.is_dir():
        raise FileNotFoundError(f"R2_R103_AUTHORITY_ALIAS_MISSING:{r103_root}")
    if not legacy_root.is_dir():
        raise FileNotFoundError(f"R2_LEGACY_R104_ROOT_MISSING:{legacy_root}")

    required_package_files = (
        "assets/operator/runtime/kronos_model_l5.safetensors",
        "assets/operator/runtime/kronos_tokenizer_encode.safetensors",
        "assets/operator/operator_reducers_v1.npz",
        "assets/medium/runtime/timesfm_layer3.safetensors",
        "authority/control_plane_r1/risk_supervisor_r1.py",
        "authority/account_physics_r0/CB16_ACCOUNT_PHYSICS_STATE_V1_R0/ACCOUNT_PHYSICS_CONTRACT_V1.json",
    )
    for rel in required_package_files:
        p = package_root / rel
        if not p.is_file():
            raise FileNotFoundError(f"R2_PACKAGE_AUTHORITY_FILE_MISSING:{rel}")

    reuse = _copy_frozen_adoption_receipt(legacy_r104_root=legacy_root, run_root=run_root)
    _atomic_json(run_root / "R2_FROZEN_ADOPTION_REUSE.json", reuse)

    market_binding = _prepare_market_cache_authority_binding(
        legacy_r104_root=legacy_root,
        run_root=run_root,
    )
    teacher_binding = _prepare_persistent_teacher_authority_binding(
        metadata_root=metadata_root,
        payload_root=payload_root,
        legacy_r104_root=legacy_root,
        explicit_root=Path(args.teacher_cache_root).resolve() if args.teacher_cache_root else None,
    )
    _atomic_json(run_root / "R2_COMPILED_TEACHER_PERSISTENT_BINDING.json", teacher_binding)
    persistent_teacher_root = Path(teacher_binding["persistent_root"])
    teacher_manifests_before = _manifest_names(persistent_teacher_root)

    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_r2_native_qualification.py"),
        "--package-root", str(package_root),
        "--r103-root", str(r103_root),
        "--legacy-r104-root", str(legacy_root),
        "--run-root", str(run_root),
        "--metadata-root", str(metadata_root),
        "--payload-root", str(payload_root),
        "--attempts", str(args.attempts),
        "--out", str(out_path),
    ]
    process = subprocess.run(cmd, check=False)

    market_cache_unchanged = _market_cache_authority_unchanged(market_binding)
    _finalize_market_cache_receipts(
        run_root=run_root,
        market_binding=market_binding,
        unchanged=market_cache_unchanged,
        require_external=process.returncode == 0,
    )
    if not market_cache_unchanged:
        if out_path.is_file():
            failed = json.loads(out_path.read_text())
            failed["status"] = "FAIL"
            failed["market_cache_source_files_unchanged"] = False
            failed["market_cache_authority_failure"] = "R2_MARKET_CACHE_AUTHORITY_CHANGED_OR_REBOUND"
            _atomic_json(out_path, failed)
        raise RuntimeError("R2_MARKET_CACHE_AUTHORITY_CHANGED_OR_REBOUND")

    if process.returncode != 0:
        return process.returncode

    teacher_receipt_path = run_root / "COMPILED_TEACHER_AUTHORITY_RECEIPT_R102.json"
    if not teacher_receipt_path.is_file():
        raise FileNotFoundError(f"R2_COMPILED_TEACHER_RECEIPT_MISSING:{teacher_receipt_path}")
    teacher_receipt = json.loads(teacher_receipt_path.read_text())
    authority_hash = str(teacher_receipt.get("authority_hash", ""))
    expected_manifest_name = f"{authority_hash}.manifest.json"
    teacher_manifests_after = _manifest_names(persistent_teacher_root)
    existed_before = expected_manifest_name in set(teacher_manifests_before)
    replay_mode = teacher_receipt.get("mode")
    replay_teacher_reused = replay_mode == "REUSED_VERIFIED_AUTHORITY"

    result = json.loads(out_path.read_text())
    result["frozen_adoption_reuse"] = reuse
    result["package_authority_mode"] = "READ_ONLY_SYSTEMD_ALIAS"
    result["parent_adoption_reexecuted"] = False
    result["package_root_modified"] = False
    result["market_cache_authority_mode"] = market_binding["mode"]
    result["market_cache_source_root"] = market_binding["source_market_cache_root"]
    result["market_cache_target_root"] = market_binding["target_market_cache_root"]
    result["market_cache_symbol_count"] = len(market_binding["market_cache_symbols"])
    result["market_cache_payload_files_copied"] = False
    result["market_cache_source_files_unchanged"] = True
    result["compiled_teacher_persistent_binding"] = teacher_binding
    result["compiled_teacher_authority_hash"] = authority_hash
    result["compiled_teacher_manifest_existed_before_native_run"] = existed_before
    result["compiled_teacher_first_invocation_mode_inferred"] = (
        "REUSED_EXISTING_PERSISTENT_AUTHORITY"
        if existed_before
        else "COLD_COMPILED_AND_PUBLISHED"
    )
    result["compiled_teacher_replay_mode"] = replay_mode
    result["compiled_teacher_replay_reused_verified_authority"] = replay_teacher_reused
    result["compiled_teacher_manifests_before"] = teacher_manifests_before
    result["compiled_teacher_manifests_after"] = teacher_manifests_after
    if not replay_teacher_reused:
        result["status"] = "FAIL"
        result["compiled_teacher_authority_failure"] = "R2_REPLAY_DID_NOT_REUSE_VERIFIED_TEACHER_AUTHORITY"
        _atomic_json(out_path, result)
        return 2

    _atomic_json(out_path, result)
    print("R2_NATIVE_V2_WRAPPER=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())