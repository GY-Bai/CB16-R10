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


def main() -> int:
    ap = argparse.ArgumentParser(description="R2 native qualification with read-only frozen package authority")
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--r103-root", required=True)
    ap.add_argument("--legacy-r104-root", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--metadata-root", required=True)
    ap.add_argument("--payload-root", required=True)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    package_root = Path(args.package_root).resolve()
    r103_root = Path(args.r103_root).resolve()
    legacy_root = Path(args.legacy_r104_root).resolve()
    run_root = Path(args.run_root).resolve()
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

    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_r2_native_qualification.py"),
        "--package-root", str(package_root),
        "--r103-root", str(r103_root),
        "--legacy-r104-root", str(legacy_root),
        "--run-root", str(run_root),
        "--metadata-root", str(Path(args.metadata_root).resolve()),
        "--payload-root", str(Path(args.payload_root).resolve()),
        "--attempts", str(args.attempts),
        "--out", str(out_path),
    ]
    p = subprocess.run(cmd, check=False)

    market_cache_unchanged = _market_cache_authority_unchanged(market_binding)
    _finalize_market_cache_receipts(
        run_root=run_root,
        market_binding=market_binding,
        unchanged=market_cache_unchanged,
        require_external=p.returncode == 0,
    )
    if not market_cache_unchanged:
        if out_path.is_file():
            failed = json.loads(out_path.read_text())
            failed["status"] = "FAIL"
            failed["market_cache_source_files_unchanged"] = False
            failed["market_cache_authority_failure"] = "R2_MARKET_CACHE_AUTHORITY_CHANGED_OR_REBOUND"
            _atomic_json(out_path, failed)
        raise RuntimeError("R2_MARKET_CACHE_AUTHORITY_CHANGED_OR_REBOUND")

    if p.returncode != 0:
        return p.returncode

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
    _atomic_json(out_path, result)
    print("R2_NATIVE_V2_WRAPPER=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
