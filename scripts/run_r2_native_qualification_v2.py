#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    (run_root / "R2_FROZEN_ADOPTION_REUSE.json").write_text(json.dumps(reuse, indent=2, sort_keys=True) + "\n")

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
        "--out", str(Path(args.out).resolve()),
    ]
    p = subprocess.run(cmd, check=False)
    if p.returncode != 0:
        return p.returncode

    result = json.loads(Path(args.out).read_text())
    result["frozen_adoption_reuse"] = reuse
    result["package_authority_mode"] = "READ_ONLY_SYSTEMD_ALIAS"
    result["parent_adoption_reexecuted"] = False
    result["package_root_modified"] = False
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("R2_NATIVE_V2_WRAPPER=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
