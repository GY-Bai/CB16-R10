from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .frozen_layer_contract_r10 import PREFIX_SPECS_R10, verify_installed_prefix
from .r102_common import (
    G0_FILE_SHA256, G0_TENSOR_SEMANTIC_SHA256, PHYSICS_CONTRACT_SHA256,
    RISK_SUPERVISOR_SHA256, V55_KERNEL_AGGREGATE_SHA256,
    atomic_write_json, hardlink_or_copy, sha256_file, torch_tensor_semantic_sha256,
)

DEFAULT_PARENT_R101 = Path("/home/bgy/m3-infra/CB16_SHANXI_FROZEN_BODY_G0_BRAIN_R10_1_THIN_V1")
DEFAULT_PARENT_G0 = Path("/home/bgy/cb16_ssd/runtime/R10_1/G0/central_brain_g0_r10_1.pt")


def _checkpoint_state(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    state = obj.get("state_dict") or obj.get("model_state") or obj.get("model") or obj
    if not isinstance(state, dict):
        raise RuntimeError("PARENT_G0_STATE_NOT_FOUND")
    return state


def adopt_parent_r101(
    *,
    package_root: str | Path,
    parent_r101_root: str | Path = DEFAULT_PARENT_R101,
    parent_g0: str | Path = DEFAULT_PARENT_G0,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(package_root).resolve()
    parent = Path(parent_r101_root).resolve()
    g0 = Path(parent_g0).resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    parent_gate_path = parent / "PRELEARNING_GATE_RECEIPT_R10_1.json"
    if not parent_gate_path.is_file():
        raise RuntimeError(f"R10_1_PRELEARNING_RECEIPT_REQUIRED:{parent_gate_path}")
    parent_gate = json.loads(parent_gate_path.read_text())
    if parent_gate.get("status") != "READY_FOR_R10_1_G0_LEARNING_PIPELINE":
        raise RuntimeError(f"R10_1_PARENT_NOT_READY:{parent_gate.get('status')}")
    if parent_gate.get("authority_resolution") != "A__INPUT_STEMS_ARE_CENTRAL_BRAIN_OWNED_AND_TRAINABLE":
        raise RuntimeError("R10_1_PARENT_AUTHORITY_A_MISMATCH")
    if bool(parent_gate.get("market_training_started", True)):
        raise RuntimeError("R10_1_PARENT_RECEIPT_SAYS_MARKET_TRAINING_ALREADY_STARTED")
    if not g0.is_file():
        # Safe fallback to the exact immutable G0 bundled in the package itself.
        bundled = root / "authority/g0_parent/central_brain_g0_r10_parent.pt"
        if bundled.is_file() and sha256_file(bundled) == G0_FILE_SHA256:
            g0 = bundled
        else:
            raise FileNotFoundError(g0)

    adopted = {}
    for prefix_id, spec in PREFIX_SPECS_R10.items():
        rel = Path(spec["output_relpath"])
        src = parent / rel
        dst = root / rel
        mode = hardlink_or_copy(src, dst)
        verification = verify_installed_prefix(root, prefix_id)
        adopted[prefix_id] = {"mode": mode, **verification}

    g0_file_sha = sha256_file(g0)
    if g0_file_sha != G0_FILE_SHA256:
        raise RuntimeError(f"PARENT_G0_FILE_SHA_MISMATCH:{g0_file_sha}")
    semantic = torch_tensor_semantic_sha256(_checkpoint_state(g0))
    if semantic != G0_TENSOR_SEMANTIC_SHA256:
        raise RuntimeError(f"PARENT_G0_TENSOR_SEMANTIC_MISMATCH:{semantic}")

    local_g0 = root / "authority/g0_parent/central_brain_g0_r10_2_parent.pt"
    mode = hardlink_or_copy(g0, local_g0)

    physics_root = root / "authority/account_physics_r0/CB16_ACCOUNT_PHYSICS_STATE_V1_R0"
    physics_contract = json.loads((physics_root / "ACCOUNT_PHYSICS_CONTRACT_V1.json").read_text())
    if physics_contract.get("contract_sha256") != PHYSICS_CONTRACT_SHA256:
        raise RuntimeError("PHYSICS_CONTRACT_AUTHORITY_MISMATCH")
    if physics_contract.get("historical_kernel", {}).get("source_aggregate_sha256") not in (None, V55_KERNEL_AGGREGATE_SHA256):
        raise RuntimeError("V55_KERNEL_AGGREGATE_AUTHORITY_MISMATCH")

    supervisor = root / "authority/control_plane_r1/risk_supervisor_r1.py"
    if sha256_file(supervisor) != RISK_SUPERVISOR_SHA256:
        raise RuntimeError("RISK_SUPERVISOR_R1_SHA_MISMATCH")

    receipt = {
        "schema": "CB16_R10_2_PARENT_ADOPTION_RECEIPT_V1",
        "status": "R10_1_PARENT_AND_FROZEN_AUTHORITIES_ADOPTED",
        "scientific_semantics_changed": False,
        "parent_r101_root": str(parent),
        "parent_prelearning_receipt_sha256": sha256_file(parent_gate_path),
        "parent_prelearning_status": parent_gate.get("status"),
        "parent_authority_resolution": parent_gate.get("authority_resolution"),
        "prefixes": adopted,
        "g0": {
            "source": str(g0), "local": str(local_g0), "adoption_mode": mode,
            "file_sha256": g0_file_sha, "tensor_semantic_sha256": semantic,
            "reinitialized": False, "tensor_values_changed": False,
        },
        "physics_contract_sha256": PHYSICS_CONTRACT_SHA256,
        "risk_supervisor_sha256": RISK_SUPERVISOR_SHA256,
    }
    if receipt_path is not None:
        atomic_write_json(receipt_path, receipt)
    return receipt
