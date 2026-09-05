from __future__ import annotations

"""R9 runtime authority binding.

R9 never re-benchmarks or reinterprets the Shanxi R8.1 qualification inside a
historical campaign.  Instead it binds the campaign to the already completed,
immutable hardware qualification receipts and to the exact local software
appliance identity.

The authority contract verifies:
- final R8.1 status is READY_WITH_LIMITS or READY_FOR_SHORT_REAL_CAMPAIGN;
- hard_failures is empty;
- bring-up/performance/burn-in receipt content hashes match the final profile;
- CUDA binary compatibility, numeric canary and representative CB16 CUDA canary passed;
- one-hour burn-in passed with no failures and no swap growth;
- runtime profile explicitly says scientific_semantics_changed == false;
- the selected profile is FP32 / AMP disabled / TIER_1 / single CUDA owner;
- optional live environment probe matches the frozen torch/CUDA/device contract.

This module does not change performance settings.  It only validates and emits a
campaign authority receipt.
"""

import dataclasses
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class RuntimeAuthorityPolicyR9:
    allowed_final_status: tuple[str, ...] = (
        "READY_WITH_LIMITS",
        "READY_FOR_SHORT_REAL_CAMPAIGN",
    )
    required_torch_version: str = "2.8.0+cu126"
    required_torch_cuda: str = "12.6"
    required_device_cc: tuple[int, int] = (6, 1)
    required_selected_cubin: str = "sm_60"
    require_fp32: bool = True
    require_amp_disabled: bool = True
    require_single_cuda_owner: bool = True
    require_tier: str = "TIER_1"
    max_burnin_gpu_temp_c: float = 82.0
    max_burnin_ram_fraction: float = 0.85
    max_allowed_swap_growth_bytes: int = 0

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class ShanxiRuntimeAuthorityR9:
    authority_version: str
    status: str
    qualification_final_status: str
    qualification_warning_codes: tuple[str, ...]
    source_file_sha256: dict[str, str]
    source_receipt_hashes: dict[str, str]
    runtime_profile_hash: str
    runtime_profile: dict[str, Any]
    torch_version: str
    torch_cuda_version: str
    gpu_name: str
    compute_capability: tuple[int, int]
    selected_cubin_arch: str
    binary_compatibility_mode: str
    burnin_duration_s: float
    burnin_max_ram_used_fraction: float
    burnin_max_gpu_temp_c: float
    burnin_swap_growth_bytes: int
    policy_hash: str
    scientific_semantics_changed: bool

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_runtime_authority_r9(
    qualification_dir: str | Path,
    *,
    policy: RuntimeAuthorityPolicyR9 | None = None,
    live_environment_check: bool = False,
) -> ShanxiRuntimeAuthorityR9:
    policy = policy or RuntimeAuthorityPolicyR9()
    qdir = Path(qualification_dir)
    paths = {
        "bringup": qdir / "01_BRINGUP_R8_1.json",
        "performance": qdir / "02_PERFORMANCE_R8_1.json",
        "burnin": qdir / "03_BURNIN_R8_1.json",
        "final": qdir / "FINAL_QUALIFICATION_R8_1.json",
    }
    bringup = _load(paths["bringup"])
    performance = _load(paths["performance"])
    burnin = _load(paths["burnin"])
    final = _load(paths["final"])

    failures: list[str] = []
    if final.get("status") not in policy.allowed_final_status:
        failures.append("FINAL_QUALIFICATION_STATUS_NOT_READY")
    if final.get("hard_failures"):
        failures.append("FINAL_QUALIFICATION_HAS_HARD_FAILURES")
    profile = final.get("runtime_profile") or {}
    if profile.get("scientific_semantics_changed") is not False:
        failures.append("RUNTIME_PROFILE_SCIENTIFIC_SEMANTICS_CHANGED")

    expected = profile.get("source_receipt_hashes") or {}
    for name, obj in (
        ("bringup", bringup),
        ("performance", performance),
        ("burnin", burnin),
    ):
        if not obj.get("content_hash"):
            failures.append(f"{name.upper()}_CONTENT_HASH_MISSING")
        elif expected.get(name) != obj.get("content_hash"):
            failures.append(f"{name.upper()}_RECEIPT_HASH_NOT_BOUND_BY_FINAL")

    if bringup.get("cuda_binary_compatibility", {}).get("status") != "PASS":
        failures.append("CUDA_BINARY_COMPATIBILITY_NOT_PASS")
    if bringup.get("cuda_canary", {}).get("status") != "PASS":
        failures.append("CUDA_NUMERIC_CANARY_NOT_PASS")
    if bringup.get("cb16_cuda_workload_canary", {}).get("status") != "PASS":
        failures.append("CB16_CUDA_WORKLOAD_CANARY_NOT_PASS")
    if performance.get("status") != "PASS":
        failures.append("PERFORMANCE_QUALIFICATION_NOT_PASS")
    if burnin.get("status") != "PASS":
        failures.append("BURNIN_NOT_PASS")
    if burnin.get("failures"):
        failures.append("BURNIN_HAS_FAILURES")

    gpu = bringup.get("hardware", {}).get("gpu", {})
    cc = tuple(gpu.get("compute_capability") or ())
    selected_cubin = bringup.get("cuda_binary_compatibility", {}).get(
        "selected_cubin_arch"
    )
    if gpu.get("torch_version") != policy.required_torch_version:
        failures.append("TORCH_VERSION_NOT_FROZEN_AUTHORITY")
    if gpu.get("torch_cuda_version") != policy.required_torch_cuda:
        failures.append("TORCH_CUDA_VERSION_NOT_FROZEN_AUTHORITY")
    if cc != policy.required_device_cc:
        failures.append("DEVICE_CC_NOT_FROZEN_AUTHORITY")
    if selected_cubin != policy.required_selected_cubin:
        failures.append("SELECTED_CUBIN_NOT_FROZEN_AUTHORITY")

    pgpu = profile.get("gpu") or {}
    if policy.require_fp32 and pgpu.get("dtype") != "fp32":
        failures.append("RUNTIME_PROFILE_NOT_FP32")
    if policy.require_amp_disabled and pgpu.get("amp_enabled") is not False:
        failures.append("RUNTIME_PROFILE_AMP_NOT_DISABLED")
    if policy.require_single_cuda_owner and pgpu.get("single_cuda_owner") is not True:
        failures.append("RUNTIME_PROFILE_NOT_SINGLE_CUDA_OWNER")
    if pgpu.get("tier") != policy.require_tier:
        failures.append("RUNTIME_PROFILE_TIER_MISMATCH")

    max_ram = float(burnin.get("max_ram_used_fraction", 1.0))
    max_temp = float(burnin.get("max_gpu_temp_c", 999.0))
    swap_growth = int(burnin.get("swap_growth_bytes", 1))
    if max_ram > policy.max_burnin_ram_fraction:
        failures.append("BURNIN_RAM_EXCEEDED_AUTHORITY_LIMIT")
    if max_temp > policy.max_burnin_gpu_temp_c:
        failures.append("BURNIN_GPU_TEMP_EXCEEDED_AUTHORITY_LIMIT")
    if swap_growth > policy.max_allowed_swap_growth_bytes:
        failures.append("BURNIN_SWAP_GROWTH_EXCEEDED_AUTHORITY_LIMIT")

    if live_environment_check:
        try:
            import torch

            live_cc = tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else ()
            if torch.__version__ != policy.required_torch_version:
                failures.append("LIVE_TORCH_VERSION_DRIFT")
            if torch.version.cuda != policy.required_torch_cuda:
                failures.append("LIVE_TORCH_CUDA_DRIFT")
            if live_cc != policy.required_device_cc:
                failures.append("LIVE_GPU_CC_DRIFT")
            if not torch.cuda.is_available():
                failures.append("LIVE_CUDA_NOT_AVAILABLE")
        except Exception:
            failures.append("LIVE_ENVIRONMENT_PROBE_ERROR")

    if failures:
        raise RuntimeError("R9_RUNTIME_AUTHORITY_FAIL:" + ",".join(sorted(set(failures))))

    authority = ShanxiRuntimeAuthorityR9(
        authority_version="CB16_SHANXI_RUNTIME_AUTHORITY_R9",
        status="SHANXI_RUNTIME_AUTHORITY_BOUND",
        qualification_final_status=str(final["status"]),
        qualification_warning_codes=tuple(final.get("warnings") or ()),
        source_file_sha256={k: sha256_file(v) for k, v in paths.items()},
        source_receipt_hashes={
            "bringup": bringup["content_hash"],
            "performance": performance["content_hash"],
            "burnin": burnin["content_hash"],
        },
        runtime_profile_hash=str(profile.get("content_hash")),
        runtime_profile=profile,
        torch_version=str(gpu.get("torch_version")),
        torch_cuda_version=str(gpu.get("torch_cuda_version")),
        gpu_name=str(gpu.get("name")),
        compute_capability=cc,
        selected_cubin_arch=str(selected_cubin),
        binary_compatibility_mode=str(
            bringup.get("cuda_binary_compatibility", {}).get("compatibility_mode")
        ),
        burnin_duration_s=float(burnin.get("duration_s", 0.0)),
        burnin_max_ram_used_fraction=max_ram,
        burnin_max_gpu_temp_c=max_temp,
        burnin_swap_growth_bytes=swap_growth,
        policy_hash=policy.content_hash,
        scientific_semantics_changed=False,
    )
    return authority


def save_runtime_authority_r9(authority: ShanxiRuntimeAuthorityR9, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(authority)
    payload["content_hash"] = authority.content_hash
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p


def load_saved_runtime_authority_r9(path: str | Path) -> ShanxiRuntimeAuthorityR9:
    obj = json.loads(Path(path).read_text())
    claimed = obj.pop("content_hash", None)
    obj["qualification_warning_codes"] = tuple(obj["qualification_warning_codes"])
    obj["compute_capability"] = tuple(obj["compute_capability"])
    r = ShanxiRuntimeAuthorityR9(**obj)
    if claimed and claimed != r.content_hash:
        raise RuntimeError("R9_RUNTIME_AUTHORITY_RECEIPT_HASH_MISMATCH")
    return r
