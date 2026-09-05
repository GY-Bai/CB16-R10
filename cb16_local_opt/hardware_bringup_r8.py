from __future__ import annotations

"""Authoritative local-host bring-up probe for the Shanxi 3700X/16GB/GTX1060 node.

The probe is intentionally fail-closed around CUDA execution.  Merely seeing the GPU in
`nvidia-smi` is not sufficient. R8.1 accepts NVIDIA-guaranteed same-major cubin binary
compatibility (for GTX1060 CC 6.1, an sm_60 cubin is compatible) but requires both a real
FP32 execution canary and the representative CB16 Encoder+Trader+AdamW workload canary.
"""

import dataclasses
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .hardware_profile import detect_hardware, recommend_runtime, validate_pascal_torch
from .pascal_cuda_compat_r81 import (
    Cb16CudaWorkloadCanaryR81,
    CudaBinaryCompatibilityReceiptR81,
    assess_cuda_binary_compatibility_r81,
    cb16_cuda_workload_canary_r81,
)


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _run(cmd: list[str], timeout: float = 10.0) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {"cmd": cmd, "rc": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"cmd": cmd, "rc": -1, "stdout": "", "stderr": repr(exc)}


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    p = Path("/proc/meminfo")
    if p.exists():
        for line in p.read_text(errors="ignore").splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                out[k] = int(v.strip().split()[0]) * 1024
            except Exception:
                pass
    return out


def _cpu_governors() -> tuple[str, ...]:
    vals = set()
    for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"):
        try:
            vals.add(p.read_text().strip())
        except Exception:
            pass
    return tuple(sorted(vals))


def _block_info(mount: str) -> dict[str, Any]:
    f = _run(["findmnt", "-no", "SOURCE,FSTYPE,OPTIONS", mount]) if shutil.which("findmnt") else None
    source = None
    if f and f["rc"] == 0 and f["stdout"]:
        source = f["stdout"].split()[0]
    ls = None
    if source and shutil.which("lsblk"):
        ls = _run(["lsblk", "-ndo", "NAME,TYPE,ROTA,TRAN,SIZE,MODEL", source])
    return {"mount": mount, "findmnt": f, "source": source, "lsblk": ls}


@dataclass(frozen=True)
class BringupPolicyR8:
    min_ram_gib: float = 14.0
    min_gpu_vram_gib: float = 5.5
    expected_compute_capability: tuple[int, int] = (6, 1)
    # Legacy R8 exact-string gate. Kept for schema compatibility but disabled by default
    # in R8.1 because NVIDIA guarantees sm_60 cubin execution on a CC 6.1 device.
    require_sm61_binary: bool = False
    require_binary_compatible_cubin: bool = True
    require_cb16_cuda_workload_canary: bool = True
    require_avx2: bool = True
    min_ssd_free_gib: float = 20.0
    min_hdd_free_gib: float = 20.0
    warn_swap_used_gib: float = 1.0

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class CudaExecutionCanaryR8:
    status: str
    device_name: str | None
    compute_capability: tuple[int, int] | None
    compiled_arches: tuple[str, ...]
    fp32_matmul_max_abs_error: float | None
    fp32_checksum: float | None
    peak_allocated_bytes: int
    error: str | None


@dataclass(frozen=True)
class BringupReceiptR8:
    schema: str
    authority: str
    status: str
    hard_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    hardware: Mapping[str, Any]
    recommendation: Mapping[str, Any]
    os_runtime: Mapping[str, Any]
    nvidia_smi: Mapping[str, Any]
    cpu_governors: tuple[str, ...]
    storage: Mapping[str, Any]
    cuda_canary: CudaExecutionCanaryR8
    cuda_binary_compatibility: CudaBinaryCompatibilityReceiptR81
    cb16_cuda_workload_canary: Cb16CudaWorkloadCanaryR81
    policy_hash: str
    created_at_unix: float

    @property
    def content_hash(self) -> str:
        d = asdict(self)
        d.pop("created_at_unix", None)
        return canonical_hash(d)


def cuda_execution_canary_r8() -> CudaExecutionCanaryR8:
    try:
        import torch
        if not torch.cuda.is_available():
            return CudaExecutionCanaryR8("FAIL", None, None, tuple(torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else ()), None, None, 0, "TORCH_CUDA_NOT_AVAILABLE")
        dev = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        cc = tuple(int(x) for x in torch.cuda.get_device_capability(0))
        arches = tuple(torch.cuda.get_arch_list())
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # deterministic-sized real FP32 CUDA kernel + CPU reference
        g = torch.Generator(device="cpu").manual_seed(12345)
        a_cpu = torch.randn(256, 256, generator=g, dtype=torch.float32)
        b_cpu = torch.randn(256, 256, generator=g, dtype=torch.float32)
        ref = a_cpu @ b_cpu
        got = a_cpu.to(dev) @ b_cpu.to(dev)
        torch.cuda.synchronize()
        got_cpu = got.cpu()
        err = float((got_cpu - ref).abs().max().item())
        checksum = float(got_cpu.double().sum().item())
        peak = int(torch.cuda.max_memory_allocated())
        if not (err < 1e-2):
            return CudaExecutionCanaryR8("FAIL", name, cc, arches, err, checksum, peak, "FP32_CUDA_NUMERIC_CANARY_FAIL")
        return CudaExecutionCanaryR8("PASS", name, cc, arches, err, checksum, peak, None)
    except Exception as exc:
        return CudaExecutionCanaryR8("FAIL", None, None, (), None, None, 0, repr(exc))


def collect_bringup_r8(*, ssd_mount: str, hdd_mount: str, policy: BringupPolicyR8 | None = None) -> BringupReceiptR8:
    policy = policy or BringupPolicyR8()
    hw = detect_hardware([("SSD", ssd_mount), ("HDD", hdd_mount)])
    rec = recommend_runtime(hw)
    cuda = cuda_execution_canary_r8()
    compatibility = assess_cuda_binary_compatibility_r81(
        device_cc=cuda.compute_capability or hw.gpu.compute_capability,
        compiled_arches=cuda.compiled_arches or hw.gpu.compiled_arches,
        expected_cc=policy.expected_compute_capability,
    )
    cb16_cuda = cb16_cuda_workload_canary_r81()
    mem = _meminfo()
    swap_total = mem.get("SwapTotal", 0)
    swap_free = mem.get("SwapFree", swap_total)
    swap_used = max(0, swap_total - swap_free)

    smi = _run([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit,utilization.gpu,pstate,clocks.sm,clocks.mem,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]) if shutil.which("nvidia-smi") else {"rc": -1, "stderr": "NVIDIA_SMI_NOT_FOUND", "stdout": "", "cmd": []}

    hard: list[str] = []
    warn: list[str] = []
    if policy.require_avx2 and not hw.cpu.avx2:
        hard.append("CPU_AVX2_REQUIRED")
    if hw.memory.total_gib < policy.min_ram_gib:
        hard.append(f"RAM_TOO_SMALL:{hw.memory.total_gib:.2f}GiB")
    if not hw.gpu.present:
        hard.append("CUDA_GPU_NOT_VISIBLE")
    if hw.gpu.compute_capability != policy.expected_compute_capability:
        hard.append(f"UNEXPECTED_GPU_COMPUTE_CAPABILITY:{hw.gpu.compute_capability}")
    if (hw.gpu.vram_gib or 0.0) < policy.min_gpu_vram_gib:
        hard.append("GPU_VRAM_TOO_SMALL")
    # Preserve legacy diagnostics but R8.1 does not treat exact sm_61 absence as a hard
    # failure when a compatible sm_60 cubin is present for the CC 6.1 device.
    issues = validate_pascal_torch(hw, strict=False)
    hard.extend(x for x in issues if x not in hard and x == "TORCH_CUDA_NOT_AVAILABLE")
    if policy.require_sm61_binary and "PYTORCH_WHEEL_MISSING_SM61" in issues:
        hard.append("PYTORCH_WHEEL_MISSING_SM61")
    if policy.require_binary_compatible_cubin and compatibility.status != "PASS":
        hard.append("PYTORCH_WHEEL_NO_BINARY_COMPATIBLE_PASCAL_CUBIN")
    if compatibility.status == "PASS" and not compatibility.native_arch_present:
        warn.append(
            "NATIVE_SM61_NOT_PRESENT_USING_" + str(compatibility.selected_cubin_arch).upper()
            + "_NVIDIA_BINARY_COMPATIBILITY"
        )
    if cuda.status != "PASS":
        hard.append("CUDA_FP32_EXECUTION_CANARY_FAIL")
    if cuda.compute_capability and tuple(cuda.compute_capability) != policy.expected_compute_capability:
        hard.append("CUDA_CANARY_WRONG_COMPUTE_CAPABILITY")
    if policy.require_sm61_binary and cuda.compiled_arches and "sm_61" not in cuda.compiled_arches:
        hard.append("CUDA_CANARY_SM61_NOT_COMPILED")
    if policy.require_cb16_cuda_workload_canary and cb16_cuda.status != "PASS":
        hard.append("CB16_CUDA_WORKLOAD_CANARY_FAIL")

    disk_by_label = {d.label.upper(): d for d in hw.disks}
    ssd = disk_by_label.get("SSD")
    hdd = disk_by_label.get("HDD")
    if ssd is None:
        hard.append("SSD_MOUNT_NOT_ACCESSIBLE")
    elif ssd.free_bytes / 1024**3 < policy.min_ssd_free_gib:
        hard.append("SSD_FREE_SPACE_BELOW_MINIMUM")
    if hdd is None:
        hard.append("HDD_MOUNT_NOT_ACCESSIBLE")
    elif hdd.free_bytes / 1024**3 < policy.min_hdd_free_gib:
        hard.append("HDD_FREE_SPACE_BELOW_MINIMUM")
    if swap_used / 1024**3 > policy.warn_swap_used_gib:
        warn.append(f"SWAP_ALREADY_USED:{swap_used/1024**3:.2f}GiB")
    governors = _cpu_governors()
    if governors and any(g not in {"performance", "schedutil"} for g in governors):
        warn.append("CPU_GOVERNOR_MAY_REDUCE_BENCHMARK_STABILITY:" + ",".join(governors))
    if "Ryzen 7 3700X" not in hw.cpu.model:
        warn.append("CPU_MODEL_DIFFERS_FROM_EXPECTED_3700X:" + hw.cpu.model)
    if hw.gpu.name and "GTX 1060" not in hw.gpu.name:
        warn.append("GPU_MODEL_DIFFERS_FROM_EXPECTED_GTX1060:" + hw.gpu.name)
    if smi.get("rc") != 0:
        warn.append("NVIDIA_SMI_QUERY_FAILED")

    status = "FAIL" if hard else ("PASS_WITH_WARNINGS" if warn else "PASS")
    authority = "SHANXI_HARDWARE_AUTHORITATIVE" if not hard and cuda.status == "PASS" else "DIAGNOSTIC_ONLY"
    return BringupReceiptR8(
        schema="CB16_SHANXI_HARDWARE_BRINGUP_R8_1",
        authority=authority,
        status=status,
        hard_failures=tuple(hard),
        warnings=tuple(warn),
        hardware=asdict(hw),
        recommendation=asdict(rec),
        os_runtime={
            "platform": platform.platform(), "kernel": platform.release(), "python": platform.python_version(),
            "hostname": platform.node(), "swap_total_bytes": swap_total, "swap_used_bytes": swap_used,
        },
        nvidia_smi=smi,
        cpu_governors=governors,
        storage={"ssd": _block_info(ssd_mount), "hdd": _block_info(hdd_mount)},
        cuda_canary=cuda,
        cuda_binary_compatibility=compatibility,
        cb16_cuda_workload_canary=cb16_cuda,
        policy_hash=policy.content_hash,
        created_at_unix=time.time(),
    )


def write_bringup_receipt_r8(receipt: BringupReceiptR8, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(receipt); payload["content_hash"] = receipt.content_hash
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path
