from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import torch
except Exception:  # pragma: no cover - hardware probe must survive broken torch
    torch = None


@dataclass(frozen=True)
class CPUProfile:
    model: str
    physical_cores: int
    logical_threads: int
    architecture: str
    avx2: bool
    l3_cache_mb: float | None = None


@dataclass(frozen=True)
class MemoryProfile:
    total_bytes: int
    available_bytes: int | None = None

    @property
    def total_gib(self) -> float:
        return self.total_bytes / (1024 ** 3)


@dataclass(frozen=True)
class GPUProfile:
    present: bool
    name: str | None = None
    compute_capability: tuple[int, int] | None = None
    vram_bytes: int | None = None
    torch_cuda_version: str | None = None
    torch_version: str | None = None
    compiled_arches: tuple[str, ...] = ()
    cuda_available: bool = False

    @property
    def vram_gib(self) -> float | None:
        return None if self.vram_bytes is None else self.vram_bytes / (1024 ** 3)

    @property
    def sm_tag(self) -> str | None:
        if self.compute_capability is None:
            return None
        return f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"


@dataclass(frozen=True)
class DiskProfile:
    label: str
    mount: str
    total_bytes: int
    free_bytes: int
    medium: str = "UNKNOWN"  # SSD / HDD / NVME / UNKNOWN


@dataclass(frozen=True)
class HardwareProfile:
    cpu: CPUProfile
    memory: MemoryProfile
    gpu: GPUProfile
    disks: tuple[DiskProfile, ...] = ()
    os: str = field(default_factory=platform.platform)
    hostname: str = field(default_factory=platform.node)


@dataclass(frozen=True)
class RuntimeRecommendation:
    train_dtype: str
    amp_default: bool
    trajectory_workers: int
    teacher_workers: int
    dataloader_workers: int
    cpu_account_batch: int
    rollout_batch_candidates: tuple[int, ...]
    train_batch_candidates: tuple[int, ...]
    max_resident_dataset_fraction: float
    notes: tuple[str, ...]


def _read_meminfo() -> MemoryProfile:
    total = None
    available = None
    p = Path("/proc/meminfo")
    if p.exists():
        kv = {}
        for line in p.read_text(errors="ignore").splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            m = re.search(r"(\d+)", v)
            if m:
                kv[k.strip()] = int(m.group(1)) * 1024
        total = kv.get("MemTotal")
        available = kv.get("MemAvailable")
    if total is None:
        # POSIX fallback.
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        total = int(page * pages)
    return MemoryProfile(total_bytes=int(total), available_bytes=available)


def _read_cpu_model_and_flags() -> tuple[str, bool]:
    model = platform.processor() or "UNKNOWN_CPU"
    avx2 = False
    p = Path("/proc/cpuinfo")
    if p.exists():
        txt = p.read_text(errors="ignore")
        for line in txt.splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
        avx2 = bool(re.search(r"\bavx2\b", txt))
    return model, avx2


def _physical_cores_linux() -> int:
    p = Path("/proc/cpuinfo")
    if not p.exists():
        return max(1, (os.cpu_count() or 1) // 2)
    pairs = set()
    physical = None
    core = None
    for line in p.read_text(errors="ignore").splitlines() + [""]:
        if not line.strip():
            if physical is not None and core is not None:
                pairs.add((physical, core))
            physical = core = None
            continue
        if line.startswith("physical id"):
            physical = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core = line.split(":", 1)[1].strip()
    return len(pairs) if pairs else max(1, (os.cpu_count() or 1) // 2)


def detect_gpu() -> GPUProfile:
    if torch is None:
        return GPUProfile(present=False, cuda_available=False)
    tv = getattr(torch, "__version__", None)
    cv = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False
    if not available:
        return GPUProfile(
            present=False,
            torch_cuda_version=cv,
            torch_version=tv,
            cuda_available=False,
        )
    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        arches = tuple(torch.cuda.get_arch_list())
        cc = tuple(torch.cuda.get_device_capability(idx))
        return GPUProfile(
            present=True,
            name=str(props.name),
            compute_capability=(int(cc[0]), int(cc[1])),
            vram_bytes=int(props.total_memory),
            torch_cuda_version=cv,
            torch_version=tv,
            compiled_arches=arches,
            cuda_available=True,
        )
    except Exception:
        return GPUProfile(
            present=True,
            torch_cuda_version=cv,
            torch_version=tv,
            cuda_available=available,
        )


def detect_disks(mounts: Iterable[tuple[str, str]] | None = None) -> tuple[DiskProfile, ...]:
    """Detect capacity.  Medium type is intentionally explicit/manual by default.

    mounts: iterable of (label, mount).  Example:
        [("SSD", "/data/ssd"), ("HDD", "/data/hdd")]
    """
    if mounts is None:
        mounts = [("ROOT", "/")]
    out: list[DiskProfile] = []
    for label, mount in mounts:
        try:
            du = shutil.disk_usage(mount)
        except OSError:
            continue
        medium = "UNKNOWN"
        upper = label.upper()
        if "NVME" in upper:
            medium = "NVME"
        elif "SSD" in upper:
            medium = "SSD"
        elif "HDD" in upper:
            medium = "HDD"
        out.append(DiskProfile(label, mount, du.total, du.free, medium))
    return tuple(out)


def detect_hardware(mounts: Iterable[tuple[str, str]] | None = None) -> HardwareProfile:
    model, avx2 = _read_cpu_model_and_flags()
    logical = int(os.cpu_count() or 1)
    physical = _physical_cores_linux() if platform.system() == "Linux" else max(1, logical // 2)
    cpu = CPUProfile(
        model=model,
        physical_cores=physical,
        logical_threads=logical,
        architecture=platform.machine(),
        avx2=avx2,
    )
    return HardwareProfile(
        cpu=cpu,
        memory=_read_meminfo(),
        gpu=detect_gpu(),
        disks=detect_disks(mounts),
    )


def shanxi_3700x_1060_profile(
    *,
    ram_gib: int = 16,
    ssd_mount: str = "/data/ssd",
    hdd_mount: str = "/data/hdd",
) -> HardwareProfile:
    """Explicit expected profile for the user's Shanxi machine.

    Disk capacities are filled from the live machine only when the paths exist.
    """
    disks = detect_disks([("SSD", ssd_mount), ("HDD", hdd_mount)])
    gpu = GPUProfile(
        present=True,
        name="NVIDIA GeForce GTX 1060 6GB",
        compute_capability=(6, 1),
        vram_bytes=6 * 1024**3,
        cuda_available=True,
    )
    return HardwareProfile(
        cpu=CPUProfile(
            model="AMD Ryzen 7 3700X",
            physical_cores=8,
            logical_threads=16,
            architecture="x86_64",
            avx2=True,
            l3_cache_mb=32.0,
        ),
        memory=MemoryProfile(total_bytes=ram_gib * 1024**3),
        gpu=gpu,
        disks=disks,
        os="EXPECTED_LOCAL_PROFILE",
        hostname="SHANXI_NODE",
    )


def recommend_runtime(profile: HardwareProfile) -> RuntimeRecommendation:
    ram = profile.memory.total_gib
    cores = profile.cpu.physical_cores
    gpu = profile.gpu

    # Conservative because 16 GiB is the limiting resource, not the 16 SMT threads.
    trajectory_workers = max(1, min(4, cores // 2))
    teacher_workers = max(1, min(2, max(1, cores // 4)))
    dataloader_workers = 2 if cores >= 6 and ram >= 12 else 1

    cpu_batch = 16384 if ram >= 16 else 8192
    rollout = (4096, 8192, 16384, 32768)
    train = (512, 1024, 2048, 4096, 8192)

    amp = False
    dtype = "fp32"
    notes: list[str] = [
        "Prefer vectorization inside each worker before increasing worker count.",
        "Keep full history off RAM; stream/mmap sharded datasets.",
        "Use SSD for active shards/snapshots/indexes and HDD for cold archives.",
    ]

    if gpu.present and gpu.compute_capability == (6, 1):
        notes.append("Pascal sm_61 detected/expected: FP32 is the default training path; AMP is benchmark-only.")
        notes.append("Verify the installed PyTorch wheel has a cubin binary-compatible with CC 6.1 and passes the representative CB16 CUDA workload canary.")
    if ram <= 16.5:
        notes.append("16 GiB RAM: cap prefetch/worker replication and keep swap as emergency-only, not steady-state.")
    return RuntimeRecommendation(
        train_dtype=dtype,
        amp_default=amp,
        trajectory_workers=trajectory_workers,
        teacher_workers=teacher_workers,
        dataloader_workers=dataloader_workers,
        cpu_account_batch=cpu_batch,
        rollout_batch_candidates=rollout,
        train_batch_candidates=train,
        max_resident_dataset_fraction=0.35,
        notes=tuple(notes),
    )


def validate_pascal_torch(profile: HardwareProfile, *, strict: bool = True) -> list[str]:
    """Return warnings or raise if an expected Pascal GPU cannot be executed by torch."""
    issues: list[str] = []
    gpu = profile.gpu
    if not gpu.present:
        issues.append("CUDA_GPU_NOT_VISIBLE")
    elif gpu.compute_capability == (6, 1):
        if gpu.compiled_arches and "sm_61" not in gpu.compiled_arches:
            issues.append("PYTORCH_WHEEL_MISSING_SM61")
        if not gpu.cuda_available:
            issues.append("TORCH_CUDA_NOT_AVAILABLE")
    if strict and issues:
        raise RuntimeError("HARDWARE_COMPATIBILITY_FAIL:" + ",".join(issues))
    return issues


def to_json(profile: HardwareProfile) -> str:
    return json.dumps(asdict(profile), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    p = detect_hardware()
    print(to_json(p))
    print(json.dumps(asdict(recommend_runtime(p)), indent=2, ensure_ascii=False))
