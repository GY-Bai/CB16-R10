from __future__ import annotations

"""Low-overhead CUDA helpers for the R11 canonical FP32 training path.

This module is runtime-only. It never changes model architecture, optimizer,
loss, batching semantics, seed/permutation semantics, or gradient authority.
The canonical hardware target is GTX 1060 / Pascal sm_61, where FP32 is the
only accelerated arithmetic path assumed by R11.
"""

import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


CUDA_GRAPH_DISABLED_FALLBACK = "CUDA_GRAPH_DISABLED_FALLBACK"
GTX1060_SM61_FP32_PATH = "GTX1060_SM61_FP32"
GENERIC_CUDA_FP32_PATH = "GENERIC_CUDA_FP32"
CPU_FP32_FALLBACK_PATH = "CPU_FP32_FALLBACK"


@dataclass(frozen=True)
class GPUExecutionProfileR11:
    device_type: str
    device_index: int | None
    device_name: str | None
    compute_capability: tuple[int, int] | None
    execution_path: str
    canonical_dtype: str = "torch.float32"
    amp_enabled: bool = False
    fp16_fast_path_assumed: bool = False
    bf16_enabled: bool = False
    tf32_enabled: bool = False
    tensor_core_assumed: bool = False
    triton_required: bool = False
    inductor_cuda_required: bool = False
    cuda_graph_api_available: bool = False

    @property
    def is_sm61(self) -> bool:
        return self.compute_capability == (6, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "compute_capability": (
                list(self.compute_capability) if self.compute_capability is not None else None
            ),
            "execution_path": self.execution_path,
            "canonical_dtype": self.canonical_dtype,
            "amp_enabled": self.amp_enabled,
            "fp16_fast_path_assumed": self.fp16_fast_path_assumed,
            "bf16_enabled": self.bf16_enabled,
            "tf32_enabled": self.tf32_enabled,
            "tensor_core_assumed": self.tensor_core_assumed,
            "triton_required": self.triton_required,
            "inductor_cuda_required": self.inductor_cuda_required,
            "cuda_graph_api_available": self.cuda_graph_api_available,
        }


def classify_cuda_capability_r11(
    capability: tuple[int, int] | None,
    *,
    device_name: str | None = None,
    device_index: int | None = 0,
    cuda_graph_api_available: bool = True,
) -> GPUExecutionProfileR11:
    """Pure classifier used by tests and by the live CUDA inspection path."""
    if capability is None:
        return GPUExecutionProfileR11(
            device_type="cpu",
            device_index=None,
            device_name=device_name,
            compute_capability=None,
            execution_path=CPU_FP32_FALLBACK_PATH,
            cuda_graph_api_available=False,
        )
    capability = (int(capability[0]), int(capability[1]))
    path = GTX1060_SM61_FP32_PATH if capability == (6, 1) else GENERIC_CUDA_FP32_PATH
    return GPUExecutionProfileR11(
        device_type="cuda",
        device_index=device_index,
        device_name=device_name,
        compute_capability=capability,
        execution_path=path,
        cuda_graph_api_available=bool(cuda_graph_api_available),
    )


def inspect_gpu_execution_profile_r11(device: str | torch.device) -> GPUExecutionProfileR11:
    dev = torch.device(device)
    if dev.type != "cuda":
        return classify_cuda_capability_r11(None, device_name=str(dev), device_index=None)
    if not torch.cuda.is_available():
        raise RuntimeError("R11_CUDA_REQUESTED_BUT_UNAVAILABLE")
    index = torch.cuda.current_device() if dev.index is None else int(dev.index)
    capability = tuple(int(x) for x in torch.cuda.get_device_capability(index))
    name = str(torch.cuda.get_device_name(index))
    graph_api = bool(hasattr(torch.cuda, "CUDAGraph") and hasattr(torch.cuda, "graph"))
    return classify_cuda_capability_r11(
        capability,
        device_name=name,
        device_index=index,
        cuda_graph_api_available=graph_api,
    )


@dataclass(frozen=True)
class H2DBenchmarkResultR11:
    sample_bytes: int
    pageable_h2d_ms: float | None
    pinned_h2d_ms: float | None
    pinned_stage_ms: float | None
    selected_pinned: bool
    selected_non_blocking: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_bytes": int(self.sample_bytes),
            "pageable_h2d_ms": self.pageable_h2d_ms,
            "pinned_h2d_ms": self.pinned_h2d_ms,
            "pinned_stage_ms": self.pinned_stage_ms,
            "selected_pinned": self.selected_pinned,
            "selected_non_blocking": self.selected_non_blocking,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class H2DTransferReceiptR11:
    strategy: str
    non_blocking: bool
    transfer_ms: float | None
    benchmark: H2DBenchmarkResultR11 | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "non_blocking": self.non_blocking,
            "transfer_ms": self.transfer_ms,
            "benchmark": None if self.benchmark is None else self.benchmark.as_dict(),
        }


def _median_cuda_copy_ms(
    source: torch.Tensor,
    destination: torch.Tensor,
    *,
    non_blocking: bool,
    repeats: int,
    warmups: int,
) -> float:
    for _ in range(max(0, int(warmups))):
        destination.copy_(source, non_blocking=non_blocking)
    torch.cuda.synchronize(destination.device)
    values: list[float] = []
    for _ in range(max(1, int(repeats))):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        destination.copy_(source, non_blocking=non_blocking)
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return float(statistics.median(values))


def benchmark_h2d_policy_r11(
    host: torch.Tensor,
    device: str | torch.device,
    *,
    repeats: int = 5,
    warmups: int = 2,
    min_total_improvement_fraction: float = 0.03,
    max_sample_bytes: int = 16 * 1024 * 1024,
) -> H2DBenchmarkResultR11:
    """Benchmark pageable vs pinned transfer without assuming pinning is a win.

    Selection includes the one-time host->pinned staging cost because R11 normally
    transfers each immutable campaign split only once. This makes the policy
    deliberately conservative for small GTX1060 workloads.
    """
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return H2DBenchmarkResultR11(
            sample_bytes=0,
            pageable_h2d_ms=None,
            pinned_h2d_ms=None,
            pinned_stage_ms=None,
            selected_pinned=False,
            selected_non_blocking=False,
            reason="CUDA_UNAVAILABLE_OR_NOT_REQUESTED",
        )
    if host.device.type != "cpu" or host.dtype != torch.float32 or not host.is_contiguous():
        raise RuntimeError("R11_H2D_BENCHMARK_REQUIRES_CONTIGUOUS_CPU_FP32")

    flat = host.reshape(-1)
    elem_size = max(1, int(host.element_size()))
    sample_elems = min(int(flat.numel()), max(1, int(max_sample_bytes) // elem_size))
    pageable = flat[:sample_elems].contiguous()
    destination = torch.empty_like(pageable, device=dev)

    pageable_ms = _median_cuda_copy_ms(
        pageable,
        destination,
        non_blocking=False,
        repeats=repeats,
        warmups=warmups,
    )
    t0 = time.perf_counter()
    pinned = pageable.pin_memory()
    pinned_stage_ms = (time.perf_counter() - t0) * 1000.0
    pinned_ms = _median_cuda_copy_ms(
        pinned,
        destination,
        non_blocking=True,
        repeats=repeats,
        warmups=warmups,
    )
    pinned_total = pinned_stage_ms + pinned_ms
    threshold = pageable_ms * (1.0 - float(min_total_improvement_fraction))
    selected = bool(pinned_total < threshold)
    reason = (
        "PINNED_TOTAL_BENCHMARK_WIN"
        if selected
        else "PAGEABLE_TOTAL_BENCHMARK_WIN_OR_TIE"
    )
    return H2DBenchmarkResultR11(
        sample_bytes=int(sample_elems * elem_size),
        pageable_h2d_ms=float(pageable_ms),
        pinned_h2d_ms=float(pinned_ms),
        pinned_stage_ms=float(pinned_stage_ms),
        selected_pinned=selected,
        selected_non_blocking=selected,
        reason=reason,
    )


def transfer_host_tensor_r11(
    host: torch.Tensor,
    device: str | torch.device,
    *,
    pin_memory: bool | None = None,
    benchmark_hook: Callable[[torch.Tensor, torch.device], H2DBenchmarkResultR11] | None = None,
) -> tuple[torch.Tensor, H2DTransferReceiptR11]:
    """Move one immutable FP32 host buffer to its campaign-resident device buffer.

    pin_memory=None selects by benchmark. Explicit True/False remains available
    for controlled qualification and differential tests.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        return host, H2DTransferReceiptR11(
            strategy="CPU_RESIDENT",
            non_blocking=False,
            transfer_ms=0.0,
            benchmark=None,
        )
    if not torch.cuda.is_available():
        raise RuntimeError("R11_CUDA_REQUESTED_BUT_UNAVAILABLE")
    if host.device.type != "cpu" or host.dtype != torch.float32 or not host.is_contiguous():
        raise RuntimeError("R11_H2D_TRANSFER_REQUIRES_CONTIGUOUS_CPU_FP32")

    benchmark: H2DBenchmarkResultR11 | None = None
    if pin_memory is None:
        hook = benchmark_hook or (lambda tensor, target: benchmark_h2d_policy_r11(tensor, target))
        benchmark = hook(host, dev)
        use_pinned = bool(benchmark.selected_pinned)
    else:
        use_pinned = bool(pin_memory)

    source = host.pin_memory() if use_pinned and not host.is_pinned() else host
    non_blocking = bool(use_pinned)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    packed = source.to(dev, dtype=torch.float32, non_blocking=non_blocking)
    end.record()
    end.synchronize()
    transfer_ms = float(start.elapsed_time(end))
    return packed, H2DTransferReceiptR11(
        strategy="PINNED_ASYNC_H2D" if use_pinned else "PAGEABLE_SYNC_H2D",
        non_blocking=non_blocking,
        transfer_ms=transfer_ms,
        benchmark=benchmark,
    )


def prepare_epoch_permutations_r11(
    *,
    rows: int,
    epochs: int,
    seed: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, int]:
    """Generate the exact CPU randperm sequence, then transfer the stack once."""
    if rows <= 0 or epochs <= 0:
        raise ValueError("R11_INVALID_PERMUTATION_SHAPE")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    cpu = torch.stack(
        [torch.randperm(int(rows), generator=generator) for _ in range(int(epochs))],
        dim=0,
    )
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("R11_CUDA_REQUESTED_BUT_UNAVAILABLE")
        return cpu.to(dev), 1
    return cpu, 0


@dataclass(frozen=True)
class TelemetryConfigR11:
    enabled: bool = False
    sample_every_steps: int = 32
    sample_nvidia_smi: bool = True

    def validate(self) -> None:
        if self.sample_every_steps <= 0:
            raise ValueError("R11_TELEMETRY_SAMPLE_INTERVAL_INVALID")


class StepPhaseTimerR11:
    _ORDER = (
        "start",
        "batch_ready",
        "forward_done",
        "backward_done",
        "clip_checks_done",
        "optimizer_done",
    )

    def __init__(self, device: torch.device):
        self.device = device
        self._cuda = device.type == "cuda" and torch.cuda.is_available()
        self._marks: dict[str, Any] = {}
        self.mark("start")

    def mark(self, name: str) -> None:
        if name not in self._ORDER:
            raise ValueError(f"R11_UNKNOWN_TELEMETRY_MARK:{name}")
        if self._cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._marks[name] = event
        else:
            self._marks[name] = time.perf_counter()

    def _delta(self, a: str, b: str) -> float | None:
        if a not in self._marks or b not in self._marks:
            return None
        if self._cuda:
            return float(self._marks[a].elapsed_time(self._marks[b]))
        return float((self._marks[b] - self._marks[a]) * 1000.0)

    def finish(self) -> dict[str, float | None]:
        if "optimizer_done" not in self._marks:
            self.mark("optimizer_done")
        if self._cuda:
            self._marks["optimizer_done"].synchronize()
        return {
            "batch_prepare_ms": self._delta("start", "batch_ready"),
            "forward_ms": self._delta("batch_ready", "forward_done"),
            "backward_ms": self._delta("forward_done", "backward_done"),
            "grad_clip_checks_ms": self._delta("backward_done", "clip_checks_done"),
            "optimizer_ms": self._delta("clip_checks_done", "optimizer_done"),
            "batch_latency_ms": self._delta("start", "optimizer_done"),
        }


def sample_nvidia_smi_r11(device_index: int | None) -> dict[str, float | None]:
    if shutil.which("nvidia-smi") is None:
        return {"gpu_busy_percent": None, "vram_used_mb": None, "vram_total_mb": None}
    index = 0 if device_index is None else int(device_index)
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=0.5,
        ).strip().splitlines()[0]
        busy, used, total = (float(x.strip()) for x in out.split(",")[:3])
        return {
            "gpu_busy_percent": busy,
            "vram_used_mb": used,
            "vram_total_mb": total,
        }
    except Exception:
        return {"gpu_busy_percent": None, "vram_used_mb": None, "vram_total_mb": None}


@dataclass
class RuntimeTelemetryR11:
    config: TelemetryConfigR11 = field(default_factory=TelemetryConfigR11)
    step_samples: list[dict[str, float | None]] = field(default_factory=list)
    device_samples: list[dict[str, float | None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.config.validate()

    def begin_step(self, step_index: int, device: torch.device) -> StepPhaseTimerR11 | None:
        if not self.config.enabled:
            return None
        if int(step_index) % int(self.config.sample_every_steps) != 0:
            return None
        return StepPhaseTimerR11(device)

    def record_step(self, timer: StepPhaseTimerR11 | None) -> None:
        if timer is not None:
            self.step_samples.append(timer.finish())

    def sample_device(self, device: torch.device) -> None:
        if not self.config.enabled or device.type != "cuda" or not torch.cuda.is_available():
            return
        sample: dict[str, float | None] = {
            "torch_memory_allocated_mb": float(torch.cuda.memory_allocated(device) / (1024**2)),
            "torch_memory_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024**2)),
            "torch_max_memory_allocated_mb": float(
                torch.cuda.max_memory_allocated(device) / (1024**2)
            ),
        }
        if self.config.sample_nvidia_smi:
            sample.update(sample_nvidia_smi_r11(device.index))
        else:
            sample.update(
                {"gpu_busy_percent": None, "vram_used_mb": None, "vram_total_mb": None}
            )
        self.device_samples.append(sample)

    @staticmethod
    def _median(samples: list[dict[str, float | None]], key: str) -> float | None:
        values = [float(x[key]) for x in samples if x.get(key) is not None]
        return float(statistics.median(values)) if values else None

    def snapshot(self) -> dict[str, Any]:
        latest = self.device_samples[-1] if self.device_samples else {}
        return {
            "enabled": bool(self.config.enabled),
            "sample_every_steps": int(self.config.sample_every_steps),
            "sampled_steps": int(len(self.step_samples)),
            "batch_latency_ms_median": self._median(self.step_samples, "batch_latency_ms"),
            "batch_prepare_ms_median": self._median(self.step_samples, "batch_prepare_ms"),
            "forward_ms_median": self._median(self.step_samples, "forward_ms"),
            "backward_ms_median": self._median(self.step_samples, "backward_ms"),
            "grad_clip_checks_ms_median": self._median(
                self.step_samples, "grad_clip_checks_ms"
            ),
            "optimizer_ms_median": self._median(self.step_samples, "optimizer_ms"),
            "gpu_busy_percent": latest.get("gpu_busy_percent"),
            "vram_used_mb": latest.get("vram_used_mb"),
            "vram_total_mb": latest.get("vram_total_mb"),
            "torch_memory_allocated_mb": latest.get("torch_memory_allocated_mb"),
            "torch_memory_reserved_mb": latest.get("torch_memory_reserved_mb"),
            "torch_max_memory_allocated_mb": latest.get("torch_max_memory_allocated_mb"),
        }
