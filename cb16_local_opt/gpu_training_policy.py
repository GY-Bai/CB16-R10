from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DataLoaderPolicy:
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass(frozen=True)
class TrainingPolicy:
    device: str = "cuda"
    dtype: str = "fp32"
    amp_enabled: bool = False
    gradient_clip_norm: float = 5.0
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    train_batch_candidates: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
    rollout_batch_candidates: tuple[int, ...] = (4096, 8192, 16384, 32768)
    cpu_torch_threads: int = 2

    def apply_cpu_thread_policy(self) -> None:
        # Prevent BLAS/OpenMP oversubscription on an 8C/16T host.
        os.environ.setdefault("OMP_NUM_THREADS", str(self.cpu_torch_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.cpu_torch_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(self.cpu_torch_threads))
        torch.set_num_threads(max(1, self.cpu_torch_threads))

    @staticmethod
    def for_gtx1060() -> "TrainingPolicy":
        return TrainingPolicy(
            device="cuda",
            dtype="fp32",
            amp_enabled=False,
            cpu_torch_threads=2,
        )


@dataclass
class ProbabilisticTrainingBatch:
    market_latent: torch.Tensor              # [B,64]
    account_state: torch.Tensor              # [B,6]
    direction_target_probs: torch.Tensor     # [B,3], probabilistic evidence
    requested_risk_target: torch.Tensor      # [B]
    direction_weight: torch.Tensor           # [B]
    sizing_weight: torch.Tensor              # [B]
    admitted: torch.Tensor                   # bool [B]

    def pin_memory(self) -> "ProbabilisticTrainingBatch":
        """Pin all CPU tensors for faster non-blocking host->CUDA transfer."""
        return ProbabilisticTrainingBatch(
            market_latent=self.market_latent.pin_memory(),
            account_state=self.account_state.pin_memory(),
            direction_target_probs=self.direction_target_probs.pin_memory(),
            requested_risk_target=self.requested_risk_target.pin_memory(),
            direction_weight=self.direction_weight.pin_memory(),
            sizing_weight=self.sizing_weight.pin_memory(),
            admitted=self.admitted.pin_memory(),
        )

    def validate(self) -> None:
        b = self.market_latent.shape[0]
        if self.account_state.shape[0] != b:
            raise ValueError("batch length mismatch")
        if self.direction_target_probs.shape != (b, 3):
            raise ValueError("direction_target_probs must be [B,3]")
        if self.requested_risk_target.shape != (b,):
            raise ValueError("requested_risk_target must be [B]")
        if self.direction_weight.shape != (b,) or self.sizing_weight.shape != (b,):
            raise ValueError("weight shape mismatch")
        if self.admitted.shape != (b,) or self.admitted.dtype != torch.bool:
            raise ValueError("admitted must be bool [B]")
        if torch.any(self.direction_target_probs < -1e-7):
            raise ValueError("negative target probability")
        sums = self.direction_target_probs.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=0):
            raise ValueError("direction target distributions must sum to one")
        if torch.any((self.requested_risk_target < 0) | (self.requested_risk_target > 1)):
            raise ValueError("risk target out of range")
        if torch.any(self.direction_weight < 0) or torch.any(self.sizing_weight < 0):
            raise ValueError("negative weights")


@dataclass(frozen=True)
class StepStats:
    loss: float
    direction_loss: float
    sizing_loss: float
    admitted_rows: int
    grad_norm: float
    batch_size: int
    elapsed_ms: float


def build_optimizer(model: nn.Module, policy: TrainingPolicy):
    if policy.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=policy.lr, weight_decay=policy.weight_decay)
    if policy.optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=policy.lr, momentum=0.9, weight_decay=policy.weight_decay)
    raise ValueError(f"unsupported optimizer {policy.optimizer}")


def probabilistic_student_loss(
    outputs: dict[str, torch.Tensor],
    batch: ProbabilisticTrainingBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference Student loss over already-admitted probabilistic evidence.

    No realized-outcome BEST_ACTION/CORRECT_DIRECTION labels are created here.
    """
    batch.validate()
    admitted = batch.admitted.to(outputs["direction_logits"].device)
    if admitted.sum().item() == 0:
        zero = outputs["direction_logits"].sum() * 0.0
        return zero, zero, zero

    target_p = batch.direction_target_probs.to(outputs["direction_logits"].device, non_blocking=True)
    dw = batch.direction_weight.to(outputs["direction_logits"].device, non_blocking=True)
    sw = batch.sizing_weight.to(outputs["direction_logits"].device, non_blocking=True)
    risk_t = batch.requested_risk_target.to(outputs["direction_logits"].device, non_blocking=True)

    mask = admitted.float()
    logp = F.log_softmax(outputs["direction_logits"], dim=-1)
    row_dir = -(target_p * logp).sum(dim=-1)
    dir_den = torch.clamp((dw * mask).sum(), min=1e-12)
    direction_loss = (row_dir * dw * mask).sum() / dir_den

    row_size = F.smooth_l1_loss(outputs["requested_risk_raw"], risk_t, reduction="none")
    size_den = torch.clamp((sw * mask).sum(), min=1e-12)
    sizing_loss = (row_size * sw * mask).sum() / size_den
    total = direction_loss + sizing_loss
    return total, direction_loss, sizing_loss


def train_step(
    *,
    model: nn.Module,
    optimizer,
    batch: ProbabilisticTrainingBatch,
    policy: TrainingPolicy,
) -> StepStats:
    policy.apply_cpu_thread_policy()
    device = torch.device(policy.device if policy.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    start = time.perf_counter()
    admitted_rows = int(batch.admitted.sum().item())
    if admitted_rows == 0:
        # Hard invariant: no admitted evidence => no optimizer step, including no
        # decoupled AdamW weight decay.
        elapsed = (time.perf_counter() - start) * 1000
        return StepStats(
            loss=0.0,
            direction_loss=0.0,
            sizing_loss=0.0,
            admitted_rows=0,
            grad_norm=0.0,
            batch_size=int(batch.market_latent.shape[0]),
            elapsed_ms=elapsed,
        )
    market = batch.market_latent.to(device, non_blocking=True)
    account = batch.account_state.to(device, non_blocking=True)

    # GTX 1060/Pascal default: full FP32. AMP is an explicit benchmark-only override.
    amp_ok = bool(policy.amp_enabled and device.type == "cuda")
    with torch.autocast(device_type=device.type, enabled=amp_ok):
        outputs = model(market, account)
        total, dloss, sloss = probabilistic_student_loss(outputs, batch)

    total.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), policy.gradient_clip_norm))
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    return StepStats(
        loss=float(total.detach().cpu()),
        direction_loss=float(dloss.detach().cpu()),
        sizing_loss=float(sloss.detach().cpu()),
        admitted_rows=admitted_rows,
        grad_norm=grad_norm,
        batch_size=int(batch.market_latent.shape[0]),
        elapsed_ms=elapsed,
    )


@torch.no_grad()
def benchmark_rollout_batches(
    *,
    model: nn.Module,
    market_dim: int,
    account_dim: int,
    candidates: Iterable[int],
    device: str | None = None,
    warmup: int = 3,
    repeats: int = 10,
) -> list[dict]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()
    results = []
    for b in candidates:
        b = int(b)
        try:
            market = torch.randn(b, market_dim, device=dev)
            account = torch.randn(b, account_dim, device=dev)
            for _ in range(warmup):
                model(market, account)
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(repeats):
                model(market, account)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0
            results.append({
                "batch": b,
                "states_per_second": b * repeats / max(dt, 1e-12),
                "ms_per_batch": dt * 1000 / repeats,
                "peak_vram_bytes": int(peak),
                "status": "PASS",
            })
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                results.append({"batch": b, "status": "OOM"})
                continue
            raise
    return results


def choose_largest_stable_batch(results: list[dict]) -> int:
    passed = [r for r in results if r.get("status") == "PASS"]
    if not passed:
        raise RuntimeError("NO_STABLE_BATCH")
    # Prefer throughput, with batch size as a deterministic tiebreaker.
    best = max(passed, key=lambda r: (r["states_per_second"], r["batch"]))
    return int(best["batch"])
