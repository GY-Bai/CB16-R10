from __future__ import annotations

"""Pascal CUDA binary-compatibility and CB16 workload canaries.

R8.1 corrects an overly strict R8 assumption: a GTX1060 (compute capability 6.1)
does not require an *exact* ``sm_61`` entry in ``torch.cuda.get_arch_list()`` when the
wheel contains a compatible same-major lower-minor cubin such as ``sm_60``.

NVIDIA guarantees cubin binary compatibility from compute capability X.y to X.z when
z >= y within the same major revision.  For the Shanxi GTX1060 6.1, ``sm_60`` is
therefore binary-compatible.  R8.1 does not trust the architecture string alone: it
also executes the actual CB16 GPU operator surface (Encoder + Trader + AdamW backward)
and requires numerical agreement with the CPU reference within explicit tolerances.
"""

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def parse_sm_arch(arch: str) -> tuple[int, int] | None:
    if not isinstance(arch, str) or not arch.startswith("sm_"):
        return None
    digits = arch[3:]
    if not digits.isdigit() or len(digits) < 2:
        return None
    value = int(digits)
    return value // 10, value % 10


def compatible_cubin_arches(
    device_cc: tuple[int, int],
    compiled_arches: Sequence[str],
) -> tuple[str, ...]:
    """Return cubin arches guaranteed compatible with ``device_cc``.

    Generic NVIDIA rule: code X.y can run on device X.z when z >= y.  The Shanxi
    qualification targets discrete-GPU Pascal 6.1.  We explicitly reject the Tegra
    Pascal 6.2 cross-family special case rather than generalizing beyond the target.
    """
    dmaj, dmin = map(int, device_cc)
    out = []
    for arch in compiled_arches:
        parsed = parse_sm_arch(arch)
        if parsed is None:
            continue
        cmaj, cmin = parsed
        if (dmaj, dmin) == (6, 2) and (cmaj, cmin) != (6, 2):
            continue
        if cmaj == dmaj and cmin <= dmin:
            out.append(arch)
    return tuple(sorted(set(out), key=lambda a: parse_sm_arch(a) or (-1, -1)))


@dataclass(frozen=True)
class CudaBinaryCompatibilityReceiptR81:
    device_compute_capability: tuple[int, int] | None
    compiled_arches: tuple[str, ...]
    compatible_cubin_arches: tuple[str, ...]
    selected_cubin_arch: str | None
    native_arch_present: bool
    compatibility_mode: str
    status: str
    reason: str | None

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def assess_cuda_binary_compatibility_r81(
    *,
    device_cc: tuple[int, int] | None,
    compiled_arches: Sequence[str],
    expected_cc: tuple[int, int] = (6, 1),
) -> CudaBinaryCompatibilityReceiptR81:
    arches = tuple(compiled_arches)
    if device_cc is None:
        return CudaBinaryCompatibilityReceiptR81(
            None, arches, (), None, False, "NONE", "FAIL", "DEVICE_CC_UNAVAILABLE"
        )
    cc = tuple(int(x) for x in device_cc)
    if cc != tuple(expected_cc):
        return CudaBinaryCompatibilityReceiptR81(
            cc, arches, (), None, False, "NONE", "FAIL",
            f"UNEXPECTED_DEVICE_CC:{cc}",
        )
    compat = compatible_cubin_arches(cc, arches)
    native = f"sm_{cc[0]}{cc[1]}" in arches
    if not compat:
        return CudaBinaryCompatibilityReceiptR81(
            cc, arches, (), None, native, "NONE", "FAIL",
            "NO_BINARY_COMPATIBLE_CUBIN_FOR_DEVICE",
        )
    selected = max(compat, key=lambda a: parse_sm_arch(a) or (-1, -1))
    mode = "NATIVE_CUBIN" if native else "SAME_MAJOR_FORWARD_MINOR_BINARY_COMPATIBILITY"
    return CudaBinaryCompatibilityReceiptR81(
        cc, arches, compat, selected, native, mode, "PASS", None
    )


@dataclass(frozen=True)
class Cb16CudaWorkloadCanaryR81:
    status: str
    device_name: str | None
    compute_capability: tuple[int, int] | None
    torch_version: str | None
    torch_cuda_version: str | None
    compiled_arches: tuple[str, ...]
    binary_compatibility_hash: str | None
    encoder_forward_max_abs_error: float | None
    trader_direction_logits_max_abs_error: float | None
    trader_risk_max_abs_error: float | None
    training_loss_cpu: float | None
    training_loss_cuda: float | None
    training_loss_abs_error: float | None
    minimum_nonzero_grad_norm: float | None
    parameter_delta_l2: float | None
    rollout_rows: int
    training_rows: int
    peak_allocated_bytes: int
    error: str | None

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def _state_l2_delta(before: dict[str, Any], after: dict[str, Any]) -> float:
    import torch
    total = 0.0
    for key in before:
        a = before[key].detach().double().cpu()
        b = after[key].detach().double().cpu()
        total += float(torch.sum((b - a) ** 2).item())
    return float(total ** 0.5)


def cb16_cuda_workload_canary_r81(
    *,
    rollout_rows: int = 4096,
    training_rows: int = 1024,
) -> Cb16CudaWorkloadCanaryR81:
    """Exercise the GPU operations actually used by the current CB16 runtime.

    Coverage includes LayerNorm, Linear, SiLU, Embedding, concat, softmax, sigmoid,
    index/select-like direction conditioning, probabilistic training loss, backward,
    gradient clipping and AdamW.  Physics/Teacher remain CPU-side and are intentionally
    outside this CUDA canary.
    """
    try:
        import torch
        import torch.nn.functional as F
        from .market_encoder_r5 import ReferenceGrammarEncoderR5
        from .trader_capacity_ladder import build_trader

        if not torch.cuda.is_available():
            return Cb16CudaWorkloadCanaryR81(
                status="FAIL",
                device_name=None,
                compute_capability=None,
                torch_version=getattr(torch, "__version__", None),
                torch_cuda_version=getattr(torch.version, "cuda", None),
                compiled_arches=tuple(torch.cuda.get_arch_list()),
                binary_compatibility_hash=None,
                encoder_forward_max_abs_error=None,
                trader_direction_logits_max_abs_error=None,
                trader_risk_max_abs_error=None,
                training_loss_cpu=None,
                training_loss_cuda=None,
                training_loss_abs_error=None,
                minimum_nonzero_grad_norm=None,
                parameter_delta_l2=None,
                rollout_rows=rollout_rows,
                training_rows=training_rows,
                peak_allocated_bytes=0,
                error="TORCH_CUDA_NOT_AVAILABLE",
            )

        dev = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        cc = tuple(int(x) for x in torch.cuda.get_device_capability(0))
        arches = tuple(torch.cuda.get_arch_list())
        compat = assess_cuda_binary_compatibility_r81(
            device_cc=cc, compiled_arches=arches, expected_cc=(6, 1)
        )
        if compat.status != "PASS":
            return Cb16CudaWorkloadCanaryR81(
                status="FAIL",
                device_name=name,
                compute_capability=cc,
                torch_version=torch.__version__,
                torch_cuda_version=torch.version.cuda,
                compiled_arches=arches,
                binary_compatibility_hash=compat.content_hash,
                encoder_forward_max_abs_error=None,
                trader_direction_logits_max_abs_error=None,
                trader_risk_max_abs_error=None,
                training_loss_cpu=None,
                training_loss_cuda=None,
                training_loss_abs_error=None,
                minimum_nonzero_grad_norm=None,
                parameter_delta_l2=None,
                rollout_rows=rollout_rows,
                training_rows=training_rows,
                peak_allocated_bytes=0,
                error=compat.reason,
            )

        torch.manual_seed(20260904)
        torch.cuda.manual_seed_all(20260904)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # ----- Encoder CPU vs CUDA -----
        enc_cpu = ReferenceGrammarEncoderR5().cpu().eval()
        enc_gpu = ReferenceGrammarEncoderR5().to(dev).eval()
        enc_gpu.load_state_dict(enc_cpu.state_dict(), strict=True)
        windows = torch.randn(256, 32, 5, dtype=torch.float32)
        with torch.inference_mode():
            z_cpu = enc_cpu(windows)
            z_gpu = enc_gpu(windows.to(dev)).cpu()
        enc_err = float((z_cpu - z_gpu).abs().max().item())

        # ----- Trader rollout CPU vs CUDA -----
        trader_cpu = build_trader("TIER_1").cpu().eval()
        trader_gpu = build_trader("TIER_1").to(dev).eval()
        trader_gpu.load_state_dict(trader_cpu.state_dict(), strict=True)
        g = torch.Generator(device="cpu").manual_seed(20260905)
        market = torch.randn(rollout_rows, 64, generator=g, dtype=torch.float32)
        account = torch.randn(rollout_rows, 6, generator=g, dtype=torch.float32)
        with torch.inference_mode():
            oc = trader_cpu(market, account)
            og = trader_gpu(market.to(dev), account.to(dev))
            logits_gpu = og["direction_logits"].cpu()
            risk_gpu = og["requested_risk_raw"].cpu()
        logit_err = float((oc["direction_logits"] - logits_gpu).abs().max().item())
        risk_err = float((oc["requested_risk_raw"] - risk_gpu).abs().max().item())

        # ----- Actual training operator surface CPU vs CUDA -----
        train_cpu = build_trader("TIER_1").cpu().train()
        train_gpu = build_trader("TIER_1").to(dev).train()
        train_gpu.load_state_dict(train_cpu.state_dict(), strict=True)
        before = {k: v.detach().clone() for k, v in train_gpu.state_dict().items()}
        opt_cpu = torch.optim.AdamW(train_cpu.parameters(), lr=3e-4, weight_decay=1e-4)
        opt_gpu = torch.optim.AdamW(train_gpu.parameters(), lr=3e-4, weight_decay=1e-4)

        g2 = torch.Generator(device="cpu").manual_seed(20260906)
        m = torch.randn(training_rows, 64, generator=g2, dtype=torch.float32)
        a = torch.randn(training_rows, 6, generator=g2, dtype=torch.float32)
        target_logits = torch.randn(training_rows, 3, generator=g2, dtype=torch.float32)
        target_probs = torch.softmax(target_logits, dim=-1)
        target_risk = torch.sigmoid(
            torch.randn(training_rows, generator=g2, dtype=torch.float32)
        )
        dweight = 0.5 + torch.rand(training_rows, generator=g2, dtype=torch.float32)
        sweight = 0.5 + torch.rand(training_rows, generator=g2, dtype=torch.float32)

        def loss_fn(model, mm, aa, tp, tr, dw, sw):
            out = model(mm, aa)
            logp = torch.log_softmax(out["direction_logits"], dim=-1)
            dir_loss = -(tp * logp).sum(dim=-1)
            size_loss = (out["requested_risk_raw"] - tr) ** 2
            return (dir_loss * dw).sum() / dw.sum() + (size_loss * sw).sum() / sw.sum()

        opt_cpu.zero_grad(set_to_none=True)
        lc = loss_fn(train_cpu, m, a, target_probs, target_risk, dweight, sweight)
        lc.backward()
        torch.nn.utils.clip_grad_norm_(train_cpu.parameters(), 10.0)
        opt_cpu.step()

        mg, ag = m.to(dev), a.to(dev)
        tpg, trg = target_probs.to(dev), target_risk.to(dev)
        dwg, swg = dweight.to(dev), sweight.to(dev)
        opt_gpu.zero_grad(set_to_none=True)
        lg = loss_fn(train_gpu, mg, ag, tpg, trg, dwg, swg)
        lg.backward()
        grad_norms = [
            float(p.grad.detach().float().norm().item())
            for p in train_gpu.parameters()
            if p.grad is not None and torch.isfinite(p.grad).all()
        ]
        torch.nn.utils.clip_grad_norm_(train_gpu.parameters(), 10.0)
        opt_gpu.step()
        torch.cuda.synchronize()

        after = train_gpu.state_dict()
        param_delta = _state_l2_delta(before, after)
        loss_cpu = float(lc.detach().item())
        loss_cuda = float(lg.detach().cpu().item())
        loss_err = abs(loss_cpu - loss_cuda)
        min_grad = min((x for x in grad_norms if x > 0), default=0.0)
        peak = int(torch.cuda.max_memory_allocated())

        # Conservative numeric gates: the GPU is not required to be bit-identical to CPU,
        # but the current small FP32 graph should be close and must update parameters.
        failures = []
        if enc_err > 5e-4:
            failures.append(f"ENCODER_NUMERIC_MISMATCH:{enc_err}")
        if logit_err > 5e-4:
            failures.append(f"TRADER_LOGIT_NUMERIC_MISMATCH:{logit_err}")
        if risk_err > 5e-5:
            failures.append(f"TRADER_RISK_NUMERIC_MISMATCH:{risk_err}")
        if loss_err > 2e-3:
            failures.append(f"TRAINING_LOSS_NUMERIC_MISMATCH:{loss_err}")
        if min_grad <= 0:
            failures.append("NO_NONZERO_FINITE_GRADIENT")
        if param_delta <= 0:
            failures.append("ADAMW_PARAMETER_DELTA_ZERO")

        return Cb16CudaWorkloadCanaryR81(
            status="FAIL" if failures else "PASS",
            device_name=name,
            compute_capability=cc,
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda,
            compiled_arches=arches,
            binary_compatibility_hash=compat.content_hash,
            encoder_forward_max_abs_error=enc_err,
            trader_direction_logits_max_abs_error=logit_err,
            trader_risk_max_abs_error=risk_err,
            training_loss_cpu=loss_cpu,
            training_loss_cuda=loss_cuda,
            training_loss_abs_error=loss_err,
            minimum_nonzero_grad_norm=float(min_grad),
            parameter_delta_l2=float(param_delta),
            rollout_rows=rollout_rows,
            training_rows=training_rows,
            peak_allocated_bytes=peak,
            error=";".join(failures) if failures else None,
        )
    except Exception as exc:
        try:
            import torch
            arches = tuple(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else ()
            cc = tuple(int(x) for x in torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
            name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            tv = torch.__version__
            cv = torch.version.cuda
            peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        except Exception:
            arches, cc, name, tv, cv, peak = (), None, None, None, None, 0
        return Cb16CudaWorkloadCanaryR81(
            status="FAIL",
            device_name=name,
            compute_capability=cc,
            torch_version=tv,
            torch_cuda_version=cv,
            compiled_arches=arches,
            binary_compatibility_hash=None,
            encoder_forward_max_abs_error=None,
            trader_direction_logits_max_abs_error=None,
            trader_risk_max_abs_error=None,
            training_loss_cpu=None,
            training_loss_cuda=None,
            training_loss_abs_error=None,
            minimum_nonzero_grad_norm=None,
            parameter_delta_l2=None,
            rollout_rows=rollout_rows,
            training_rows=training_rows,
            peak_allocated_bytes=peak,
            error=repr(exc),
        )
