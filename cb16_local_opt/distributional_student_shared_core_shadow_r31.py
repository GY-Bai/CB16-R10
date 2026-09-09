from __future__ import annotations

"""R11 Science G0 R3.1: disposable Shared Decision Core distributional shadow.

R3 qualified a distributional belief head over a *detached* production shared
representation. R3.1 remains shadow-only but asks the next architectural
question: can distributional Teacher evidence legally reshape a disposable copy
of the Shared Decision Core, and what happens to the existing Direction/Risk
objective if it does?

Only a disposable Shared Decision Core and a disposable distributional head are
gradient owners here. Production Student, three Brain stems, Direction Head,
Requested-Risk Head, Teacher, future outcomes, Supervisor and Physics remain
outside the R3.1 gradient graph.
"""

import copy
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


R31_RUNTIME = "CB16_R11_DISTRIBUTIONAL_STUDENT_SHARED_CORE_SHADOW_R3_1_V1"
R31_SHARED_OWNER = "Disposable Shared Decision Core"
R31_HEAD_OWNER = "Distributional Student Shadow Head"
R31_AUTHORIZED_OWNERS = frozenset({R31_SHARED_OWNER, R31_HEAD_OWNER})
R31_HIDDEN_DIM = 64
R31_INIT_GAP_BIAS = -6.0


class DistributionalStudentSharedCoreHeadR31(nn.Module):
    """Monotone distributional head that intentionally permits input gradient."""

    def __init__(
        self,
        *,
        shared_dim: int,
        action_count: int,
        quantile_count: int,
        hidden_dim: int = R31_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if shared_dim <= 0 or action_count <= 0 or quantile_count < 2 or hidden_dim <= 0:
            raise ValueError("R11_R31_INVALID_SHADOW_HEAD_DIMENSION")
        self.shared_dim = int(shared_dim)
        self.action_count = int(action_count)
        self.quantile_count = int(quantile_count)
        self.hidden_dim = int(hidden_dim)
        self.body = nn.Sequential(nn.Linear(self.shared_dim, self.hidden_dim), nn.SiLU())
        self.out = nn.Linear(self.hidden_dim, self.action_count * self.quantile_count)
        self._reset_parameters_r31()

    def _reset_parameters_r31(self) -> None:
        with torch.no_grad():
            bias = self.out.bias.view(self.action_count, self.quantile_count)
            bias[:, 1:].fill_(R31_INIT_GAP_BIAS)

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        if shared.ndim != 2 or shared.shape[1] != self.shared_dim:
            raise RuntimeError("R11_R31_SHARED_REPRESENTATION_SHAPE_DRIFT")
        raw = self.out(self.body(shared)).view(
            shared.shape[0], self.action_count, self.quantile_count
        )
        base = raw[..., :1]
        positive_steps = F.softplus(raw[..., 1:])
        return torch.cat([base, base + torch.cumsum(positive_steps, dim=-1)], dim=-1)


def make_disposable_shared_core_student_r31(production_model: nn.Module) -> nn.Module:
    """Deep-copy production Student and authorize gradients only in shared_core."""
    shadow = copy.deepcopy(production_model)
    for _name, param in shadow.named_parameters():
        param.requires_grad_(False)
        param.grad = None
    if not hasattr(shadow, "shared_core"):
        raise RuntimeError("R11_R31_SHARED_CORE_MISSING")
    for param in shadow.shared_core.parameters():
        param.requires_grad_(True)
    shadow.train()
    return shadow


def shared_core_parameter_report_r31(model: nn.Module) -> dict[str, int | str]:
    total = 0
    trainable = 0
    bytes_fp32 = 0
    trainable_names: list[str] = []
    for name, p in model.named_parameters():
        total += int(p.numel())
        bytes_fp32 += int(p.numel() * p.element_size())
        if p.requires_grad:
            trainable += int(p.numel())
            trainable_names.append(name)
            if not name.startswith("shared_core"):
                raise RuntimeError(f"R11_R31_UNAUTHORIZED_TRAINABLE_DISPOSABLE_PARAMETER:{name}")
        if p.dtype != torch.float32:
            raise RuntimeError("R11_R31_NON_FP32_DISPOSABLE_STUDENT_PARAMETER")
    if not trainable_names:
        raise RuntimeError("R11_R31_SHARED_CORE_NOT_TRAINABLE")
    return {
        "model_parameter_count": total,
        "authorized_shared_core_parameter_count": trainable,
        "model_parameter_bytes": bytes_fp32,
        "authorized_owner": R31_SHARED_OWNER,
    }


def head_parameter_report_r31(head: nn.Module) -> dict[str, int | str]:
    count = sum(int(p.numel()) for p in head.parameters())
    trainable = sum(int(p.numel()) for p in head.parameters() if p.requires_grad)
    byte_count = sum(int(p.numel() * p.element_size()) for p in head.parameters())
    if count != trainable:
        raise RuntimeError("R11_R31_FROZEN_DISTRIBUTIONAL_HEAD_PARAMETER")
    if any(p.dtype != torch.float32 for p in head.parameters()):
        raise RuntimeError("R11_R31_NON_FP32_DISTRIBUTIONAL_HEAD_PARAMETER")
    return {
        "parameter_count": count,
        "trainable_parameter_count": trainable,
        "parameter_bytes": byte_count,
        "authorized_owner": R31_HEAD_OWNER,
    }


def forward_disposable_r31(model: nn.Module, batch: Any) -> Mapping[str, torch.Tensor]:
    return model(batch.operator48, batch.medium48, batch.account6)


def assert_r31_gradient_ownership(
    *,
    production_model: nn.Module,
    disposable_model: nn.Module,
    head: nn.Module,
) -> dict[str, Any]:
    production_grad_names = [name for name, p in production_model.named_parameters() if p.grad is not None]
    if production_grad_names:
        raise RuntimeError(f"R11_R31_PRODUCTION_STUDENT_RECEIVED_GRADIENT:{production_grad_names}")

    shared_nonzero: list[str] = []
    shared_missing: list[str] = []
    forbidden_grad_names: list[str] = []
    nonfinite: list[str] = []
    for name, p in disposable_model.named_parameters():
        if name.startswith("shared_core"):
            if p.grad is None:
                shared_missing.append(name)
            else:
                if not bool(torch.isfinite(p.grad).all().item()):
                    nonfinite.append(name)
                if bool(torch.any(p.grad != 0).item()):
                    shared_nonzero.append(name)
        elif p.grad is not None:
            forbidden_grad_names.append(name)
    if shared_missing:
        raise RuntimeError(f"R11_R31_SHARED_CORE_PARAMETER_WITHOUT_GRADIENT:{shared_missing}")
    if not shared_nonzero:
        raise RuntimeError("R11_R31_ZERO_SHARED_CORE_GRADIENT_PATH")
    if forbidden_grad_names:
        raise RuntimeError(f"R11_R31_FORBIDDEN_DISPOSABLE_GRADIENT:{forbidden_grad_names}")

    head_nonzero: list[str] = []
    head_missing: list[str] = []
    for name, p in head.named_parameters():
        if p.grad is None:
            head_missing.append(name)
            continue
        if not bool(torch.isfinite(p.grad).all().item()):
            nonfinite.append(f"head:{name}")
        if bool(torch.any(p.grad != 0).item()):
            head_nonzero.append(name)
    if head_missing:
        raise RuntimeError(f"R11_R31_HEAD_PARAMETER_WITHOUT_GRADIENT:{head_missing}")
    if not head_nonzero:
        raise RuntimeError("R11_R31_ZERO_HEAD_GRADIENT_PATH")
    if nonfinite:
        raise RuntimeError(f"R11_R31_NONFINITE_AUTHORIZED_GRADIENT:{nonfinite}")

    return {
        "authorized_gradient_owners": sorted(R31_AUTHORIZED_OWNERS),
        "production_student_gradient_parameter_count": 0,
        "shared_core_gradient_parameter_tensors": len(shared_nonzero),
        "distributional_head_gradient_parameter_tensors": len(head_nonzero),
        "forbidden_disposable_gradient_parameter_count": 0,
        "teacher_target_autograd": False,
    }


def state_update_audit_r31(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    if set(before) != set(after):
        raise RuntimeError("R11_R31_DISPOSABLE_STATE_KEY_DRIFT")
    shared_sq = 0.0
    forbidden_changed: list[str] = []
    for name in before:
        old = before[name].detach().cpu()
        new = after[name].detach().cpu()
        if name.startswith("shared_core"):
            d = new.double() - old.double()
            shared_sq += float(torch.sum(d * d))
        elif not torch.equal(old, new):
            forbidden_changed.append(name)
    if forbidden_changed:
        raise RuntimeError(f"R11_R31_FORBIDDEN_DISPOSABLE_PARAMETER_UPDATE:{forbidden_changed}")
    shared_delta = math.sqrt(shared_sq)
    if not math.isfinite(shared_delta) or shared_delta <= 0.0:
        raise RuntimeError("R11_R31_SHARED_CORE_DID_NOT_UPDATE")
    return {
        "shared_core_parameter_l2_delta": shared_delta,
        "forbidden_parameter_update_count": 0,
    }
