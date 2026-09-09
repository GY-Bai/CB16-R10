from __future__ import annotations

"""R11 Science G0 R3.4 — shadow distributional-belief to decision bridge.

The production Student and its Shared Decision Core remain frozen.  A trained
shadow distributional belief is detached, then consumed by a zero-residual
bridge that can only perturb disposable Direction/Requested-Risk outputs.
"""

import copy
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn


R34_RUNTIME = "CB16_R11_DISTRIBUTIONAL_BELIEF_DECISION_BRIDGE_SHADOW_R3_4_V1"
R34_OWNER = "Disposable Belief-to-Decision Residual Bridge"
R34_HIDDEN_DIM = 64
R34_STEPS = 32
R34_LR = 2e-3
R34_WEIGHT_DECAY = 0.0
R34_GRAD_CLIP = 10.0
R34_SEEDS = (31_337, 271_828, 314_159, 161_803, 424_242)
R34_RISK_EPS = 1e-5


class DistributionalBeliefDecisionBridgeR34(nn.Module):
    """Small zero-residual adapter over frozen shared state + detached belief."""

    def __init__(self, *, shared_dim: int, belief_dim: int, hidden_dim: int = R34_HIDDEN_DIM) -> None:
        super().__init__()
        if shared_dim <= 0 or belief_dim <= 0 or hidden_dim <= 0:
            raise ValueError("R11_R34_INVALID_BRIDGE_DIMENSION")
        self.shared_dim = int(shared_dim)
        self.belief_dim = int(belief_dim)
        self.hidden_dim = int(hidden_dim)
        self.body = nn.Sequential(
            nn.Linear(self.shared_dim + self.belief_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.direction_delta = nn.Linear(self.hidden_dim, 3)
        self.risk_logit_delta = nn.Linear(self.hidden_dim, 1)
        self._zero_residual_outputs()

    def _zero_residual_outputs(self) -> None:
        with torch.no_grad():
            self.direction_delta.weight.zero_()
            self.direction_delta.bias.zero_()
            self.risk_logit_delta.weight.zero_()
            self.risk_logit_delta.bias.zero_()

    def forward(self, shared: torch.Tensor, belief_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        if shared.ndim != 2 or shared.shape[1] != self.shared_dim:
            raise RuntimeError("R11_R34_SHARED_SHAPE_DRIFT")
        if belief_flat.ndim != 2 or belief_flat.shape[1] != self.belief_dim:
            raise RuntimeError("R11_R34_BELIEF_SHAPE_DRIFT")
        if shared.shape[0] != belief_flat.shape[0]:
            raise RuntimeError("R11_R34_SHARED_BELIEF_ROW_DRIFT")
        x = torch.cat([shared, belief_flat], dim=-1)
        h = self.body(x)
        return {
            "direction_logit_delta": self.direction_delta(h),
            "risk_logit_delta": self.risk_logit_delta(h),
        }


def bridge_parameter_report_r34(bridge: nn.Module) -> dict[str, int | str]:
    total = sum(int(p.numel()) for p in bridge.parameters())
    trainable = sum(int(p.numel()) for p in bridge.parameters() if p.requires_grad)
    byte_count = sum(int(p.numel() * p.element_size()) for p in bridge.parameters())
    if total != trainable:
        raise RuntimeError("R11_R34_FROZEN_BRIDGE_PARAMETER")
    if any(p.dtype != torch.float32 for p in bridge.parameters()):
        raise RuntimeError("R11_R34_NON_FP32_BRIDGE_PARAMETER")
    return {
        "parameter_count": total,
        "trainable_parameter_count": trainable,
        "parameter_bytes": byte_count,
        "authorized_owner": R34_OWNER,
    }


def clone_bridge_r34(bridge: DistributionalBeliefDecisionBridgeR34) -> DistributionalBeliefDecisionBridgeR34:
    out = copy.deepcopy(bridge)
    if any(not torch.equal(a, b) for a, b in zip(bridge.state_dict().values(), out.state_dict().values())):
        raise RuntimeError("R11_R34_BRIDGE_CLONE_DRIFT")
    return out


def apply_bridge_r34(
    base_outputs: Mapping[str, torch.Tensor],
    bridge_outputs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    base_logits = base_outputs["direction_logits"].detach()
    base_risk = base_outputs["requested_risk_raw"].detach()
    if base_logits.ndim != 2 or base_logits.shape[1] != 3:
        raise RuntimeError("R11_R34_BASE_DIRECTION_SHAPE_DRIFT")
    if base_risk.ndim != 2 or base_risk.shape[1] != 1:
        raise RuntimeError("R11_R34_BASE_RISK_SHAPE_DRIFT")
    if base_logits.requires_grad or base_risk.requires_grad:
        raise RuntimeError("R11_R34_PRODUCTION_OUTPUT_AUTOGRAD_TAINT")
    risk = torch.clamp(base_risk, R34_RISK_EPS, 1.0 - R34_RISK_EPS)
    risk_logit = torch.log(risk) - torch.log1p(-risk)
    direction_logits = base_logits + bridge_outputs["direction_logit_delta"]
    requested_risk_raw = torch.sigmoid(risk_logit + bridge_outputs["risk_logit_delta"])
    return {
        "direction_logits": direction_logits,
        "direction_probs": torch.softmax(direction_logits, dim=-1),
        "requested_risk_raw": requested_risk_raw,
    }


def assert_zero_residual_equivalence_r34(
    bridge: DistributionalBeliefDecisionBridgeR34,
    shared: torch.Tensor,
    belief: torch.Tensor,
    base_outputs: Mapping[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        delta = bridge(shared, belief)
        out = apply_bridge_r34(base_outputs, delta)
    if not torch.equal(out["direction_logits"], base_outputs["direction_logits"].detach()):
        raise RuntimeError("R11_R34_ZERO_RESIDUAL_DIRECTION_NOT_EXACT")
    # sigmoid(logit(x)) is numerically close but not guaranteed byte-identical.
    if not torch.allclose(out["requested_risk_raw"], base_outputs["requested_risk_raw"].detach(), atol=2e-7, rtol=2e-7):
        raise RuntimeError("R11_R34_ZERO_RESIDUAL_RISK_NOT_EQUIVALENT")


def group_block_shuffle_r34(
    belief: torch.Tensor,
    dependence_group_ids: Sequence[str],
    *,
    shift: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Rotate whole equal-sized dependence-group blocks; never shuffle rows within a group."""
    if belief.ndim != 2 or len(dependence_group_ids) != int(belief.shape[0]):
        raise RuntimeError("R11_R34_GROUP_SHUFFLE_ROW_DRIFT")
    order: list[str] = []
    members: dict[str, list[int]] = {}
    for idx, gid0 in enumerate(dependence_group_ids):
        gid = str(gid0)
        if gid not in members:
            order.append(gid)
            members[gid] = []
        members[gid].append(idx)
    if len(order) < 2:
        raise RuntimeError("R11_R34_GROUP_SHUFFLE_NEEDS_MULTIPLE_GROUPS")
    sizes = {len(members[g]) for g in order}
    if len(sizes) != 1:
        raise RuntimeError(f"R11_R34_NONUNIFORM_GROUP_SIZE_FOR_CONTROL:{sorted(sizes)}")
    n = len(order)
    k = int(shift) % n
    if k == 0:
        raise RuntimeError("R11_R34_IDENTITY_GROUP_SHUFFLE_FORBIDDEN")
    out = torch.empty_like(belief)
    mapping: list[dict[str, str]] = []
    for dst_pos, dst_gid in enumerate(order):
        src_gid = order[(dst_pos + k) % n]
        if src_gid == dst_gid:
            raise RuntimeError("R11_R34_GROUP_SHUFFLE_FIXED_POINT")
        dst = torch.tensor(members[dst_gid], dtype=torch.long, device=belief.device)
        src = torch.tensor(members[src_gid], dtype=torch.long, device=belief.device)
        out.index_copy_(0, dst, belief.index_select(0, src))
        mapping.append({"destination_group": dst_gid, "source_group": src_gid})
    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError("R11_R34_NONFINITE_SHUFFLED_BELIEF")
    return out, {
        "group_count": n,
        "rows_per_group": next(iter(sizes)),
        "shift": k,
        "fixed_point_groups": 0,
        "whole_group_block_shuffle": True,
        "mapping": mapping,
    }


def assert_bridge_gradient_ownership_r34(
    *,
    production_model: nn.Module,
    distributional_head: nn.Module,
    bridge: nn.Module,
) -> dict[str, Any]:
    production = [name for name, p in production_model.named_parameters() if p.grad is not None]
    head = [name for name, p in distributional_head.named_parameters() if p.grad is not None]
    if production:
        raise RuntimeError(f"R11_R34_PRODUCTION_RECEIVED_BRIDGE_GRADIENT:{production}")
    if head:
        raise RuntimeError(f"R11_R34_DISTRIBUTIONAL_HEAD_RECEIVED_BRIDGE_GRADIENT:{head}")
    bridge_nonzero: list[str] = []
    bridge_missing: list[str] = []
    for name, p in bridge.named_parameters():
        if p.grad is None:
            bridge_missing.append(name)
            continue
        if not bool(torch.isfinite(p.grad).all().item()):
            raise RuntimeError(f"R11_R34_NONFINITE_BRIDGE_GRADIENT:{name}")
        if bool(torch.any(p.grad != 0).item()):
            bridge_nonzero.append(name)
    if bridge_missing:
        raise RuntimeError(f"R11_R34_BRIDGE_PARAMETER_WITHOUT_GRADIENT:{bridge_missing}")
    if not bridge_nonzero:
        raise RuntimeError("R11_R34_ZERO_BRIDGE_GRADIENT_PATH")
    return {
        "authorized_gradient_owner": R34_OWNER,
        "production_student_gradient_parameter_count": 0,
        "distributional_head_gradient_parameter_count": 0,
        "bridge_gradient_parameter_tensors": len(bridge_nonzero),
        "teacher_target_autograd": False,
    }


def state_l2_delta_r34(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    if set(before) != set(after):
        raise RuntimeError("R11_R34_STATE_KEY_DRIFT")
    total = 0.0
    for name in before:
        d = after[name].detach().cpu().double() - before[name].detach().cpu().double()
        total += float(torch.sum(d * d))
    delta = math.sqrt(total)
    if not math.isfinite(delta):
        raise RuntimeError("R11_R34_NONFINITE_STATE_DELTA")
    return delta
