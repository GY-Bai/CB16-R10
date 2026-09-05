from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class TraderTierSpec:
    name: str
    market_dim: int
    account_dim: int
    account_hidden: tuple[int, ...]
    account_latent: int
    shared_hidden: tuple[int, ...]
    direction_hidden: tuple[int, ...]
    sizing_hidden: tuple[int, ...]
    direction_embedding_dim: int = 8


TIER_SPECS: dict[str, TraderTierSpec] = {
    "TIER_1": TraderTierSpec(
        name="TIER_1",
        market_dim=64,
        account_dim=6,
        account_hidden=(32,),
        account_latent=32,
        shared_hidden=(256, 256),
        direction_hidden=(128,),
        sizing_hidden=(128, 64),
    ),
    "TIER_2": TraderTierSpec(
        name="TIER_2",
        market_dim=64,
        account_dim=6,
        account_hidden=(64,),
        account_latent=32,
        shared_hidden=(512, 512, 256),
        direction_hidden=(256, 128),
        sizing_hidden=(256, 128, 64),
    ),
    "TIER_3": TraderTierSpec(
        name="TIER_3",
        market_dim=64,
        account_dim=6,
        account_hidden=(128, 64),
        account_latent=64,
        shared_hidden=(1024, 1024, 512, 256),
        direction_hidden=(256, 128),
        sizing_hidden=(256, 128, 64),
    ),
    "TIER_4": TraderTierSpec(
        name="TIER_4",
        market_dim=64,
        account_dim=6,
        account_hidden=(256, 128),
        account_latent=128,
        shared_hidden=(1536, 1536, 1024, 512),
        direction_hidden=(512, 256),
        sizing_hidden=(512, 256, 128),
        direction_embedding_dim=16,
    ),
}


def _mlp(in_dim: int, hidden: Iterable[int], out_dim: int | None = None) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    d = int(in_dim)
    for h in hidden:
        h = int(h)
        layers.extend([nn.Linear(d, h), nn.SiLU()])
        d = h
    if out_dim is not None:
        layers.append(nn.Linear(d, int(out_dim)))
        d = int(out_dim)
    return nn.Sequential(*layers), d


class JointDirectionSizingTrader(nn.Module):
    """Shared-core, dual-head Trader.

    Input:
      market_latent  [B, market_dim]
      account_state6 [B, account_dim]

    Output:
      direction logits over [SHORT, FLAT, LONG]
      requested risk in [0,1]

    The sizing head is direction-conditioned through a differentiable expected
    direction embedding during training.  At action composition, FLAT forces risk=0.
    """

    DIRECTIONS = (-1, 0, 1)

    def __init__(self, spec: TraderTierSpec):
        super().__init__()
        self.spec = spec
        self.account_encoder, _ = _mlp(
            spec.account_dim,
            spec.account_hidden,
            spec.account_latent,
        )
        fusion_dim = spec.market_dim + spec.account_latent
        self.shared_core, shared_dim = _mlp(fusion_dim, spec.shared_hidden)
        self.direction_body, direction_dim = _mlp(shared_dim, spec.direction_hidden)
        self.direction_out = nn.Linear(direction_dim, 3)

        self.direction_embedding = nn.Embedding(3, spec.direction_embedding_dim)
        self.sizing_body, sizing_dim = _mlp(
            shared_dim + spec.direction_embedding_dim,
            spec.sizing_hidden,
        )
        self.sizing_out = nn.Linear(sizing_dim, 1)

    def forward(self, market_latent: torch.Tensor, account_state: torch.Tensor) -> dict[str, torch.Tensor]:
        if market_latent.ndim != 2 or market_latent.shape[-1] != self.spec.market_dim:
            raise ValueError("market_latent shape mismatch")
        if account_state.ndim != 2 or account_state.shape[-1] != self.spec.account_dim:
            raise ValueError("account_state shape mismatch")
        a = self.account_encoder(account_state)
        h = self.shared_core(torch.cat([market_latent, a], dim=-1))
        dbody = self.direction_body(h)
        logits = self.direction_out(dbody)
        probs = torch.softmax(logits, dim=-1)

        # Expected embedding keeps sizing differentiable with respect to direction logits.
        emb_matrix = self.direction_embedding.weight  # [3,E]
        expected_emb = probs @ emb_matrix
        s = self.sizing_body(torch.cat([h, expected_emb], dim=-1))
        risk_raw = torch.sigmoid(self.sizing_out(s)).squeeze(-1)
        return {
            "shared": h,
            "direction_logits": logits,
            "direction_probs": probs,
            "requested_risk_raw": risk_raw,
        }

    @torch.no_grad()
    def compose_action(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cls = outputs["direction_probs"].argmax(dim=-1)  # 0 short,1 flat,2 long
        direction = cls.to(torch.int8) - 1
        risk = outputs["requested_risk_raw"].clamp(0, 1)
        risk = torch.where(direction == 0, torch.zeros_like(risk), risk)
        return {
            "direction_class": cls,
            "direction": direction,
            "requested_risk": risk,
        }


def build_trader(tier: str = "TIER_1") -> JointDirectionSizingTrader:
    try:
        spec = TIER_SPECS[tier]
    except KeyError:
        raise ValueError(f"unknown tier {tier}; available={sorted(TIER_SPECS)}")
    return JointDirectionSizingTrader(spec)


def parameter_report(model: nn.Module) -> dict[str, int]:
    groups = {
        "account_encoder": 0,
        "shared_core": 0,
        "direction_head": 0,
        "sizing_head": 0,
        "total": 0,
        "trainable": 0,
    }
    for name, p in model.named_parameters():
        n = p.numel()
        groups["total"] += n
        if p.requires_grad:
            groups["trainable"] += n
        if name.startswith("account_encoder"):
            groups["account_encoder"] += n
        elif name.startswith("shared_core"):
            groups["shared_core"] += n
        elif name.startswith(("direction_body", "direction_out")):
            groups["direction_head"] += n
        elif name.startswith(("direction_embedding", "sizing_body", "sizing_out")):
            groups["sizing_head"] += n
    return groups


def ladder_report() -> dict[str, dict[str, int]]:
    return {name: parameter_report(build_trader(name)) for name in TIER_SPECS}


def require_capacity_escalation_receipt(
    *,
    current_tier: str,
    next_tier: str,
    evidence_of_underfitting: bool,
    receipt_id: str | None,
) -> None:
    order = list(TIER_SPECS)
    if current_tier not in order or next_tier not in order:
        raise ValueError("unknown tier")
    if order.index(next_tier) <= order.index(current_tier):
        return
    if not evidence_of_underfitting or not receipt_id:
        raise RuntimeError("CAPACITY_ESCALATION_NOT_AUTHORIZED")
