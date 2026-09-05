from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch
from torch import nn


@dataclass(frozen=True)
class TypedBrainTierSpecR10:
    name: str
    operator_dim: int = 48
    medium_dim: int = 48
    account_dim: int = 6
    operator_hidden: tuple[int,...] = (64,)
    medium_hidden: tuple[int,...] = (64,)
    account_hidden: tuple[int,...] = (32,)
    shared_hidden: tuple[int,...] = (256,256)
    direction_hidden: tuple[int,...] = (128,)
    sizing_hidden: tuple[int,...] = (128,64)
    direction_embedding_dim: int = 8

TIERS_R10 = {
    "TIER_1": TypedBrainTierSpecR10("TIER_1"),
    "TIER_2": TypedBrainTierSpecR10("TIER_2", operator_hidden=(96,64), medium_hidden=(96,64), account_hidden=(64,32), shared_hidden=(512,512,256), direction_hidden=(256,128), sizing_hidden=(256,128,64)),
    "TIER_3": TypedBrainTierSpecR10("TIER_3", operator_hidden=(128,96), medium_hidden=(128,96), account_hidden=(96,64), shared_hidden=(1024,1024,512,256), direction_hidden=(256,128), sizing_hidden=(256,128,64)),
}


def _mlp(in_dim: int, hidden: Iterable[int]):
    layers=[]; d=in_dim
    for h in hidden:
        layers += [nn.Linear(d,h), nn.SiLU()]; d=h
    return nn.Sequential(*layers), d


class TypedCentralBrainR10(nn.Module):
    """Trainable G0+ nominal economic brain over frozen typed organs.

    Operator48, Medium48 and AccountState6 remain distinct at the input boundary and are
    separately read by Central-Brain-owned trainable stems before the Shared Decision Core.
    Historical attribute names ``*_encoder`` are checkpoint keys only; R10.1 authority classifies
    them as trainable Brain stems, not frozen sensory/account encoders. Ordered4H30 is absent.
    """
    def __init__(self, spec: TypedBrainTierSpecR10):
        super().__init__(); self.spec=spec
        self.operator_encoder, od = _mlp(spec.operator_dim, spec.operator_hidden)
        self.medium_encoder, md = _mlp(spec.medium_dim, spec.medium_hidden)
        self.account_encoder, ad = _mlp(spec.account_dim, spec.account_hidden)
        self.shared_core, sd = _mlp(od+md+ad, spec.shared_hidden)
        self.direction_body, dd = _mlp(sd, spec.direction_hidden); self.direction_out=nn.Linear(dd,3)
        self.direction_embedding=nn.Embedding(3,spec.direction_embedding_dim)
        self.sizing_body, zd = _mlp(sd+spec.direction_embedding_dim,spec.sizing_hidden); self.sizing_out=nn.Linear(zd,1)

    def forward(self, operator48: torch.Tensor, medium48: torch.Tensor, account_state6: torch.Tensor):
        if operator48.ndim!=2 or operator48.shape[1]!=48: raise ValueError("operator48 shape")
        if medium48.ndim!=2 or medium48.shape[1]!=48: raise ValueError("medium48 shape")
        if account_state6.ndim!=2 or account_state6.shape[1]!=6: raise ValueError("account_state6 shape")
        # Frozen organ outputs are treated as values, not gradient-owned parameters.
        o=self.operator_encoder(operator48.detach()); m=self.medium_encoder(medium48.detach()); a=self.account_encoder(account_state6.detach())
        h=self.shared_core(torch.cat([o,m,a],-1)); logits=self.direction_out(self.direction_body(h)); probs=torch.softmax(logits,-1)
        emb=probs @ self.direction_embedding.weight
        risk=torch.sigmoid(self.sizing_out(self.sizing_body(torch.cat([h,emb],-1)))).squeeze(-1)
        return {"shared":h,"direction_logits":logits,"direction_probs":probs,"requested_risk_raw":risk}

    @torch.no_grad()
    def compose_action(self, out):
        cls=out["direction_probs"].argmax(-1); direction=cls.to(torch.int8)-1
        risk=out["requested_risk_raw"].clamp(0,1); risk=torch.where(direction==0,torch.zeros_like(risk),risk)
        return {"direction_class":cls,"direction":direction,"requested_risk":risk}


def build_g0_brain_r10(tier: str="TIER_1", *, seed: int=0, device: str="cpu") -> TypedCentralBrainR10:
    if tier not in TIERS_R10: raise ValueError(f"unknown tier {tier}")
    torch.manual_seed(int(seed)); model=TypedCentralBrainR10(TIERS_R10[tier]).to(device)
    return model


def parameter_report_r10(model: nn.Module):
    groups={k:0 for k in ["operator_input","medium_input","account_input","shared_core","direction_head","sizing_head","total","trainable"]}
    for n,p in model.named_parameters():
        z=p.numel(); groups["total"]+=z; groups["trainable"]+=z if p.requires_grad else 0
        if n.startswith("operator_encoder"):groups["operator_input"]+=z
        elif n.startswith("medium_encoder"):groups["medium_input"]+=z
        elif n.startswith("account_encoder"):groups["account_input"]+=z
        elif n.startswith("shared_core"):groups["shared_core"]+=z
        elif n.startswith(("direction_body","direction_out")):groups["direction_head"]+=z
        elif n.startswith(("direction_embedding","sizing_body","sizing_out")):groups["sizing_head"]+=z
    return groups

# R10.1 logical role aliases. These are functions rather than registered module aliases so the
# original R10 state_dict key-space remains exactly unchanged.
def brain_stem_aliases_r101(model: TypedCentralBrainR10):
    return {
        "operator_brain_stem": model.operator_encoder,
        "medium_brain_stem": model.medium_encoder,
        "account_brain_stem": model.account_encoder,
        "shared_decision_core": model.shared_core,
        "direction_head": (model.direction_body, model.direction_out),
        "requested_risk_head": (model.direction_embedding, model.sizing_body, model.sizing_out),
    }
