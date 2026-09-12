from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
import torch
from torch import nn

@dataclass(frozen=True)
class BrainBindings:
    observation_hash: str
    normalizer_hash: str
    distribution_hash: str
    science_hash: str
    def semantic_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__,sort_keys=True,separators=(",",":")).encode()).hexdigest()

class CCCentralBrain(nn.Module):
    def __init__(self, market_dim:int, account_dim:int, execution_dim:int, hidden:int, bindings:BrainBindings):
        super().__init__(); self.bindings=bindings
        self.market_organ = nn.Linear(market_dim, hidden, bias=False)
        for p in self.market_organ.parameters(): p.requires_grad_(False)
        self.account_stem = nn.Linear(account_dim+execution_dim, hidden)
        self.fusion = nn.Sequential(nn.Tanh(), nn.Linear(hidden*2,hidden), nn.Tanh())
        self.direction_head = nn.Linear(hidden,3)
        self.risk_loc_head = nn.Linear(hidden,3)
        self.risk_log_scale = nn.Parameter(torch.full((3,), -0.5))

    def forward(self, market:torch.Tensor, account:torch.Tensor, execution:torch.Tensor):
        m=self.market_organ(market)
        a=self.account_stem(torch.cat([account,execution],dim=-1))
        h=self.fusion(torch.cat([m,a],dim=-1))
        return self.direction_head(h), self.risk_loc_head(h), self.risk_log_scale.expand_as(self.risk_loc_head(h))

    def assert_gradient_ownership(self) -> None:
        if any(p.requires_grad for p in self.market_organ.parameters()):
            raise RuntimeError("FROZEN_MARKET_ORGAN_TRAINABLE")
        required=(self.account_stem,self.fusion,self.direction_head,self.risk_loc_head)
        if not all(any(p.requires_grad for p in mod.parameters()) for mod in required):
            raise RuntimeError("TRAINABLE_PATH_FROZEN")
