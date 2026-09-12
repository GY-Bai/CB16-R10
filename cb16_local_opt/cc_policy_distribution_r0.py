from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor
from .cc_policy_rng_r0 import PolicyRNG

DIRECTION_ORDER = ("SHORT", "FLAT", "LONG")
INDEX = {d:i for i,d in enumerate(DIRECTION_ORDER)}
RISK_ENDPOINT_POLICY = "FLAT_ZERO_POINT_MASS_NONFLAT_ENDPOINT_MASSES_DISALLOWED_V1_R1"

@dataclass(frozen=True)
class NominalAction:
    direction: str
    target_risk: float
    log_prob: float
    risk_measure_kind: str
    rng_stream_id: str
    rng_counter: int


def stable_direction_log_probs(logits: Tensor) -> Tensor:
    if logits.shape[-1] != 3:
        raise ValueError("DIRECTION_LOGITS_SHAPE")
    if not torch.isfinite(logits).all():
        raise ValueError("DIRECTION_LOGITS_NONFINITE")
    return torch.log_softmax(logits, dim=-1)


def _logit(r: Tensor) -> Tensor:
    return torch.log(r) - torch.log1p(-r)


def nonflat_risk_log_density(risk: Tensor, loc: Tensor, log_scale: Tensor) -> Tensor:
    if torch.any((risk <= 0) | (risk >= 1)):
        raise ValueError("NONFLAT_ENDPOINT_DISALLOWED")
    scale = torch.exp(log_scale)
    if torch.any(scale <= 0) or not torch.isfinite(scale).all():
        raise ValueError("BAD_RISK_SCALE")
    z = _logit(risk)
    normal = -0.5*((z-loc)/scale)**2 - log_scale - 0.5*math.log(2*math.pi)
    jac = -torch.log(risk) - torch.log1p(-risk)
    return normal + jac


def joint_log_prob(direction_logits: Tensor, risk_loc: Tensor, risk_log_scale: Tensor,
                   direction: str, risk: float) -> Tensor:
    lp = stable_direction_log_probs(direction_logits)[INDEX[direction]]
    if direction == "FLAT":
        if float(risk) != 0.0:
            raise ValueError("FLAT_RISK_NOT_ZERO")
        return lp
    if not 0.0 < float(risk) < 1.0:
        raise ValueError("NONFLAT_ENDPOINT_DISALLOWED")
    i = INDEX[direction]
    rt = torch.as_tensor(float(risk), dtype=direction_logits.dtype, device=direction_logits.device)
    return lp + nonflat_risk_log_density(rt, risk_loc[i], risk_log_scale[i])


def sample_nominal(direction_logits: Tensor, risk_loc: Tensor, risk_log_scale: Tensor, rng: PolicyRNG) -> NominalAction:
    probs = torch.softmax(direction_logits.detach().cpu(), dim=-1).tolist()
    u = rng.random(); acc = 0.0; idx = 2
    for i,p in enumerate(probs):
        acc += float(p)
        if u <= acc:
            idx = i; break
    direction = DIRECTION_ORDER[idx]
    if direction == "FLAT":
        risk = 0.0; kind = "point_mass"
    else:
        z = float(risk_loc[idx].detach().cpu()) + math.exp(float(risk_log_scale[idx].detach().cpu())) * rng.normal()
        risk = 1.0/(1.0+math.exp(-max(-40.0,min(40.0,z))))
        # transformed Normal is continuous, exact endpoints are never emitted
        risk = min(1.0-1e-12, max(1e-12, risk)); kind = "continuous_density"
    lp = float(joint_log_prob(direction_logits, risk_loc, risk_log_scale, direction, risk).detach().cpu())
    sid,cnt = rng.provenance()
    return NominalAction(direction,risk,lp,kind,sid,cnt)
