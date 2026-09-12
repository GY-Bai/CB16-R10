from __future__ import annotations
from dataclasses import dataclass
import hashlib, math
import torch
from .cc_policy_distribution_r0 import DIRECTION_ORDER

@dataclass(frozen=True)
class DeterministicEvaluationPolicy:
    parent_policy_id: str
    policy_sha256: str
    evaluation_id: str

    @classmethod
    def from_behavior(cls, policy_id: str, policy_sha256: str) -> "DeterministicEvaluationPolicy":
        eid = hashlib.sha256(f"eval|{policy_id}|{policy_sha256}".encode()).hexdigest()
        return cls(policy_id, policy_sha256, eid)

    def action(self, direction_logits: torch.Tensor, risk_loc: torch.Tensor) -> tuple[str,float]:
        idx = int(torch.argmax(direction_logits).item())
        d = DIRECTION_ORDER[idx]
        if d == "FLAT": return d,0.0
        return d, float(torch.sigmoid(risk_loc[idx]).detach().cpu())
