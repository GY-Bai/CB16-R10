from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class EconomicCohort:
    cohort_id: str
    account_start_states: Mapping[str, str]
    weights: Mapping[str, float]
    capital_denominator_id: str
    common_horizon_id: str
    policy_object_type: str
    policy_identity: str
    realization_status: str

    def validate(self) -> "EconomicCohort":
        ids=set(self.account_start_states)
        if not ids or set(self.weights)!=ids: raise ValueError("weights must exactly cover cohort")
        if any(w<0 for w in self.weights.values()) or sum(self.weights.values()) <= 0: raise ValueError("invalid cohort weights")
        if self.policy_object_type not in {"frozen_checkpoint","generation_chain","baseline"}: raise ValueError("invalid policy object")
        if self.realization_status not in {"REALIZED","TRUNCATED","ESTIMATED"}: raise ValueError("invalid realization status")
        return self
