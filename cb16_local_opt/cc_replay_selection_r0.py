from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class ReplaySelectionMetadata:
    sequence_id: str
    selection_rng_identity: str
    sampling_probability_or_weight: float
    source_class: str
    generation_class: str
    support_health_summary: Mapping[str, float]
    selection_policy_version: str

    def validate(self) -> "ReplaySelectionMetadata":
        if not self.selection_rng_identity or not self.selection_policy_version: raise ValueError("selection provenance required")
        if not 0 < self.sampling_probability_or_weight <= 1: raise ValueError("sampling probability/weight must be in (0,1]")
        forbidden={"realized_pnl", "future_return", "survived", "winner"}
        if forbidden & set(self.support_health_summary):
            raise ValueError("outcome-derived selection metadata forbidden")
        return self
