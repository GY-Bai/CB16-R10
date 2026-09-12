from __future__ import annotations
from dataclasses import dataclass, field

VALID_VIEWS = {"raw", "replay", "demonstration", "economic"}

@dataclass
class ExperienceViews:
    raw_ids: set[str] = field(default_factory=set)
    replay_ids: set[str] = field(default_factory=set)
    demonstration_ids: set[str] = field(default_factory=set)
    economic_ids: set[str] = field(default_factory=set)

    def ingest_raw(self, fact_id: str) -> None: self.raw_ids.add(fact_id)
    def set_membership(self, fact_id: str, view: str, included: bool) -> None:
        if view not in VALID_VIEWS - {"raw"}: raise ValueError("invalid derived view")
        if fact_id not in self.raw_ids: raise KeyError("derived membership requires raw fact")
        target = getattr(self, f"{view}_ids")
        (target.add if included else target.discard)(fact_id)
    def assert_economic_cohort(self, preregistered_ids: set[str]) -> None:
        if self.economic_ids != preregistered_ids:
            raise ValueError("economic cohort differs from preregistration; survivor filtering forbidden")
