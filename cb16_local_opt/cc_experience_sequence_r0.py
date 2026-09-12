from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .cc_experience_transition_r0 import ImmutableExperienceTransitionV1
from .cc_experience_wire_r0 import CCExperienceSequenceV1, content_sha256

@dataclass(frozen=True)
class PolicySwitch:
    decision_index: int
    from_policy_generation: str
    to_policy_generation: str

@dataclass(frozen=True)
class OrderedExperienceSequence:
    wire: CCExperienceSequenceV1
    transitions: tuple[ImmutableExperienceTransitionV1, ...]
    policy_switches: tuple[PolicySwitch, ...]

    @classmethod
    def build(cls, sequence_id: str, transitions: Sequence[ImmutableExperienceTransitionV1], *,
              science_semantic_version: str, market_lineage_id: str, source_classification: str,
              normalizer_identities: Sequence[str], chunk_boundary_type: str = "CHUNK_END",
              bootstrap_state_ref_or_null: str | None = None) -> "OrderedExperienceSequence":
        ts = tuple(transitions)
        if not ts:
            raise ValueError("empty sequence")
        for t in ts:
            t.validate()
        lineage = ts[0].environment.account_lineage_id
        if any(t.environment.account_lineage_id != lineage for t in ts):
            raise ValueError("cross-account splice")
        indices = [t.environment.decision_index for t in ts]
        if any(b != a + 1 for a, b in zip(indices, indices[1:])):
            raise ValueError("decision indices must be strictly contiguous; reorder/drop/duplicate forbidden")
        if len({t.transition_id for t in ts}) != len(ts):
            raise ValueError("duplicate transition ID")
        for prev, cur in zip(ts, ts[1:]):
            if prev.environment.post_account_truth_hash != cur.environment.pre_account_truth_hash:
                raise ValueError("account truth discontinuity")
        switches = tuple(
            PolicySwitch(cur.environment.decision_index, prev.policy_generation, cur.policy_generation)
            for prev, cur in zip(ts, ts[1:]) if prev.policy_generation != cur.policy_generation
        )
        refs = tuple(t.transition_id for t in ts)
        raw_hash = content_sha256([t.content_sha256 for t in ts])
        wire = CCExperienceSequenceV1(
            sequence_id=sequence_id,
            account_lineage_id=lineage,
            science_semantic_version=science_semantic_version,
            market_lineage_id=market_lineage_id,
            source_classification=source_classification,
            transition_refs=refs,
            first_decision_index=indices[0], last_decision_index=indices[-1],
            behavior_policy_identities=tuple(dict.fromkeys(t.policy_id for t in ts)),
            normalizer_identities=tuple(normalizer_identities),
            chunk_boundary_type=chunk_boundary_type,
            bootstrap_state_ref_or_null=bootstrap_state_ref_or_null,
            raw_fact_content_sha256=raw_hash,
        ).validate()
        return cls(wire=wire, transitions=ts, policy_switches=switches)

    def pure_generation(self) -> str | None:
        generations = {t.policy_generation for t in self.transitions}
        return next(iter(generations)) if len(generations) == 1 else None
