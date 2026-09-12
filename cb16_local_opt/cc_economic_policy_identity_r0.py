from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class EconomicPolicyIdentity:
    object_type: str
    identity: str
    generations: tuple[str, ...]

    @classmethod
    def frozen_checkpoint(cls, checkpoint_sha: str, generation: str) -> "EconomicPolicyIdentity":
        return cls("frozen_checkpoint", checkpoint_sha, (generation,))
    @classmethod
    def generation_chain(cls, strategy_id: str, generations: tuple[str,...]) -> "EconomicPolicyIdentity":
        if len(set(generations)) < 2: raise ValueError("generation chain must identify mixed generations")
        return cls("generation_chain", strategy_id, generations)
    @classmethod
    def baseline(cls, name: str) -> "EconomicPolicyIdentity": return cls("baseline", name, ())
    def assert_pure_generation(self, generation: str) -> None:
        if self.object_type!="frozen_checkpoint" or self.generations!=(generation,):
            raise ValueError("mixed-generation result cannot be credited to final checkpoint")
