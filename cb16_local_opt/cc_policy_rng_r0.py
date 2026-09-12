from __future__ import annotations
from dataclasses import dataclass
import base64, hashlib, pickle, random

@dataclass
class PolicyRNG:
    policy_id: str
    account_lineage_id: str
    seed: int
    counter: int = 0

    def __post_init__(self) -> None:
        material = f"{self.seed}|{self.policy_id}|{self.account_lineage_id}".encode()
        derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        self._rng = random.Random(derived)
        for _ in range(self.counter):
            self._rng.random()

    @property
    def stream_id(self) -> str:
        return hashlib.sha256(f"{self.policy_id}|{self.account_lineage_id}|{self.seed}".encode()).hexdigest()

    def random(self) -> float:
        self.counter += 1
        return self._rng.random()

    def normal(self) -> float:
        # exactly two uniform draws, so provenance remains explicit
        import math
        u1 = max(self.random(), 2.0**-53)
        u2 = self.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def state_dict(self) -> dict[str, object]:
        return {"policy_id": self.policy_id, "account_lineage_id": self.account_lineage_id,
                "seed": self.seed, "counter": self.counter,
                "state_b64": base64.b64encode(pickle.dumps(self._rng.getstate(), protocol=4)).decode()}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "PolicyRNG":
        obj = cls(str(state["policy_id"]), str(state["account_lineage_id"]), int(state["seed"]), 0)
        obj.counter = int(state["counter"])
        obj._rng.setstate(pickle.loads(base64.b64decode(str(state["state_b64"]))))
        return obj

    def provenance(self) -> tuple[str, int]:
        return self.stream_id, self.counter
