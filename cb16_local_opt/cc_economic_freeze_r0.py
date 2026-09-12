from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
import json

@dataclass(frozen=True)
class EvaluationFreeze:
    cohort_id: str
    common_horizon_id: str
    weights_hash: str
    capital_denominator_id: str
    baseline_definition_id: str
    policy_identity: str
    data_lineage_id: str
    version: str
    @property
    def fingerprint(self)->str:
        return sha256(json.dumps(asdict(self),sort_keys=True,separators=(",",":")).encode()).hexdigest()

def require_new_version_if_changed(before: EvaluationFreeze, after: EvaluationFreeze) -> None:
    if before.fingerprint != after.fingerprint and before.version == after.version:
        raise ValueError("post-observation evaluation rescue forbidden; changed contract requires new version")
