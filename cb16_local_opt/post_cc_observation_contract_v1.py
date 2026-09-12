from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .cc_experience_wire_r0 import content_sha256

OBSERVATION_CONTRACT_ID = "PostCCObservationFactV1"


@dataclass(frozen=True)
class PostCCObservationFactV1:
    science_semantic_version: str
    observation_schema: str
    observation_hash: str
    market_payload: Mapping[str, Any]
    account_payload: Mapping[str, Any]
    execution_payload: Mapping[str, Any]
    market_source_identity: str
    market_source_version: str
    market_visible_through_time: str
    account_lineage_id: str
    decision_index: int
    environment_time: str
    normalizer_identity: str

    def payload_for_hash(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload.pop("observation_hash")
        return payload

    def expected_hash(self) -> str:
        return content_sha256(
            {"contract_id": OBSERVATION_CONTRACT_ID, **self.payload_for_hash()}
        )

    def validate(self) -> "PostCCObservationFactV1":
        for name in (
            "science_semantic_version",
            "observation_schema",
            "observation_hash",
            "market_source_identity",
            "market_source_version",
            "market_visible_through_time",
            "account_lineage_id",
            "environment_time",
            "normalizer_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if self.decision_index < 0:
            raise ValueError("decision_index must be >= 0")
        for name in ("market_payload", "account_payload", "execution_payload"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"{name} must be a mapping")
        if self.market_visible_through_time > self.environment_time:
            raise ValueError("FUTURE_MARKET_INFORMATION_FORBIDDEN")
        if self.observation_hash != self.expected_hash():
            raise ValueError("OBSERVATION_HASH_MISMATCH")
        return self


def build_observation_fact(
    *,
    science_semantic_version: str,
    observation_schema: str,
    market_payload: Mapping[str, Any],
    account_payload: Mapping[str, Any],
    execution_payload: Mapping[str, Any],
    market_source_identity: str,
    market_source_version: str,
    market_visible_through_time: str,
    account_lineage_id: str,
    decision_index: int,
    environment_time: str,
    normalizer_identity: str,
) -> PostCCObservationFactV1:
    provisional = PostCCObservationFactV1(
        science_semantic_version,
        observation_schema,
        "PENDING",
        dict(market_payload),
        dict(account_payload),
        dict(execution_payload),
        market_source_identity,
        market_source_version,
        market_visible_through_time,
        account_lineage_id,
        int(decision_index),
        environment_time,
        normalizer_identity,
    )
    digest = provisional.expected_hash()
    return PostCCObservationFactV1(
        provisional.science_semantic_version,
        provisional.observation_schema,
        digest,
        provisional.market_payload,
        provisional.account_payload,
        provisional.execution_payload,
        provisional.market_source_identity,
        provisional.market_source_version,
        provisional.market_visible_through_time,
        provisional.account_lineage_id,
        provisional.decision_index,
        provisional.environment_time,
        provisional.normalizer_identity,
    ).validate()
