"""S0-v2 durable observation fact codec.

This module completes the frozen ``PostCCObservationFactV1`` successor contract
with a deterministic codec plus the canonical Brain observation builder used by
the durable collection path.  It deliberately does not define a second
observation contract: every fact is an instance of
``post_cc_observation_contract_v1.PostCCObservationFactV1``.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from typing import Any, Mapping

from .cc_experience_wire_r0 import canonical_json_bytes, content_sha256
from .post_cc_observation_contract_v1 import (
    OBSERVATION_CONTRACT_ID,
    PostCCObservationFactV1,
    build_observation_fact,
)

CANONICAL_OBSERVATION_SCHEMA_V1 = "CB16_S0V2_CANONICAL_BRAIN_OBSERVATION_V1"
CANONICAL_NORMALIZER_IDENTITY_V1 = "CB16_S0V2_CANONICAL_NORMALIZER_V1"
CANONICAL_MARKET_DIM_V1 = 2
CANONICAL_ACCOUNT_DIM_V1 = 3
CANONICAL_EXECUTION_DIM_V1 = 2
CANONICAL_PAYLOAD_KEY_V1 = "values"

# Keys that would admit information unavailable at the policy decision.  The
# codec is fail-closed: it refuses to encode or decode any observation payload
# carrying these names at any nesting depth.
FORBIDDEN_FUTURE_INFORMATION_KEY_FRAGMENTS_V1 = (
    "future",
    "lookahead",
    "oracle",
    "ground_truth",
    "groundtruth",
    "answer",
    "label",
    "next_",
    "nextprice",
    "outcome",
    "realized_reward",
    "future_return",
)

_OBSERVATION_FACT_FIELDS = (
    "science_semantic_version",
    "observation_schema",
    "observation_hash",
    "market_payload",
    "account_payload",
    "execution_payload",
    "market_source_identity",
    "market_source_version",
    "market_visible_through_time",
    "account_lineage_id",
    "decision_index",
    "environment_time",
    "normalizer_identity",
)
_CANONICAL_PAYLOAD_NAMES = ("market_payload", "account_payload", "execution_payload")
_STRING_FIELDS = (
    "science_semantic_version",
    "observation_schema",
    "observation_hash",
    "market_source_identity",
    "market_source_version",
    "market_visible_through_time",
    "account_lineage_id",
    "environment_time",
    "normalizer_identity",
)


def _require_nonempty_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number")
    return 0.0 if out == 0.0 else out


def _check_no_future_information(name: str, value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            low = key_text.lower()
            if any(fragment in low for fragment in FORBIDDEN_FUTURE_INFORMATION_KEY_FRAGMENTS_V1):
                raise ValueError(f"FUTURE_INFORMATION_FIELD_FORBIDDEN:{name}.{key_text}")
            _check_no_future_information(f"{name}.{key_text}", nested)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _check_no_future_information(f"{name}[{index}]", nested)


def _canonical_payload_values(name: str, payload: Mapping[str, Any], expected_dim: int) -> tuple[float, ...]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if set(payload.keys()) != {CANONICAL_PAYLOAD_KEY_V1}:
        raise ValueError(f"{name} must contain exactly the canonical key '{CANONICAL_PAYLOAD_KEY_V1}'")
    raw_values = payload[CANONICAL_PAYLOAD_KEY_V1]
    if not isinstance(raw_values, (list, tuple)):
        raise ValueError(f"{name}.{CANONICAL_PAYLOAD_KEY_V1} must be a sequence")
    values = tuple(_require_finite_number(f"{name}.values[{i}]", value) for i, value in enumerate(raw_values))
    if len(values) != int(expected_dim):
        raise ValueError(f"{name} must contain exactly {int(expected_dim)} canonical values")
    return values


def canonical_observation_bytes_v1(fact: PostCCObservationFactV1) -> bytes:
    """Deterministic byte representation of a validated observation fact."""
    if not isinstance(fact, PostCCObservationFactV1):
        raise TypeError("fact must be a PostCCObservationFactV1")
    fact.validate()
    for name in _CANONICAL_PAYLOAD_NAMES:
        _check_no_future_information(name, getattr(fact, name))
    return canonical_json_bytes(asdict(fact))


def encode_observation_fact_v1(fact: PostCCObservationFactV1) -> bytes:
    return canonical_observation_bytes_v1(fact)


def decode_observation_fact_v1(payload: bytes) -> PostCCObservationFactV1:
    """Decode a durable observation fact and fail closed on any corruption."""
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    try:
        raw = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("OBSERVATION_FACT_BYTES_INVALID") from exc
    if not isinstance(raw, dict):
        raise ValueError("OBSERVATION_FACT_MUST_BE_OBJECT")
    if set(raw.keys()) != set(_OBSERVATION_FACT_FIELDS):
        raise ValueError("OBSERVATION_FACT_FIELD_SET_INVALID")
    try:
        fact = PostCCObservationFactV1(
            science_semantic_version=raw["science_semantic_version"],
            observation_schema=raw["observation_schema"],
            observation_hash=raw["observation_hash"],
            market_payload=raw["market_payload"],
            account_payload=raw["account_payload"],
            execution_payload=raw["execution_payload"],
            market_source_identity=raw["market_source_identity"],
            market_source_version=raw["market_source_version"],
            market_visible_through_time=raw["market_visible_through_time"],
            account_lineage_id=raw["account_lineage_id"],
            decision_index=raw["decision_index"],
            environment_time=raw["environment_time"],
            normalizer_identity=raw["normalizer_identity"],
        )
    except (TypeError, KeyError) as exc:
        raise ValueError("OBSERVATION_FACT_FIELDS_INVALID") from exc
    fact.validate()
    for name in _CANONICAL_PAYLOAD_NAMES:
        _check_no_future_information(name, getattr(fact, name))
    return fact


def observation_logical_id_from_identity_v1(
    *,
    account_lineage_id: str,
    decision_index: int,
    environment_time: str,
    observation_hash: str,
) -> str:
    """Compute the observation logical identity without rebuilding the fact."""
    for name, value in (
        ("account_lineage_id", account_lineage_id),
        ("environment_time", environment_time),
        ("observation_hash", observation_hash),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty text")
    if len(observation_hash) != 64:
        raise ValueError("observation_hash must be 64-hex text")
    if isinstance(decision_index, bool) or int(decision_index) < 0:
        raise ValueError("decision_index must be >= 0")
    return content_sha256(
        {
            "contract_id": OBSERVATION_CONTRACT_ID,
            "account_lineage_id": account_lineage_id,
            "decision_index": int(decision_index),
            "environment_time": environment_time,
            "observation_hash": observation_hash,
        }
    )


def observation_logical_id_v1(fact: PostCCObservationFactV1) -> str:
    """Identity of a decision observation: one lineage/decision/time/hash tuple."""
    fact.validate()
    return observation_logical_id_from_identity_v1(
        account_lineage_id=fact.account_lineage_id,
        decision_index=int(fact.decision_index),
        environment_time=fact.environment_time,
        observation_hash=fact.observation_hash,
    )


def observation_content_sha256_v1(fact: PostCCObservationFactV1) -> str:
    return hashlib.sha256(encode_observation_fact_v1(fact)).hexdigest()


def canonical_observation_vectors_v1(fact: PostCCObservationFactV1) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return the canonical (market, account, execution) Brain inputs from a fact."""
    if fact.observation_schema != CANONICAL_OBSERVATION_SCHEMA_V1:
        raise ValueError("CANONICAL_OBSERVATION_SCHEMA_MISMATCH")
    fact.validate()
    market = _canonical_payload_values("market_payload", fact.market_payload, CANONICAL_MARKET_DIM_V1)
    account = _canonical_payload_values("account_payload", fact.account_payload, CANONICAL_ACCOUNT_DIM_V1)
    execution = _canonical_payload_values("execution_payload", fact.execution_payload, CANONICAL_EXECUTION_DIM_V1)
    return market, account, execution


def build_canonical_observation_fact_v1(
    *,
    science_semantic_version: str,
    market_values: tuple[float, ...] | list[float],
    account_values: tuple[float, ...] | list[float],
    execution_values: tuple[float, ...] | list[float],
    market_source_identity: str,
    market_source_version: str,
    market_visible_through_time: str,
    account_lineage_id: str,
    decision_index: int,
    environment_time: str,
    normalizer_identity: str = CANONICAL_NORMALIZER_IDENTITY_V1,
    observation_schema: str = CANONICAL_OBSERVATION_SCHEMA_V1,
) -> PostCCObservationFactV1:
    """Build a fact from already-canonicalized numeric Brain inputs."""
    market = tuple(_require_finite_number(f"market_values[{i}]", value) for i, value in enumerate(market_values))
    account = tuple(_require_finite_number(f"account_values[{i}]", value) for i, value in enumerate(account_values))
    execution = tuple(_require_finite_number(f"execution_values[{i}]", value) for i, value in enumerate(execution_values))
    if len(market) != CANONICAL_MARKET_DIM_V1:
        raise ValueError(f"market_values must have dimension {CANONICAL_MARKET_DIM_V1}")
    if len(account) != CANONICAL_ACCOUNT_DIM_V1:
        raise ValueError(f"account_values must have dimension {CANONICAL_ACCOUNT_DIM_V1}")
    if len(execution) != CANONICAL_EXECUTION_DIM_V1:
        raise ValueError(f"execution_values must have dimension {CANONICAL_EXECUTION_DIM_V1}")
    if isinstance(decision_index, bool) or int(decision_index) < 0:
        raise ValueError("decision_index must be >= 0")
    fact = build_observation_fact(
        science_semantic_version=_require_nonempty_text("science_semantic_version", science_semantic_version),
        observation_schema=_require_nonempty_text("observation_schema", observation_schema),
        market_payload={CANONICAL_PAYLOAD_KEY_V1: list(market)},
        account_payload={CANONICAL_PAYLOAD_KEY_V1: list(account)},
        execution_payload={CANONICAL_PAYLOAD_KEY_V1: list(execution)},
        market_source_identity=_require_nonempty_text("market_source_identity", market_source_identity),
        market_source_version=_require_nonempty_text("market_source_version", market_source_version),
        market_visible_through_time=_require_nonempty_text("market_visible_through_time", market_visible_through_time),
        account_lineage_id=_require_nonempty_text("account_lineage_id", account_lineage_id),
        decision_index=int(decision_index),
        environment_time=_require_nonempty_text("environment_time", environment_time),
        normalizer_identity=_require_nonempty_text("normalizer_identity", normalizer_identity),
    )
    for name in _CANONICAL_PAYLOAD_NAMES:
        _check_no_future_information(name, getattr(fact, name))
    return fact


def canonical_environment_time_text_v1(environment_time: Any) -> str:
    """Canonical zero-padded text for a W-01 integer / persisted string time."""
    if isinstance(environment_time, bool):
        raise ValueError("environment_time must be an integer or canonical text")
    if isinstance(environment_time, int):
        if environment_time < 0:
            raise ValueError("environment_time must be >= 0")
        return f"{environment_time:020d}"
    if isinstance(environment_time, str):
        text = environment_time.strip()
        if text.isdigit():
            return f"{int(text):020d}"
        return text
    raise ValueError("environment_time must be an integer or canonical text")


def assert_w01_observation_identity_v1(decision: Any, fact: PostCCObservationFactV1) -> None:
    """Fail closed unless the W-01 decision carries this fact's observation hash."""
    fact.validate()
    decision_hash = getattr(decision, "observation_hash", None)
    if decision_hash != fact.observation_hash:
        raise ValueError("W01_OBSERVATION_HASH_MISMATCH")
    if getattr(decision, "account_lineage_id", None) != fact.account_lineage_id:
        raise ValueError("W01_OBSERVATION_LINEAGE_MISMATCH")
    if int(getattr(decision, "decision_index", -1)) != int(fact.decision_index):
        raise ValueError("W01_OBSERVATION_DECISION_INDEX_MISMATCH")
    environment_time = getattr(decision, "environment_time", None)
    if environment_time is None:
        raise ValueError("W01_OBSERVATION_ENVIRONMENT_TIME_MISSING")
    if canonical_environment_time_text_v1(environment_time) != fact.environment_time:
        raise ValueError("W01_OBSERVATION_ENVIRONMENT_TIME_MISMATCH")
