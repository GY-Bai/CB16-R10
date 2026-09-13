from __future__ import annotations

from dataclasses import replace

import pytest

from cb16_local_opt.cc_runtime_wire_r0 import CCPolicyDecisionV1
from cb16_local_opt.post_cc_observation_fact_v1 import (
    CANONICAL_ACCOUNT_DIM_V1,
    CANONICAL_EXECUTION_DIM_V1,
    CANONICAL_MARKET_DIM_V1,
    FORBIDDEN_FUTURE_INFORMATION_KEY_FRAGMENTS_V1,
    assert_w01_observation_identity_v1,
    build_canonical_observation_fact_v1,
    canonical_observation_bytes_v1,
    canonical_observation_vectors_v1,
    decode_observation_fact_v1,
    observation_content_sha256_v1,
    observation_logical_id_v1,
)
from cb16_local_opt.post_cc_observation_contract_v1 import PostCCObservationFactV1


def fact() -> PostCCObservationFactV1:
    return build_canonical_observation_fact_v1(
        science_semantic_version="CB16_R11_POST_CC_SCIENCE_SEMANTIC_V1",
        market_values=(0.02, 1.0),
        account_values=(1.0, 0.0, 0.0),
        execution_values=(0.0, 0.0),
        market_source_identity="SYN",
        market_source_version="V1",
        market_visible_through_time="00000000000000000003",
        account_lineage_id="acct-1",
        decision_index=3,
        environment_time="00000000000000000003",
    )


def test_canonical_dimensions_and_vector_round_trip():
    assert (CANONICAL_MARKET_DIM_V1, CANONICAL_ACCOUNT_DIM_V1, CANONICAL_EXECUTION_DIM_V1) == (2, 3, 2)
    market, account, execution = canonical_observation_vectors_v1(fact())
    assert market == (0.02, 1.0)
    assert account == (1.0, 0.0, 0.0)
    assert execution == (0.0, 0.0)


def test_deterministic_serialization_and_round_trip():
    first = fact()
    second = fact()
    assert first == second
    assert observation_logical_id_v1(first) == observation_logical_id_v1(second)
    assert observation_content_sha256_v1(first) == observation_content_sha256_v1(second)
    bytes_a = canonical_observation_bytes_v1(first)
    bytes_b = canonical_observation_bytes_v1(second)
    assert bytes_a == bytes_b
    assert decode_observation_fact_v1(bytes_a) == first


def test_same_semantic_observation_same_hash_and_payload_change_changes_hash():
    first = fact()
    changed = build_canonical_observation_fact_v1(
        science_semantic_version="CB16_R11_POST_CC_SCIENCE_SEMANTIC_V1",
        market_values=(0.03, 1.0),
        account_values=(1.0, 0.0, 0.0),
        execution_values=(0.0, 0.0),
        market_source_identity="SYN",
        market_source_version="V1",
        market_visible_through_time="00000000000000000003",
        account_lineage_id="acct-1",
        decision_index=3,
        environment_time="00000000000000000003",
    )
    assert first.observation_hash != changed.observation_hash
    assert observation_logical_id_v1(first) != observation_logical_id_v1(changed)


def test_payload_and_hash_corruption_fail_closed():
    good = fact()
    corrupted = replace(good, account_payload={"values": [999.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="OBSERVATION_HASH_MISMATCH"):
        corrupted.validate()
    with pytest.raises(ValueError, match="OBSERVATION_HASH_MISMATCH"):
        canonical_observation_bytes_v1(corrupted)


@pytest.mark.parametrize(
    "payload",
    (
        {"values": [0.0, 1.0], "future_return": 1.0},
        {"values": [0.0, 1.0], "nested": {"oracle": 1.0}},
        {"values": [0.0, 1.0], "labels": [1.0]},
    ),
)
def test_future_information_fields_fail_closed_at_any_depth(payload):
    from cb16_local_opt.post_cc_observation_fact_v1 import _check_no_future_information  # type: ignore

    with pytest.raises(ValueError, match="FUTURE_INFORMATION_FIELD_FORBIDDEN"):
        _check_no_future_information("payload", payload)


def test_w01_observation_identity_agreement_and_mismatch():
    good = fact()
    decision = CCPolicyDecisionV1(
        science_semantic_version="CB16_R11_CC_SCIENCE_SEMANTIC_V1",
        account_lineage_id=good.account_lineage_id,
        decision_index=good.decision_index,
        environment_time=3,
        policy_generation="0",
        policy_id="p",
        policy_sha256="a" * 64,
        observation_schema=good.observation_schema,
        observation_hash=good.observation_hash,
        normalizer_id=good.normalizer_identity,
        nominal_direction="LONG",
        nominal_target_risk=0.4,
        log_mu=-1.0,
        risk_measure_kind="continuous_density",
        rng_stream_id="rng",
        rng_position_or_counter=0,
    )
    assert_w01_observation_identity_v1(decision, good)
    with pytest.raises(ValueError, match="W01_OBSERVATION_HASH_MISMATCH"):
        assert_w01_observation_identity_v1(replace(decision, observation_hash="b" * 64), good)
    with pytest.raises(ValueError, match="W01_OBSERVATION_ENVIRONMENT_TIME_MISMATCH"):
        assert_w01_observation_identity_v1(replace(decision, environment_time=4), good)
