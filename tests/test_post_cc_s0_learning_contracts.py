from dataclasses import replace

import pytest

from cb16_local_opt.post_cc_joint_replay_contract_v1 import PostCCJointReplaySampleV1
from cb16_local_opt.post_cc_observation_contract_v1 import build_observation_fact


def observation():
    return build_observation_fact(
        science_semantic_version="CB16_R11_POST_CC_SCIENCE_SEMANTIC_V1",
        observation_schema="POST_CC_SYNTHETIC_OBSERVATION_V1",
        market_payload={"x": [1.0, 2.0]},
        account_payload={"equity": 1000.0},
        execution_payload={"fees": 0.0},
        market_source_identity="S1_SYNTHETIC",
        market_source_version="V1",
        market_visible_through_time="00000007",
        account_lineage_id="acct",
        decision_index=3,
        environment_time="00000007",
        normalizer_identity="NORM_V1",
    )


def test_observation_hash_is_content_addressed_and_corruption_fails_closed():
    obs = observation()
    assert len(obs.observation_hash) == 64
    obs.validate()
    with pytest.raises(ValueError, match="OBSERVATION_HASH_MISMATCH"):
        replace(obs, account_payload={"equity": 999.0}).validate()
    with pytest.raises(ValueError, match="FUTURE_MARKET_INFORMATION_FORBIDDEN"):
        build_observation_fact(
            science_semantic_version="s",
            observation_schema="o",
            market_payload={},
            account_payload={},
            execution_payload={},
            market_source_identity="m",
            market_source_version="v",
            market_visible_through_time="00000008",
            account_lineage_id="a",
            decision_index=0,
            environment_time="00000007",
            normalizer_identity="n",
        )


def test_joint_replay_preserves_nominal_action_true_log_mu_and_observation_link():
    obs = observation()
    sample = PostCCJointReplaySampleV1(
        "seq",
        "ref",
        "acct",
        3,
        "00000007",
        obs,
        "LONG",
        0.4,
        "TARGET_RISK",
        -1.2,
        "pi-g0",
        "0",
        0.03,
        0.99,
        "CONTINUE",
        None,
        1.0,
        "pi-target",
        ("a" * 64,),
        {"executed_direction": "FLAT"},
    ).validate()
    assert sample.nominal_direction == "LONG"
    assert sample.consequence_context["executed_direction"] == "FLAT"
    with pytest.raises(ValueError, match="FLAT nominal_target_risk"):
        replace(sample, nominal_direction="FLAT", nominal_target_risk=0.4).validate()
    with pytest.raises(ValueError, match="ACCOUNT_LINEAGE_MISMATCH"):
        replace(sample, account_lineage_id="other").validate()
