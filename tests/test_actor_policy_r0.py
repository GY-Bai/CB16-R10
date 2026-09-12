from __future__ import annotations

from dataclasses import replace

import pytest

from cb16_local_opt.action_contract_r0 import LONG, FLAT, make_target_position_action_r0
from cb16_local_opt.actor_critic_contract_r0 import ACTOR_DISTRIBUTION_VERSION_R0
from cb16_local_opt.actor_policy_r0 import (
    ACTOR_POLICY_DISTRIBUTION_DESCRIPTOR_SCHEMA_R0,
    ACTOR_POLICY_INTERFACE_VERSION_R0,
    ActorPolicyDistributionR0,
    ActorPolicyR0,
    make_actor_policy_distribution_r0,
    validate_actor_policy_interface_r0,
)


POLICY_HASH = "3" * 64


class CountingRng:
    def __init__(self) -> None:
        self.draws = 0

    def draw(self) -> int:
        self.draws += 1
        return self.draws


class StubActorPolicy(ActorPolicyR0[dict[str, float], CountingRng, float]):
    def __init__(self) -> None:
        self._descriptor = make_actor_policy_distribution_r0(
            policy_id="policy-r0",
            policy_version="policy-version-r0",
            policy_hash=POLICY_HASH,
        )

    @property
    def distribution_descriptor(self) -> ActorPolicyDistributionR0:
        return self._descriptor

    def sample(
        self,
        observation: dict[str, float],
        *,
        rng: CountingRng,
    ):
        draw = rng.draw()
        return make_target_position_action_r0(
            action_id=f"sample-{draw}",
            policy_id=self._descriptor.policy_id,
            policy_version=self._descriptor.policy_version,
            target_direction=LONG,
            requested_target_risk=0.25,
        )

    def log_prob(self, observation: dict[str, float], action):
        action.validate()
        return -0.25

    def deterministic_action(self, observation: dict[str, float]):
        return make_target_position_action_r0(
            action_id="deterministic",
            policy_id=self._descriptor.policy_id,
            policy_version=self._descriptor.policy_version,
            target_direction=FLAT,
            requested_target_risk=0.0,
        )


class BadSampleInterfacePolicy(StubActorPolicy):
    def sample(self, observation: dict[str, float]):  # type: ignore[override]
        return self.deterministic_action(observation)


class BadDeterministicInterfacePolicy(StubActorPolicy):
    def deterministic_action(  # type: ignore[override]
        self,
        observation: dict[str, float],
        *,
        rng=None,
    ):
        return super().deterministic_action(observation)


class BadLogProbInterfacePolicy(StubActorPolicy):
    def log_prob(  # type: ignore[override]
        self,
        observation: dict[str, float],
        action,
        *,
        rng=None,
    ):
        return super().log_prob(observation, action)


def test_distribution_descriptor_freezes_policy_and_science_versions() -> None:
    descriptor = make_actor_policy_distribution_r0(
        policy_id="policy-r0",
        policy_version="weights-v1",
        policy_hash=POLICY_HASH,
    )

    assert descriptor.to_payload() == {
        "schema_version": ACTOR_POLICY_DISTRIBUTION_DESCRIPTOR_SCHEMA_R0,
        "interface_version": ACTOR_POLICY_INTERFACE_VERSION_R0,
        "actor_distribution_version": ACTOR_DISTRIBUTION_VERSION_R0,
        "policy_id": "policy-r0",
        "policy_version": "weights-v1",
        "policy_hash": POLICY_HASH,
    }


@pytest.mark.parametrize(
    ("field_name", "bad_value", "error"),
    [
        ("schema_version", "wrong", "ACPOL_DISTRIBUTION_SCHEMA_VERSION_MISMATCH"),
        ("interface_version", "wrong", "ACPOL_INTERFACE_VERSION_MISMATCH"),
        (
            "actor_distribution_version",
            "wrong",
            "ACPOL_ACTOR_DISTRIBUTION_VERSION_MISMATCH",
        ),
        ("policy_id", "", "ACPOL_POLICY_ID_INVALID"),
        ("policy_version", "", "ACPOL_POLICY_VERSION_INVALID"),
        ("policy_hash", "ABC", "ACPOL_POLICY_HASH_INVALID"),
    ],
)
def test_distribution_descriptor_fails_closed_on_invalid_identity(
    field_name: str,
    bad_value: str,
    error: str,
) -> None:
    descriptor = make_actor_policy_distribution_r0(
        policy_id="policy-r0",
        policy_version="weights-v1",
        policy_hash=POLICY_HASH,
    )
    candidate = replace(descriptor, **{field_name: bad_value})

    with pytest.raises(RuntimeError, match=error):
        candidate.validate()


def test_sample_log_prob_and_deterministic_action_are_distinct_interfaces() -> None:
    policy = StubActorPolicy()
    descriptor = validate_actor_policy_interface_r0(policy)

    assert descriptor is policy.distribution_descriptor
    assert type(policy).sample is not type(policy).log_prob
    assert type(policy).sample is not type(policy).deterministic_action
    assert type(policy).log_prob is not type(policy).deterministic_action


def test_deterministic_evaluation_and_log_prob_do_not_consume_sampling_rng() -> None:
    policy = StubActorPolicy()
    validate_actor_policy_interface_r0(policy)
    rng = CountingRng()
    observation = {"feature": 1.0}

    deterministic = policy.deterministic_action(observation)
    assert deterministic.target_direction == FLAT
    assert rng.draws == 0

    assert policy.log_prob(observation, deterministic) == -0.25
    assert rng.draws == 0

    sampled = policy.sample(observation, rng=rng)
    assert sampled.target_direction == LONG
    assert rng.draws == 1

    # Evaluation after stochastic collection still receives no RNG handle.
    policy.deterministic_action(observation)
    assert rng.draws == 1


def test_interface_validator_requires_explicit_rng_only_on_sample() -> None:
    with pytest.raises(RuntimeError, match="ACPOL_SAMPLE_INTERFACE_INVALID"):
        validate_actor_policy_interface_r0(BadSampleInterfacePolicy())

    with pytest.raises(RuntimeError, match="ACPOL_DETERMINISTIC_INTERFACE_INVALID"):
        validate_actor_policy_interface_r0(BadDeterministicInterfacePolicy())

    with pytest.raises(RuntimeError, match="ACPOL_LOG_PROB_INTERFACE_INVALID"):
        validate_actor_policy_interface_r0(BadLogProbInterfacePolicy())


def test_interface_validator_rejects_missing_distribution_descriptor() -> None:
    class MissingDescriptor:
        def sample(self, observation, *, rng):
            raise AssertionError("not called")

        def log_prob(self, observation, action):
            raise AssertionError("not called")

        def deterministic_action(self, observation):
            raise AssertionError("not called")

    with pytest.raises(RuntimeError, match="ACPOL_DISTRIBUTION_DESCRIPTOR_REQUIRED"):
        validate_actor_policy_interface_r0(MissingDescriptor())
