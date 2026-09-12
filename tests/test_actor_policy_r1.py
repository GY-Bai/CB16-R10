from __future__ import annotations

from dataclasses import fields, replace
import inspect

import pytest
import torch

import cb16_local_opt.actor_policy_r1 as actor_policy_module
from cb16_local_opt.account_economics_r0 import make_account_economics_state_r0
from cb16_local_opt.account_observation_r0 import project_account_observation_r0
from cb16_local_opt.action_contract_r1 import FLAT, make_target_position_action_r1
from cb16_local_opt.actor_observation_r0 import build_actor_observation_r0
from cb16_local_opt.actor_policy_r1 import (
    ACTOR_POLICY_INTERFACE_VERSION_R1,
    ActorPolicyDistributionIdentityR1,
    ActorPolicyR1,
    make_actor_policy_distribution_identity_r1,
    make_policy_tensor_contract_r1,
    validate_actor_policy_interface_r1,
    validate_actor_policy_observation_r1,
    validate_policy_action_binding_r1,
    validate_policy_tensor_r1,
)


class CountingRng:
    def __init__(self) -> None:
        self.calls = 0

    def consume(self) -> int:
        self.calls += 1
        return self.calls


def _account_observation():
    state = make_account_economics_state_r0(
        account_id="actor-policy-account",
        cash=100.0,
        position_quantity=0.0,
        position_cost_basis=0.0,
        mark_price=10.0,
        realized_pnl_cumulative=0.0,
        fees_cumulative=0.0,
        funding_cumulative=0.0,
        margin_collateral=0.0,
        liabilities=0.0,
        external_capital_flows_cumulative=100.0,
        economic_responsibility_open=True,
    )
    return project_account_observation_r0(state)


def _observation():
    return build_actor_observation_r0(
        market_sensory_version="MARKET_CAUSAL_V1",
        market_causal_features=(0.1, -0.2, 0.3),
        account_observation=_account_observation(),
        legal_execution_version="LEGAL_CAUSAL_V1",
        legal_execution_causal_features=(1.0, 0.5),
    )


def _identity(*, device_type="cpu", device_index=None, dtype="float32"):
    return make_actor_policy_distribution_identity_r1(
        distribution_id="round2-actor-distribution",
        distribution_version="INTERFACE_ONLY_BC034",
        policy_id="behavior-policy",
        policy_version="v1",
        policy_sha256="a" * 64,
        tensor_contract=make_policy_tensor_contract_r1(
            device_type=device_type,
            device_index=device_index,
            dtype=dtype,
        ),
    )


class StubPolicy(ActorPolicyR1):
    """Interface fixture only; BC-034 intentionally has no probability math."""

    def __init__(self, identity: ActorPolicyDistributionIdentityR1 | None = None) -> None:
        self._identity = _identity() if identity is None else identity

    @property
    def distribution_identity(self) -> ActorPolicyDistributionIdentityR1:
        return self._identity

    def _tensorize(self, observation) -> torch.Tensor:
        validate_actor_policy_observation_r1(self._identity, observation)
        tensor = torch.tensor(
            observation.market_causal_features,
            dtype=torch.float32,
            device="cpu",
        )
        return validate_policy_tensor_r1(tensor, self._identity.tensor_contract)

    def sample(self, observation, *, rng: CountingRng):
        self._tensorize(observation)
        rng.consume()
        action = make_target_position_action_r1(
            action_id=f"sample-{rng.calls}",
            policy_id=self._identity.policy_id,
            policy_version=self._identity.policy_version,
            target_direction=FLAT,
            requested_target_risk=0.0,
        )
        return validate_policy_action_binding_r1(self._identity, action)

    def log_prob(self, observation, action):
        self._tensorize(observation)
        validate_policy_action_binding_r1(self._identity, action)
        # Placeholder interface value only. BC-035–038 own real likelihood math.
        value = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        return validate_policy_tensor_r1(value, self._identity.tensor_contract)

    def deterministic_action(self, observation):
        self._tensorize(observation)
        action = make_target_position_action_r1(
            action_id="deterministic",
            policy_id=self._identity.policy_id,
            policy_version=self._identity.policy_version,
            target_direction=FLAT,
            requested_target_risk=0.0,
        )
        return validate_policy_action_binding_r1(self._identity, action)


def test_sample_scoring_and_deterministic_evaluation_are_distinct_apis() -> None:
    policy = StubPolicy()
    identity = validate_actor_policy_interface_r1(policy)
    assert identity.interface_version == ACTOR_POLICY_INTERFACE_VERSION_R1

    rng = CountingRng()
    observation = _observation()
    sampled = policy.sample(observation, rng=rng)
    assert rng.calls == 1
    score = policy.log_prob(observation, sampled)
    assert rng.calls == 1
    deterministic = policy.deterministic_action(observation)
    assert rng.calls == 1
    assert sampled.action_id.startswith("sample-")
    assert deterministic.action_id == "deterministic"
    assert score.item() == 0.0


def test_interface_validator_freezes_exact_rng_boundary() -> None:
    policy = StubPolicy()
    validate_actor_policy_interface_r1(policy)
    assert tuple(inspect.signature(policy.sample).parameters) == ("observation", "rng")
    assert tuple(inspect.signature(policy.log_prob).parameters) == ("observation", "action")
    assert tuple(inspect.signature(policy.deterministic_action).parameters) == ("observation",)
    assert inspect.signature(policy.sample).parameters["rng"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "rng" not in inspect.signature(policy.log_prob).parameters
    assert "rng" not in inspect.signature(policy.deterministic_action).parameters


def test_bad_scoring_or_evaluation_rng_signature_fails_closed() -> None:
    identity = _identity()

    class BadLogProb:
        distribution_identity = identity

        def sample(self, observation, *, rng):
            return None

        def log_prob(self, observation, action, rng=None):
            return None

        def deterministic_action(self, observation):
            return None

    with pytest.raises(RuntimeError, match="LOG_PROB_INTERFACE_INVALID"):
        validate_actor_policy_interface_r1(BadLogProb())

    class BadDeterministic:
        distribution_identity = identity

        def sample(self, observation, *, rng):
            return None

        def log_prob(self, observation, action):
            return None

        def deterministic_action(self, observation, *, rng=None):
            return None

    with pytest.raises(RuntimeError, match="DETERMINISTIC_INTERFACE_INVALID"):
        validate_actor_policy_interface_r1(BadDeterministic())


def test_varargs_or_implicit_sampling_rng_fail_closed() -> None:
    identity = _identity()

    class BadSample:
        distribution_identity = identity

        def sample(self, observation, rng):
            return None

        def log_prob(self, observation, action):
            return None

        def deterministic_action(self, observation):
            return None

    with pytest.raises(RuntimeError, match="SAMPLE_RNG_INTERFACE_INVALID"):
        validate_actor_policy_interface_r1(BadSample())

    class VarArgs:
        distribution_identity = identity

        def sample(self, observation, *, rng):
            return None

        def log_prob(self, *args):
            return None

        def deterministic_action(self, observation):
            return None

    with pytest.raises(RuntimeError, match="LOG_PROB_INTERFACE_INVALID"):
        validate_actor_policy_interface_r1(VarArgs())


def test_distribution_identity_is_versioned_and_semantically_hashed() -> None:
    identity = _identity()
    identity.validate()
    assert len(identity.semantic_sha256) == 64
    assert identity.semantic_sha256 == _identity().semantic_sha256
    assert identity.tensor_contract.semantic_sha256 == _identity().tensor_contract.semantic_sha256

    with pytest.raises(RuntimeError, match="DISTRIBUTION_SCIENCE_VERSION_MISMATCH"):
        replace(identity, actor_distribution_version="OLD").validate()
    with pytest.raises(RuntimeError, match="OBSERVATION_SCIENCE_VERSION_MISMATCH"):
        replace(identity, observation_science_version="OLD").validate()
    with pytest.raises(RuntimeError, match="OBSERVATION_SCHEMA_VERSION_MISMATCH"):
        replace(identity, observation_schema_version="OLD").validate()
    with pytest.raises(RuntimeError, match="ACTION_SCHEMA_VERSION_MISMATCH"):
        replace(identity, action_schema_version="OLD").validate()
    with pytest.raises(RuntimeError, match="POLICY_HASH_INVALID"):
        replace(identity, policy_sha256="not-a-sha").validate()


def test_bc034_identity_does_not_predefine_bc035_or_bc036_distribution_math() -> None:
    identity_fields = {field.name for field in fields(ActorPolicyDistributionIdentityR1)}
    premature_fields = {
        "logits",
        "direction_logits",
        "direction_probs",
        "short_probability",
        "flat_probability",
        "long_probability",
        "risk_alpha",
        "risk_beta",
        "endpoint_mass_zero",
        "endpoint_mass_one",
    }
    assert premature_fields.isdisjoint(identity_fields)


def test_incompatible_actor_observation_version_fails_closed() -> None:
    identity = _identity()
    observation = _observation()
    validate_actor_policy_observation_r1(identity, observation)
    incompatible = replace(observation, schema_version="INCOMPATIBLE_OBSERVATION")
    with pytest.raises(RuntimeError):
        validate_actor_policy_observation_r1(identity, incompatible)
    with pytest.raises(RuntimeError, match="OBSERVATION_TYPE_INVALID"):
        validate_actor_policy_observation_r1(identity, {"market": (1.0,)})


def test_tensor_device_dtype_and_finiteness_contracts_fail_closed() -> None:
    cpu32 = make_policy_tensor_contract_r1(device_type="cpu", dtype="float32")
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert validate_policy_tensor_r1(tensor, cpu32) is tensor

    with pytest.raises(RuntimeError, match="TENSOR_DTYPE_MISMATCH"):
        validate_policy_tensor_r1(torch.tensor([1.0], dtype=torch.float64), cpu32)
    with pytest.raises(RuntimeError, match="TENSOR_DTYPE_MISMATCH"):
        validate_policy_tensor_r1(torch.tensor([1], dtype=torch.int64), cpu32)
    with pytest.raises(RuntimeError, match="TENSOR_NONFINITE"):
        validate_policy_tensor_r1(torch.tensor([float("nan")], dtype=torch.float32), cpu32)
    with pytest.raises(RuntimeError, match="TENSOR_TYPE_INVALID"):
        validate_policy_tensor_r1([1.0], cpu32)

    cuda_contract = make_policy_tensor_contract_r1(device_type="cuda", device_index=0, dtype="float32")
    with pytest.raises(RuntimeError, match="TENSOR_DEVICE_MISMATCH"):
        validate_policy_tensor_r1(tensor, cuda_contract)


def test_tensor_contract_rejects_noncanonical_or_unsupported_dtype_and_cpu_index() -> None:
    with pytest.raises(RuntimeError, match="DTYPE_INVALID"):
        make_policy_tensor_contract_r1(device_type="cpu", dtype="int64")
    with pytest.raises(RuntimeError, match="CPU_DEVICE_INDEX_FORBIDDEN"):
        make_policy_tensor_contract_r1(device_type="cpu", device_index=0, dtype="float32")
    with pytest.raises(RuntimeError, match="DEVICE_INDEX_INVALID"):
        make_policy_tensor_contract_r1(device_type="cuda", device_index=-1, dtype="float32")


def test_action_binding_requires_exact_behavior_policy_identity() -> None:
    identity = _identity()
    action = make_target_position_action_r1(
        action_id="a",
        policy_id=identity.policy_id,
        policy_version=identity.policy_version,
        target_direction=FLAT,
        requested_target_risk=0.0,
    )
    validate_policy_action_binding_r1(identity, action)
    with pytest.raises(RuntimeError, match="POLICY_ID_MISMATCH"):
        validate_policy_action_binding_r1(identity, replace(action, policy_id="other-policy"))
    with pytest.raises(RuntimeError, match="POLICY_VERSION_MISMATCH"):
        validate_policy_action_binding_r1(identity, replace(action, policy_version="other-version"))


def test_r1_policy_module_does_not_depend_on_unmerged_r0_candidate() -> None:
    source = inspect.getsource(actor_policy_module)
    assert "actor_policy_r0" not in source
    assert "AC-015" not in source
