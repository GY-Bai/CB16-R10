from __future__ import annotations

from collections import Counter
from dataclasses import fields, replace
import inspect
import math

import pytest
import torch

import cb16_local_opt.actor_policy_r1 as actor_policy_module
from cb16_local_opt.account_economics_r0 import make_account_economics_state_r0
from cb16_local_opt.account_observation_r0 import project_account_observation_r0
from cb16_local_opt.action_contract_r1 import FLAT, LONG, SHORT, make_target_position_action_r1
from cb16_local_opt.actor_observation_r0 import build_actor_observation_r0
from cb16_local_opt.actor_policy_r1 import (
    ACTOR_POLICY_INTERFACE_VERSION_R1,
    CONTINUOUS_DENSITY,
    DIRECTION_ORDER_R1,
    POINT_MASS,
    RISK_ENDPOINT_POLICY_R1,
    ActorPolicyDistributionIdentityR1,
    ActorPolicyR1,
    ConditionalBoundedRiskR1,
    DirectionCategoricalR1,
    JointNominalActionLikelihoodR1,
    RiskLikelihoodTermR1,
    SampledNominalActionR1,
    joint_nominal_action_likelihood_r1,
    make_actor_policy_distribution_identity_r1,
    make_conditional_bounded_risk_r1,
    make_direction_categorical_r1,
    make_policy_tensor_contract_r1,
    recompute_sampled_log_mu_r1,
    sample_nominal_action_r1,
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
        distribution_version="JOINT_NOMINAL_ACTION_BC038",
        policy_id="behavior-policy",
        policy_version="v1",
        policy_sha256="a" * 64,
        tensor_contract=make_policy_tensor_contract_r1(
            device_type=device_type,
            device_index=device_index,
            dtype=dtype,
        ),
    )


def _direction_distribution(probabilities=(0.2, 0.3, 0.5)) -> DirectionCategoricalR1:
    contract = make_policy_tensor_contract_r1(device_type="cpu", dtype="float32")
    probs = torch.tensor(probabilities, dtype=torch.float32)
    return make_direction_categorical_r1(logits=torch.log(probs), tensor_contract=contract)


def _conditional_risk(
    *,
    short_location: float = -0.75,
    short_log_scale: float = math.log(0.5),
    long_location: float = 0.75,
    long_log_scale: float = math.log(0.5),
) -> ConditionalBoundedRiskR1:
    contract = make_policy_tensor_contract_r1(device_type="cpu", dtype="float32")
    return make_conditional_bounded_risk_r1(
        short_location=torch.tensor(short_location, dtype=torch.float32),
        short_log_scale=torch.tensor(short_log_scale, dtype=torch.float32),
        long_location=torch.tensor(long_location, dtype=torch.float32),
        long_log_scale=torch.tensor(long_log_scale, dtype=torch.float32),
        tensor_contract=contract,
    )


class StubPolicy(ActorPolicyR1):
    """Interface fixture; component-level joint likelihood is tested separately."""

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


def test_bc034_identity_does_not_predefine_distribution_parameters() -> None:
    identity_fields = {field.name for field in fields(ActorPolicyDistributionIdentityR1)}
    parameter_fields = {
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
    assert parameter_fields.isdisjoint(identity_fields)


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


def test_direction_categorical_has_exact_canonical_order_and_known_probabilities() -> None:
    distribution = _direction_distribution()
    assert DIRECTION_ORDER_R1 == (SHORT, FLAT, LONG)
    expected = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    assert torch.allclose(distribution.probabilities, expected, rtol=1e-6, atol=1e-7)
    for direction, probability in zip(DIRECTION_ORDER_R1, expected):
        assert torch.allclose(
            distribution.log_prob(direction),
            torch.log(probability),
            rtol=1e-6,
            atol=1e-7,
        )


def test_direction_categorical_extreme_logits_keep_probabilities_and_log_probs_finite() -> None:
    contract = make_policy_tensor_contract_r1(device_type="cpu", dtype="float32")
    distribution = make_direction_categorical_r1(
        logits=torch.tensor([1000.0, 0.0, -1000.0], dtype=torch.float32),
        tensor_contract=contract,
    )
    probabilities = distribution.probabilities
    log_probabilities = distribution.log_probabilities
    assert bool(torch.isfinite(probabilities).all().item())
    assert bool(torch.isfinite(log_probabilities).all().item())
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0, dtype=torch.float32))
    assert probabilities[0].item() == 1.0
    assert log_probabilities.tolist() == [0.0, -1000.0, -2000.0]


def test_direction_categorical_log_prob_retains_gradient_path_to_logits() -> None:
    contract = make_policy_tensor_contract_r1(device_type="cpu", dtype="float32")
    logits = torch.tensor([-0.5, 0.1, 0.7], dtype=torch.float32, requires_grad=True)
    distribution = make_direction_categorical_r1(logits=logits, tensor_contract=contract)
    loss = -distribution.log_prob(LONG)
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all().item())
    assert bool((logits.grad != 0).any().item())


def test_direction_categorical_sampled_frequencies_match_known_probabilities() -> None:
    distribution = _direction_distribution()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260912)
    sample_count = 50_000
    samples = distribution.sample_directions(sample_count, generator=generator)
    counts = Counter(samples)
    expected = {SHORT: 0.2, FLAT: 0.3, LONG: 0.5}
    assert len(samples) == sample_count
    assert set(counts) == set(DIRECTION_ORDER_R1)
    for direction, probability in expected.items():
        frequency = counts[direction] / sample_count
        assert abs(frequency - probability) < 0.01


def test_direction_categorical_rejects_bad_shape_direction_count_and_generator() -> None:
    contract = make_policy_tensor_contract_r1(device_type="cpu", dtype="float32")
    with pytest.raises(RuntimeError, match="DIRECTION_LOGITS_SHAPE_INVALID"):
        make_direction_categorical_r1(
            logits=torch.tensor([0.0, 1.0], dtype=torch.float32),
            tensor_contract=contract,
        )
    with pytest.raises(RuntimeError, match="DIRECTION_LOGITS_SHAPE_INVALID"):
        make_direction_categorical_r1(
            logits=torch.zeros((1, 3), dtype=torch.float32),
            tensor_contract=contract,
        )
    with pytest.raises(RuntimeError, match="TENSOR_NONFINITE"):
        make_direction_categorical_r1(
            logits=torch.tensor([0.0, float("inf"), 1.0], dtype=torch.float32),
            tensor_contract=contract,
        )

    distribution = _direction_distribution()
    with pytest.raises(RuntimeError, match="DIRECTION_INVALID"):
        distribution.log_prob("HOLD")
    generator = torch.Generator(device="cpu")
    with pytest.raises(RuntimeError, match="DIRECTION_SAMPLE_COUNT_INVALID"):
        distribution.sample_directions(0, generator=generator)
    with pytest.raises(RuntimeError, match="GENERATOR_INVALID"):
        distribution.sample_directions(1, generator=object())


def test_bc035_direction_distribution_does_not_predefine_bc036_risk_math() -> None:
    distribution_fields = {field.name for field in fields(DirectionCategoricalR1)}
    forbidden = {
        "risk",
        "risk_alpha",
        "risk_beta",
        "risk_logits",
        "endpoint_mass_zero",
        "endpoint_mass_one",
    }
    assert forbidden.isdisjoint(distribution_fields)


def test_conditional_risk_samples_are_strictly_bounded_and_direction_conditioned() -> None:
    distribution = _conditional_risk()
    short_generator = torch.Generator(device="cpu")
    long_generator = torch.Generator(device="cpu")
    short_generator.manual_seed(1234)
    long_generator.manual_seed(1234)

    short_first = distribution.sample_risk(SHORT, generator=short_generator)
    long_first = distribution.sample_risk(LONG, generator=long_generator)
    assert 0.0 < short_first.item() < 1.0
    assert 0.0 < long_first.item() < 1.0
    assert long_first.item() > short_first.item()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260912)
    for direction in (SHORT, LONG):
        for _ in range(256):
            sample = distribution.sample_risk(direction, generator=generator)
            assert bool(torch.isfinite(sample).item())
            assert 0.0 < sample.item() < 1.0


def test_flat_risk_is_exact_zero_and_consumes_no_random_draw() -> None:
    distribution = _conditional_risk()
    with_flat = torch.Generator(device="cpu")
    direct = torch.Generator(device="cpu")
    with_flat.manual_seed(777)
    direct.manual_seed(777)

    flat = distribution.sample_risk(FLAT, generator=with_flat)
    after_flat = distribution.sample_risk(LONG, generator=with_flat)
    direct_long = distribution.sample_risk(LONG, generator=direct)
    assert flat.item() == 0.0
    assert torch.equal(after_flat, direct_long)
    assert distribution.log_prob(FLAT, torch.tensor(0.0, dtype=torch.float32)).item() == 0.0
    with pytest.raises(RuntimeError, match="FLAT_RISK_MUST_BE_ZERO"):
        distribution.log_prob(FLAT, torch.tensor(0.1, dtype=torch.float32))


def test_conditional_risk_known_answer_log_density_is_reconstructable() -> None:
    distribution = _conditional_risk(
        short_location=0.0,
        short_log_scale=0.0,
        long_location=0.0,
        long_log_scale=0.0,
    )
    risk = torch.tensor(0.5, dtype=torch.float32)
    expected = math.log(4.0) - 0.5 * math.log(2.0 * math.pi)
    for direction in (SHORT, LONG):
        term = distribution.likelihood_term(direction, risk)
        assert term.measure_kind == CONTINUOUS_DENSITY
        assert term.endpoint_policy == RISK_ENDPOINT_POLICY_R1
        assert term.log_likelihood.item() == pytest.approx(expected, rel=1e-6, abs=1e-6)
        assert torch.equal(term.log_likelihood, distribution.log_prob(direction, risk))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(99)
    sampled = distribution.sample_risk(LONG, generator=generator)
    reconstructed_a = distribution.log_prob(LONG, sampled)
    reconstructed_b = distribution.log_prob(LONG, sampled)
    assert torch.equal(reconstructed_a, reconstructed_b)


def test_bc037_endpoint_likelihood_known_answers() -> None:
    distribution = _conditional_risk()
    flat = distribution.likelihood_term(FLAT, torch.tensor(0.0, dtype=torch.float32))
    assert flat.measure_kind == POINT_MASS
    assert flat.endpoint_policy == RISK_ENDPOINT_POLICY_R1
    assert flat.log_likelihood.item() == 0.0

    for direction in (SHORT, LONG):
        for endpoint in (0.0, 1.0):
            risk = torch.tensor(endpoint, dtype=torch.float32)
            term = distribution.likelihood_term(direction, risk)
            assert term.measure_kind == POINT_MASS
            assert term.endpoint_policy == RISK_ENDPOINT_POLICY_R1
            assert term.log_likelihood.item() == float("-inf")
            assert distribution.log_prob(direction, risk).item() == float("-inf")


def test_bc037_endpoints_never_evaluate_continuous_density(monkeypatch) -> None:
    distribution = _conditional_risk()

    def forbidden_density(*args, **kwargs):
        raise AssertionError("continuous density evaluated at a point-mass endpoint")

    monkeypatch.setattr(ConditionalBoundedRiskR1, "_interior_log_density", forbidden_density)
    for direction in (SHORT, LONG):
        for endpoint in (0.0, 1.0):
            term = distribution.likelihood_term(
                direction,
                torch.tensor(endpoint, dtype=torch.float32),
            )
            assert term.measure_kind == POINT_MASS
            assert term.log_likelihood.item() == float("-inf")


def test_bc037_likelihood_term_tampering_fails_closed() -> None:
    distribution = _conditional_risk()
    flat = distribution.likelihood_term(FLAT, torch.tensor(0.0, dtype=torch.float32))
    with pytest.raises(RuntimeError, match="FLAT_POINT_MASS_REQUIRED"):
        replace(flat, measure_kind=CONTINUOUS_DENSITY).validate()
    with pytest.raises(RuntimeError, match="RISK_ENDPOINT_POLICY_MISMATCH"):
        replace(flat, endpoint_policy="OTHER").validate()

    endpoint = distribution.likelihood_term(LONG, torch.tensor(1.0, dtype=torch.float32))
    with pytest.raises(RuntimeError, match="NONFLAT_ENDPOINT_ZERO_MASS_REQUIRED"):
        replace(endpoint, log_likelihood=torch.tensor(0.0, dtype=torch.float32)).validate()
    with pytest.raises(RuntimeError, match="NONFLAT_ENDPOINT_ZERO_MASS_REQUIRED"):
        replace(endpoint, measure_kind=CONTINUOUS_DENSITY).validate()

    interior = distribution.likelihood_term(LONG, torch.tensor(0.5, dtype=torch.float32))
    with pytest.raises(RuntimeError, match="INTERIOR_DENSITY_REQUIRED"):
        replace(interior, measure_kind=POINT_MASS).validate()


def test_conditional_risk_rejects_bad_shapes_scale_overflow_and_generator() -> None:
    contract = make_policy_tensor_contract_r1(device_type="cpu", dtype="float32")
    scalar = torch.tensor(0.0, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="RISK_LOCATION_SHAPE_INVALID"):
        make_conditional_bounded_risk_r1(
            short_location=torch.tensor([0.0], dtype=torch.float32),
            short_log_scale=scalar,
            long_location=scalar,
            long_log_scale=scalar,
            tensor_contract=contract,
        )
    with pytest.raises(RuntimeError, match="RISK_LOG_SCALE_SHAPE_INVALID"):
        make_conditional_bounded_risk_r1(
            short_location=scalar,
            short_log_scale=torch.tensor([0.0], dtype=torch.float32),
            long_location=scalar,
            long_log_scale=scalar,
            tensor_contract=contract,
        )
    with pytest.raises(RuntimeError, match="RISK_SCALE_INVALID"):
        make_conditional_bounded_risk_r1(
            short_location=scalar,
            short_log_scale=torch.tensor(1000.0, dtype=torch.float32),
            long_location=scalar,
            long_log_scale=scalar,
            tensor_contract=contract,
        )

    distribution = _conditional_risk()
    with pytest.raises(RuntimeError, match="GENERATOR_INVALID"):
        distribution.sample_risk(LONG, generator=object())
    with pytest.raises(RuntimeError, match="DIRECTION_INVALID"):
        distribution.sample_risk("HOLD", generator=torch.Generator(device="cpu"))
    with pytest.raises(RuntimeError, match="RISK_OUT_OF_BOUNDS"):
        distribution.log_prob(LONG, torch.tensor(-0.1, dtype=torch.float32))


def test_conditional_risk_log_prob_retains_gradient_paths() -> None:
    contract = make_policy_tensor_contract_r1(device_type="cpu", dtype="float32")
    short_location = torch.tensor(-0.4, dtype=torch.float32, requires_grad=True)
    short_log_scale = torch.tensor(-0.2, dtype=torch.float32, requires_grad=True)
    long_location = torch.tensor(0.6, dtype=torch.float32, requires_grad=True)
    long_log_scale = torch.tensor(-0.3, dtype=torch.float32, requires_grad=True)
    distribution = make_conditional_bounded_risk_r1(
        short_location=short_location,
        short_log_scale=short_log_scale,
        long_location=long_location,
        long_log_scale=long_log_scale,
        tensor_contract=contract,
    )
    loss = -distribution.log_prob(LONG, torch.tensor(0.7, dtype=torch.float32))
    loss.backward()
    assert long_location.grad is not None
    assert long_log_scale.grad is not None
    assert bool(torch.isfinite(long_location.grad).item())
    assert bool(torch.isfinite(long_log_scale.grad).item())
    assert long_location.grad.item() != 0.0
    assert long_log_scale.grad.item() != 0.0
    assert short_location.grad is None
    assert short_log_scale.grad is None


def test_bc038_joint_known_answers_and_flat_has_no_risk_density() -> None:
    direction = _direction_distribution((0.2, 0.3, 0.5))
    risk = _conditional_risk(
        short_location=0.0,
        short_log_scale=0.0,
        long_location=0.0,
        long_log_scale=0.0,
    )
    identity = _identity()
    interior_risk_log = math.log(4.0) - 0.5 * math.log(2.0 * math.pi)

    long_action = make_target_position_action_r1(
        action_id="long-half",
        policy_id=identity.policy_id,
        policy_version=identity.policy_version,
        target_direction=LONG,
        requested_target_risk=0.5,
    )
    long_joint = joint_nominal_action_likelihood_r1(
        direction_distribution=direction,
        risk_distribution=risk,
        action=long_action,
    )
    assert isinstance(long_joint, JointNominalActionLikelihoodR1)
    assert long_joint.risk_likelihood.measure_kind == CONTINUOUS_DENSITY
    assert long_joint.joint_log_likelihood.item() == pytest.approx(
        math.log(0.5) + interior_risk_log,
        rel=1e-6,
        abs=1e-6,
    )

    flat_action = make_target_position_action_r1(
        action_id="flat",
        policy_id=identity.policy_id,
        policy_version=identity.policy_version,
        target_direction=FLAT,
        requested_target_risk=0.0,
    )
    flat_joint = joint_nominal_action_likelihood_r1(
        direction_distribution=direction,
        risk_distribution=risk,
        action=flat_action,
    )
    assert flat_joint.risk_likelihood.measure_kind == POINT_MASS
    assert flat_joint.risk_likelihood.log_likelihood.item() == 0.0
    assert flat_joint.joint_log_likelihood.item() == pytest.approx(math.log(0.3), rel=1e-6)


def test_bc038_nonflat_zero_mass_endpoint_makes_joint_log_likelihood_negative_infinity() -> None:
    direction = _direction_distribution()
    risk = _conditional_risk()
    identity = _identity()
    for direction_name in (SHORT, LONG):
        for endpoint in (0.0, 1.0):
            action = make_target_position_action_r1(
                action_id=f"{direction_name}-{endpoint}",
                policy_id=identity.policy_id,
                policy_version=identity.policy_version,
                target_direction=direction_name,
                requested_target_risk=endpoint,
            )
            joint = joint_nominal_action_likelihood_r1(
                direction_distribution=direction,
                risk_distribution=risk,
                action=action,
            )
            assert joint.risk_likelihood.measure_kind == POINT_MASS
            assert joint.joint_log_likelihood.item() == float("-inf")


def test_bc038_sampled_log_mu_exactly_matches_later_recomputation() -> None:
    identity = _identity()
    direction = _direction_distribution()
    risk = _conditional_risk()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260912)

    saw_flat = False
    saw_nonflat = False
    for index in range(64):
        sample = sample_nominal_action_r1(
            identity=identity,
            direction_distribution=direction,
            risk_distribution=risk,
            action_id=f"sample-{index}",
            generator=generator,
        )
        assert isinstance(sample, SampledNominalActionR1)
        assert sample.behavior_policy_sha256 == identity.policy_sha256
        assert sample.distribution_identity_sha256 == identity.semantic_sha256
        recomputed = recompute_sampled_log_mu_r1(
            sample=sample,
            identity=identity,
            direction_distribution=direction,
            risk_distribution=risk,
        )
        assert torch.equal(sample.log_mu, recomputed)
        saw_flat |= sample.action.target_direction == FLAT
        saw_nonflat |= sample.action.target_direction in (SHORT, LONG)
    assert saw_flat
    assert saw_nonflat


def test_bc038_recomputation_fails_closed_on_behavior_checkpoint_identity_change() -> None:
    identity = _identity()
    direction = _direction_distribution()
    risk = _conditional_risk()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    sample = sample_nominal_action_r1(
        identity=identity,
        direction_distribution=direction,
        risk_distribution=risk,
        action_id="frozen",
        generator=generator,
    )

    changed_checkpoint = replace(identity, policy_sha256="b" * 64)
    with pytest.raises(RuntimeError, match="BEHAVIOR_CHECKPOINT_HASH_MISMATCH"):
        recompute_sampled_log_mu_r1(
            sample=sample,
            identity=changed_checkpoint,
            direction_distribution=direction,
            risk_distribution=risk,
        )

    changed_identity = replace(identity, distribution_version="OTHER")
    with pytest.raises(RuntimeError, match="DISTRIBUTION_IDENTITY_MISMATCH"):
        recompute_sampled_log_mu_r1(
            sample=sample,
            identity=changed_identity,
            direction_distribution=direction,
            risk_distribution=risk,
        )


def test_bc038_joint_tampering_and_tensor_contract_mismatch_fail_closed() -> None:
    identity = _identity()
    direction = _direction_distribution()
    risk = _conditional_risk()
    action = make_target_position_action_r1(
        action_id="joint",
        policy_id=identity.policy_id,
        policy_version=identity.policy_version,
        target_direction=LONG,
        requested_target_risk=0.6,
    )
    joint = joint_nominal_action_likelihood_r1(
        direction_distribution=direction,
        risk_distribution=risk,
        action=action,
    )
    with pytest.raises(RuntimeError, match="JOINT_LOG_LIKELIHOOD_MISMATCH"):
        replace(
            joint,
            joint_log_likelihood=joint.joint_log_likelihood + torch.tensor(1.0),
        ).validate()

    cpu64 = make_policy_tensor_contract_r1(device_type="cpu", dtype="float64")
    risk64 = make_conditional_bounded_risk_r1(
        short_location=torch.tensor(0.0, dtype=torch.float64),
        short_log_scale=torch.tensor(0.0, dtype=torch.float64),
        long_location=torch.tensor(0.0, dtype=torch.float64),
        long_log_scale=torch.tensor(0.0, dtype=torch.float64),
        tensor_contract=cpu64,
    )
    with pytest.raises(RuntimeError, match="JOINT_COMPONENT_CONTRACT_MISMATCH"):
        joint_nominal_action_likelihood_r1(
            direction_distribution=direction,
            risk_distribution=risk64,
            action=action,
        )


def test_bc038_still_does_not_serialize_rng_state() -> None:
    source = inspect.getsource(actor_policy_module)
    assert "get_state" not in source
    assert "set_state" not in source
    assert "rng_position" not in source
    assert "trajectory_rng" not in source
