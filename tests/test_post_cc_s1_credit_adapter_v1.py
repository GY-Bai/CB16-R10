"""Durable decision-interval credit adapter tests (delayed-credit known answer)."""

from __future__ import annotations

from dataclasses import replace
import random
import tempfile

import pytest

from cb16_local_opt import post_cc_s1_tasks_v1 as T
from cb16_local_opt.post_cc_s1_credit_adapter_v1 import (
    CreditCorruption,
    apply_credit_views_v1,
    assert_sample_log_mu_bound_to_durable_v1,
    load_credit_v1,
    record_decision_interval_credit_v1,
    verify_credits_for_sequences_v1,
)
from cb16_local_opt.post_cc_generation_continuity_v1 import parameter_state_sha256_v1
from cb16_local_opt.post_cc_observation_store_v1 import ImmutableContentStore
from cb16_local_opt.post_cc_replay_materializer_v1 import ReplayMaterializerV1
from cb16_local_opt.cc_policy_rng_r0 import PolicyRNG


def _collect_delayed_episode(root: str, *, seed: int = 1701):
    spec = T.build_task_specs_v1()[T.TASK_DELAYED_CONSEQUENCE_CREDIT]
    context = spec.contexts[0]
    lineage = "cc-s1-credit-test-lineage"
    account, _provenance = T.establish_account_context_v1(
        spec=spec, context=context, lineage=lineage, setup_root=f"{root}/setup"
    )
    actor = T.make_s1_brain_v1()
    evidence = T.collect_episode_v1(
        spec=spec,
        context=context,
        account=account,
        root=root,
        lineage=lineage,
        policy_generation="0",
        policy_id="cc-s1-credit-policy",
        policy_sha256=parameter_state_sha256_v1(actor),
        actor=actor,
        action_rng=PolicyRNG("cc-s1-credit-policy", lineage, seed),
        env_rng=random.Random(seed),
        sequence_id=f"{lineage}-{seed}",
    )
    return spec, evidence


def test_credit_view_reproduces_downstream_equity_delta_and_drops_truncation_bootstrap():
    with tempfile.TemporaryDirectory() as root:
        _spec, evidence = _collect_delayed_episode(root)
        assert evidence.credit_view_id is not None
        credit = load_credit_v1(root, evidence.sequence_id, 0)
        assert credit is not None
        assert credit.reward == pytest.approx(evidence.reward, abs=1e-12)
        assert credit.boundary_type == "OBJECTIVE_HORIZON_REACHED"
        assert credit.mechanical_terminal is False
        materialized = ReplayMaterializerV1.from_durable_state_v1(root).materialize_sequence(
            evidence.sequence_id, target_policy_identity="cc-s1-target", restart_verified=True
        )
        raw_sample = materialized.samples[0]
        assert raw_sample.boundary_type == "COMPUTE_CHUNK"
        assert raw_sample.bootstrap_state_ref_or_null is not None
        samples, bootstrap_map = apply_credit_views_v1(root, materialized)
        assert len(samples) == 1
        assert samples[0].boundary_type == "OBJECTIVE_HORIZON_REACHED"
        assert samples[0].bootstrap_state_ref_or_null is None
        assert samples[0].reward == pytest.approx(credit.reward, abs=1e-12)
        assert bootstrap_map == {}
        assert credit.credit_id in samples[0].consequence_context["credit_view_id"]


def test_credit_restart_rederivation_is_stable_and_verified():
    with tempfile.TemporaryDirectory() as root:
        _spec, evidence = _collect_delayed_episode(root)
        first = load_credit_v1(root, evidence.sequence_id, 0)
        second = load_credit_v1(root, evidence.sequence_id, 0)
        assert first is not None and second is not None
        assert first.content_sha256 == second.content_sha256
        counts = verify_credits_for_sequences_v1(root, (evidence.sequence_id,))
        assert counts["credits_verified"] == 1
        assert first.final_equity == pytest.approx(evidence.final_equity, abs=1e-12)
        assert first.decision_equity_before == pytest.approx(1000.0, abs=1e-12)


def test_credit_rejects_tampered_downstream_raw_advance():
    with tempfile.TemporaryDirectory() as root:
        _spec, evidence = _collect_delayed_episode(root)
        raw_id = evidence.raw_advance_ids[0]
        store = ImmutableContentStore(root, "replay_raw_advances")
        import json

        original = store.get_bytes(raw_id)
        store.delete_for_test_only(raw_id)
        payload = json.loads(original.decode("utf-8"))
        payload["post_equity"] = float(payload["post_equity"]) + 1000.0
        tampered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        store.put_bytes(raw_id, tampered)
        with pytest.raises(CreditCorruption):
            load_credit_v1(root, evidence.sequence_id, 0)


def test_credit_reward_must_match_equity_delta():
    with tempfile.TemporaryDirectory() as root:
        _spec, evidence = _collect_delayed_episode(root)
        credit = load_credit_v1(root, evidence.sequence_id, 0)
        assert credit is not None
        with pytest.raises(CreditCorruption):
            replace(credit, reward=credit.reward + 0.5).validate()
        with pytest.raises(CreditCorruption):
            replace(credit, credit_id="0" * 64).validate()


def test_credit_view_cannot_be_bound_to_another_transition():
    with tempfile.TemporaryDirectory() as root:
        _spec, evidence = _collect_delayed_episode(root)
        materialized = ReplayMaterializerV1.from_durable_state_v1(root).materialize_sequence(
            evidence.sequence_id, target_policy_identity="cc-s1-target", restart_verified=True
        )
        tampered = replace(materialized.samples[0], transition_record_sha256="a" * 64)
        corrupted = replace(materialized, samples=(tampered,))
        with pytest.raises((CreditCorruption, ValueError)):
            apply_credit_views_v1(root, corrupted)


def test_sample_log_mu_binding_rejects_fabricated_value():
    with tempfile.TemporaryDirectory() as root:
        _spec, evidence = _collect_delayed_episode(root)
        materialized = ReplayMaterializerV1.from_durable_state_v1(root).materialize_sequence(
            evidence.sequence_id, target_policy_identity="cc-s1-target", restart_verified=True
        )
        sample = materialized.samples[0]
        assert_sample_log_mu_bound_to_durable_v1(root, sample)
        fabricated = replace(sample, behavior_log_mu=float(sample.behavior_log_mu) + 0.25)
        with pytest.raises(CreditCorruption):
            assert_sample_log_mu_bound_to_durable_v1(root, fabricated)
        executed_substitution = replace(sample, nominal_target_risk=0.123456)
        with pytest.raises(CreditCorruption):
            assert_sample_log_mu_bound_to_durable_v1(root, executed_substitution)
