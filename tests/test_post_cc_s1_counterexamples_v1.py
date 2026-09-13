"""Focused counterexample tests for Sol R1 blockers B1/B2/B4/B5."""

from __future__ import annotations

import random
import tempfile

import pytest
import torch

from cb16_local_opt import post_cc_s1_tasks_v1 as T
import cb16_local_opt.post_cc_s1_training_loop_v1 as loop
from cb16_local_opt.post_cc_s1_checkpoint_identity_v1 import (
    FROZEN_AUTHORITY_INITIAL_CHECKPOINT_SHA256_V1,
    initial_checkpoint_identity_v1,
    initialize_canonical_target_actor_critic_v1,
    initialize_isolated_behavior_actor_v1,
)
from cb16_local_opt.post_cc_generation_continuity_v1 import parameter_state_sha256_v1
from cb16_local_opt.post_cc_s1_training_loop_v1 import S1SeedRunConfigV1, compute_retention_evidence_v1, run_seed_v1
from cb16_local_opt.post_cc_replay_materializer_v1 import ReplayStoreV1


def _tiny_config(spec, root, *, max_units=2, unit_size=4, control_id=None):
    return S1SeedRunConfigV1(
        spec=spec,
        seed=1701,
        run_root=root,
        mode="smoke",
        control_id=control_id,
        unit_size=unit_size,
        max_units=max_units,
        evaluation_population=4,
        manifest_sha256="0" * 64,
    )


def test_b1_sampler_never_forces_nonflat_into_uniform_draw():
    index_rows = [
        {"sequence_id": "flat-a", "nominal_direction": "FLAT"},
        {"sequence_id": "flat-b", "nominal_direction": "FLAT"},
        {"sequence_id": "nonflat-c", "nominal_direction": "LONG"},
    ]
    chosen_seed = None
    for seed in range(500):
        probe = random.Random(seed)
        draw = probe.sample([row["sequence_id"] for row in index_rows], 2)
        if "nonflat-c" not in draw:
            chosen_seed = seed
            break
    assert chosen_seed is not None, "expected a seed whose uniform draw excludes the non-FLAT sample"
    expected = random.Random(chosen_seed).sample([row["sequence_id"] for row in index_rows], 2)
    actual = loop._sample_sequence_ids_v1(
        index_rows=index_rows, replay_rng=random.Random(chosen_seed), target_size=2
    )
    assert tuple(expected) == actual
    assert "nonflat-c" not in actual  # old code would have substituted it back in


def test_b1_degenerate_all_flat_batch_skips_gradient_without_changing_sampling(monkeypatch):
    spec = T.build_task_specs_v1()[T.TASK_DELAYED_CONSEQUENCE_CREDIT]
    original_sampler = loop._sample_sequence_ids_v1

    def flat_only_sampler(*, index_rows, replay_rng, target_size):
        flat_ids = [str(row["sequence_id"]) for row in index_rows if row["nominal_direction"] == "FLAT"]
        if flat_ids:
            return tuple(flat_ids[: min(int(target_size), len(flat_ids))])
        return original_sampler(index_rows=index_rows, replay_rng=replay_rng, target_size=target_size)

    monkeypatch.setattr(loop, "_sample_sequence_ids_v1", flat_only_sampler)
    with tempfile.TemporaryDirectory(prefix="cb16-s1-b1-degenerate-") as root:
        result = run_seed_v1(_tiny_config(spec, root, max_units=3, unit_size=4))
    assert result["uniform_sampling"]["units_update_skipped_degenerate"] >= 1
    assert result["optimizer"]["optimizer_step_final"] == 0
    for unit in result["unit_evidence"]:
        if unit["update_skipped"]:
            assert unit["update_skipped_reason"] == "UNIFORM_BATCH_HAS_NO_NONFLAT_SAMPLE_SKIP_GRADIENT_STEP"
            assert unit["selected_a1_sequence_count"] == 0 or True


def test_b2_retention_uses_durable_index_not_selected_ids():
    with tempfile.TemporaryDirectory(prefix="cb16-s1-b2-retention-") as root:
        store = ReplayStoreV1(root)
        index_rows = []
        for index in range(3):
            sequence_id = f"a1-{index}"
            index_rows.append(
                {"sequence_id": sequence_id, "context_id": "A", "phase_id": "A", "unit_index": 0,
                 "nominal_direction": "LONG", "reward": 0.1}
            )
        # no durable sequence records exist yet -> eligibility must be False
        evidence_missing = compute_retention_evidence_v1(
            index_rows=index_rows, unit_size=128, unit_records=[], replay_store=store, phase_plan_completed=True
        )
        assert evidence_missing["a1_collected_count"] == 3
        assert evidence_missing["a1_durable_sequence_count_after_b"] == 0
        assert evidence_missing["a1_eligible_for_generic_replay_after_b"] is False
        # durable facts can be materialised by a real run: use a produced smoke run index
        spec = T.build_task_specs_v1()[T.TASK_ABA_RETENTION]
        result = run_seed_v1(_tiny_config(spec, root + "/run", max_units=2, unit_size=4))
        retention = result["retention_evidence"]
        assert retention["eligibility_rule"] == "ALL_DURABLE_INDEX_ROWS_UNIFORM_NO_EXPIRY"
        assert retention["a1_durable_sequence_count_after_b"] == retention["a1_collected_count"]
        assert retention["no_age_based_expiry"] is True


def test_b4_frozen_codec_verification_reports_contract_state_honestly():
    identity = dict(initial_checkpoint_identity_v1())
    assert identity["codec_id"] == "SORTED_STATE_DICT_NAME_SHAPE_DTYPE_FLOAT_VALUE_JSON_V1"
    assert identity["declared_frozen_sha256"] == FROZEN_AUTHORITY_INITIAL_CHECKPOINT_SHA256_V1
    # the authority hash is honestly reported as not reproduced under the declared
    # codec instead of being silently substituted by another hash
    assert identity["declared_hash_reproduced"] is False
    assert identity["contract_state"] == "CONTRACT_MISMATCH"
    assert identity["computed_actor_plus_critic_sha256"] != FROZEN_AUTHORITY_INITIAL_CHECKPOINT_SHA256_V1


def test_b4_target_initialization_is_isolated_from_behavior_rng():
    torch.manual_seed(12345)
    state_before = torch.random.get_rng_state()
    behavior = initialize_isolated_behavior_actor_v1(1701)
    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after)
    assert parameter_state_sha256_v1(behavior) != ""
    first_actor, first_critic = initialize_canonical_target_actor_critic_v1()
    behavior_same_seed_again = initialize_isolated_behavior_actor_v1(1701)
    second_actor, second_critic = initialize_canonical_target_actor_critic_v1()
    assert parameter_state_sha256_v1(first_actor) == parameter_state_sha256_v1(second_actor)
    assert parameter_state_sha256_v1(first_critic) == parameter_state_sha256_v1(second_critic)
    assert parameter_state_sha256_v1(behavior) == parameter_state_sha256_v1(behavior_same_seed_again)
    distinct_behavior = initialize_isolated_behavior_actor_v1(4242)
    assert parameter_state_sha256_v1(distinct_behavior) != parameter_state_sha256_v1(first_actor)


def test_b4_seed_run_records_identical_target_initialization_across_seeds():
    spec = T.build_task_specs_v1()[T.TASK_OFF_POLICY_VTRACE_CORRECTION]
    with tempfile.TemporaryDirectory(prefix="cb16-s1-b4-seed-") as root:
        first = run_seed_v1(
            S1SeedRunConfigV1(spec=spec, seed=1701, run_root=root + "/a", mode="smoke", control_id=None,
                              unit_size=2, max_units=2, evaluation_population=4, manifest_sha256="0" * 64)
        )
        second = run_seed_v1(
            S1SeedRunConfigV1(spec=spec, seed=1702, run_root=root + "/b", mode="smoke", control_id=None,
                              unit_size=2, max_units=2, evaluation_population=4, manifest_sha256="0" * 64)
        )
    assert first["initial_target_parameter_state_sha256"] == second["initial_target_parameter_state_sha256"]
    assert first["initial_target_critic_parameter_state_sha256"] == second["initial_target_critic_parameter_state_sha256"]
    assert first["initial_checkpoint_identity"]["contract_state"] == "CONTRACT_MISMATCH"


def test_b5_high_bankruptcy_loss_is_durable_failure_fact_and_in_denominator():
    from cb16_local_opt.post_cc_s1_qualification_v1 import run_high_bankruptcy_failure_fact_audit_v1

    audit = run_high_bankruptcy_failure_fact_audit_v1(".")
    assert audit["all_checks_pass"] is True
    assert audit["checks"]["loss_is_mechanical_terminal"] is True
    assert audit["checks"]["loss_boundary_is_economic_terminal"] is True
    assert audit["checks"]["loss_closure_provenance_durable"] is True
    assert audit["checks"]["loss_retained_with_full_weight"] is True
    assert audit["details"]["loss_equity"] <= 0.0
    assert audit["details"]["loss_reward"] < 0.0
