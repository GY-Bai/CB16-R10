"""S0-v2 bounded foundation qualification gate compiler.

The compiler runs the bounded synthetic durable-learning chain and emits
machine-readable gate evidence.  It never accesses FINAL or fresh market data.
"""

from __future__ import annotations

from dataclasses import replace
import ast
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable

import torch

from .cc_account_recovery_r0 import restore_runtime_r0, seal_runtime_r0
from .cc_critic_value_r0 import SeparateCritic
from .cc_policy_distribution_r0 import joint_log_prob
from .cc_policy_rng_r0 import PolicyRNG
from .cc_runtime_decision_schedule_r0 import CCDecisionScheduleR0
from .post_cc_critic_vtrace_v1 import (
    bootstrap_decision_v1,
    bootstrap_value_v1,
    joint_actor_critic_losses_v1,
)
from .post_cc_durable_collection_v1 import (
    DurableObservationCollectorV1,
    canonical_brain_vectors_from_account_v1,
)
from .post_cc_generation_continuity_v1 import (
    GenerationContinuityError,
    assert_same_logical_account_continuation_v1,
    commit_child_generation_v1,
    parameter_state_sha256_v1,
)
from .post_cc_joint_batch_v1 import (
    DurableBatchProvenanceV1,
    MATERIALIZER_CONTRACT_ID,
    boundary_semantics_v1,
    build_joint_batch_v1,
    observation_to_tensors_v1,
)
from .post_cc_joint_policy_loss_v1 import (
    assert_persisted_log_mu_used_v1,
    target_joint_log_probs_v1,
    target_likelihoods_v1,
)
from .post_cc_learner_v1 import (
    InjectedUpdateFaultV1,
    PostCCDurableReplayLearnerV1,
)
from .post_cc_observation_fact_v1 import (
    _check_no_future_information,
    assert_w01_observation_identity_v1,
    build_canonical_observation_fact_v1,
    canonical_observation_bytes_v1,
    decode_observation_fact_v1,
    observation_logical_id_v1,
)
from .post_cc_observation_store_v1 import (
    ImmutableContentStore,
    ObservationSemanticConflict,
    ObservationStoreCorruption,
    ObservationStoreV1,
)
from .post_cc_replay_materializer_v1 import (
    MaterializedReplayV1,
    ReplayCorruption,
    ReplayMaterializerV1,
    ReplayStoreV1,
)
from .post_cc_update_transaction_v1 import (
    DurableUpdateCorruptionError,
    STATUS_COMMITTED,
    STATUS_PREPARED,
    STATUS_STAGED,
    DurableUpdateStoreV1,
)
from .post_cc_s0v2_synthetic_fixture_v1 import (
    DEMO_LINEAGE_ID,
    DEMO_MARKET_SOURCE_IDENTITY,
    DEMO_MARKET_SOURCE_VERSION,
    SCIENCE_SEMANTIC_VERSION,
    collect_durable_sequence_v1,
    make_account,
    make_batch_v1,
    make_brain,
    make_canonical_observation_v1,
    make_executor,
    make_interval,
    make_joint_sample_v1,
    make_policy_rng,
    make_runtime,
    materialize_v1,
)

S0V2_QUALIFICATION_COMPILER_ID = "CB16_R11_S0V2_FOUNDATION_QUALIFICATION_V1"
S0V2_GATES = (
    "S0V2_BASELINE_IDENTITY_PASS",
    "S0V2_HISTORICAL_RECEIPT_IMMUTABILITY_PASS",
    "S0V2_OBSERVATION_FACT_INTEGRITY_PASS",
    "S0V2_OBSERVATION_STORE_RESTART_PASS",
    "S0V2_OBSERVATION_RUNTIME_LINKAGE_PASS",
    "S0V2_DURABLE_REPLAY_RECONSTRUCTION_PASS",
    "S0V2_JOINT_BATCH_INVARIANTS_PASS",
    "S0V2_NOMINAL_ACTION_LIKELIHOOD_PASS",
    "S0V2_TRUE_LOG_MU_INTEGRITY_PASS",
    "S0V2_CRITIC_BOOTSTRAP_VTRACE_PASS",
    "S0V2_GRADIENT_OWNERSHIP_PASS",
    "S0V2_EXACTLY_ONCE_UPDATE_RECOVERY_PASS",
    "S0V2_CHILD_CHECKPOINT_IMMUTABILITY_PASS",
    "S0V2_SAME_ACCOUNT_GENERATION_CONTINUITY_PASS",
    "S0V2_FINAL_FRESH_FIREWALL_PASS",
    "S0V2_LEGACY_PERFORMANCE_FIREWALL_PASS",
)

NEW_SURFACES = (
    "cb16_local_opt/post_cc_observation_fact_v1.py",
    "cb16_local_opt/post_cc_observation_store_v1.py",
    "cb16_local_opt/post_cc_durable_collection_v1.py",
    "cb16_local_opt/post_cc_joint_replay_v1.py",
    "cb16_local_opt/post_cc_replay_materializer_v1.py",
    "cb16_local_opt/post_cc_joint_batch_v1.py",
    "cb16_local_opt/post_cc_joint_policy_loss_v1.py",
    "cb16_local_opt/post_cc_critic_vtrace_v1.py",
    "cb16_local_opt/post_cc_update_transaction_v1.py",
    "cb16_local_opt/post_cc_learner_v1.py",
    "cb16_local_opt/post_cc_generation_continuity_v1.py",
    "scripts/run_r11_s0v2_foundation_qualification.py",
)
LEGACY_PERFORMANCE_MODULES = (
    "gpu_inference_broker",
    "multiprocess_trajectory_farm",
    "vectorized_physics",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_blob_sha(repo_root: Path, relative_path: str) -> str:
    return subprocess.check_output(["git", "hash-object", relative_path], cwd=repo_root, text=True).strip()


def _raised(fn: Callable[[], Any], exception_type: type[BaseException]) -> bool:
    try:
        fn()
    except exception_type:
        return True
    return False


def _gate_baseline_identity(repo_root: Path) -> tuple[bool, dict[str, Any]]:
    baseline = _load_json(repo_root / "authority/rearchitecture_r11/CB16_R11_S0V2_BASELINE_V1.json")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline["scientific_code_baseline_sha"], "HEAD"],
        cwd=repo_root,
    ).returncode
    child_contracts = baseline["frozen_successor_contracts"]
    observation_ok = (
        _git_blob_sha(repo_root, child_contracts["observation_contract"]["path"])
        == child_contracts["observation_contract"]["git_blob_sha1"]
    )
    joint_ok = (
        _git_blob_sha(repo_root, child_contracts["joint_replay_contract"]["path"])
        == child_contracts["joint_replay_contract"]["git_blob_sha1"]
    )
    brain = baseline["canonical_cc_implementation"]["brain"]
    brain_ok = _git_blob_sha(repo_root, brain["path"]) == brain["git_blob_sha1"]
    passed = (
        baseline["stage_identity"] == "CB16_R11_S0V2_DURABLE_LEARNABILITY_FOUNDATION_R0"
        and baseline["status"] == "FROZEN"
        and ancestor == 0
        and baseline["scientific_code_baseline_sha"] == "392063881a6ef0dd1776ac579f1a290a134fc49e"
        and observation_ok
        and joint_ok
        and brain_ok
    )
    return passed, {
        "baseline_manifest": "authority/rearchitecture_r11/CB16_R11_S0V2_BASELINE_V1.json",
        "scientific_code_baseline_sha": baseline["scientific_code_baseline_sha"],
        "task_document_handoff_sha": baseline["task_document_handoff_sha"],
        "observation_contract_blob_ok": observation_ok,
        "joint_replay_contract_blob_ok": joint_ok,
        "canonical_brain_blob_ok": brain_ok,
    }


def _gate_historical_receipt_immutability(repo_root: Path) -> tuple[bool, dict[str, Any]]:
    s0_baseline = _load_json(repo_root / "authority/rearchitecture_r11/CB16_R11_S0V2_BASELINE_V1.json")
    checks: dict[str, bool] = {}
    checks["post_cc_s0_receipt"] = (
        _git_blob_sha(repo_root, s0_baseline["historical_post_cc_s0_receipt"]["path"])
        == s0_baseline["historical_post_cc_s0_receipt"]["git_blob_sha1"]
    )
    for key in ("integration_spec", "integration_receipt", "thread_c_receipt"):
        item = s0_baseline["parent_cc"][key]
        checks[key] = _git_blob_sha(repo_root, item["path"]) == item["git_blob_sha1"]
    return all(checks.values()), checks


def _gate_observation_fact_integrity() -> tuple[bool, dict[str, Any]]:
    fact = build_canonical_observation_fact_v1(
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        market_values=(0.02, 1.0),
        account_values=(1.0, 0.0, 0.0),
        execution_values=(0.0, 0.0),
        market_source_identity=DEMO_MARKET_SOURCE_IDENTITY,
        market_source_version=DEMO_MARKET_SOURCE_VERSION,
        market_visible_through_time="00000000000000000000",
        account_lineage_id=DEMO_LINEAGE_ID,
        decision_index=0,
        environment_time="00000000000000000000",
    )
    payload = canonical_observation_bytes_v1(fact)
    round_trip = decode_observation_fact_v1(payload) == fact
    corrupted = replace(fact, account_payload={"values": [9.0, 0.0, 0.0]})
    corruption_rejected = _raised(corrupted.validate, ValueError)
    future_fields_rejected = _raised(
        lambda: _check_no_future_information("payload", {"values": [0.0, 1.0], "future_return": 1.0}),
        ValueError,
    )
    decision = __import__("cb16_local_opt.cc_runtime_wire_r0", fromlist=["CCPolicyDecisionV1"]).CCPolicyDecisionV1(
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        account_lineage_id=fact.account_lineage_id,
        decision_index=fact.decision_index,
        environment_time=0,
        policy_generation="0",
        policy_id="p",
        policy_sha256="a" * 64,
        observation_schema=fact.observation_schema,
        observation_hash=fact.observation_hash,
        normalizer_id=fact.normalizer_identity,
        nominal_direction="LONG",
        nominal_target_risk=0.4,
        log_mu=-1.0,
        risk_measure_kind="continuous_density",
        rng_stream_id="rng",
        rng_position_or_counter=0,
    )
    identity_ok = True
    try:
        assert_w01_observation_identity_v1(decision, fact)
    except ValueError:
        identity_ok = False
    passed = bool(round_trip and corruption_rejected and future_fields_rejected and identity_ok)
    return passed, {
        "observation_hash": fact.observation_hash,
        "logical_id": observation_logical_id_v1(fact),
        "round_trip": round_trip,
        "corruption_rejected": corruption_rejected,
        "future_fields_rejected": future_fields_rejected,
        "w01_identity_agreement": identity_ok,
    }


def _gate_observation_store_restart() -> tuple[bool, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="cb16-s0v2-store-") as td:
        store = ObservationStoreV1(td)
        fact = make_canonical_observation_v1()
        receipt = store.put(fact)
        store.put(fact)
        store.close()
        reopened = ObservationStoreV1(td)
        restart_ok = reopened.count() == 1 and reopened.get(receipt.logical_id) == fact
        content = ImmutableContentStore(td, "conflict-check")
        content.put_bytes("logical", b'{"a":1}')
        conflict_rejected = _raised(lambda: content.put_bytes("logical", b'{"a":2}'), ObservationSemanticConflict)
        object_paths = list((Path(td) / "observations" / "objects").rglob("*.bin"))
        object_paths[0].write_bytes(b'{"corrupted":true}')
        corruption_rejected = _raised(lambda: reopened.get(receipt.logical_id), ObservationStoreCorruption)
        return bool(restart_ok and conflict_rejected and corruption_rejected), {
            "restart_reconstruction": restart_ok,
            "semantic_conflict_rejected": conflict_rejected,
            "corruption_rejected": corruption_rejected,
            "content_sha256": receipt.content_sha256,
        }


def _gate_observation_runtime_linkage() -> tuple[bool, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="cb16-s0v2-linkage-") as td:
        runtime = make_runtime(account=make_account())
        runtime.schedule = CCDecisionScheduleR0(2)
        collector = DurableObservationCollectorV1(
            td,
            science_semantic_version=SCIENCE_SEMANTIC_VERSION,
            market_source_identity=DEMO_MARKET_SOURCE_IDENTITY,
            market_source_version=DEMO_MARKET_SOURCE_VERSION,
        )
        brain = make_brain()
        rng = make_policy_rng()
        box: dict[str, Any] = {}

        def callback(account_state, clocks):
            market, account_values, execution_values = canonical_brain_vectors_from_account_v1(account_state)
            logits, risk_loc, risk_scale = brain(
                torch.tensor(market), torch.tensor(account_values), torch.tensor(execution_values)
            )
            from .cc_policy_distribution_r0 import sample_nominal

            nominal = sample_nominal(logits, risk_loc, risk_scale, rng)
            decision, fact = collector.capture_policy_decision(
                account_state,
                account_lineage_id=runtime.account_lineage_id,
                decision_index=clocks.policy_decision_index,
                environment_time=clocks.environment_time,
                policy_generation=runtime.policy_generation,
                policy_id=runtime.policy_id,
                policy_sha256=runtime.policy_sha256,
                nominal=nominal,
            )
            box["decision"] = decision
            box["fact"] = fact
            return decision

        transition = runtime.step(make_interval(101.0), callback, expected_predecessor_token=runtime.predecessor_token)
        decision_persisted = ObservationStoreV1(td).count() == 1
        hash_match = box["decision"].observation_hash == box["fact"].observation_hash
        record = collector.persist_transition(
            sequence_id="linkage-seq", transition=transition, decision=box["decision"], reward=0.0, discount=0.99
        )
        persisted_log_mu = record.behavior_log_mu == box["decision"].log_mu
        observations_before = ObservationStoreV1(td).count()
        no_decision = runtime.step(make_interval(99.0), None, expected_predecessor_token=runtime.predecessor_token)
        collector.persist_no_decision_advance(no_decision)
        no_fabricated_observation = ObservationStoreV1(td).count() == observations_before
        raw_only = ReplayStoreV1(td).count_raw_advances() == 1
        passed = bool(decision_persisted and hash_match and persisted_log_mu and no_fabricated_observation and raw_only)
        return passed, {
            "decision_observation_persisted": decision_persisted,
            "w01_hash_match": hash_match,
            "persisted_log_mu": persisted_log_mu,
            "no_decision_observation_not_fabricated": no_fabricated_observation,
            "raw_only_advance_persisted": raw_only,
        }


def _gate_durable_replay_reconstruction() -> tuple[bool, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="cb16-s0v2-replay-") as td:
        collected = collect_durable_sequence_v1(td)
        sequence_id = collected.sequence_id
        phase_a = materialize_v1(td, sequence_id, restart_verified=False)
        expected_id = phase_a.manifest.stable_materialization_id
        expected_hashes = phase_a.manifest.ordered_sample_hashes
        ReplayStoreV1(td).put_manifest(phase_a.manifest)
        del collected, phase_a
        gc.collect()
        phase_b = materialize_v1(
            td,
            sequence_id,
            restart_verified=True,
            expected_materialization_id=expected_id,
        )
        persisted = ReplayStoreV1(td).get_manifest(expected_id)
        passed = (
            phase_b.manifest.ordered_sample_hashes == expected_hashes
            and persisted.ordered_sample_hashes == expected_hashes
        )
        return passed, {
            "sequence_id": sequence_id,
            "sample_count": len(expected_hashes),
            "stable_materialization_id": expected_id,
            "restart_sample_identity_equal": phase_b.manifest.ordered_sample_hashes == expected_hashes,
            "persisted_manifest_equal": persisted.ordered_sample_hashes == expected_hashes,
        }


def _gate_joint_batch_invariants() -> tuple[bool, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="cb16-s0v2-batch-") as td:
        collected = collect_durable_sequence_v1(td)
        batch = materialize_v1(td, collected.sequence_id, restart_verified=True).to_batch()
        sample = batch.samples[0]
        checks = {
            "flat_nonzero_rejected": _raised(
                lambda: replace(
                    sample, nominal_direction="FLAT", nominal_target_risk=0.4, risk_measure_kind="point_mass"
                ).validate(),
                ValueError,
            ),
            "nonflat_endpoint_rejected": _raised(lambda: replace(sample, nominal_target_risk=1.0).validate(), ValueError),
            "nonfinite_log_mu_rejected": _raised(
                lambda: replace(sample, behavior_log_mu=float("nan")).validate(), ValueError
            ),
            "missing_behavior_identity_rejected": _raised(
                lambda: replace(sample, behavior_policy_id="").validate(), ValueError
            ),
            "missing_source_reference_rejected": _raised(
                lambda: replace(sample, source_fact_hashes=()).validate(), ValueError
            ),
        }
        if len(batch.samples) >= 2:
            reversed_samples = (batch.samples[1], batch.samples[0])
            checks["time_order_violation_rejected"] = _raised(
                lambda: build_joint_batch_v1(
                    samples=reversed_samples,
                    provenance=batch.provenance,
                    target_policy_identity=batch.target_policy_identity,
                ),
                ValueError,
            )
        else:
            checks["time_order_violation_rejected"] = False
        corrupted_hashes = ("0" * 64,) + tuple(batch.observation_hashes[1:])
        checks["observation_hash_mismatch_rejected"] = _raised(
            lambda: replace(batch, observation_hashes=corrupted_hashes).validate(), ValueError
        )
        wrong_indices = torch.zeros_like(batch.direction_indices)
        checks["direction_index_mapping_rejected"] = _raised(
            lambda: replace(batch, direction_indices=wrong_indices).validate(), ValueError
        )
        return all(checks.values()), checks


def _force_exact_same_policy(actor, samples, bootstrap_map=None):
    placeholders = tuple(replace(sample, behavior_log_mu=0.0) for sample in samples)
    placeholder_batch = make_batch_v1(placeholders, bootstrap_observations_by_sequence=bootstrap_map)
    with torch.no_grad():
        values = target_joint_log_probs_v1(actor, placeholder_batch)
    return tuple(replace(sample, behavior_log_mu=float(values[index].item())) for index, sample in enumerate(samples))


def _gate_nominal_action_likelihood() -> tuple[bool, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="cb16-s0v2-likelihood-") as td:
        actor = make_brain()
        checks: dict[str, bool] = {}
        for direction, risk in (("FLAT", 0.0), ("LONG", 0.4), ("SHORT", 0.6)):
            sample = make_joint_sample_v1(root=td, nominal_direction=direction, nominal_target_risk=risk)
            batch = make_batch_v1((sample,))
            market, account, execution = observation_to_tensors_v1(sample.observation)
            logits, risk_loc, risk_scale = actor(
                market.reshape(1, -1), account.reshape(1, -1), execution.reshape(1, -1)
            )
            expected = joint_log_prob(
                logits.reshape(-1), risk_loc.reshape(-1), risk_scale.reshape(-1), direction, risk
            )
            actual = target_joint_log_probs_v1(actor, batch)[0]
            checks[f"{direction.lower()}_density_matches_scalar_reference"] = bool(
                torch.allclose(actual, expected.reshape(()), atol=1e-6, rtol=1e-6)
            )
        sample = make_joint_sample_v1(root=td, nominal_direction="LONG", nominal_target_risk=0.4)
        exact = _force_exact_same_policy(actor, (sample,))[0]
        same_policy_batch = make_batch_v1((exact,))
        likelihoods = target_likelihoods_v1(actor, same_policy_batch)
        checks["same_policy_log_pi_equals_log_mu"] = bool(
            torch.allclose(likelihoods.log_pi, likelihoods.log_mu, atol=1e-6, rtol=1e-6)
        )
        executed = replace(sample, consequence_context={"executed_direction": "FLAT", "executed_quantity": 0.0})
        base_log_pi = target_joint_log_probs_v1(actor, make_batch_v1((sample,)))
        executed_log_pi = target_joint_log_probs_v1(actor, make_batch_v1((executed,)))
        checks["execution_context_does_not_replace_nominal_action"] = bool(
            torch.allclose(base_log_pi, executed_log_pi, atol=1e-7, rtol=1e-7)
        )
        return all(checks.values()), checks


def _gate_true_log_mu_integrity() -> tuple[bool, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="cb16-s0v2-logmu-") as td:
        collected = collect_durable_sequence_v1(td)
        materialized = materialize_v1(td, collected.sequence_id, restart_verified=True)
        store = ReplayStoreV1(td)
        values_match = all(
            sample.behavior_log_mu == store.get_transition(sample.transition_id).behavior_log_mu
            for sample in materialized.samples
        )
        sample = materialized.samples[0]
        reconstructed_rejected = _raised(
            lambda: replace(sample, log_mu_source="RECONSTRUCTED_FROM_EXECUTION").validate(), ValueError
        )
        batch = materialized.to_batch()
        fabricated_batch = replace(batch, behavior_log_mu=torch.zeros_like(batch.behavior_log_mu))
        fabricated_rejected = _raised(lambda: fabricated_batch.validate(), ValueError)
        persisted_source_checked = True
        try:
            assert_persisted_log_mu_used_v1(batch)
        except ValueError:
            persisted_source_checked = False
        passed = bool(values_match and reconstructed_rejected and fabricated_rejected and persisted_source_checked)
        return passed, {
            "all_materialized_log_mu_equal_persisted": values_match,
            "reconstructed_log_mu_rejected": reconstructed_rejected,
            "fabricated_batch_log_mu_rejected": fabricated_rejected,
            "decision_time_source_checked": persisted_source_checked,
        }


def _gate_critic_bootstrap_vtrace() -> tuple[bool, dict[str, Any]]:
    checks: dict[str, bool] = {
        "boundary_semantics_mechanically_distinct": (
            boundary_semantics_v1("ECONOMIC_TERMINAL") == "ECONOMIC_TERMINAL"
            and boundary_semantics_v1("ECONOMIC_TERMINAL", mechanical_terminal=True) == "MECHANICAL_TERMINAL"
            and boundary_semantics_v1("OBJECTIVE_HORIZON_REACHED") == "TASK_HORIZON"
            and boundary_semantics_v1("COMPUTE_CHUNK") == "COMPUTE_TRUNCATION"
            and boundary_semantics_v1("DATA_END_TRUNCATION") == "DATASET_TRUNCATION"
            and boundary_semantics_v1("TRADING_DISABLED_PENDING_SETTLEMENT") == "PENDING_SETTLEMENT"
        ),
        "economic_terminal_bootstrap_zero": bootstrap_value_v1("ECONOMIC_TERMINAL", 7.0) == 0.0,
        "objective_horizon_bootstrap_zero": bootstrap_value_v1("OBJECTIVE_HORIZON_REACHED", 7.0) == 0.0,
        "compute_chunk_bootstrap_next": bootstrap_value_v1("COMPUTE_CHUNK", 7.0) == 7.0,
        "unknown_boundary_rejected": _raised(lambda: bootstrap_value_v1("UNKNOWN", 1.0), ValueError),
        "terminal_must_not_supply_bootstrap": _raised(
            lambda: bootstrap_decision_v1("ECONOMIC_TERMINAL", mechanical_terminal=True, next_value=1.0), ValueError
        ),
    }
    with tempfile.TemporaryDirectory(prefix="cb16-s0v2-vtrace-") as td:
        actor = make_brain()
        first = make_joint_sample_v1(
            root=td,
            decision_index=0,
            environment_time=0,
            nominal_direction="LONG",
            nominal_target_risk=0.4,
            reward=0.01,
            boundary_type="CONTINUE",
        )
        second = make_joint_sample_v1(
            root=td,
            transition_id="qual-vtrace-terminal",
            decision_index=1,
            environment_time=1,
            nominal_direction="SHORT",
            nominal_target_risk=0.6,
            reward=0.02,
            boundary_type="ECONOMIC_TERMINAL",
        )
        first, second = _force_exact_same_policy(actor, (first, second))
        batch = make_batch_v1((first, second))
        losses = joint_actor_critic_losses_v1(actor, SeparateCritic(7, 8), batch)
        checks["same_policy_rho_unit"] = bool(torch.allclose(losses.vtrace.rhos, torch.ones_like(losses.vtrace.rhos)))
        manual_last = batch.rewards[1]
        manual_first = batch.rewards[0] + batch.discounts[0] * manual_last
        checks["terminal_vtrace_recurrence"] = bool(
            torch.allclose(losses.vtrace.vs, torch.stack([manual_first, manual_last]), atol=1e-6, rtol=1e-6)
        )
        truncation = replace(
            second,
            transition_id="qual-vtrace-truncation",
            boundary_type="COMPUTE_CHUNK",
            bootstrap_state_ref_or_null="b" * 64,
        )
        bootstrap_fact = make_canonical_observation_v1(decision_index=2, environment_time=2)
        bootstrap_map = {truncation.sequence_id: bootstrap_fact}
        first_t, truncation_t = _force_exact_same_policy(actor, (first, truncation), bootstrap_map)
        trunc_batch = make_batch_v1((first_t, truncation_t), bootstrap_observations_by_sequence=bootstrap_map)
        critic = SeparateCritic(7, 8)
        with torch.no_grad():
            critic.net[-1].bias.fill_(0.5)
        trunc_losses = joint_actor_critic_losses_v1(actor, critic, trunc_batch)
        market, account, execution = observation_to_tensors_v1(bootstrap_fact)
        with torch.no_grad():
            next_value = float(critic(torch.cat([market, account, execution]).reshape(1, -1)).item())
        expected_truncation_last = (
            float(trunc_batch.rewards[-1].item())
            + float(trunc_batch.discounts[-1].item()) * next_value
        )
        checks["truncation_uses_durable_next_observation"] = bool(
            abs(trunc_losses.vtrace.vs[-1].item() - expected_truncation_last) < 1e-6
        )
        checks["terminal_has_no_future_bootstrap"] = (
            abs(losses.vtrace.vs[-1].item() - float(batch.rewards[-1].item())) < 1e-6
        )
    return all(checks.values()), checks


def _make_update_context(root: str):
    collected = collect_durable_sequence_v1(root)
    batch = materialize_v1(root, collected.sequence_id, restart_verified=True).to_batch()
    target = make_brain()
    target.load_state_dict(collected.behavior_brain.state_dict())
    target.assert_gradient_ownership()
    learner = PostCCDurableReplayLearnerV1(
        target_actor=target,
        target_critic=SeparateCritic(7, 8),
        update_store=DurableUpdateStoreV1(Path(root) / "updates"),
        parent_policy_identity="cc-s0v2-policy-g0",
        target_policy_identity="cc-s0v2-target-policy-g1",
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
    )
    return collected, learner, batch


def _gate_gradient_ownership_child_immutability() -> tuple[bool, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="cb16-s0v2-gradient-") as td:
        collected, learner, batch = _make_update_context(td)
        behavior_sha = parameter_state_sha256_v1(collected.behavior_brain)
        result = learner.apply_durable_update_v1(batch=batch)
        behavior_immutable = parameter_state_sha256_v1(collected.behavior_brain) == behavior_sha
        child_distinct = result.child_checkpoint_sha256 != result.record.parent_checkpoint_sha256
        gradient_summary = dict(result.gradient_ownership_summary)
        gradient_ok = (
            gradient_summary.get("frozen_market_organ_grad_zero") is True
            and int(gradient_summary.get("critic_nonzero_gradient_parameter_count", 0)) > 0
            and int(gradient_summary.get("account_stem_nonzero_gradient_parameter_count", 0)) > 0
        )
        checkpoint_path = learner.update_store._checkpoint_path(result.child_checkpoint_sha256)
        checkpoint_bytes = checkpoint_path.read_bytes()
        checkpoint_path.write_bytes(b'{"corrupted":true}')
        tamper_rejected = _raised(
            lambda: learner.update_store.load_child_checkpoint(result.update_id),
            DurableUpdateCorruptionError,
        )
        checkpoint_path.write_bytes(checkpoint_bytes)
        passed = bool(behavior_immutable and child_distinct and gradient_ok and tamper_rejected)
        return passed, {
            "behavior_checkpoint_immutable": behavior_immutable,
            "child_checkpoint_distinct_from_parent": child_distinct,
            "gradient_ownership_ok": gradient_ok,
            "checkpoint_tamper_rejected": tamper_rejected,
            "gradient_ownership_summary": gradient_summary,
            "child_checkpoint_sha256": result.child_checkpoint_sha256,
        }


def _run_fault_scenario(fault_at: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"cb16-s0v2-recovery-{fault_at}-") as td:
        collected, learner, batch = _make_update_context(td)
        parent_payload = learner.export_parent_checkpoint_bytes()
        raised = False
        try:
            learner.apply_durable_update_v1(batch=batch, fault_at=fault_at)
        except InjectedUpdateFaultV1:
            raised = True
        update_id = next(learner.update_store.updates_dir.glob("*.json")).stem
        status_after_fault = learner.update_store.get_record(update_id).commit_status
        restarted = PostCCDurableReplayLearnerV1.from_parent_checkpoint_bytes_v1(
            payload=parent_payload,
            target_actor=make_brain(),
            target_critic=SeparateCritic(7, 8),
            update_store=DurableUpdateStoreV1(Path(td) / "updates"),
            parent_policy_identity=learner.parent_policy_identity,
            target_policy_identity=learner.target_policy_identity,
            science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        )
        result = restarted.apply_durable_update_v1(batch=batch)
        expected_status = {
            "after_gradient_before_stage": STATUS_PREPARED,
            "after_stage_before_commit": STATUS_STAGED,
            "after_commit_before_ack": STATUS_COMMITTED,
        }[fault_at]
        expected_gradient_applications = 0 if fault_at != "after_gradient_before_stage" else 1
        passed = (
            raised
            and status_after_fault == expected_status
            and result.optimizer_step_after == 1
            and restarted.gradient_applications == expected_gradient_applications
        )
        return {
            "fault_at": fault_at,
            "status_after_fault": status_after_fault,
            "final_status": learner.update_store.get_record(update_id).commit_status,
            "final_optimizer_step": result.optimizer_step_after,
            "recovery_gradient_applications": restarted.gradient_applications,
            "passed": passed,
        }


def _gate_exactly_once_recovery() -> tuple[bool, dict[str, Any]]:
    evidence = {
        fault: _run_fault_scenario(fault)
        for fault in ("after_gradient_before_stage", "after_stage_before_commit", "after_commit_before_ack")
    }
    return all(item["passed"] for item in evidence.values()), evidence


def _gate_same_account_generation_continuity() -> tuple[bool, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="cb16-s0v2-generation-") as td:
        collected, learner, batch = _make_update_context(td)
        result = learner.apply_durable_update_v1(batch=batch)
        runtime_seal = seal_runtime_r0(collected.runtime)
        del collected
        gc.collect()
        runtime, _memory_token = restore_runtime_r0(runtime_seal, executor=make_executor())
        account_snapshot = runtime.account
        lineage_before = runtime.account_lineage_id
        receipt = commit_child_generation_v1(
            runtime,
            child_checkpoint_sha256=result.child_checkpoint_sha256,
            new_policy_generation="1",
            new_policy_id="cc-s0v2-policy-g1",
            new_policy_sha256=result.child_checkpoint_sha256,
            expected_account_snapshot=account_snapshot,
            boundary_type="OBJECTIVE_HORIZON_REACHED",
            committed_update_record=result.record,
        )
        account_preserved = runtime.account == account_snapshot and runtime.account_lineage_id == lineage_before
        child_rng = PolicyRNG("cc-s0v2-policy-g1", lineage_before, 20260913)
        collector = DurableObservationCollectorV1(
            td,
            science_semantic_version=SCIENCE_SEMANTIC_VERSION,
            market_source_identity=DEMO_MARKET_SOURCE_IDENTITY,
            market_source_version=DEMO_MARKET_SOURCE_VERSION,
        )
        child_box: dict[str, Any] = {}

        def child_callback(account_state, clocks):
            market, account_values, execution_values = canonical_brain_vectors_from_account_v1(account_state)
            logits, risk_loc, risk_scale = learner.target_actor(
                torch.tensor(market), torch.tensor(account_values), torch.tensor(execution_values)
            )
            from .cc_policy_distribution_r0 import sample_nominal

            nominal = sample_nominal(logits, risk_loc, risk_scale, child_rng)
            decision, _fact = collector.capture_policy_decision(
                account_state,
                account_lineage_id=runtime.account_lineage_id,
                decision_index=clocks.policy_decision_index,
                environment_time=clocks.environment_time,
                policy_generation="1",
                policy_id="cc-s0v2-policy-g1",
                policy_sha256=result.child_checkpoint_sha256,
                nominal=nominal,
            )
            child_box["decision"] = decision
            return decision

        runtime.step(
            make_interval(104.0, boundary_type="COMPUTE_CHUNK"),
            child_callback,
            expected_predecessor_token=runtime.predecessor_token,
        )
        child_acted = (
            child_box["decision"].policy_id == "cc-s0v2-policy-g1"
            and child_box["decision"].policy_sha256 == result.child_checkpoint_sha256
        )
        flat_rejected = _raised(
            lambda: assert_same_logical_account_continuation_v1(
                account_lineage_id_before=lineage_before,
                account_lineage_id_after=lineage_before,
                account_before=account_snapshot,
                account_after=make_account(account_id="fresh-flat-account"),
            ),
            GenerationContinuityError,
        )
        passed = bool(account_preserved and child_acted and flat_rejected and receipt.account_fully_preserved)
        return passed, {
            "restart_restored_runtime": True,
            "account_preserved": account_preserved,
            "child_policy_acted": child_acted,
            "flat_account_substitution_rejected": flat_rejected,
            "generation_switch_id": receipt.switch_id,
        }


def _gate_final_fresh_firewall(repo_root: Path) -> tuple[bool, dict[str, Any]]:
    forbidden_paths = (
        repo_root / "final_holdout",
        repo_root / "data" / "2025-09",
    )
    paths_absent = all(not path.exists() for path in forbidden_paths)
    forbidden_tokens = ("2025-09", "final_holdout")
    token_hits: dict[str, list[str]] = {}
    for relative in NEW_SURFACES:
        path = repo_root / relative
        text = path.read_text(encoding="utf-8")
        hits = [token for token in forbidden_tokens if token in text]
        if hits:
            token_hits[relative] = hits
    passed = bool(paths_absent and not token_hits)
    return passed, {"forbidden_paths_absent": paths_absent, "forbidden_token_hits": token_hits}


def _gate_legacy_performance_firewall(repo_root: Path) -> tuple[bool, dict[str, Any]]:
    violations: dict[str, list[str]] = {}
    for relative in NEW_SURFACES:
        path = repo_root / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            violations[relative] = ["SYNTAX_ERROR"]
            continue
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        hits = [module for module in LEGACY_PERFORMANCE_MODULES if any(module in item for item in imports)]
        if hits:
            violations[relative] = hits
    return not violations, {"violations": violations}


def compile_s0v2_gates(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    gate_results: dict[str, bool] = {}
    evidence: dict[str, Any] = {}

    def record(gate: str, result: tuple[bool, dict[str, Any]]) -> None:
        gate_results[gate] = bool(result[0])
        evidence[gate] = result[1]

    record("S0V2_BASELINE_IDENTITY_PASS", _gate_baseline_identity(root))
    record("S0V2_HISTORICAL_RECEIPT_IMMUTABILITY_PASS", _gate_historical_receipt_immutability(root))
    record("S0V2_OBSERVATION_FACT_INTEGRITY_PASS", _gate_observation_fact_integrity())
    record("S0V2_OBSERVATION_STORE_RESTART_PASS", _gate_observation_store_restart())
    record("S0V2_OBSERVATION_RUNTIME_LINKAGE_PASS", _gate_observation_runtime_linkage())
    record("S0V2_DURABLE_REPLAY_RECONSTRUCTION_PASS", _gate_durable_replay_reconstruction())
    record("S0V2_JOINT_BATCH_INVARIANTS_PASS", _gate_joint_batch_invariants())
    record("S0V2_NOMINAL_ACTION_LIKELIHOOD_PASS", _gate_nominal_action_likelihood())
    record("S0V2_TRUE_LOG_MU_INTEGRITY_PASS", _gate_true_log_mu_integrity())
    record("S0V2_CRITIC_BOOTSTRAP_VTRACE_PASS", _gate_critic_bootstrap_vtrace())
    gradient_result = _gate_gradient_ownership_child_immutability()
    record("S0V2_GRADIENT_OWNERSHIP_PASS", gradient_result)
    record("S0V2_CHILD_CHECKPOINT_IMMUTABILITY_PASS", gradient_result)
    record("S0V2_EXACTLY_ONCE_UPDATE_RECOVERY_PASS", _gate_exactly_once_recovery())
    record("S0V2_SAME_ACCOUNT_GENERATION_CONTINUITY_PASS", _gate_same_account_generation_continuity())
    record("S0V2_FINAL_FRESH_FIREWALL_PASS", _gate_final_fresh_firewall(root))
    record("S0V2_LEGACY_PERFORMANCE_FIREWALL_PASS", _gate_legacy_performance_firewall(root))

    missing_gates = sorted(set(S0V2_GATES) - set(gate_results))
    status = "PASS" if all(gate_results.get(gate, False) for gate in S0V2_GATES) and not missing_gates else "FAIL"
    return {
        "schema": "CB16_R11_S0V2_FOUNDATION_QUALIFICATION_V1",
        "compiler_id": S0V2_QUALIFICATION_COMPILER_ID,
        "status": status,
        "gates": gate_results,
        "missing_gates": missing_gates,
        "evidence": evidence,
        "firewall": {
            "FINAL_opened": False,
            "fresh_data_used": False,
            "ECONOMIC_evidence_claimed": False,
            "TRANSFER_evidence_claimed": False,
            "end_to_end_learnability_claimed": False,
        },
        "evidence_ceiling": "POST_CC_DURABLE_LEARNING_FOUNDATION_QUALIFIED",
    }
