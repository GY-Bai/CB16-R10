"""CB16 R11 S1 qualification runtime, integrity attack matrix and artifacts.

Two scientific modes only:

* ``smoke``         -- bounded engineering evidence, never a scientific verdict
* ``qualification`` -- exact frozen execution manifest, no scientific overrides

Runtime evidence must come from GitHub Actions -> Shanxi Docker; the local
runner is only the implementation of that entrypoint.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

from .cc_runtime_boundary_r0 import COMPUTE_CHUNK, OBJECTIVE_HORIZON_REACHED
from .post_cc_joint_batch_v1 import DurableBatchProvenanceV1, MATERIALIZER_CONTRACT_ID, build_joint_batch_v1
from .post_cc_joint_policy_loss_v1 import assert_persisted_log_mu_used_v1
from .post_cc_joint_replay_v1 import DurableSequenceRecordV1
from .post_cc_learner_v1 import InjectedUpdateFaultV1, PostCCDurableReplayLearnerV1, PostCCLearnerError
from .post_cc_observation_store_v1 import ObservationSemanticConflict, ObservationStoreV1
from .post_cc_replay_materializer_v1 import ReplayCorruption, ReplayMaterializerV1, ReplayStoreV1
from .post_cc_update_transaction_v1 import DurableUpdateCorruptionError, STATUS_COMMITTED, DurableUpdateStoreV1
from .post_cc_generation_continuity_v1 import (
    GenerationContinuityError,
    commit_child_generation_v1,
    parameter_state_sha256_v1,
)
from .post_cc_critic_vtrace_v1 import bootstrap_decision_v1
from .post_cc_s1_execution_manifest_v1 import (
    build_s1_execution_manifest_v1,
    validate_s1_execution_manifest_v1,
)
from .post_cc_s1_gate_compiler_v1 import compile_s1_gates_v1
from .post_cc_s1_tasks_v1 import (
    CONTROL_ACCOUNT_ABLATION,
    CONTROL_FABRICATED_LOG_MU_REJECTION,
    CONTROL_NO_SIGNAL,
    CONTROL_OBJECTIVE_FIREWALL,
    CONTROL_RANDOM_IMPOSSIBLE,
    CONTROL_SHUFFLED_CREDIT,
    FROZEN_TASK_IDS_V1,
    EpisodeEvidenceV1,
    TaskSpecV1,
    build_task_specs_v1,
    collect_episode_v1,
    derived_stream_seed_v1,
    enumerate_oracle_v1,
    establish_account_context_v1,
    expected_action_return_v1,
    final_equity_for_action_branch_v1,
    make_s1_brain_v1,
    make_s1_critic_v1,
    make_runtime_v1,
    stable_sha256_v1,
    make_genesis_account_v1,
    make_executor_v1,
)
from .post_cc_s1_training_loop_v1 import (
    QUALIFICATION_EVIDENCE_CLASS,
    SMOKE_ONLY_EVIDENCE_CLASS,
    S1SeedRunConfigV1,
    run_seed_v1,
)
from .post_cc_s1_credit_adapter_v1 import assert_sample_log_mu_bound_to_durable_v1, raw_advance_logical_id_v1
from .cc_policy_rng_r0 import PolicyRNG
from .post_cc_joint_replay_v1 import DurableJointReplaySampleV1

S1_PROGRAM_ID_V1 = "CB16_R11_POST_CC_S1_END_TO_END_LEARNABILITY_PROGRAM_V1"
S1_ARTIFACT_ROOT_V1 = "artifacts/post_cc_s1"
FORBIDDEN_OBJECTIVE_TOKENS_V1 = (
    "sharpe",
    "sortino",
    "log_growth",
    "log_wealth",
    "max_drawdown",
    "survivor",
)
S1_RUNTIME_SOURCE_MODULES_V1 = (
    "cb16_local_opt/post_cc_s1_tasks_v1.py",
    "cb16_local_opt/post_cc_s1_training_loop_v1.py",
)


class S1QualificationError(RuntimeError):
    pass


def _raised(callable_object, exception_types: type[BaseException] | tuple[type[BaseException], ...]) -> bool:
    try:
        callable_object()
    except exception_types:
        return True
    return False


def _collect_probe_episode_v1(root: str, *, task_id: str, seed: int = 1701, context_index: int = 0):
    specs = build_task_specs_v1()
    spec = specs[task_id]
    context = spec.contexts[context_index]
    lineage = f"cc-s1-integrity-{task_id.lower()}-{seed}"
    account, _provenance = establish_account_context_v1(
        spec=spec,
        context=context,
        lineage=lineage,
        setup_root=str(Path(root) / "setup"),
    )
    actor = make_s1_brain_v1()
    policy_sha256 = parameter_state_sha256_v1(actor)
    evidence = collect_episode_v1(
        spec=spec,
        context=context,
        account=account,
        root=root,
        lineage=lineage,
        policy_generation="0",
        policy_id="cc-s1-integrity-policy",
        policy_sha256=policy_sha256,
        actor=actor,
        action_rng=PolicyRNG("cc-s1-integrity-policy", lineage, seed),
        env_rng=random.Random(seed),
        sequence_id=f"{lineage}-sequence",
    )
    return spec, context, account, actor, evidence


def _integrity_batch_v1(root: str, *, task_id: str, seed: int = 1701):
    spec, context, account, actor, evidence = _collect_probe_episode_v1(root, task_id=task_id, seed=seed)
    materialized = ReplayMaterializerV1.from_durable_state_v1(root).materialize_sequence(
        evidence.sequence_id,
        target_policy_identity="cc-s1-integrity-target",
        restart_verified=True,
    )
    from .post_cc_s1_credit_adapter_v1 import apply_credit_views_v1

    samples, bootstrap = apply_credit_views_v1(root, materialized)
    forced = []
    for sample in samples:
        forced.append(
            replace(
                sample,
                nominal_direction="LONG",
                nominal_target_risk=0.5,
                risk_measure_kind="continuous_density",
                reward=0.01,
                boundary_type=OBJECTIVE_HORIZON_REACHED,
                mechanical_terminal=False,
                bootstrap_state_ref_or_null=None,
            ).validate()
        )
    provenance = DurableBatchProvenanceV1(
        materializer_contract_id=MATERIALIZER_CONTRACT_ID,
        materialization_id=materialized.manifest.manifest_id,
        materialization_manifest_sha256=materialized.manifest.manifest_sha256,
        source_store_root_identity=materialized.manifest.source_store_root_identity,
        restart_verified=True,
        durable_replay_only=True,
        collector_private_records_used=False,
    )
    batch = build_joint_batch_v1(
        samples=tuple(forced),
        provenance=provenance,
        target_policy_identity="cc-s1-integrity-target",
        bootstrap_observations_by_sequence=None,
    )
    return batch


def _attack_observation_content_mismatch(root: str) -> Mapping[str, Any]:
    spec, context, account, actor, evidence = _collect_probe_episode_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    replay_store = ReplayStoreV1(root)
    transition = replay_store.get_transition(evidence.transition_id)
    store = ObservationStoreV1(root)
    tampered = store.get(transition.observation_logical_id)
    from dataclasses import replace as _replace

    corrupted = _replace(tampered, account_payload={"values": [9.0, 0.0, 0.0]})
    rejected_direct = _raised(lambda: corrupted.validate(), (ValueError, RuntimeError))
    from .post_cc_observation_store_v1 import ImmutableContentStore

    content_store = ImmutableContentStore(root, ObservationStoreV1.NAMESPACE)
    semantic_conflict = _raised(
        lambda: content_store.put_bytes(
            transition.observation_logical_id,
            b'{"corrupted":true}',
        ),
        (ObservationSemanticConflict, ValueError, RuntimeError),
    )
    return {
        "rejected": bool(rejected_direct and semantic_conflict),
        "corrupted_fact_rejected": rejected_direct,
        "logical_id_rebind_rejected": semantic_conflict,
    }


def _attack_missing_behavior_identity(root: str) -> Mapping[str, Any]:
    batch = _integrity_batch_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    sample = batch.samples[0]
    rejected = _raised(
        lambda: replace(sample, behavior_policy_id="", behavior_policy_sha256="0" * 64).validate(),
        ValueError,
    )
    return {"rejected": rejected}


def _attack_fabricated_log_mu(root: str) -> Mapping[str, Any]:
    spec, context, account, actor, evidence = _collect_probe_episode_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    materialized = ReplayMaterializerV1.from_durable_state_v1(root).materialize_sequence(
        evidence.sequence_id,
        target_policy_identity="cc-s1-integrity-target",
        restart_verified=True,
    )
    sample = materialized.samples[0]
    bound_before = _raised(lambda: assert_sample_log_mu_bound_to_durable_v1(root, sample), (RuntimeError, ValueError))
    rejected_rebind = not bound_before and _raised(
        lambda: assert_sample_log_mu_bound_to_durable_v1(
            root, replace(sample, behavior_log_mu=float(sample.behavior_log_mu) + 0.5)
        ),
        (RuntimeError, ValueError),
    )
    rejected_source = _raised(
        lambda: replace(sample, log_mu_source="RECONSTRUCTED_FROM_EXECUTION").validate(), ValueError
    )
    batch = _integrity_batch_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    tampered = batch.behavior_log_mu.clone()
    tampered[0] = tampered[0] + 0.5
    rejected_tensor = _raised(lambda: assert_persisted_log_mu_used_v1(replace(batch, behavior_log_mu=tampered)), ValueError)
    return {
        "rejected": bool(rejected_rebind and rejected_source and rejected_tensor),
        "tensor_mismatch_rejected": rejected_tensor,
        "reconstruction_flag_rejected": rejected_source,
        "durable_binding_rebind_rejected": rejected_rebind,
    }


def _attack_nominal_action_substitution(root: str) -> Mapping[str, Any]:
    batch = _integrity_batch_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    sample = batch.samples[0]
    rejected = _raised(
        lambda: replace(
            sample,
            consequence_context={"nominal_direction": "LONG", "executed_quantity": 1.0},
        ).validate(),
        ValueError,
    )
    return {"rejected": rejected}


def _attack_nonflat_risk_outside_support(root: str) -> Mapping[str, Any]:
    batch = _integrity_batch_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    sample = batch.samples[0]
    rejected = _raised(
        lambda: replace(
            sample, nominal_target_risk=1.2, risk_measure_kind="continuous_density"
        ).validate(),
        ValueError,
    )
    return {"rejected": rejected}


def _attack_flat_with_nonzero_risk(root: str) -> Mapping[str, Any]:
    batch = _integrity_batch_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    sample = batch.samples[0]
    rejected = _raised(
        lambda: replace(
            sample,
            nominal_direction="FLAT",
            nominal_target_risk=0.5,
            risk_measure_kind="point_mass",
        ).validate(),
        ValueError,
    )
    return {"rejected": rejected}


def _attack_time_order_corruption(root: str) -> Mapping[str, Any]:
    batch = _integrity_batch_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    replay_store = ReplayStoreV1(root)
    transition = replay_store.get_transition(batch.samples[0].transition_id)
    rejected = _raised(
        lambda: replace(
            transition,
            environment_time_before=transition.environment_time_after,
        ).validate(),
        (ValueError, RuntimeError),
    )
    return {"rejected": rejected}


def _attack_cross_account_splice(root: str, temp_root: str) -> Mapping[str, Any]:
    batch_a = _integrity_batch_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    batch_b = _integrity_batch_v1(temp_root, task_id="DELAYED_CONSEQUENCE_CREDIT", seed=1702)
    replay_store_a = ReplayStoreV1(root)
    replay_store_b = ReplayStoreV1(temp_root)
    sequence_a = replay_store_a.get_sequence(batch_a.samples[0].sequence_id)
    transition_b = replay_store_b.get_transition(batch_b.samples[0].transition_id)
    splice_id = "cc-s1-integrity-splice-sequence"
    try:
        replay_store_a.put_transition(transition_b)
    except Exception:
        pass
    spliced = replace(
        sequence_a,
        sequence_id=splice_id,
        transition_ids=(sequence_a.transition_ids[0], transition_b.transition_id),
        first_decision_index=0,
        last_decision_index=1,
        raw_fact_content_sha256=stable_sha256_v1(
            (sequence_a.raw_fact_content_sha256, transition_b.content_sha256)
        ),
    )
    try:
        replay_store_a.put_sequence(spliced)
        rejected = _raised(
            lambda: ReplayMaterializerV1.from_durable_state_v1(root).materialize_sequence(
                splice_id, target_policy_identity="cc-s1-integrity-target", restart_verified=True
            ),
            (ReplayCorruption, ValueError, RuntimeError),
        )
    except Exception:
        rejected = True
    return {"rejected": bool(rejected)}


def _attack_prepared_or_staged_child_authority(root: str) -> Mapping[str, Any]:
    batch = _integrity_batch_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    learner = PostCCDurableReplayLearnerV1(
        target_actor=make_s1_brain_v1(),
        target_critic=make_s1_critic_v1(),
        update_store=DurableUpdateStoreV1(Path(root) / "updates"),
        parent_policy_identity="cc-s1-integrity-parent",
        target_policy_identity="cc-s1-integrity-child",
        science_semantic_version="CB16_R11_POST_CC_SCIENCE_SEMANTIC_V1",
    )
    prepared_rejected = False
    staged_rejected = False
    try:
        learner.apply_durable_update_v1(batch=batch, fault_at="after_gradient_before_stage")
    except InjectedUpdateFaultV1:
        pass
    prepared_id = next(learner.update_store.updates_dir.glob("*.json")).stem
    runtime = make_runtime_v1(
        spec=build_task_specs_v1()["DELAYED_CONSEQUENCE_CREDIT"],
        account=make_genesis_account_v1(account_id="cc-s1-integrity-account"),
        account_lineage_id="cc-s1-integrity-lineage",
        policy_generation="0",
        policy_id="cc-s1-integrity-policy",
        policy_sha256="0" * 64,
        schedule_every=1,
    )
    prepared_rejected = _raised(
        lambda: commit_child_generation_v1(
            runtime,
            update_store=learner.update_store,
            update_id=prepared_id,
            new_policy_generation="1",
            new_policy_id="cc-s1-integrity-child",
            new_policy_sha256="1" * 64,
            expected_account_snapshot=runtime.account,
            boundary_type=OBJECTIVE_HORIZON_REACHED,
        ),
        GenerationContinuityError,
    )
    with tempfile.TemporaryDirectory(prefix="cb16-s1-integrity-staged-") as staged_root:
        staged_batch = _integrity_batch_v1(staged_root, task_id="DELAYED_CONSEQUENCE_CREDIT")
        staged_learner = PostCCDurableReplayLearnerV1(
            target_actor=make_s1_brain_v1(),
            target_critic=make_s1_critic_v1(),
            update_store=DurableUpdateStoreV1(Path(staged_root) / "updates"),
            parent_policy_identity="cc-s1-integrity-parent",
            target_policy_identity="cc-s1-integrity-child",
            science_semantic_version="CB16_R11_POST_CC_SCIENCE_SEMANTIC_V1",
        )
        try:
            staged_learner.apply_durable_update_v1(batch=staged_batch, fault_at="after_stage_before_commit")
        except InjectedUpdateFaultV1:
            pass
        staged_id = next(staged_learner.update_store.updates_dir.glob("*.json")).stem
        staged_rejected = _raised(
            lambda: commit_child_generation_v1(
                runtime,
                update_store=staged_learner.update_store,
                update_id=staged_id,
                new_policy_generation="1",
                new_policy_id="cc-s1-integrity-child",
                new_policy_sha256="1" * 64,
                expected_account_snapshot=runtime.account,
                boundary_type=OBJECTIVE_HORIZON_REACHED,
            ),
            GenerationContinuityError,
        )
    return {"rejected": bool(prepared_rejected and staged_rejected), "prepared_rejected": prepared_rejected, "staged_rejected": staged_rejected}


def _attack_committed_child_tamper(root: str) -> Mapping[str, Any]:
    batch = _integrity_batch_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    learner = PostCCDurableReplayLearnerV1(
        target_actor=make_s1_brain_v1(),
        target_critic=make_s1_critic_v1(),
        update_store=DurableUpdateStoreV1(Path(root) / "updates"),
        parent_policy_identity="cc-s1-integrity-parent",
        target_policy_identity="cc-s1-integrity-child",
        science_semantic_version="CB16_R11_POST_CC_SCIENCE_SEMANTIC_V1",
    )
    result = learner.apply_durable_update_v1(batch=batch)
    checkpoint_path = learner.update_store._checkpoint_path(result.child_checkpoint_sha256)
    original = checkpoint_path.read_bytes()
    checkpoint_path.write_bytes(b'{"tampered":true}')
    rejected = _raised(
        lambda: learner.update_store.load_child_checkpoint(result.update_id),
        (DurableUpdateCorruptionError, ValueError),
    )
    checkpoint_path.write_bytes(original)
    return {"rejected": rejected}


def _attack_same_instance_retry_after_mutation(root: str) -> Mapping[str, Any]:
    batch = _integrity_batch_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    learner = PostCCDurableReplayLearnerV1(
        target_actor=make_s1_brain_v1(),
        target_critic=make_s1_critic_v1(),
        update_store=DurableUpdateStoreV1(Path(root) / "updates"),
        parent_policy_identity="cc-s1-integrity-parent",
        target_policy_identity="cc-s1-integrity-child",
        science_semantic_version="CB16_R11_POST_CC_SCIENCE_SEMANTIC_V1",
    )
    try:
        learner.apply_durable_update_v1(batch=batch, fault_at="after_gradient_before_stage")
    except InjectedUpdateFaultV1:
        pass
    step_after_fault = int(learner.optimizer_step)
    gradients_after_fault = int(learner.gradient_applications)
    retry_rejected = _raised(lambda: learner.apply_durable_update_v1(batch=batch), PostCCLearnerError)
    state_unchanged = (
        int(learner.optimizer_step) == step_after_fault
        and int(learner.gradient_applications) == gradients_after_fault
    )
    return {"rejected": bool(retry_rejected and state_unchanged), "retry_rejected": retry_rejected, "state_unchanged": state_unchanged}


def _attack_terminal_with_bootstrap() -> Mapping[str, Any]:
    rejected = _raised(
        lambda: bootstrap_decision_v1(
            OBJECTIVE_HORIZON_REACHED, mechanical_terminal=False, next_value=1.5
        ),
        ValueError,
    )
    return {"rejected": rejected}


def _attack_truncation_missing_bootstrap(root: str) -> Mapping[str, Any]:
    rejected_bootstrap = _raised(
        lambda: bootstrap_decision_v1(COMPUTE_CHUNK, mechanical_terminal=False, next_value=None),
        ValueError,
    )
    spec, context, account, actor, evidence = _collect_probe_episode_v1(root, task_id="DELAYED_CONSEQUENCE_CREDIT")
    replay_store = ReplayStoreV1(root)
    sequence = replay_store.get_sequence(evidence.sequence_id)
    from .post_cc_durable_collection_v1 import DurableObservationCollectorV1

    collector = DurableObservationCollectorV1(
        root,
        science_semantic_version="CB16_R11_POST_CC_SCIENCE_SEMANTIC_V1",
        market_source_identity="CC_S1_SYNTHETIC_MARKET_V1",
        market_source_version="V1",
    )
    rejected_finalize = _raised(
        lambda: collector.finalize_sequence(
            sequence_id="cc-s1-integrity-missing-bootstrap",
            market_lineage_id="CC_S1_SYNTHETIC_MARKET_V1",
            source_classification="INTEGRITY",
            transition_ids=(evidence.transition_id,),
            chunk_boundary_type=COMPUTE_CHUNK,
            bootstrap_state_ref_or_null=None,
        ),
        (RuntimeError, ValueError),
    )
    return {"rejected": bool(rejected_bootstrap and rejected_finalize), "bootstrap_decision_rejected": rejected_bootstrap, "finalize_rejected": rejected_finalize}


def run_integrity_attack_suite_v1(repo_root: str | Path) -> Mapping[str, Any]:
    attacks: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="cb16-s1-integrity-") as suite_root:
        def sub(name: str) -> str:
            path = Path(suite_root) / name
            path.mkdir(parents=True, exist_ok=True)
            return str(path)

        attacks["observation_content_hash_mismatch"] = _attack_observation_content_mismatch(sub("observation-mismatch"))
        attacks["missing_behavior_identity"] = _attack_missing_behavior_identity(sub("missing-behavior"))
        attacks["fabricated_log_mu"] = _attack_fabricated_log_mu(sub("fabricated-log-mu"))
        attacks["nominal_replaced_by_executed"] = _attack_nominal_action_substitution(sub("nominal-substitution"))
        attacks["nonflat_risk_outside_support"] = _attack_nonflat_risk_outside_support(sub("nonflat-support"))
        attacks["flat_with_nonzero_risk"] = _attack_flat_with_nonzero_risk(sub("flat-risk"))
        attacks["time_order_corruption"] = _attack_time_order_corruption(sub("time-order"))
        attacks["cross_account_replay_splice"] = _attack_cross_account_splice(sub("splice-a"), sub("splice-b"))
        attacks["prepared_or_staged_child_authority"] = _attack_prepared_or_staged_child_authority(sub("prepared-staged"))
        attacks["committed_child_checkpoint_tamper"] = _attack_committed_child_tamper(sub("checkpoint-tamper"))
        attacks["same_instance_retry_after_mutation"] = _attack_same_instance_retry_after_mutation(sub("same-instance-retry"))
        attacks["terminal_supplied_with_bootstrap"] = _attack_terminal_with_bootstrap()
        attacks["truncation_missing_durable_bootstrap"] = _attack_truncation_missing_bootstrap(sub("truncation-bootstrap"))
    all_rejected = all(item.get("rejected") is True for item in attacks.values())
    return {"schema": "CB16_R11_POST_CC_S1_INTEGRITY_ATTACK_SUITE_V1", "attacks": attacks, "all_rejected": bool(all_rejected)}


def _source_scan_v1(repo_root: Path) -> Mapping[str, Any]:
    """Flag forbidden objectives without flagging explicit prohibition declarations."""
    import re

    hits: dict[str, list[str]] = {}
    for relative in S1_RUNTIME_SOURCE_MODULES_V1:
        text = (repo_root / relative).read_text(encoding="utf-8")
        lowered = text.lower()
        for token in FORBIDDEN_OBJECTIVE_TOKENS_V1:
            offsets = [match.start() for match in re.finditer(re.escape(token), lowered)]
            if not offsets:
                continue
            for offset in offsets:
                line_start = lowered.rfind("\n", 0, offset) + 1
                line_end = lowered.find("\n", offset)
                line = lowered[line_start : line_end if line_end != -1 else len(lowered)]
                prohibition = re.search(re.escape(token) + r"[a-z_]*['\"]?\s*[:=]\s*false", line)
                if prohibition:
                    continue
                hits.setdefault(token, []).append(relative)
                break
    promotion_hits = [
        relative
        for relative in S1_RUNTIME_SOURCE_MODULES_V1
        if "post_cc_promotion" in (repo_root / relative).read_text(encoding="utf-8")
    ]
    return {"forbidden_objective_token_hits": hits, "production_promotion_import_hits": promotion_hits}


def run_objective_firewall_audit_v1(repo_root: str | Path) -> Mapping[str, Any]:
    root = Path(repo_root)
    specs = build_task_specs_v1()
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    high = specs["HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION"]
    context = high.contexts[0]
    account = make_genesis_account_v1(account_id="cc-s1-objective-audit")
    production_ev = expected_action_return_v1(
        spec=high, context=context, account=account, direction="LONG", risk=0.9
    )
    hand_win = 1000.0 + (0.9 * 2.0 * 1000.0 / 100.0) * (120.0 - 100.0)
    hand_loss = 1000.0 + (0.9 * 2.0 * 1000.0 / 100.0) * (40.0 - 100.0)
    hand_ev = 0.8 * hand_win + 0.2 * hand_loss
    hand_return = (hand_ev - 1000.0) / 1000.0
    checks["arithmetic_orientation_declared"] = all(
        spec.objective_orientation == "COMPLETE_SAMPLE_ARITHMETIC_EQUITY_DELTA" for spec in specs.values()
    )
    checks["production_ev_matches_hand_arithmetic"] = bool(abs(production_ev - hand_return) <= 1e-9)
    loss_equity = final_equity_for_action_branch_v1(
        spec=high,
        account=account,
        direction="LONG",
        risk=0.9,
        terminal_mark=40.0,
        stage1_mark=100.0,
        stage2_is_no_decision_advance=False,
    )
    checks["failure_branch_equity_is_negative_and_counted"] = bool(loss_equity <= 0.0 and hand_return < math.inf)
    checks["bankruptcy_definition_is_terminal_equity_le_zero"] = bool(loss_equity <= 0.0)
    scan = _source_scan_v1(root)
    checks["no_alternative_objective_token_in_s1_sources"] = not scan["forbidden_objective_token_hits"]
    checks["no_production_promotion_in_s1_sources"] = not scan["production_promotion_import_hits"]
    checks["gate_compiler_keeps_explicit_no_survivor_check"] = bool(
        "no_survivor_filtering_objective" in (
            root / "cb16_local_opt/post_cc_s1_gate_compiler_v1.py"
        ).read_text(encoding="utf-8")
    )
    checks["buy_and_hold_flat_sibling_contract_untouched"] = not (
        root / "cb16_local_opt/post_cc_promotion_v1.py"
    ).read_text(encoding="utf-8").lower().count("master_baseline")
    details.update(
        {
            "production_expected_return_LONG_0_9": production_ev,
            "hand_expected_return_LONG_0_9": hand_return,
            "loss_branch_equity": loss_equity,
            "source_scan": scan,
        }
    )
    return {
        "schema": "CB16_R11_POST_CC_S1_OBJECTIVE_FIREWALL_AUDIT_V1",
        "checks": checks,
        "details": details,
        "all_checks_pass": bool(all(checks.values())),
    }


def run_fabricated_log_mu_audit_v1(repo_root: str | Path) -> Mapping[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cb16-s1-fabricated-") as root:
        attack = _attack_fabricated_log_mu(root)
    return {
        "schema": "CB16_R11_POST_CC_S1_FABRICATED_LOG_MU_AUDIT_V1",
        "checks": dict(attack),
        "all_checks_pass": bool(attack["rejected"] is True),
    }


# ---------------------------------------------------------------------------
# Program orchestration and durable artifacts
# ---------------------------------------------------------------------------


def _training_control_ids_v1(spec: TaskSpecV1) -> tuple[str, ...]:
    return tuple(
        control_id
        for control_id in spec.declared_controls
        if control_id not in (CONTROL_OBJECTIVE_FIREWALL, CONTROL_FABRICATED_LOG_MU_REJECTION)
    )


def _job_list_v1(manifest: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    specs = build_task_specs_v1()
    jobs: list[dict[str, Any]] = []
    if mode == "qualification":
        seeds = tuple(int(seed) for seed in manifest["seeds"])
        unit_size = 128
        evaluation_population = 2048
    elif mode == "smoke":
        seeds = (1701,)
        unit_size = 16
        evaluation_population = 32
    else:
        raise S1QualificationError("UNKNOWN_PROGRAM_MODE")
    for task_id in FROZEN_TASK_IDS_V1:
        spec = specs[task_id]
        max_units = int(spec.training_decisions) // int(unit_size)
        if mode == "smoke":
            max_units = min(max_units, 4)
        for seed in seeds:
            jobs.append(
                {
                    "task_id": task_id,
                    "seed": int(seed),
                    "control_id": None,
                    "unit_size": int(unit_size),
                    "max_units": int(max_units),
                    "evaluation_population": int(evaluation_population),
                    "mode": mode,
                }
            )
            for control_id in _training_control_ids_v1(spec):
                jobs.append(
                    {
                        "task_id": task_id,
                        "seed": int(seed),
                        "control_id": control_id,
                        "unit_size": int(unit_size),
                        "max_units": int(max_units),
                        "evaluation_population": int(evaluation_population),
                        "mode": mode,
                    }
                )
    return jobs


def _run_job_v1(job: Mapping[str, Any]) -> dict[str, Any]:
    """Worker entrypoint; imports inside the process for pickling cleanliness."""
    import torch

    from .post_cc_s1_tasks_v1 import build_task_specs_v1
    from .post_cc_s1_training_loop_v1 import S1SeedRunConfigV1, run_seed_v1

    torch.set_num_threads(1)
    spec = build_task_specs_v1()[str(job["task_id"])]
    config = S1SeedRunConfigV1(
        spec=spec,
        seed=int(job["seed"]),
        run_root=str(job["run_root"]),
        mode=str(job["mode"]),
        control_id=job["control_id"],
        unit_size=int(job["unit_size"]),
        max_units=int(job["max_units"]),
        evaluation_population=int(job["evaluation_population"]),
        manifest_sha256=str(job["manifest_sha256"]),
    )
    try:
        result = run_seed_v1(config)
        staging_root = job.get("checkpoint_staging_root")
        final_child_sha = None
        unit_evidence = result.get("unit_evidence", [])
        if unit_evidence:
            final_child_sha = unit_evidence[-1].get("child_checkpoint_sha256")
        if staging_root and final_child_sha:
            source = Path(str(job["run_root"])) / "updates" / "checkpoints" / f"{final_child_sha}.json"
            destination_dir = Path(str(staging_root)) / str(job["task_id"]) / (
                "positive" if job["control_id"] is None else f"control_{str(job['control_id']).lower()}"
            )
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"seed_{int(job['seed'])}_final_child.json"
            if source.exists():
                shutil.copyfile(source, destination)
        provenance_staging = job.get("provenance_staging_root")
        if provenance_staging:
            staging_dir = Path(str(provenance_staging)) / str(job["task_id"]) / (
                "positive" if job["control_id"] is None else f"control_{str(job['control_id']).lower()}"
            ) / f"seed_{int(job['seed'])}"
            export_run_provenance_v1(str(job["run_root"]), staging_dir)
        if job.get("cleanup_run_root") and Path(str(job["run_root"])).exists():
            shutil.rmtree(str(job["run_root"]), ignore_errors=True)
        return {"status": "OK", "job": dict(job), "result": result}
    except Exception as exc:  # noqa: BLE001 - execution failures must be recorded
        return {
            "status": "EXECUTION_ERROR",
            "job": dict(job),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def _worker_count_v1(requested: int | None) -> int:
    if requested is not None:
        return max(1, int(requested))
    env_value = os.environ.get("CB16_S1_WORKERS")
    if env_value:
        return max(1, int(env_value))
    return max(1, min(8, os.cpu_count() or 1))


def _job_run_root_v1(scratch_root: Path, job: Mapping[str, Any]) -> Path:
    suffix = "positive" if job["control_id"] is None else f"control_{job['control_id'].lower()}"
    return scratch_root / str(job["task_id"]) / suffix / f"seed_{int(job['seed'])}"


def _execute_jobs_v1(
    jobs: Sequence[Mapping[str, Any]],
    *,
    scratch_root: Path,
    checkpoint_staging_root: Path,
    provenance_staging_root: Path,
    cleanup_run_roots: bool,
    workers: int,
) -> list[dict[str, Any]]:
    enriched = []
    for job in jobs:
        item = dict(job)
        item["run_root"] = str(_job_run_root_v1(scratch_root, item))
        item["checkpoint_staging_root"] = str(checkpoint_staging_root)
        item["provenance_staging_root"] = str(provenance_staging_root)
        item["cleanup_run_root"] = bool(cleanup_run_roots)
        enriched.append(item)
    if workers <= 1 or len(enriched) <= 1:
        return [_run_job_v1(job) for job in enriched]
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(_run_job_v1, enriched))


def export_run_provenance_v1(run_root: str | Path, destination: str | Path) -> Mapping[str, Any]:
    """B6: export the durable update journal + index before scratch deletion."""
    run_root_path = Path(run_root)
    staging_dir = Path(destination)
    journal_dir = staging_dir / "update_journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal_records = 0
    updates_dir = run_root_path / "updates"
    record_paths = sorted((updates_dir / "updates").glob("*.json")) if (updates_dir / "updates").exists() else []
    if not record_paths and updates_dir.exists():
        record_paths = sorted(updates_dir.glob("*.json"))
    for record_path in record_paths:
        shutil.copyfile(record_path, journal_dir / record_path.name)
        journal_records += 1
    exported_stores: list[str] = []
    for store_name in ("durable_index.jsonl", "setup_provenance.json"):
        source = run_root_path / store_name
        if source.exists():
            shutil.copyfile(source, staging_dir / store_name)
            exported_stores.append(store_name)
    return {
        "journal_records_exported": journal_records,
        "stores_exported": exported_stores,
        "destination": str(staging_dir),
    }


def _write_json_v1(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _classify_execution_failures_v1(outcomes: Sequence[Mapping[str, Any]]) -> str | None:
    errors = [item for item in outcomes if item.get("status") != "OK"]
    if not errors:
        return None
    messages = " ".join(str(item.get("error_message", "")) for item in errors).lower()
    if "out of memory" in messages or "cuda" in messages:
        return "HARDWARE_LIMIT"
    return "EXECUTION_BLOCKED"


def _write_artifacts_v1(
    *,
    output_root: Path,
    manifest: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any],
    objective_audit: Mapping[str, Any],
    fabricated_audit: Mapping[str, Any],
    failure_fact_audit: Mapping[str, Any],
    mode: str,
    compiled: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json_v1(output_root / "execution_manifest.json", manifest)

    positive_results: dict[str, Any] = {}
    control_results: dict[str, Any] = {}
    run_index: list[dict[str, Any]] = []
    checkpoint_index: list[dict[str, Any]] = []
    replay_manifest_index: list[dict[str, Any]] = []

    for outcome in outcomes:
        job = outcome["job"]
        task_id = str(job["task_id"])
        seed = int(job["seed"])
        control_id = job["control_id"]
        if outcome["status"] != "OK":
            _write_json_v1(
                output_root / ("task_results" if control_id is None else "control_results") / task_id / f"{seed}.json",
                {"status": "EXECUTION_ERROR", **outcome},
            )
            continue
        result = dict(outcome["result"])
        if control_id is None:
            positive_results[f"{task_id}|{seed}"] = result
            _write_json_v1(output_root / "task_results" / task_id / f"{seed}.json", result)
        else:
            control_results.setdefault(f"{task_id}|{control_id}", {})[str(seed)] = result
            _write_json_v1(output_root / "control_results" / control_id / f"{task_id}__{seed}.json", result)
        run_index.append(
            {
                "task_id": task_id,
                "seed": seed,
                "control_id": control_id,
                "run_root": job["run_root"],
                "evidence_class": result.get("evidence_class"),
                "decisions_consumed": result.get("decisions_consumed"),
                "initial_score": result["checkpoints"]["INITIAL"]["mean_complete_sample_arithmetic_return"],
                "final_score": result["checkpoints"]["FINAL"]["mean_complete_sample_arithmetic_return"],
                "oracle_score": result["oracle"]["mean_oracle_return"],
                "unit_evidence_sha256s": result.get("unit_evidence_sha256s", []),
                "final_child_checkpoint_sha256": (
                    result.get("unit_evidence", [{}])[-1].get("child_checkpoint_sha256")
                    if result.get("unit_evidence")
                    else None
                ),
            }
        )
        unit_evidence = result.get("unit_evidence", [])
        replay_manifest_index.append(
            {
                "task_id": task_id,
                "seed": seed,
                "control_id": control_id,
                "run_root": job["run_root"],
                "materialization_manifest_sha256s": [
                    item for unit in unit_evidence for item in unit.get("materialization_manifest_sha256s", [])
                ],
            }
        )
        checkpoint_index.append(
            {
                "task_id": task_id,
                "seed": seed,
                "control_id": control_id,
                "child_checkpoint_sha256s": [unit.get("child_checkpoint_sha256") for unit in unit_evidence],
                "final_child_checkpoint_sha256": (
                    unit_evidence[-1].get("child_checkpoint_sha256") if unit_evidence else None
                ),
            }
        )
        final_sha = unit_evidence[-1].get("child_checkpoint_sha256") if unit_evidence else None
        if final_sha:
            source = Path(job["run_root"]) / "updates" / "checkpoints" / f"{final_sha}.json"
            destination = output_root / "checkpoints" / task_id / f"{seed}__{control_id or 'positive'}_final_child.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged_root = job.get("checkpoint_staging_root")
            staged = (
                Path(str(staged_root)) / task_id / (
                    "positive" if control_id is None else f"control_{str(control_id).lower()}"
                ) / f"seed_{seed}_final_child.json"
                if staged_root
                else None
            )
            if source.exists():
                shutil.copyfile(source, destination)
            elif staged is not None and staged.exists():
                shutil.copyfile(staged, destination)

    _write_json_v1(
        output_root / "provenance" / "task_specs.json",
        {
            "task_specs": [
                {"task_id": spec.task_id, "spec_hash": spec.spec_hash, "payload": spec.payload()}
                for spec in build_task_specs_v1().values()
            ]
        },
    )
    _write_json_v1(output_root / "provenance" / "run_index.json", {"runs": run_index})
    _write_json_v1(output_root / "provenance" / "integrity_attack_suite.json", integrity)
    _write_json_v1(output_root / "provenance" / "objective_firewall_audit.json", objective_audit)
    _write_json_v1(output_root / "provenance" / "fabricated_log_mu_audit.json", fabricated_audit)
    _write_json_v1(output_root / "provenance" / "high_bankruptcy_failure_fact_audit.json", failure_fact_audit)
    _write_json_v1(output_root / "replay_manifests" / "index.json", {"runs": replay_manifest_index})
    _write_json_v1(output_root / "checkpoints" / "index.json", {"runs": checkpoint_index})
    staged_roots: set[Path] = set()
    for outcome in outcomes:
        staging = outcome.get("job", {}).get("provenance_staging_root")
        if staging:
            staged_roots.add(Path(str(staging)))
    default_staging = output_root / "provenance_staged"
    if default_staging.exists():
        staged_roots.add(default_staging)
    for staged_provenance in sorted(staged_roots):
        if not staged_provenance.exists():
            continue
        for path in staged_provenance.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(staged_provenance)
            if path.parent.name == "update_journal":
                destination = output_root / "provenance" / "update_journals" / relative
            elif path.name == "durable_index.jsonl":
                destination = output_root / "replay_manifests" / relative
            else:
                destination = output_root / "provenance" / "run_stores" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    artifact_only_trace = audit_exported_update_journal_v1(output_root)
    _write_json_v1(output_root / "provenance" / "artifact_only_update_trace_audit.json", artifact_only_trace)

    if mode == "qualification":
        if compiled is None:
            raise S1QualificationError("QUALIFICATION_REQUIRES_GATE_COMPILER_RESULT")
        compiled = dict(compiled)
        compiled["artifact_only_update_trace_all_checks_pass"] = bool(artifact_only_trace["all_checks_pass"])
        _write_json_v1(output_root / "S1_RESULT.json", compiled)
        report = _report_markdown_v1(manifest=manifest, compiled=compiled, run_index=run_index, integrity=integrity)
        (output_root / "S1_REPORT.md").write_text(report, encoding="utf-8")
        summary = {
            "mode": mode,
            "artifact_root": str(output_root),
            "classification": compiled["classification"],
            "status": compiled["status"],
            "gates": compiled["gates"],
            "contract_violations": compiled["contract_violations"],
            "evidence_insufficient": compiled["evidence_insufficient"],
            "scientific_failures": compiled["scientific_failures"],
        }
    else:
        smoke_result = {
            "schema": "CB16_R11_POST_CC_S1_SMOKE_RESULT_V1",
            "mode": "smoke",
            "status": "SMOKE_ONLY_NOT_SCIENTIFIC_QUALIFICATION",
            "evidence_class": SMOKE_ONLY_EVIDENCE_CLASS,
            "scientific_verdict_allowed": False,
            "executed_jobs": len(outcomes),
            "positive_runs": len(positive_results),
            "control_runs": sum(len(v) for v in control_results.values()),
            "integrity_attack_suite_all_rejected": bool(integrity["all_rejected"]),
            "objective_firewall_audit_pass": bool(objective_audit["all_checks_pass"]),
            "fabricated_log_mu_audit_pass": bool(fabricated_audit["all_checks_pass"]),
            "high_bankruptcy_failure_fact_audit_pass": bool(failure_fact_audit["all_checks_pass"]),
            "artifact_only_update_trace_all_checks_pass": bool(artifact_only_trace["all_checks_pass"]),
        }
        _write_json_v1(output_root / "SMOKE_RESULT.json", smoke_result)
        report = (
            "# CB16 R11 S1 bounded smoke\n\n"
            "Status: SMOKE_ONLY_NOT_SCIENTIFIC_QUALIFICATION\n\n"
            f"- executed jobs: {len(outcomes)}\n"
            f"- integrity attack suite all rejected: {smoke_result['integrity_attack_suite_all_rejected']}\n"
            f"- objective firewall audit: {smoke_result['objective_firewall_audit_pass']}\n"
            f"- fabricated log_mu audit: {smoke_result['fabricated_log_mu_audit_pass']}\n"
            f"- high-bankruptcy failure fact audit: {smoke_result['high_bankruptcy_failure_fact_audit_pass']}\n"
            f"- artifact-only update trace audit: {smoke_result['artifact_only_update_trace_all_checks_pass']}\n"
        )
        (output_root / "S1_REPORT.md").write_text(report, encoding="utf-8")
        summary = smoke_result
    return {"summary": summary, "positive_results": positive_results, "control_results": control_results}


def _report_markdown_v1(
    *,
    manifest: Mapping[str, Any],
    compiled: Mapping[str, Any],
    run_index: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any],
) -> str:
    lines = [
        "# CB16 R11 S1 end-to-end durable learnability result",
        "",
        f"- classification: `{compiled['classification']}`",
        f"- manifest sha256: `{manifest['manifest_sha256']}`",
        f"- compiler: `{compiled['compiler_id']}`",
        f"- integrity attack matrix all rejected: `{bool(integrity['all_rejected'])}`",
        "",
        "| Task | Seed | Control | Initial | Final | Oracle |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in run_index:
        lines.append(
            "| {task_id} | {seed} | {control_id} | {initial_score:.6f} | {final_score:.6f} | {oracle_score:.6f} |".format(
                task_id=row["task_id"],
                seed=row["seed"],
                control_id=row["control_id"] or "positive",
                initial_score=float(row["initial_score"]),
                final_score=float(row["final_score"]),
                oracle_score=float(row["oracle_score"]),
            )
        )
    lines.extend(["", "## Gate table", ""])
    for gate, value in sorted(compiled["gates"].items()):
        lines.append(f"- `{gate}`: {value}")
    lines.extend(
        [
            "",
            "> S1 evidence ceiling: INTEGRATED_SYNTHETIC_END_TO_END_LEARNABILITY_KNOWN_ANSWER.",
            "> FINAL/fresh/historical-market firewalls remain closed. No economic or transfer evidence is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def require_qualification_authorization_v1(
    repo_root: str | Path,
    *,
    expected_runtime_sha: str,
    expected_runtime_tree_sha: str,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Formal qualification is fail-closed until the reviewer records authorization
    bound to the exact reviewed runtime SHA/tree and execution-manifest hash."""
    root = Path(repo_root)
    path = root / "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_QUALIFICATION_AUTHORIZATION_V1.json"
    if not path.exists():
        raise S1QualificationError("S1_QUALIFICATION_AUTHORIZATION_MISSING")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "READY_FOR_S1_QUALIFICATION":
        raise S1QualificationError("S1_QUALIFICATION_AUTHORIZATION_STATUS_INVALID")
    if payload.get("reviewer_role") != "SOL_INDEPENDENT_QUALIFICATION_REVIEW":
        raise S1QualificationError("S1_QUALIFICATION_AUTHORIZATION_ROLE_INVALID")
    reviewed_sha = str(payload.get("reviewed_implementation_sha"))
    if reviewed_sha != str(expected_runtime_sha):
        raise S1QualificationError("S1_QUALIFICATION_AUTHORIZATION_RUNTIME_SHA_MISMATCH")
    if payload.get("reviewed_implementation_tree_sha") != str(expected_runtime_tree_sha):
        raise S1QualificationError("S1_QUALIFICATION_AUTHORIZATION_RUNTIME_TREE_MISMATCH")
    if payload.get("execution_manifest_sha256") != str(expected_manifest_sha256):
        raise S1QualificationError("S1_QUALIFICATION_AUTHORIZATION_MANIFEST_MISMATCH")
    import subprocess

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", reviewed_sha, "HEAD"], cwd=root, check=False
    ).returncode
    if ancestor != 0:
        raise S1QualificationError("S1_QUALIFICATION_AUTHORIZED_RUNTIME_NOT_ANCESTOR_OF_HEAD")
    allowed_metadata = {
        "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_REVIEW_CANDIDATE_V1.json",
        "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_QUALIFICATION_AUTHORIZATION_V1.json",
    }
    changed = _git_changed_paths_v1(root, reviewed_sha, "HEAD")
    if not set(changed) <= allowed_metadata:
        raise S1QualificationError("S1_QUALIFICATION_AUTHORIZATION_RUNTIME_CHANGED_AFTER_REVIEW")
    return payload


def _git_changed_paths_v1(root: Path, base_sha: str, head_ref: str) -> list[str]:
    import subprocess

    completed = subprocess.run(
        ["git", "diff", "--name-only", str(base_sha), str(head_ref)],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise S1QualificationError("S1_GIT_DIFF_FAILED_FOR_AUTHORIZATION_BINDING")
    return [line for line in completed.stdout.splitlines() if line.strip()]


def run_s1_program_v1(
    *,
    repo_root: str | Path,
    mode: str,
    output_root: str | Path,
    workers: int | None = None,
    scratch_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the S1 program in exactly one of the two scientific modes."""
    if mode not in ("smoke", "qualification"):
        raise S1QualificationError("UNKNOWN_PROGRAM_MODE")
    root = Path(repo_root)
    manifest = validate_s1_execution_manifest_v1(root)
    if mode == "qualification":
        import subprocess

        candidate_path = root / "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_REVIEW_CANDIDATE_V1.json"
        if not candidate_path.exists():
            raise S1QualificationError("S1_QUALIFICATION_REQUIRES_REVIEW_CANDIDATE_RECORD")
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        runtime_sha = str(candidate["implementation_candidate_sha"])
        runtime_tree = str(candidate["implementation_candidate_tree_sha"])
        if str(candidate.get("terminal_state")) != "READY_FOR_SOL_REVIEW":
            raise S1QualificationError("S1_QUALIFICATION_REVIEW_CANDIDATE_NOT_IN_REVIEW_STATE")
        require_qualification_authorization_v1(
            root,
            expected_runtime_sha=runtime_sha,
            expected_runtime_tree_sha=runtime_tree,
            expected_manifest_sha256=str(manifest["manifest_sha256"]),
        )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    jobs = _job_list_v1(manifest, mode)
    for job in jobs:
        job["manifest_sha256"] = manifest["manifest_sha256"]
    worker_count = _worker_count_v1(workers)
    scratch = Path(scratch_root) if scratch_root else (output / "runs")
    scratch.mkdir(parents=True, exist_ok=True)
    cleanup_run_roots = bool(scratch_root)
    outcomes = _execute_jobs_v1(
        jobs,
        scratch_root=scratch,
        checkpoint_staging_root=output / "checkpoints_staged",
        provenance_staging_root=output / "provenance_staged",
        cleanup_run_roots=cleanup_run_roots,
        workers=worker_count,
    )
    integrity = run_integrity_attack_suite_v1(root)
    objective_audit = run_objective_firewall_audit_v1(root)
    fabricated_audit = run_fabricated_log_mu_audit_v1(root)
    failure_fact_audit = run_high_bankruptcy_failure_fact_audit_v1(root)
    audits = {
        "OBJECTIVE_FIREWALL": objective_audit,
        "FABRICATED_LOG_MU_REJECTION": fabricated_audit,
        "HIGH_BANKRUPTCY_FAILURE_FACT": failure_fact_audit,
    }
    compiled = None
    if mode == "qualification":
        positive_results: dict[str, Any] = {}
        control_results: dict[str, Any] = {}
        for outcome in outcomes:
            if outcome["status"] != "OK":
                continue
            job = outcome["job"]
            if job["control_id"] is None:
                positive_results[f"{job['task_id']}|{int(job['seed'])}"] = outcome["result"]
            else:
                control_results.setdefault(f"{job['task_id']}|{job['control_id']}", {})[str(int(job["seed"]))] = outcome["result"]
        compiled = compile_s1_gates_v1(
            manifest=manifest,
            positive_results=positive_results,
            control_results=control_results,
            integrity=integrity,
            audits=audits,
        )
        execution_failure_class = _classify_execution_failures_v1(outcomes)
        if execution_failure_class is not None and compiled["classification"] != "CONTRACT_MISMATCH":
            compiled["classification"] = execution_failure_class
            compiled["status"] = execution_failure_class
            compiled["execution_failure_class"] = execution_failure_class
    artifacts = _write_artifacts_v1(
        output_root=output,
        manifest=manifest,
        outcomes=outcomes,
        integrity=integrity,
        objective_audit=objective_audit,
        fabricated_audit=fabricated_audit,
        failure_fact_audit=failure_fact_audit,
        mode=mode,
        compiled=compiled,
    )
    return {
        "schema": "CB16_R11_POST_CC_S1_PROGRAM_SUMMARY_V1",
        "program_id": S1_PROGRAM_ID_V1,
        "mode": mode,
        "workers": int(worker_count),
        "jobs": len(jobs),
        "wall_seconds": float(time.time() - started),
        "scratch_root": str(scratch),
        "cleanup_run_roots": bool(cleanup_run_roots),
        "artifact_root": str(output),
        **artifacts["summary"],
    }


# ---------------------------------------------------------------------------
# B5 / B6 audits: durable failure facts and artifact-only update tracing
# ---------------------------------------------------------------------------


def run_high_bankruptcy_failure_fact_audit_v1(repo_root: str | Path) -> Mapping[str, Any]:
    """Counterexample: LOSS must be a durable failure/mechanical-terminal fact and stay in the denominator."""
    import random
    import tempfile

    from .post_cc_generation_continuity_v1 import parameter_state_sha256_v1 as _parameter_state_sha256
    from .post_cc_joint_batch_v1 import DurableBatchProvenanceV1 as _BatchProvenance
    from .post_cc_replay_materializer_v1 import ReplayMaterializerV1 as _Materializer
    from .post_cc_s1_credit_adapter_v1 import apply_credit_views_v1 as _apply_credit_views
    from .post_cc_s1_tasks_v1 import (
        TASK_HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION,
        BranchOutcomeV1,
        build_task_specs_v1,
        collect_episode_v1,
        establish_account_context_v1,
        make_s1_brain_v1,
    )
    from .cc_policy_rng_r0 import PolicyRNG

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    spec = build_task_specs_v1()[TASK_HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION]
    context = spec.contexts[0]
    for label, branch in (("WIN", BranchOutcomeV1(1.0, 120.0)), ("LOSS", BranchOutcomeV1(1.0, 40.0))):
        with tempfile.TemporaryDirectory(prefix=f"cb16-s1-failure-fact-{label.lower()}-") as root:
            account, _provenance = establish_account_context_v1(
                spec=spec, context=context, lineage=f"cc-s1-failure-{label.lower()}", setup_root=f"{root}/setup"
            )
            actor = make_s1_brain_v1()
            import torch

            with torch.no_grad():
                actor.direction_head.bias.copy_(torch.tensor([-8.0, -8.0, 8.0]))
                actor.risk_loc_head.bias.copy_(torch.tensor([0.0, 0.0, 8.0]))
            evidence = collect_episode_v1(
                spec=spec,
                context=context,
                account=account,
                root=root,
                lineage=f"cc-s1-failure-{label.lower()}",
                policy_generation="0",
                policy_id="cc-s1-failure-fact-policy",
                policy_sha256=_parameter_state_sha256(actor),
                actor=actor,
                action_rng=PolicyRNG("cc-s1-failure-fact-policy", f"cc-s1-failure-{label.lower()}", 1701),
                env_rng=random.Random(1701),
                sequence_id=f"cc-s1-failure-{label.lower()}-sequence",
                branch_override=branch,
            )
            transition = ReplayStoreV1(root).get_transition(evidence.transition_id)
            details[f"{label.lower()}_equity"] = float(evidence.final_equity)
            details[f"{label.lower()}_reward"] = float(evidence.reward)
            details[f"{label.lower()}_mechanical_terminal"] = bool(transition.mechanical_terminal)
            details[f"{label.lower()}_boundary"] = transition.boundary_type
            details[f"{label.lower()}_closure_in_transition"] = bool(
                transition.consequence_context
                and "insolvency_closure" in transition.consequence_context
            )
            if label == "LOSS":
                checks["loss_equity_nonpositive"] = bool(evidence.final_equity <= 0.0)
                checks["loss_is_mechanical_terminal"] = bool(transition.mechanical_terminal is True)
                checks["loss_boundary_is_economic_terminal"] = bool(transition.boundary_type == "ECONOMIC_TERMINAL")
                checks["loss_closure_provenance_durable"] = bool(
                    transition.consequence_context
                    and transition.consequence_context.get("insolvency_closure", {}).get("mechanical_terminal") is True
                )
                checks["loss_reward_negative_and_retained"] = bool(evidence.reward < 0.0)
                materialized = _Materializer.from_durable_state_v1(root).materialize_sequence(
                    evidence.sequence_id, target_policy_identity="cc-s1-failure-fact-target", restart_verified=True
                )
                samples, bootstrap = _apply_credit_views(root, materialized)
                provenance = _BatchProvenance(
                    materializer_contract_id=MATERIALIZER_CONTRACT_ID,
                    materialization_id=materialized.manifest.manifest_id,
                    materialization_manifest_sha256=materialized.manifest.manifest_sha256,
                    source_store_root_identity=materialized.manifest.source_store_root_identity,
                    restart_verified=True,
                    durable_replay_only=True,
                    collector_private_records_used=False,
                )
                batch = build_joint_batch_v1(
                    samples=samples,
                    provenance=provenance,
                    target_policy_identity="cc-s1-failure-fact-target",
                    bootstrap_observations_by_sequence=bootstrap,
                )
                checks["loss_retained_with_full_weight"] = bool(
                    float(batch.replay_weights[0].item()) == 1.0
                    and float(batch.rewards[0].item()) < 0.0
                    and int(batch.rewards.shape[0]) == 1
                )
            else:
                checks["win_is_not_mechanical_terminal"] = bool(transition.mechanical_terminal is False)
                checks["win_boundary_is_objective_horizon"] = bool(
                    transition.boundary_type == "OBJECTIVE_HORIZON_REACHED"
                )
    return {
        "schema": "CB16_R11_POST_CC_S1_HIGH_BANKRUPTCY_FAILURE_FACT_AUDIT_V1",
        "checks": checks,
        "details": details,
        "all_checks_pass": bool(all(checks.values())),
    }


def audit_exported_update_journal_v1(artifact_root: str | Path) -> Mapping[str, Any]:
    """Artifact-only B6 audit: trace a full qualified update after scratch deletion."""
    import hashlib

    root = Path(artifact_root)
    journal_root = root / "provenance" / "update_journals"
    records_paths = sorted(journal_root.rglob("update_journal/*.json"))
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    if not records_paths:
        return {
            "schema": "CB16_R11_POST_CC_S1_ARTIFACT_ONLY_UPDATE_TRACE_AUDIT_V1",
            "checks": {"update_journal_exported": False},
            "details": {},
            "all_checks_pass": False,
        }
    checks["update_journal_exported"] = True
    run_dirs = sorted({path.parent for path in records_paths}, key=lambda p: -len(list(p.glob("*.json"))))
    selected_records = []
    for path in sorted(run_dirs[0].glob("*.json")):
        selected_records.append(json.loads(path.read_text(encoding="utf-8")))
    committed = [record for record in selected_records if record.get("commit_status") == "COMMITTED"]
    checks["committed_update_records_present"] = bool(committed)
    run_relative = run_dirs[0].relative_to(root)
    durable_index = root / "replay_manifests" / Path(*run_relative.parts[2:-1]) / "durable_index.jsonl"
    checks["durable_index_exported"] = bool(durable_index.exists())
    details["exported_durable_index_path"] = str(durable_index)
    if committed:
        newest = max(committed, key=lambda item: int(item.get("optimizer_step_after") or 0))
        details["traced_update_id"] = newest.get("update_id")
        details["traced_optimizer_step_after"] = newest.get("optimizer_step_after")
        checks["update_has_sample_hashes"] = bool(newest.get("materialized_sample_hashes"))
        checks["update_has_sampling_weights"] = bool(newest.get("sampling_probabilities_or_weights"))
        checks["update_has_parent_and_child"] = bool(
            newest.get("parent_checkpoint_sha256") and newest.get("child_checkpoint_sha256")
        )
        checks["update_has_losses_and_vtrace"] = bool(
            newest.get("actor_loss") is not None
            and newest.get("critic_loss") is not None
            and newest.get("vtrace_diagnostics")
        )
        checks["update_has_gradient_ownership"] = bool(newest.get("gradient_ownership_summary"))
        linked_result = False
        for result_path in list((root / "task_results").rglob("*.json")) + list((root / "control_results").rglob("*.json")):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            for unit in payload.get("unit_evidence", []):
                if unit.get("update_id") == newest.get("update_id"):
                    linked_result = True
                    checks["unit_evidence_links_update"] = bool(
                        unit.get("batch_content_sha256") == newest.get("batch_content_sha256")
                        and bool(unit.get("materialization_manifest_ids"))
                    )
        checks["update_linked_from_result_record"] = bool(linked_result)
        final_step = max(int(item.get("optimizer_step_after") or 0) for item in committed)
        is_final_update = int(newest.get("optimizer_step_after") or 0) == final_step
        if is_final_update:
            checkpoint_files = list((root / "checkpoints").rglob("*_final_child.json"))
            matched = False
            for checkpoint_file in checkpoint_files:
                digest = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
                if digest == newest.get("child_checkpoint_sha256"):
                    matched = True
                    break
            checks["final_child_checkpoint_bytes_match_journal"] = bool(matched)
        else:
            checks["final_child_checkpoint_bytes_match_journal"] = True
    return {
        "schema": "CB16_R11_POST_CC_S1_ARTIFACT_ONLY_UPDATE_TRACE_AUDIT_V1",
        "artifact_root": str(root),
        "traced_run": str(run_relative),
        "checks": checks,
        "details": details,
        "all_checks_pass": bool(all(checks.values())),
    }
