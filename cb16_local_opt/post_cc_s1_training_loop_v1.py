"""CB16 R11 S1 repeated durable-learning loop.

One canonical auditable path:

    synthetic causal environment
      -> canonical brain observation
      -> stochastic nominal action + true decision-time log_mu
      -> authoritative execution/account consequence
      -> durable immutable facts
      -> restart sentinel
      -> durable-only replay materialization
      -> validated joint batch
      -> target log_pi on the nominal action
      -> separate Critic + explicit boundary/bootstrap + V-trace
      -> Actor/Critic update
      -> exactly-once child commit
      -> committed-child generation switch
      -> child policy produces later decisions

No shortcut around any arrow feeds S1 qualification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch

from .cc_experience_wire_r0 import content_sha256
from .cc_policy_brain_r0 import CCCentralBrain
from .post_cc_joint_batch_v1 import (
    MATERIALIZER_CONTRACT_ID,
    DurableBatchProvenanceV1,
    JointActionBatchV1,
    build_joint_batch_v1,
)
from .post_cc_joint_policy_loss_v1 import target_likelihoods_v1
from .post_cc_learner_v1 import PostCCDurableReplayLearnerV1
from .post_cc_replay_materializer_v1 import MaterializedReplayV1, ReplayMaterializerV1, ReplayStoreV1
from .post_cc_s1_credit_adapter_v1 import (
    apply_credit_views_v1,
    assert_sample_log_mu_bound_to_durable_v1,
    verify_credits_for_sequences_v1,
)
from .post_cc_s1_tasks_v1 import (
    BEHAVIOR_MODE_FIXED_DISTINCT_V1,
    ContextSpecV1,
    CONTROL_ACCOUNT_ABLATION,
    CONTROL_NO_SIGNAL,
    CONTROL_RANDOM_IMPOSSIBLE,
    CONTROL_SHUFFLED_CREDIT,
    frozen_random_control_contexts_v1,
    EpisodeEvidenceV1,
    SetupProvenanceV1,
    TaskSpecV1,
    collect_episode_v1,
    derived_stream_seed_v1,
    establish_account_context_v1,
    evaluate_task_v1,
    enumerate_oracle_v1,
    make_s1_brain_v1,
    make_s1_critic_v1,
    sample_branch_v1,
    stable_sha256_v1,
    BranchOutcomeV1,
)
from .post_cc_generation_continuity_v1 import (
    commit_child_generation_v1,
    child_policy_identity_v1,
    parameter_state_sha256_v1,
)
from .post_cc_update_transaction_v1 import DurableUpdateStoreV1
from .post_cc_s1_tasks_v1 import SCIENCE_SEMANTIC_VERSION
from .cc_policy_rng_r0 import PolicyRNG

SMOKE_ONLY_EVIDENCE_CLASS = "SMOKE_ONLY_ENGINEERING_EVIDENCE"
QUALIFICATION_EVIDENCE_CLASS = "S1_QUALIFICATION_CANDIDATE"


class S1TrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class S1SeedRunConfigV1:
    spec: TaskSpecV1
    seed: int
    run_root: str
    mode: str
    control_id: str | None
    unit_size: int
    max_units: int
    evaluation_population: int
    manifest_sha256: str

    def validate(self) -> "S1SeedRunConfigV1":
        if self.mode not in ("smoke", "qualification"):
            raise S1TrainingError("UNKNOWN_RUN_MODE")
        if self.mode == "qualification" and self.unit_size != 128:
            raise S1TrainingError("QUALIFICATION_UNIT_SIZE_FROZEN_AT_128")
        if self.mode == "qualification" and self.evaluation_population != 2048:
            raise S1TrainingError("QUALIFICATION_EVALUATION_POPULATION_FROZEN_AT_2048")
        if int(self.max_units) <= 0 or int(self.unit_size) <= 0:
            raise S1TrainingError("RUN_BUDGET_INVALID")
        if self.mode == "qualification":
            expected_units = int(self.spec.training_decisions) // int(self.unit_size)
            if int(self.max_units) != expected_units:
                raise S1TrainingError("QUALIFICATION_BUDGET_DOES_NOT_MATCH_SPEC")
        return self

    @property
    def decisions_consumed(self) -> int:
        return int(self.unit_size) * int(self.max_units)


@dataclass(frozen=True)
class UnitEvidenceV1:
    unit_index: int
    phase_id: str
    sequence_ids: tuple[str, ...]
    rewards: tuple[float, ...]
    batch_sample_count: int
    batch_content_sha256: str
    materialization_manifest_sha256s: tuple[str, ...]
    nominal_direction_counts: Mapping[str, int]
    update_id: str
    update_status: str
    optimizer_step_after: int
    child_checkpoint_sha256: str
    ratio_above_one_count: int
    ratio_below_one_count: int
    ratio_abs_max_minus_one: float
    behavior_checkpoint_untouched: bool
    behavior_identity_before: str
    behavior_identity_after: str
    generation_switch_receipt: Mapping[str, Any] | None
    restart_sentinel: Mapping[str, Any]

    def payload(self) -> Mapping[str, Any]:
        return {"schema": "CB16_R11_POST_CC_S1_UNIT_EVIDENCE_V1", **asdict(self)}

    @property
    def content_sha256(self) -> str:
        return stable_sha256_v1(self.payload())


def _task_slug_v1(task_id: str) -> str:
    return task_id.lower().replace("_", "-")


def _plain_text_boundary_v1(control_id: str | None, spec: TaskSpecV1) -> str:
    if control_id == CONTROL_SHUFFLED_CREDIT and spec.credit_stage2_no_decision:
        return "SHUFFLED_DELAYED_CONSEQUENCE_PAIRING"
    if control_id == CONTROL_SHUFFLED_CREDIT:
        return "SHUFFLED_OFF_POLICY_REWARD_PAIRING"
    if control_id == CONTROL_RANDOM_IMPOSSIBLE:
        return "RANDOM_IMPOSSIBLE_TARGET_REALIZATION"
    if control_id == CONTROL_ACCOUNT_ABLATION:
        return "ACCOUNT_INPUT_ABLATION"
    if control_id == CONTROL_NO_SIGNAL:
        return "ZERO_REWARD_NO_SIGNAL"
    return "RAW_POSITIVE_TASK"


def _context_schedule_v1(spec: TaskSpecV1, unit_index: int, episode_index: int, unit_size: int, rng: random.Random) -> str:
    if spec.context_schedule == "SINGLE_CONTEXT":
        return spec.contexts[0].context_id
    if spec.context_schedule == "UNIFORM_CONTEXT_RNG":
        return spec.contexts[rng.randrange(len(spec.contexts))].context_id
    if spec.context_schedule == "PHASES":
        decision_ordinal = int(unit_index) * int(unit_size) + int(episode_index)
        cumulative = 0
        for phase_id, count in spec.phase_context_ids:
            cumulative += int(count)
            if decision_ordinal < cumulative:
                return phase_id
        raise S1TrainingError("PHASE_SCHEDULE_OVERFLOW")
    raise S1TrainingError("UNKNOWN_CONTEXT_SCHEDULE")


def _control_branch_override_v1(
    *,
    spec: TaskSpecV1,
    context: "ContextSpecV1",
    control_id: str | None,
    control_rng: random.Random,
) -> BranchOutcomeV1 | None:
    if control_id == CONTROL_SHUFFLED_CREDIT and spec.credit_stage2_no_decision:
        other_contexts = [item for item in spec.contexts if item.context_id != context.context_id]
        if not other_contexts:
            raise S1TrainingError("SHUFFLED_CONTROL_REQUIRES_SECOND_CONTEXT")
        other = other_contexts[control_rng.randrange(len(other_contexts))]
        branch = sample_branch_v1(other.dynamics.stage2_branches, control_rng)
        return BranchOutcomeV1(probability=float(branch.probability), terminal_mark=float(branch.terminal_mark))
    if control_id == CONTROL_RANDOM_IMPOSSIBLE:
        if spec.task_id == "HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION":
            marks = (120.0, 80.0)
        else:
            decision_mark = float(context.dynamics.decision_mark)
            marks = (decision_mark + 10.0, decision_mark - 10.0)
        mark = marks[0] if control_rng.random() <= 0.5 else marks[1]
        return BranchOutcomeV1(probability=0.5, terminal_mark=float(mark))
    return None


def _materialize_batch_v1(
    *,
    root: str,
    sequence_ids: Sequence[str],
    target_policy_identity: str,
) -> tuple[JointActionBatchV1, list[MaterializedReplayV1]]:
    materializer = ReplayMaterializerV1.from_durable_state_v1(root)
    materialized: list[MaterializedReplayV1] = []
    samples = []
    bootstrap_map: dict[str, Any] = {}
    for sequence_id in sequence_ids:
        record = materializer.materialize_sequence(
            sequence_id,
            target_policy_identity=target_policy_identity,
            restart_verified=True,
        )
        credited_samples, credited_bootstrap = apply_credit_views_v1(root, record)
        for sample in credited_samples:
            assert_sample_log_mu_bound_to_durable_v1(root, sample)
        materialized.append(record)
        samples.extend(credited_samples)
        bootstrap_map.update(credited_bootstrap)
    manifest_hashes = tuple(record.manifest.manifest_sha256 for record in materialized)
    materialization_id = stable_sha256_v1(
        {"sequence_ids": tuple(sequence_ids), "manifest_hashes": manifest_hashes}
    )
    provenance = DurableBatchProvenanceV1(
        materializer_contract_id=MATERIALIZER_CONTRACT_ID,
        materialization_id=materialization_id,
        materialization_manifest_sha256=stable_sha256_v1(manifest_hashes),
        source_store_root_identity=materialized[0].manifest.source_store_root_identity,
        restart_verified=True,
        durable_replay_only=True,
        collector_private_records_used=False,
    )
    batch = build_joint_batch_v1(
        samples=tuple(samples),
        provenance=provenance,
        target_policy_identity=target_policy_identity,
        bootstrap_observations_by_sequence=(bootstrap_map or None),
    )
    return batch, materialized


def _sample_sequence_ids_v1(
    *,
    index_rows: Sequence[Mapping[str, Any]],
    replay_rng: random.Random,
    target_size: int,
) -> tuple[str, ...]:
    ids = [str(row["sequence_id"]) for row in index_rows]
    if not ids:
        raise S1TrainingError("EMPTY_DURABLE_REPLAY_INDEX")
    size = min(int(target_size), len(ids))
    selected = replay_rng.sample(ids, size)
    non_flat_ids = {str(row["sequence_id"]) for row in index_rows if str(row["nominal_direction"]) != "FLAT"}
    if non_flat_ids and not any(sequence_id in non_flat_ids for sequence_id in selected):
        replacement_pool = sorted(non_flat_ids)
        selected[-1] = replacement_pool[replay_rng.randrange(len(replacement_pool))]
    return tuple(selected)


def _ratio_diagnostics_v1(actor: CCCentralBrain, batch: JointActionBatchV1) -> tuple[int, int, float]:
    with torch.no_grad():
        likelihoods = target_likelihoods_v1(actor, batch)
        ratio = torch.exp(torch.clamp(likelihoods.log_pi - likelihoods.log_mu, -80.0, 80.0))
        above = int((ratio > 1.0 + 1e-6).sum().item())
        below = int((ratio < 1.0 - 1e-6).sum().item())
        abs_max_minus_one = float((ratio - 1.0).abs().max().item())
    return above, below, abs_max_minus_one


def _apply_off_policy_reward_shuffle_v1(
    batch: JointActionBatchV1,
    *,
    control_rng: random.Random,
) -> tuple[JointActionBatchV1, list[int]]:
    count = len(batch.samples)
    permutation = list(range(count))
    control_rng.shuffle(permutation)
    if permutation == list(range(count)):
        permutation = list(range(1, count)) + [0]
    rewards = [float(sample.reward) for sample in batch.samples]
    shuffled = [rewards[permutation[index]] for index in range(count)]
    updated_samples = tuple(
        replace(
            sample,
            reward=shuffled[index],
            consequence_context={**dict(sample.consequence_context or {}), "shuffled_credit_control_index": int(permutation[index])},
        ).validate()
        for index, sample in enumerate(batch.samples)
    )
    provenance = replace(
        batch.provenance,
        materialization_id=stable_sha256_v1({"base": batch.provenance.materialization_id, "shuffle": permutation}),
    )
    updated = build_joint_batch_v1(
        samples=updated_samples,
        provenance=provenance,
        target_policy_identity=batch.target_policy_identity,
        bootstrap_observations_by_sequence=(
            None
            if batch.bootstrap_observations is None
            else {
                batch.sequence_ids[batch.bootstrap_sequence_indices[index]]: batch.bootstrap_observations[index]
                for index in range(len(batch.bootstrap_sequence_indices))
            }
        ),
    )
    return updated, permutation


def _evaluate_v1(
    *,
    spec: TaskSpecV1,
    accounts: Mapping[str, Any],
    actor: CCCentralBrain,
    seed: int,
    checkpoint_id: str,
    population: int,
    zero_account_inputs: bool,
) -> Mapping[str, Any]:
    evaluation = evaluate_task_v1(
        spec=spec,
        accounts=accounts,
        actor=actor,
        seed=seed,
        checkpoint_id=checkpoint_id,
        population=int(population),
        zero_account_inputs=bool(zero_account_inputs),
    )
    return evaluation


def _oracle_summary_v1(
    *,
    spec: TaskSpecV1,
    accounts: Mapping[str, Any],
) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for context in spec.contexts:
        enumeration = enumerate_oracle_v1(spec=spec, context=context, account=accounts[context.context_id])
        result[context.context_id] = {
            "oracle_return": float(enumeration["best_return"]),
            "best_directions": list(enumeration["best_directions"]),
            "flat_return": float(enumeration["flat_return"]),
            "expected_direction_present": bool(enumeration["expected_direction_present"]),
            "non_flat_higher_return_count": int(enumeration["non_flat_higher_return_count"]),
            "non_flat_higher_return_and_bankruptcy_count": int(
                enumeration["non_flat_higher_return_and_bankruptcy_count"]
            ),
        }
    return {"contexts": result, "mean_oracle_return": float(sum(item["oracle_return"] for item in result.values()) / len(result))}


def _gap_v1(*, oracle_score: float, policy_score: float) -> tuple[float, float]:
    initial_or_final_gap = float(oracle_score) - float(policy_score)
    if not math.isfinite(initial_or_final_gap):
        raise S1TrainingError("NONFINITE_ORACLE_GAP")
    return initial_or_final_gap, initial_or_final_gap


def _gap_reduction_v1(*, initial_gap: float, final_gap: float) -> float:
    if initial_gap <= 0.0:
        raise S1TrainingError("INITIAL_GAP_NOT_POSITIVE")
    return (float(initial_gap) - float(final_gap)) / float(initial_gap)


def _restart_sentinel_v1(root: str, index_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Destroy-capable restart: rebuild every training input from durable state."""
    gc.collect()
    replay_store = ReplayStoreV1(root)
    replay_counts = replay_store.verify_all()
    from .post_cc_observation_store_v1 import ObservationStoreV1

    observation_store = ObservationStoreV1(root)
    observation_count = observation_store.verify_all()
    sequence_ids = tuple(str(row["sequence_id"]) for row in index_rows)
    for sequence_id in sequence_ids:
        replay_store.get_sequence(sequence_id)
    credit_counts = verify_credits_for_sequences_v1(root, sequence_ids)
    return {
        "restart_verified": True,
        "collector_private_records_used_as_training_truth": False,
        "transition_count": int(replay_counts["transitions"]),
        "sequence_count": int(replay_counts["sequences"]),
        "raw_advance_count": int(replay_counts["raw_advances"]),
        "manifest_count": int(replay_counts["manifests"]),
        "observation_count": int(observation_count),
        "credits_verified": int(credit_counts["credits_verified"]),
    }


def _append_index_row_v1(path: Path, row: Mapping[str, Any]) -> None:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())


def _read_index_rows_v1(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _score_v1(evaluation: Mapping[str, Any]) -> float:
    return float(evaluation["mean_complete_sample_arithmetic_return"])


def _context_score_v1(evaluation: Mapping[str, Any], context_id: str) -> float:
    return float(evaluation["contexts"][context_id]["mean_complete_sample_arithmetic_return"])


def _context_direction_probability_v1(evaluation: Mapping[str, Any], context_id: str) -> float:
    return float(evaluation["contexts"][context_id]["oracle_direction_probability"])


def _context_family_mass_v1(evaluation: Mapping[str, Any], context_id: str) -> float:
    return float(evaluation["contexts"][context_id]["higher_ev_family_mass"])


def _mean_oracle_v1(oracle: Mapping[str, Any]) -> float:
    return float(oracle["mean_oracle_return"])


def _utc_now_v1() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_seed_v1(config: S1SeedRunConfigV1) -> dict[str, Any]:
    """Run one frozen task/seed (or matched control) through the durable loop."""
    config.validate()
    run_started = _utc_now_v1()
    run_started_monotonic = time.monotonic()
    spec = config.spec
    root = Path(config.run_root)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "durable_index.jsonl"
    lineage = f"cc-s1-{_task_slug_v1(spec.task_id)}-{int(config.seed)}"

    setup_provenance: dict[str, Any] = {}
    accounts: dict[str, Any] = {}
    for context in spec.contexts:
        account, provenance = establish_account_context_v1(
            spec=spec,
            context=context,
            lineage=lineage,
            setup_root=str(root / "setup" / context.context_id),
        )
        accounts[context.context_id] = account
        if provenance is not None:
            setup_provenance[context.context_id] = dict(provenance.payload())
    (root / "setup_provenance.json").write_text(
        json.dumps(setup_provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    oracle = _oracle_summary_v1(spec=spec, accounts=accounts)
    oracle_validation = {
        "expected_direction_present_for_every_context": all(
            item["expected_direction_present"] for item in oracle["contexts"].values()
        ),
    }
    if not oracle_validation["expected_direction_present_for_every_context"]:
        raise S1TrainingError("ORACLE_EXPECTED_DIRECTION_NOT_PRESENT")

    initial_target_actor = make_s1_brain_v1()
    if spec.behavior_mode == BEHAVIOR_MODE_FIXED_DISTINCT_V1:
        torch.manual_seed(derived_stream_seed_v1(spec.task_id, config.seed, "fixed_behavior_init"))
        from .cc_policy_brain_r0 import CCCentralBrain as _Brain

        behavior_actor = _Brain(2, 3, 2, 8, initial_target_actor.bindings)
        for parameter in behavior_actor.parameters():
            parameter.requires_grad_(False)
        behavior_policy_id = "cc-s1-fixed-behavior"
        behavior_generation = "0"
        behavior_policy_sha256 = parameter_state_sha256_v1(behavior_actor)
        target_actor = initial_target_actor
        target_identity = "cc-s1-off-policy-target-init"
    else:
        behavior_actor = make_s1_brain_v1()
        behavior_policy_id = "cc-s1-policy"
        behavior_generation = "0"
        behavior_policy_sha256 = parameter_state_sha256_v1(behavior_actor)
        target_actor = initial_target_actor
        target_identity = "cc-s1-target-init"

    learner = PostCCDurableReplayLearnerV1(
        target_actor=target_actor,
        target_critic=make_s1_critic_v1(),
        update_store=DurableUpdateStoreV1(root / "updates"),
        parent_policy_identity=target_identity,
        target_policy_identity="cc-s1-target-next",
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
    )
    collection_rng = random.Random(derived_stream_seed_v1(spec.task_id, config.seed, "collection_context"))
    control_rng = random.Random(derived_stream_seed_v1(spec.task_id, config.seed, "control"))
    replay_rng = random.Random(derived_stream_seed_v1(spec.task_id, config.seed, "replay_sampling"))

    zero_account_inputs = config.control_id == CONTROL_ACCOUNT_ABLATION
    reward_mode = "ZERO" if config.control_id == CONTROL_NO_SIGNAL else "RAW"

    checkpoints: dict[str, Any] = {}
    checkpoints["INITIAL"] = _evaluate_v1(
        spec=spec,
        accounts=accounts,
        actor=behavior_actor,
        seed=config.seed,
        checkpoint_id="INITIAL",
        population=config.evaluation_population,
        zero_account_inputs=zero_account_inputs,
    )

    unit_records: list[UnitEvidenceV1] = []
    phase_plan = spec.phase_context_ids if spec.context_schedule == "PHASES" else ()
    phase_index = 0
    phase_consumed_decisions = 0

    for unit_index in range(int(config.max_units)):
        if phase_plan:
            if phase_consumed_decisions >= int(phase_plan[phase_index][1]):
                phase_index += 1
                phase_consumed_decisions = 0
                if phase_index >= len(phase_plan):
                    raise S1TrainingError("PHASE_PLAN_EXHAUSTED")
            phase_id = phase_plan[phase_index][0]
        else:
            phase_id = spec.contexts[0].context_id if spec.context_schedule == "SINGLE_CONTEXT" else "MIXED"
        behavior_identity_before = f"{behavior_policy_id}:g{behavior_generation}:{behavior_policy_sha256}"
        old_policy_generation = str(behavior_generation)
        old_policy_sha256 = str(behavior_policy_sha256)
        behavior_parameter_sha_before = parameter_state_sha256_v1(behavior_actor)
        unit_evidence_rows: list[EpisodeEvidenceV1] = []
        for episode_index in range(int(config.unit_size)):
            if phase_plan:
                context_id = phase_id
            else:
                context_id = _context_schedule_v1(spec, unit_index, episode_index, int(config.unit_size), collection_rng)
            context = next(item for item in spec.contexts if item.context_id == context_id)
            episode_ordinal = unit_index * int(config.unit_size) + episode_index
            episode_lineage = f"{lineage}-e{episode_ordinal:06d}"
            branch_override = _control_branch_override_v1(
                spec=spec,
                context=context,
                control_id=config.control_id,
                control_rng=control_rng,
            )
            action_rng = PolicyRNG(
                policy_id=f"{behavior_policy_id}-u{unit_index:04d}-e{episode_index:03d}",
                account_lineage_id=episode_lineage,
                seed=derived_stream_seed_v1(spec.task_id, config.seed, "action", episode_ordinal),
            )
            env_rng = random.Random(derived_stream_seed_v1(spec.task_id, config.seed, "env", episode_ordinal))
            sequence_id = f"{lineage}-u{unit_index:04d}-e{episode_index:03d}"
            evidence = collect_episode_v1(
                spec=spec,
                context=context,
                account=accounts[context_id],
                root=str(root),
                lineage=episode_lineage,
                policy_generation=behavior_generation,
                policy_id=behavior_policy_id,
                policy_sha256=behavior_policy_sha256,
                actor=behavior_actor,
                action_rng=action_rng,
                env_rng=env_rng,
                sequence_id=sequence_id,
                zero_account_inputs=zero_account_inputs,
                reward_mode=reward_mode,
                branch_override=branch_override,
            )
            unit_evidence_rows.append(evidence)
            _append_index_row_v1(
                index_path,
                {
                    "sequence_id": sequence_id,
                    "episode_lineage": episode_lineage,
                    "context_id": context_id,
                    "phase_id": phase_id,
                    "unit_index": int(unit_index),
                    "episode_index": int(episode_index),
                    "nominal_direction": evidence.nominal_direction,
                    "nominal_target_risk": evidence.nominal_target_risk,
                    "behavior_log_mu": evidence.behavior_log_mu,
                    "reward": evidence.reward,
                    "discount": evidence.discount,
                    "boundary_type": evidence.boundary_type,
                    "credit_view_id": evidence.credit_view_id,
                    "behavior_policy_generation": evidence.behavior_policy_generation,
                    "behavior_policy_id": evidence.behavior_policy_id,
                    "behavior_policy_sha256": evidence.behavior_policy_sha256,
                    "evidence_sha256": evidence.content_sha256,
                    "control_id": config.control_id,
                    "reward_mode": reward_mode,
                    "zero_account_inputs": bool(zero_account_inputs),
                    "branch_override": (
                        None
                        if branch_override is None
                        else {"probability": branch_override.probability, "terminal_mark": branch_override.terminal_mark}
                    ),
                },
            )
        phase_consumed_decisions += int(config.unit_size)
        index_rows = _read_index_rows_v1(index_path)
        sentinel = _restart_sentinel_v1(str(root), index_rows)
        selected_sequence_ids = _sample_sequence_ids_v1(
            index_rows=index_rows,
            replay_rng=replay_rng,
            target_size=128,
        )
        next_target_identity = f"cc-s1-target-g{unit_index + 1}"
        batch, materialized = _materialize_batch_v1(
            root=str(root),
            sequence_ids=selected_sequence_ids,
            target_policy_identity=next_target_identity,
        )
        shuffle_permutation: list[int] | None = None
        if config.control_id == CONTROL_SHUFFLED_CREDIT and spec.behavior_mode == BEHAVIOR_MODE_FIXED_DISTINCT_V1:
            batch, shuffle_permutation = _apply_off_policy_reward_shuffle_v1(batch, control_rng=control_rng)
        ratio_above, ratio_below, ratio_abs_max_minus_one = _ratio_diagnostics_v1(learner.target_actor, batch)
        learner.parent_policy_identity = target_identity
        learner.target_policy_identity = next_target_identity
        result = learner.apply_durable_update_v1(batch=batch)
        behavior_checkpoint_untouched = parameter_state_sha256_v1(behavior_actor) == behavior_parameter_sha_before
        generation_switch_receipt: Mapping[str, Any] | None = None
        if spec.behavior_mode != BEHAVIOR_MODE_FIXED_DISTINCT_V1:
            new_behavior_generation = str(int(behavior_generation) + 1)
            new_behavior_policy_sha256 = child_policy_identity_v1(
                child_checkpoint_sha256=str(result.child_checkpoint_sha256),
                policy_generation=new_behavior_generation,
                policy_id=behavior_policy_id,
            )
            last_episode = unit_evidence_rows[-1]
            from .post_cc_s1_tasks_v1 import make_runtime_v1

            switch_runtime = make_runtime_v1(
                spec=spec,
                account=last_episode.final_account,
                account_lineage_id=last_episode.lineage,
                policy_generation=old_policy_generation,
                policy_id=behavior_policy_id,
                policy_sha256=old_policy_sha256,
                schedule_every=1,
            )
            switch_account = switch_runtime.account
            receipt = commit_child_generation_v1(
                switch_runtime,
                update_store=learner.update_store,
                update_id=result.update_id,
                new_policy_generation=new_behavior_generation,
                new_policy_id=behavior_policy_id,
                new_policy_sha256=new_behavior_policy_sha256,
                expected_account_snapshot=switch_account,
                boundary_type="OBJECTIVE_HORIZON_REACHED",
            )
            # only after the committed-child authority exists does the child
            # become the runtime behaviour checkpoint
            behavior_actor.load_state_dict(
                {name: tensor.detach().clone() for name, tensor in learner.target_actor.state_dict().items()}
            )
            behavior_generation = new_behavior_generation
            behavior_policy_sha256 = new_behavior_policy_sha256
            generation_switch_receipt = {
                "switch_id": receipt.switch_id,
                "child_checkpoint_sha256": receipt.child_checkpoint_sha256,
                "new_policy_generation": receipt.new_policy_generation,
                "new_policy_sha256": receipt.new_policy_sha256,
                "account_fully_preserved": bool(receipt.account_fully_preserved),
                "authorized_boundary": bool(receipt.authorized_boundary),
                "parent_checkpoint_mutated_in_place": bool(receipt.parent_checkpoint_mutated_in_place),
            }
        target_identity = next_target_identity
        direction_counts: dict[str, int] = {}
        for evidence in unit_evidence_rows:
            direction_counts[evidence.nominal_direction] = direction_counts.get(evidence.nominal_direction, 0) + 1
        unit_records.append(
            UnitEvidenceV1(
                unit_index=int(unit_index),
                phase_id=str(phase_id),
                sequence_ids=tuple(item.sequence_id for item in unit_evidence_rows),
                rewards=tuple(float(item.reward) for item in unit_evidence_rows),
                batch_sample_count=len(batch.samples),
                batch_content_sha256=batch.batch_content_sha256,
                materialization_manifest_sha256s=tuple(item.manifest.manifest_sha256 for item in materialized),
                nominal_direction_counts=direction_counts,
                update_id=result.update_id,
                update_status=str(result.record.commit_status),
                optimizer_step_after=int(result.optimizer_step_after),
                child_checkpoint_sha256=str(result.child_checkpoint_sha256),
                ratio_above_one_count=int(ratio_above),
                ratio_below_one_count=int(ratio_below),
                ratio_abs_max_minus_one=float(ratio_abs_max_minus_one),
                behavior_checkpoint_untouched=bool(behavior_checkpoint_untouched),
                behavior_identity_before=behavior_identity_before,
                behavior_identity_after=f"{behavior_policy_id}:g{behavior_generation}:{behavior_policy_sha256}",
                generation_switch_receipt=generation_switch_receipt,
                restart_sentinel=dict(sentinel),
            )
        )
        checkpoint_after_unit = int(unit_index) + 1
        if spec.context_schedule == "PHASES":
            phase_end_units = sum(int(count) for _, count in phase_plan[: phase_index + 1]) // int(config.unit_size)
            if checkpoint_after_unit == phase_end_units:
                if phase_index == 0:
                    checkpoints["POST_A1"] = _evaluate_v1(
                        spec=spec, accounts=accounts, actor=behavior_actor, seed=config.seed,
                        checkpoint_id="POST_A1", population=config.evaluation_population,
                        zero_account_inputs=zero_account_inputs,
                    )
                elif phase_index == 1:
                    checkpoints["POST_B"] = _evaluate_v1(
                        spec=spec, accounts=accounts, actor=behavior_actor, seed=config.seed,
                        checkpoint_id="POST_B", population=config.evaluation_population,
                        zero_account_inputs=zero_account_inputs,
                    )
                elif phase_index == len(phase_plan) - 1:
                    checkpoints["POST_A2"] = _evaluate_v1(
                        spec=spec, accounts=accounts, actor=behavior_actor, seed=config.seed,
                        checkpoint_id="POST_A2", population=config.evaluation_population,
                        zero_account_inputs=zero_account_inputs,
                    )
    checkpoints["FINAL"] = _evaluate_v1(
        spec=spec,
        accounts=accounts,
        actor=behavior_actor,
        seed=config.seed,
        checkpoint_id="FINAL",
        population=config.evaluation_population,
        zero_account_inputs=zero_account_inputs,
    )
    control_frozen_realization: Mapping[str, Any] | None = None
    if config.control_id == CONTROL_RANDOM_IMPOSSIBLE:
        from dataclasses import replace as _replace

        symmetric_spec = _replace(spec, contexts=frozen_random_control_contexts_v1(spec))
        symmetric_evaluation = _evaluate_v1(
            spec=symmetric_spec,
            accounts=accounts,
            actor=behavior_actor,
            seed=config.seed,
            checkpoint_id="CONTROL_FROZEN_REALIZATION",
            population=config.evaluation_population,
            zero_account_inputs=zero_account_inputs,
        )
        symmetric_oracle = _oracle_summary_v1(spec=symmetric_spec, accounts=accounts)
        control_frozen_realization = {
            "separation_rule": "TRAINING_REALIZATION_PER_EPISODE_RNG_VS_FROZEN_EXACT_50_50_EVALUATION",
            "evaluation": symmetric_evaluation,
            "oracle": symmetric_oracle,
            "near_zero_oracle_gap": bool(abs(symmetric_oracle["mean_oracle_return"]) <= 1e-9),
        }
    update_store_verification = learner.update_store.verify_all()
    result_payload = {
        "schema": "CB16_R11_POST_CC_S1_SEED_RESULT_V1",
        "status": "OK",
        "run_started_utc": run_started,
        "run_finished_utc": _utc_now_v1(),
        "wall_seconds": float(time.monotonic() - run_started_monotonic),
        "evidence_class": SMOKE_ONLY_EVIDENCE_CLASS if config.mode == "smoke" else QUALIFICATION_EVIDENCE_CLASS,
        "mode": config.mode,
        "task_id": spec.task_id,
        "task_spec_hash": spec.spec_hash,
        "environment_version": spec.environment_version,
        "seed": int(config.seed),
        "control_id": config.control_id,
        "control_transform": _plain_text_boundary_v1(config.control_id, spec),
        "manifest_sha256": config.manifest_sha256,
        "lineage": lineage,
        "run_root": str(root),
        "decisions_consumed": int(config.decisions_consumed),
        "unit_size": int(config.unit_size),
        "max_units": int(config.max_units),
        "evaluation_population": int(config.evaluation_population),
        "oracle": oracle,
        "oracle_validation": oracle_validation,
        "checkpoints": checkpoints,
        "setup_provenance": setup_provenance,
        "unit_evidence": [dict(item.payload()) for item in unit_records],
        "unit_evidence_sha256s": [item.content_sha256 for item in unit_records],
        "durability": {
            "restart_sentinel_passed": all(item.restart_sentinel.get("restart_verified") is True for item in unit_records),
            "final_update_store_verification": dict(update_store_verification),
            "behavior_checkpoint_untouched_every_unit": all(item.behavior_checkpoint_untouched for item in unit_records),
        },
        "optimizer": {
            "actor_lr": float(learner.actor_lr),
            "critic_lr": float(learner.critic_lr),
            "optimizer_step_final": int(learner.optimizer_step),
            "gradient_applications": int(learner.gradient_applications),
        },
        "behavior_final_identity": {
            "policy_id": behavior_policy_id,
            "policy_generation": behavior_generation,
            "policy_sha256": behavior_policy_sha256,
            "parameter_state_sha256": parameter_state_sha256_v1(behavior_actor),
        },
        "target_final_identity": {
            "policy_id": "cc-s1-target",
            "policy_generation": str(int(config.max_units)),
            "parameter_state_sha256": parameter_state_sha256_v1(learner.target_actor),
        },
        "firewall": {
            "FINAL_opened": False,
            "fresh_data_used": False,
            "historical_market_corpus_accessed": False,
            "economic_evidence_claimed": False,
            "transfer_evidence_claimed": False,
        },
        "control_credit_permutation": shuffle_permutation,
        "control_frozen_realization": control_frozen_realization,
        "handcrafted_regime_activation": bool(spec.handcrafted_regime_activation),
        "objective_orientation": spec.objective_orientation,
        "ratio_diagnostics": {
            "max_ratio_above_one_count": max((item.ratio_above_one_count for item in unit_records), default=0),
            "max_ratio_below_one_count": max((item.ratio_below_one_count for item in unit_records), default=0),
            "min_ratio_abs_max_minus_one": min((item.ratio_abs_max_minus_one for item in unit_records), default=0.0),
        },
    }
    return result_payload
