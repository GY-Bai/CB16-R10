"""CB16 R11 S1 execution manifest: frozen scientific concretization.

The manifest is generated from the task-spec module and validated fail-closed
against the frozen V1 registry/run spec/S0-v2 receipt before any training is
allowed.  In qualification mode nothing about seeds, thresholds, reward,
model, optimizer or budget may be overridden by CLI.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .post_cc_s1_tasks_v1 import (
    ACTOR_LEARNING_RATE_V1,
    ANALYTIC_VTRACE_TOLERANCE_V1,
    CONTROL_ACCOUNT_ABLATION,
    CONTROL_FABRICATED_LOG_MU_REJECTION,
    CONTROL_NO_SIGNAL,
    CONTROL_OBJECTIVE_FIREWALL,
    CONTROL_RANDOM_IMPOSSIBLE,
    CONTROL_SHUFFLED_CREDIT,
    CRITIC_LEARNING_RATE_V1,
    FROZEN_SEEDS_V1,
    GAP_REDUCTION_THRESHOLD_V1,
    MAXIMUM_FALSE_POSITIVE_CONTROL_SEEDS_V1,
    MAXIMUM_POLICY_DECISIONS_V1,
    MINIMUM_PASSING_SEEDS_V1,
    MODEL_INITIALIZATION_SEED_V1,
    QUALIFICATION_COLLECTION_UNIT_V1,
    QUALIFICATION_EVALUATION_ACTIONS_V1,
    QUALIFICATION_POLICY_DECISIONS_V1,
    QUALIFICATION_REPLAY_BATCH_V1,
    SCIENCE_SEMANTIC_VERSION,
    STRICT_TOLERANCE_V1,
    build_task_specs_v1,
)

S1_MANIFEST_SCHEMA_V1 = "CB16_R11_POST_CC_S1_EXECUTION_MANIFEST_V1"
S1_MANIFEST_STATUS_V1 = "FROZEN_EXECUTION_CONCRETIZATION_UNDER_S1_V1_CONTRACTS"
S0V2_ACCEPTED_MERGE_SHA_V1 = "fc80b472236e7a4df8094563f8adac826fc42231"
FROZEN_INITIAL_CHECKPOINT_SEMANTIC_SHA256_V1 = (
    "32928a6b2fea2346d303c9b61e8f86dee66099e973e7391372836b4f8a706021"
)

AUTHORITY_PATHS_V1 = {
    "baseline": "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_BASELINE_V1.json",
    "s0v2_receipt": "authority/rearchitecture_r11/CB16_R11_S0V2_RECEIPT_V1.json",
    "task_registry": "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_TASK_REGISTRY_V1.json",
    "run_spec": "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_RUN_SPEC_V1.json",
}

EXPECTED_AUTHORITY_BLOB_SHA1_V1 = {
    "task_registry": "d21aece618403322bfc9bb5752b18960ecf73ec4",
    "run_spec": "ed4aba2271aacea38f1184550e6f29bda3178a53",
}

EXPECTED_MODEL_SOURCE_BLOB_SHA1_V1 = {
    "cb16_local_opt/cc_policy_brain_r0.py": "9f63082aacc4c03c877996527465cabe28185e90",
    "cb16_local_opt/cc_critic_value_r0.py": "3c6ed457913e45b431902b8ba00f98c00c6b526d",
    "cb16_local_opt/cc_policy_distribution_r0.py": "3c97f6a548fbd0969a54ce1a0ec2823cb5d87f94",
}


class S1ManifestError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def git_blob_sha1_v1(path: str | Path) -> str:
    payload = Path(path).read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def execution_manifest_payload_v1() -> dict[str, Any]:
    specs = build_task_specs_v1()
    task_entries = []
    for spec in specs.values():
        task_entries.append(
            {
                "task_id": spec.task_id,
                "environment_version": spec.environment_version,
                "spec_hash": spec.spec_hash,
                "observation_semantics": spec.observation_semantics,
                "action_semantics": spec.action_semantics,
                "reward_semantics": spec.reward_semantics,
                "horizon_semantics": spec.horizon_semantics,
                "known_answer": spec.known_answer,
                "context_schedule": spec.context_schedule,
                "phase_context_ids": [list(item) for item in spec.phase_context_ids],
                "training_decisions": int(spec.training_decisions),
                "gap_reduction_threshold": float(spec.gap_reduction_threshold),
                "behavior_mode": spec.behavior_mode,
                "credit_stage2_no_decision": bool(spec.credit_stage2_no_decision),
                "objective_orientation": spec.objective_orientation,
                "handcrafted_regime_activation": bool(spec.handcrafted_regime_activation),
                "execution": {
                    "fee_rate": float(spec.fee_rate),
                    "slippage_bps": float(spec.slippage_bps),
                    "initial_margin_rate": float(spec.initial_margin_rate),
                    "maintenance_margin_rate": float(spec.maintenance_margin_rate),
                    "max_gross_leverage": float(spec.max_gross_leverage),
                    "reward_reference_equity": float(spec.reward_reference_equity),
                },
                "declared_controls": list(spec.declared_controls),
                "higher_ev_family": {
                    "directions": list(spec.higher_ev_family_directions),
                    "risk_low": float(spec.higher_ev_family_risk_low),
                    "risk_high": float(spec.higher_ev_family_risk_high),
                },
                "contexts": [
                    {
                        "context_id": context.context_id,
                        "account_kind": context.account_kind,
                        "setup_direction": context.setup_direction,
                        "setup_risk": context.setup_risk,
                        "oracle_direction_expected": context.oracle_direction_expected,
                        "decision_mark": float(context.dynamics.decision_mark),
                        "stage1_mark": float(context.dynamics.stage1_mark),
                        "stage2_is_no_decision_advance": bool(context.dynamics.stage2_is_no_decision_advance),
                        "stage2_branches": [
                            {"probability": float(branch.probability), "terminal_mark": float(branch.terminal_mark)}
                            for branch in context.dynamics.stage2_branches
                        ],
                    }
                    for context in spec.contexts
                ],
            }
        )
    return {
        "schema": S1_MANIFEST_SCHEMA_V1,
        "status": S1_MANIFEST_STATUS_V1,
        "science_semantic_version": SCIENCE_SEMANTIC_VERSION,
        "scope": "SYNTHETIC_ONLY",
        "parent": {
            "accepted_s0v2_merge_sha": S0V2_ACCEPTED_MERGE_SHA_V1,
            "s0v2_receipt_required_status": "QUALIFIED",
            "s0v2_receipt_required_verdict": "PASS",
        },
        "authority": dict(AUTHORITY_PATHS_V1),
        "firewall": {
            "FINAL_opened": False,
            "fresh_data_used": False,
            "historical_market_corpus_accessed": False,
            "economic_evidence_claimed": False,
            "transfer_evidence_claimed": False,
        },
        "seeds": list(FROZEN_SEEDS_V1),
        "minimum_positive_seeds_passing": int(MINIMUM_PASSING_SEEDS_V1),
        "maximum_false_positive_control_seeds": int(MAXIMUM_FALSE_POSITIVE_CONTROL_SEEDS_V1),
        "strict_tolerance": float(STRICT_TOLERANCE_V1),
        "analytic_vtrace_tolerance": float(ANALYTIC_VTRACE_TOLERANCE_V1),
        "gap_reduction_threshold": float(GAP_REDUCTION_THRESHOLD_V1),
        "maximum_policy_decisions_per_task_per_seed": int(MAXIMUM_POLICY_DECISIONS_V1),
        "oracle_comparator_class": {
            "flat_risk": 0.0,
            "risk_values": [0.1, 0.3, 0.5, 0.7, 0.9],
            "directions": ["LONG", "SHORT"],
            "comparator": "HIGHEST_COMPLETE_SAMPLE_ARITHMETIC_RETURN_MEMBER",
        },
        "training_schedule": {
            "qualification_policy_decisions": int(QUALIFICATION_POLICY_DECISIONS_V1),
            "collection_unit_policy_decisions": int(QUALIFICATION_COLLECTION_UNIT_V1),
            "durable_learner_updates_per_collection_unit": 1,
            "replay_batch_target_samples": int(QUALIFICATION_REPLAY_BATCH_V1),
            "replay_sampling": "UNIFORM_FROM_ALL_ELIGIBLE_DURABLE_REPLAY",
            "replay_age_expiry": False,
            "behavior_checkpoint_fixed_within_collection_unit": True,
            "child_becomes_behavior_only_through_committed_generation_switch": True,
            "policy_decisions_early_stop_on_success": False,
        },
        "evaluation_population": {
            "nominal_actions_per_context": int(QUALIFICATION_EVALUATION_ACTIONS_V1),
            "rng": "DEDICATED_DETERMINISTIC_EVALUATION_STREAM",
            "environment_branches": "EXACT_FINITE_BRANCH_ENUMERATION",
            "failures_and_bankruptcies_remain_in_denominator": True,
            "evaluation_decisions_enter_training_replay": False,
        },
        "model": {
            "actor_class": "CCCentralBrain",
            "actor_source_path": "cb16_local_opt/cc_policy_brain_r0.py",
            "actor_dimensions": {"market": 2, "account": 3, "execution": 2, "hidden": 8},
            "actor_trainable_parameter_count": 241,
            "actor_frozen_parameter_count": 16,
            "critic_class": "SeparateCritic",
            "critic_source_path": "cb16_local_opt/cc_critic_value_r0.py",
            "critic_input_dim": 7,
            "critic_hidden": 8,
            "critic_trainable_parameter_count": 73,
            "trainable_parameter_total": 314,
            "frozen_market_organ": True,
            "initialization_seed": int(MODEL_INITIALIZATION_SEED_V1),
            "frozen_initial_checkpoint_semantic_sha256_declared": FROZEN_INITIAL_CHECKPOINT_SEMANTIC_SHA256_V1,
            "optimizer": {
                "actor": {"type": "SGD", "learning_rate": float(ACTOR_LEARNING_RATE_V1)},
                "critic": {"type": "SGD", "learning_rate": float(CRITIC_LEARNING_RATE_V1)},
                "optimizer_change_after_results": False,
            },
        },
        "off_policy": {
            "algorithm": "V_TRACE",
            "true_behavior_log_mu_required": True,
            "target_log_pi_recomputed_from_nominal_action": True,
            "executed_action_substitution_for_policy_likelihood": False,
            "rho_bar": 1.0,
            "c_bar": 1.0,
            "pg_rho_bar": 1.0,
        },
        "controls": [
            {
                "control_id": CONTROL_NO_SIGNAL,
                "kind": "TRAINING_MATCHED",
                "rule": "ZERO_DECLARED_TASK_REWARD_NO_LEARNABLE_ACTION_TO_REWARD_RELATION",
            },
            {
                "control_id": CONTROL_SHUFFLED_CREDIT,
                "kind": "TRAINING_MATCHED",
                "rule": "BREAK_DECISION_TO_CONSEQUENCE_PAIRING_PRESERVING_MARGINALS",
            },
            {
                "control_id": CONTROL_RANDOM_IMPOSSIBLE,
                "kind": "TRAINING_MATCHED",
                "rule": "TARGET_ASSIGNMENT_INDEPENDENT_OF_ACTOR_VISIBLE_CAUSAL_INPUTS",
            },
            {
                "control_id": CONTROL_ACCOUNT_ABLATION,
                "kind": "TRAINING_MATCHED",
                "rule": "ZERO_ACCOUNT_AND_EXECUTION_INPUTS_MARKET_ONLY_ABLATION",
            },
            {
                "control_id": CONTROL_OBJECTIVE_FIREWALL,
                "kind": "INTEGRITY_ONLY",
                "rule": "ARITHMETIC_COMPLETE_SAMPLE_OBJECTIVE_DENOMINATOR_AUDIT",
            },
            {
                "control_id": CONTROL_FABRICATED_LOG_MU_REJECTION,
                "kind": "INTEGRITY_ONLY",
                "rule": "FABRICATED_RECONSTRUCTED_LOG_MU_MUST_FAIL_CLOSED",
            },
        ],
        "tasks": task_entries,
    }


def build_s1_execution_manifest_v1() -> dict[str, Any]:
    payload = execution_manifest_payload_v1()
    manifest = dict(payload)
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return manifest


def write_s1_execution_manifest_v1(path: str | Path) -> dict[str, Any]:
    manifest = build_s1_execution_manifest_v1()
    Path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_s1_execution_manifest_v1(repo_root: str | Path) -> dict[str, Any]:
    """Fail-closed validation.  Raises S1ManifestError on any mismatch."""
    root = Path(repo_root)
    path = root / "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_EXECUTION_MANIFEST_V1.json"
    if not path.exists():
        raise S1ManifestError("EXECUTION_MANIFEST_MISSING")
    committed = _load_json(path)
    payload = {key: value for key, value in committed.items() if key != "manifest_sha256"}
    computed_sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if committed.get("manifest_sha256") != computed_sha:
        raise S1ManifestError("EXECUTION_MANIFEST_HASH_MISMATCH")
    expected = build_s1_execution_manifest_v1()
    if committed != expected:
        raise S1ManifestError("EXECUTION_MANIFEST_DOES_NOT_MATCH_CODE_CONSTANTS")

    for key, relative in AUTHORITY_PATHS_V1.items():
        authority_path = root / relative
        if not authority_path.exists():
            raise S1ManifestError(f"AUTHORITY_MISSING:{key}")
    if EXPECTED_AUTHORITY_BLOB_SHA1_V1["task_registry"] != git_blob_sha1_v1(root / AUTHORITY_PATHS_V1["task_registry"]):
        raise S1ManifestError("TASK_REGISTRY_BLOB_MISMATCH")
    if EXPECTED_AUTHORITY_BLOB_SHA1_V1["run_spec"] != git_blob_sha1_v1(root / AUTHORITY_PATHS_V1["run_spec"]):
        raise S1ManifestError("RUN_SPEC_BLOB_MISMATCH")
    for relative, expected_sha in EXPECTED_MODEL_SOURCE_BLOB_SHA1_V1.items():
        if git_blob_sha1_v1(root / relative) != expected_sha:
            raise S1ManifestError(f"MODEL_SOURCE_BLOB_MISMATCH:{relative}")

    baseline = _load_json(root / AUTHORITY_PATHS_V1["baseline"])
    if baseline["status"] != "FROZEN_READY_FOR_S1_IMPLEMENTATION_NOT_STARTED":
        raise S1ManifestError("S1_BASELINE_STATUS_MISMATCH")
    if baseline["exact_parent_main"]["sha"] != S0V2_ACCEPTED_MERGE_SHA_V1:
        raise S1ManifestError("S1_BASELINE_PARENT_MISMATCH")
    if baseline["firewall"]["FINAL_opened"] is not False:
        raise S1ManifestError("S1_BASELINE_FIREWALL_VIOLATION")
    if baseline["firewall"]["fresh_data_used"] is not False:
        raise S1ManifestError("S1_BASELINE_FRESH_DATA_VIOLATION")
    if baseline["firewall"]["historical_market_corpus_access"] is not False:
        raise S1ManifestError("S1_BASELINE_HISTORICAL_CORPUS_VIOLATION")

    receipt = _load_json(root / AUTHORITY_PATHS_V1["s0v2_receipt"])
    if receipt["status"] != "QUALIFIED" or receipt["review_verdict"] != "PASS":
        raise S1ManifestError("S0V2_RECEIPT_NOT_QUALIFIED")

    registry = _load_json(root / AUTHORITY_PATHS_V1["task_registry"])
    run_spec = _load_json(root / AUTHORITY_PATHS_V1["run_spec"])
    registry_task_ids = tuple(task["task_id"] for task in registry["tasks"])
    manifest_task_ids = tuple(task["task_id"] for task in committed["tasks"])
    if registry_task_ids != manifest_task_ids:
        raise S1ManifestError("TASK_ID_SET_MISMATCH")
    if tuple(registry["common_seed_policy"]["seeds"]) != tuple(committed["seeds"]):
        raise S1ManifestError("SEED_SET_MISMATCH")
    if int(registry["common_learning_rule"]["maximum_policy_decisions_per_task_per_seed"]) != int(
        committed["maximum_policy_decisions_per_task_per_seed"]
    ):
        raise S1ManifestError("MAXIMUM_DECISION_BUDGET_MISMATCH")
    if float(registry["common_learning_rule"]["known_answer_oracle_gap_reduction_fraction"]) != float(
        committed["gap_reduction_threshold"]
    ):
        raise S1ManifestError("GAP_THRESHOLD_MISMATCH")
    if tuple(run_spec["seed_policy"]["seeds"]) != tuple(committed["seeds"]):
        raise S1ManifestError("RUN_SPEC_SEED_MISMATCH")
    if int(run_spec["model"]["actor"]["P_actor_trainable"]) != int(committed["model"]["actor_trainable_parameter_count"]):
        raise S1ManifestError("ACTOR_PARAMETER_COUNT_MISMATCH")
    if int(run_spec["model"]["critic"]["P_critic_trainable"]) != int(committed["model"]["critic_trainable_parameter_count"]):
        raise S1ManifestError("CRITIC_PARAMETER_COUNT_MISMATCH")
    if float(run_spec["optimizer"]["actor"]["learning_rate"]) != float(
        committed["model"]["optimizer"]["actor"]["learning_rate"]
    ):
        raise S1ManifestError("ACTOR_LR_MISMATCH")
    if float(run_spec["optimizer"]["critic"]["learning_rate"]) != float(
        committed["model"]["optimizer"]["critic"]["learning_rate"]
    ):
        raise S1ManifestError("CRITIC_LR_MISMATCH")
    declared_hash = run_spec["model"]["initialization"]["initial_checkpoint_semantic_sha256"]
    if declared_hash != committed["model"]["frozen_initial_checkpoint_semantic_sha256_declared"]:
        raise S1ManifestError("INITIAL_CHECKPOINT_DECLARATION_MISMATCH")
    if run_spec["firewall"]["FINAL_opened"] is not False or run_spec["firewall"]["historical_market_corpus_access"] is not False:
        raise S1ManifestError("RUN_SPEC_FIREWALL_VIOLATION")
    contract = {"schema": committed["schema"], "manifest_sha256": committed["manifest_sha256"]}
    if contract["schema"] != S1_MANIFEST_SCHEMA_V1:
        raise S1ManifestError("EXECUTION_MANIFEST_SCHEMA_MISMATCH")
    return committed
