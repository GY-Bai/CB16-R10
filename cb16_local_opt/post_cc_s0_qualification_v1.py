from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch

from .cc_critic_value_r0 import SeparateCritic
from .cc_economic_promotion_r0 import UNRESOLVED_OWNER_DECISION, assess as legacy_assess
from .cc_experience_wire_r0 import CCEconomicResultV1
from .cc_policy_brain_r0 import BrainBindings, CCCentralBrain
from .post_cc_baseline_components_v1 import FAIL, PASS, from_deltas
from .post_cc_economic_adapter_v1 import adapt_w05
from .post_cc_economic_migration_v1 import (
    LEGACY_PROMOTION_CONTRACT_ID,
    migrate_legacy_result,
    require_post_cc_promotion_authority,
)
from .post_cc_economic_ordering_v1 import LEFT_HIGHER, compare_policy_measures
from .post_cc_joint_replay_contract_v1 import PostCCJointReplaySampleV1
from .post_cc_observation_contract_v1 import build_observation_fact
from .post_cc_promotion_v1 import (
    CANDIDATE_OVER_INCUMBENT,
    PROMOTE,
    REQUIRE_ANY_BASELINE_COMPONENT,
    REQUIRE_BOTH_BASELINE_COMPONENTS,
    assess_promotion,
)

S0_QUALIFICATION_COMPILER_ID = "CB16_R11_POST_CC_S0_QUALIFICATION_V1"
REQUIRED_GATES = (
    "S0_BASELINE_IDENTITY_PASS",
    "NO_MASTER_BASELINE_PRECEDENCE_PASS",
    "MODEL_ORDERING_SEPARATION_PASS",
    "PARALLEL_BASELINE_COMPONENTS_PASS",
    "PROMOTION_RULE_SEPARATION_PASS",
    "LEGACY_MIGRATION_PASS",
    "HISTORICAL_RECEIPT_IMMUTABILITY_PASS",
    "W05_ADAPTER_PASS",
    "S1_OBSERVATION_EXTENSION_CONTRACT_PASS",
    "S1_JOINT_REPLAY_CONTRACT_PASS",
    "S1_TASK_REGISTRY_FROZEN_PASS",
    "S1_RUN_SPEC_FROZEN_PASS",
    "S1_LEARNING_EVIDENCE_RULE_FROZEN_PASS",
    "NEGATIVE_CONTROL_PREREGISTRATION_PASS",
    "FINAL_FRESH_FIREWALL_PASS",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _git_blob_sha(repo_root: Path, relative_path: str) -> str:
    return subprocess.check_output(
        ["git", "hash-object", relative_path], cwd=repo_root, text=True
    ).strip()


def _economic_result(policy: str, mean: float, bh: float, flat: float) -> CCEconomicResultV1:
    return CCEconomicResultV1(
        evaluation_id=f"s0-{policy}",
        policy_object_type="frozen_checkpoint",
        policy_identity=policy,
        cohort_id="S0_SYNTHETIC_COHORT",
        common_horizon_id="S0_T",
        capital_denominator_id="S0_CAPITAL",
        account_results=({"account_lineage_id": "a", "arithmetic_return": mean},),
        mean_arithmetic_return=mean,
        buy_hold_delta=bh,
        flat_delta=flat,
        failure_counts={"failed": 0},
        tail_diagnostics={"median": mean},
        result_scope="SYNTHETIC",
    ).validate()


def _semantic_parameter_hash(actor: torch.nn.Module, critic: torch.nn.Module) -> str:
    def encode(module: torch.nn.Module, prefix: str) -> list[dict[str, Any]]:
        rows = []
        for name, tensor in sorted(module.state_dict().items()):
            rows.append(
                {
                    "name": f"{prefix}.{name}",
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "values": [
                        float(x) for x in tensor.detach().cpu().reshape(-1).tolist()
                    ],
                }
            )
        return rows

    payload = {
        "actor": encode(actor, "actor"),
        "critic": encode(critic, "critic"),
        "optimizer_step": 0,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _model_identity_check(run_spec: dict[str, Any]) -> bool:
    torch.manual_seed(int(run_spec["model"]["initialization"]["torch_manual_seed"]))
    bindings = BrainBindings("1" * 64, "2" * 64, "3" * 64, "4" * 64)
    actor_cfg = run_spec["model"]["actor"]
    critic_cfg = run_spec["model"]["critic"]
    actor = CCCentralBrain(
        int(actor_cfg["market_dim"]),
        int(actor_cfg["account_dim"]),
        int(actor_cfg["execution_dim"]),
        int(actor_cfg["hidden"]),
        bindings,
    )
    critic = SeparateCritic(int(critic_cfg["input_dim"]), hidden=int(critic_cfg["hidden"]))
    p_actor_trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    p_actor_frozen = sum(p.numel() for p in actor.parameters() if not p.requires_grad)
    p_critic = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    expected_hash = run_spec["model"]["initialization"]["initial_checkpoint_semantic_sha256"]
    return (
        p_actor_trainable == int(actor_cfg["P_actor_trainable"])
        and p_actor_frozen == int(actor_cfg["P_actor_frozen"])
        and p_critic == int(critic_cfg["P_critic_trainable"])
        and p_actor_trainable + p_critic == int(run_spec["model"]["P_trainable_total"])
        and _semantic_parameter_hash(actor, critic) == expected_hash
    )


def compile_s0_gates(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    baseline = _load_json(root / "authority/rearchitecture_r11/CB16_R11_POST_CC_S0_BASELINE_V1.json")
    spec = _load_json(root / "authority/rearchitecture_r11/CB16_R11_POST_CC_S0_SPEC_V1.json")
    registry = _load_json(root / "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_TASK_REGISTRY_V1.json")
    run_spec = _load_json(root / "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_RUN_SPEC_V1.json")

    gates: dict[str, bool] = {}
    gates["S0_BASELINE_IDENTITY_PASS"] = (
        baseline.get("frozen_implementation_base_sha")
        == "fc7102442e91a1c27cf705487c6d06bd64b8ea09"
        and baseline.get("status") == "FROZEN"
    )

    successor_sources = (
        "cb16_local_opt/post_cc_economic_ordering_v1.py",
        "cb16_local_opt/post_cc_baseline_components_v1.py",
        "cb16_local_opt/post_cc_promotion_v1.py",
        "cb16_local_opt/post_cc_economic_migration_v1.py",
        "cb16_local_opt/post_cc_economic_adapter_v1.py",
    )
    forbidden_precedence_tokens = (
        "master_baseline_winner",
        "B&H_overrides_FLAT",
        "FLAT_overrides_B&H",
    )
    source_text = "\n".join((root / path).read_text(encoding="utf-8") for path in successor_sources)
    gates["NO_MASTER_BASELINE_PRECEDENCE_PASS"] = (
        spec["economic_semantics"]["master_baseline"] is None
        and spec["economic_semantics"]["baseline_precedence"] is None
        and not any(token in source_text for token in forbidden_precedence_tokens)
    )

    left_a = adapt_w05(_economic_result("candidate", 0.10, -10.0, 10.0)).ordering_measure
    incumbent_a = adapt_w05(_economic_result("incumbent", 0.05, 10.0, -10.0)).ordering_measure
    left_b = adapt_w05(_economic_result("candidate", 0.10, 999.0, -999.0)).ordering_measure
    incumbent_b = adapt_w05(_economic_result("incumbent", 0.05, -999.0, 999.0)).ordering_measure
    gates["MODEL_ORDERING_SEPARATION_PASS"] = (
        compare_policy_measures(left_a, incumbent_a).status == LEFT_HIGHER
        and compare_policy_measures(left_b, incumbent_b).status == LEFT_HIGHER
    )

    combos = [
        (1.0, 1.0, PASS, PASS),
        (1.0, -1.0, PASS, FAIL),
        (-1.0, 1.0, FAIL, PASS),
        (-1.0, -1.0, FAIL, FAIL),
    ]
    gates["PARALLEL_BASELINE_COMPONENTS_PASS"] = all(
        (
            from_deltas(buy_hold_delta=bh, flat_delta=flat).buy_and_hold.status == bs
            and from_deltas(buy_hold_delta=bh, flat_delta=flat).flat.status == fs
        )
        for bh, flat, bs, fs in combos
    )

    mixed = from_deltas(buy_hold_delta=-0.1, flat_delta=0.1)
    ordering = compare_policy_measures(left_a, incumbent_a)
    gates["PROMOTION_RULE_SEPARATION_PASS"] = (
        assess_promotion(
            promotion_rule_id=REQUIRE_BOTH_BASELINE_COMPONENTS,
            candidate_policy_identity="candidate",
            components=mixed,
        ).status
        != assess_promotion(
            promotion_rule_id=REQUIRE_ANY_BASELINE_COMPONENT,
            candidate_policy_identity="candidate",
            components=mixed,
        ).status
        and assess_promotion(
            promotion_rule_id=CANDIDATE_OVER_INCUMBENT,
            candidate_policy_identity="candidate",
            ordering=ordering,
            improvement_evidence_pass=True,
        ).status
        == PROMOTE
    )

    legacy_result = _economic_result("candidate", 0.10, -0.1, 0.1)
    migration = migrate_legacy_result(legacy_result)
    legacy_route_rejected = False
    try:
        require_post_cc_promotion_authority(LEGACY_PROMOTION_CONTRACT_ID)
    except ValueError as exc:
        legacy_route_rejected = "CONTRACT_MISMATCH" in str(exc)
    gates["LEGACY_MIGRATION_PASS"] = (
        legacy_assess(legacy_result).status == UNRESOLVED_OWNER_DECISION
        and migration.legacy_status == UNRESOLVED_OWNER_DECISION
        and migration.current_owner_uncertainty is False
        and migration.promotion_rule_required is True
        and legacy_route_rejected
    )

    immutable_paths = (
        ("authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json", baseline["parent_cc"]["integration_receipt"]["blob_sha"]),
        ("authority/rearchitecture_r11/CB16_R11_CC_THREAD_C_RECEIPT_V1.json", baseline["parent_cc"]["thread_c_receipt"]["blob_sha"]),
    )
    gates["HISTORICAL_RECEIPT_IMMUTABILITY_PASS"] = all(
        _git_blob_sha(root, path) == expected for path, expected in immutable_paths
    )

    adapted = adapt_w05(_economic_result("candidate", 0.10, -0.2, 0.3))
    gates["W05_ADAPTER_PASS"] = (
        adapted.ordering_measure.mean_arithmetic_return == 0.10
        and adapted.baseline_components.buy_and_hold.delta == -0.2
        and adapted.baseline_components.flat.delta == 0.3
    )

    observation = build_observation_fact(
        science_semantic_version=spec["science_semantic_version"],
        observation_schema="S0_KNOWN_ANSWER_OBSERVATION_V1",
        market_payload={"x": [1.0, 2.0]},
        account_payload={"equity": 1000.0},
        execution_payload={"fees": 0.0},
        market_source_identity="S0_SYNTHETIC",
        market_source_version="V1",
        market_visible_through_time="00000007",
        account_lineage_id="acct",
        decision_index=3,
        environment_time="00000007",
        normalizer_identity="POST_CC_S1_SYNTHETIC_NORMALIZER_V1",
    )
    corruption_rejected = False
    try:
        replace(observation, account_payload={"equity": 999.0}).validate()
    except ValueError as exc:
        corruption_rejected = "OBSERVATION_HASH_MISMATCH" in str(exc)
    gates["S1_OBSERVATION_EXTENSION_CONTRACT_PASS"] = corruption_rejected

    sample = PostCCJointReplaySampleV1(
        sequence_id="seq",
        transition_ref="ref",
        account_lineage_id="acct",
        decision_index=3,
        environment_time="00000007",
        observation=observation,
        nominal_direction="LONG",
        nominal_target_risk=0.4,
        risk_measure_kind="continuous_density",
        behavior_log_mu=-1.2,
        behavior_policy_identity="pi-g0",
        behavior_generation="0",
        reward=0.03,
        discount=1.0,
        boundary_type="CONTINUE",
        bootstrap_state_ref_or_null=None,
        sampling_probability_or_weight=1.0,
        target_policy_identity="pi-target",
        source_fact_hashes=("a" * 64,),
        consequence_context={"executed_direction": "FLAT"},
    ).validate()
    flat_nonzero_rejected = False
    try:
        replace(sample, nominal_direction="FLAT", nominal_target_risk=0.4).validate()
    except ValueError:
        flat_nonzero_rejected = True
    gates["S1_JOINT_REPLAY_CONTRACT_PASS"] = (
        sample.nominal_direction == "LONG"
        and sample.consequence_context["executed_direction"] == "FLAT"
        and flat_nonzero_rejected
    )

    mandatory_tasks = {
        "ACCOUNT_DEPENDENT_ACTION",
        "DELAYED_CONSEQUENCE_CREDIT",
        "HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION",
        "OFF_POLICY_VTRACE_CORRECTION",
        "A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION",
    }
    registered = {task["task_id"] for task in registry.get("tasks", [])}
    gates["S1_TASK_REGISTRY_FROZEN_PASS"] = (
        registry.get("status") == "FROZEN_BEFORE_S1_EXECUTION"
        and mandatory_tasks == registered
        and len(registry["common_seed_policy"]["seeds"]) == 5
        and registry["common_seed_policy"]["minimum_positive_seeds_passing"] == 4
    )

    gates["S1_RUN_SPEC_FROZEN_PASS"] = (
        run_spec.get("status") == "FROZEN_BEFORE_S1_EXECUTION"
        and run_spec.get("data_scope") == "SYNTHETIC_ONLY"
        and run_spec["collection_and_learning"]["persistent_replay_only_for_qualification"] is True
        and run_spec["collection_and_learning"]["collector_in_memory_records_as_training_truth"] is False
        and _model_identity_check(run_spec)
    )

    gates["S1_LEARNING_EVIDENCE_RULE_FROZEN_PASS"] = (
        registry["common_learning_rule"]["behavioral_or_return_improvement_required"] is True
        and registry["common_learning_rule"]["known_answer_oracle_gap_reduction_fraction"] == 0.5
        and registry["common_learning_rule"]["training_loss_alone_is_success"] is False
        and run_spec["evaluation"]["training_loss_is_diagnostic_only"] is True
    )

    controls = {control["control_id"] for control in registry.get("controls", [])}
    gates["NEGATIVE_CONTROL_PREREGISTRATION_PASS"] = {
        "NO_SIGNAL_OR_ZERO_REWARD_CONTROL",
        "SHUFFLED_CREDIT_CONTROL",
        "RANDOM_OR_IMPOSSIBLE_LABEL_CONTROL",
    }.issubset(controls) and registry["common_learning_rule"]["no_rescue_after_results"] is True

    gates["FINAL_FRESH_FIREWALL_PASS"] = (
        baseline["firewall"]["FINAL_opened"] is False
        and baseline["firewall"]["fresh_data_used"] is False
        and spec["firewall"]["FINAL_opened"] is False
        and spec["firewall"]["fresh_data_used"] is False
        and run_spec["firewall"]["FINAL_opened"] is False
        and run_spec["firewall"]["fresh_data_used"] is False
        and run_spec["firewall"]["historical_market_corpus_access"] is False
    )

    missing = [name for name in REQUIRED_GATES if not gates.get(name, False)]
    return {
        "compiler_id": S0_QUALIFICATION_COMPILER_ID,
        "status": "PASS" if not missing else "FAIL",
        "classification": "PASS" if not missing else "CONTRACT_MISMATCH",
        "gates": gates,
        "failed_gates": missing,
        "evidence_if_pass": "POST_CC_CONTRACT_MIGRATION_QUALIFIED",
        "FINAL_opened": False,
        "fresh_data_used": False,
        "ECONOMIC_evidence_claimed": False,
        "TRANSFER_evidence_claimed": False,
    }
