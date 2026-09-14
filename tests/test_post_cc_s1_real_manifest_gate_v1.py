"""R2-C2: compile the real execution manifest with the real runner audit mapping."""

from __future__ import annotations

from cb16_local_opt.post_cc_s1_execution_manifest_v1 import build_s1_execution_manifest_v1
from cb16_local_opt.post_cc_s1_gate_compiler_v1 import CLASSIFICATION_PASS, compile_s1_gates_v1
from cb16_local_opt.post_cc_s1_qualification_v1 import (
    run_fabricated_log_mu_audit_v1,
    run_high_bankruptcy_failure_fact_audit_v1,
    run_integrity_attack_suite_v1,
    run_objective_firewall_audit_v1,
)
from cb16_local_opt.post_cc_s1_tasks_v1 import (
    TASK_ABA_RETENTION,
    TASK_ACCOUNT_DEPENDENT_ACTION,
    TASK_DELAYED_CONSEQUENCE_CREDIT,
    TASK_HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION,
    TASK_OFF_POLICY_VTRACE_CORRECTION,
    build_task_specs_v1,
)

MANIFEST = build_s1_execution_manifest_v1()
SEEDS = [1701, 1702, 1703, 1704, 1705]


def _context(return_value: float, probability: float, family_mass: float = 0.0) -> dict:
    return {
        "mean_complete_sample_arithmetic_return": return_value,
        "oracle_direction_probability": probability,
        "higher_ev_family_mass": family_mass,
    }


def _checkpoint(task_id: str, value: float, probability: float = 0.8, family_mass: float = 0.0) -> dict:
    spec = build_task_specs_v1()[task_id]
    if task_id == TASK_ABA_RETENTION:
        return {
            "mean_complete_sample_arithmetic_return": value,
            "contexts": {"A": _context(value, probability), "B": _context(value, probability)},
        }
    context_ids = [context.context_id for context in spec.contexts]
    return {
        "mean_complete_sample_arithmetic_return": value,
        "contexts": {context_id: _context(value, probability, family_mass) for context_id in context_ids},
    }


def _unit(index: int, *, switch: bool = True) -> dict:
    return {
        "unit_index": index,
        "sequence_ids": [f"seq-{index}"],
        "restart_sentinel": {"restart_verified": True},
        "generation_switch_receipt": (
            {"account_fully_preserved": True, "parent_checkpoint_mutated_in_place": False} if switch else None
        ),
        "update_skipped": False,
    }


def _oracle(task_id: str, value: float) -> dict:
    spec = build_task_specs_v1()[task_id]
    return {
        "mean_oracle_return": value,
        "contexts": {
            context.context_id: {"oracle_return": value, "best_directions": ["LONG"]}
            for context in spec.contexts
        },
    }


def _passing_positive(task_id: str, seed: int) -> dict:
    oracle = 0.18
    result = {
        "task_id": task_id,
        "seed": seed,
        "manifest_sha256": MANIFEST["manifest_sha256"],
        "unit_size": 128,
        "oracle": _oracle(task_id, oracle),
        "durability": {"restart_sentinel_passed": True, "behavior_checkpoint_untouched_every_unit": True},
        "unit_evidence": [_unit(0, switch=task_id != TASK_OFF_POLICY_VTRACE_CORRECTION)],
        "firewall": {
            "FINAL_opened": False,
            "fresh_data_used": False,
            "historical_market_corpus_accessed": False,
            "economic_evidence_claimed": False,
            "transfer_evidence_claimed": False,
        },
        "handcrafted_regime_activation": False,
        "objective_orientation": "COMPLETE_SAMPLE_ARITHMETIC_EQUITY_DELTA",
        "ratio_diagnostics": {"max_ratio_above_one_count": 3, "max_ratio_below_one_count": 2},
        "failure_facts": {
            "all_failures_retained_in_complete_arithmetic_denominator": True,
            "survivor_filtering": False,
        },
    }
    if task_id == TASK_ABA_RETENTION:

        def aba_checkpoint(a_value: float, b_value: float) -> dict:
            return {
                "mean_complete_sample_arithmetic_return": (a_value + b_value) / 2.0,
                "contexts": {"A": _context(a_value, 0.8), "B": _context(b_value, 0.8)},
            }

        result["checkpoints"] = {
            "INITIAL": aba_checkpoint(0.0, 0.0),
            "POST_A1": aba_checkpoint(0.144, 0.0),
            "POST_B": aba_checkpoint(0.144, 0.144),
            "POST_A2": aba_checkpoint(0.144, 0.144),
        }
        result["unit_evidence"] = [_unit(index) for index in range(96)]
        result["retention_evidence"] = {
            "phase_plan_completed": True,
            "a1_collected_count": 32,
            "a1_durable_sequence_count_after_b": 32,
            "a1_eligible_for_generic_replay_after_b": True,
            "a1_index_rows_still_in_uniform_eligible_pool": 32,
            "no_age_based_expiry": True,
            "eligibility_rule": "ALL_DURABLE_INDEX_ROWS_UNIFORM_NO_EXPIRY",
            "a1_selected_during_b_phase_count": 32,
        }
    else:
        family_final = 0.5 if task_id == TASK_HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION else 0.0
        result["checkpoints"] = {
            "INITIAL": _checkpoint(task_id, 0.0, probability=0.2, family_mass=0.0),
            "FINAL": _checkpoint(task_id, 0.12, probability=0.8, family_mass=family_final),
        }
    return result


def _failing_control(task_id: str, seed: int) -> dict:
    result = _passing_positive(task_id, seed)
    if task_id == TASK_ABA_RETENTION:
        for name in ("POST_A1", "POST_B", "POST_A2"):
            contexts = result["checkpoints"][name]["contexts"]
            for context_id in contexts:
                contexts[context_id]["mean_complete_sample_arithmetic_return"] = 0.0
            result["checkpoints"][name]["mean_complete_sample_arithmetic_return"] = 0.0
        result["retention_evidence"]["a1_eligible_for_generic_replay_after_b"] = False
    else:
        result["checkpoints"]["FINAL"] = _checkpoint(task_id, 0.0, probability=0.21)
    return result


def test_real_manifest_with_runner_audit_mapping_has_no_spurious_contract_mismatch():
    integrity = run_integrity_attack_suite_v1(".")
    audits = {
        "OBJECTIVE_FIREWALL": run_objective_firewall_audit_v1("."),
        "FABRICATED_LOG_MU_REJECTION": run_fabricated_log_mu_audit_v1("."),
        "HIGH_BANKRUPTCY_FAILURE_FACT": run_high_bankruptcy_failure_fact_audit_v1("."),
    }
    positive_results = {}
    control_results: dict[str, dict[str, dict]] = {}
    for task_entry in MANIFEST["tasks"]:
        task_id = task_entry["task_id"]
        for seed in SEEDS:
            positive_results[f"{task_id}|{seed}"] = _passing_positive(task_id, seed)
        for control_id in task_entry["declared_controls"]:
            if control_id in ("OBJECTIVE_FIREWALL", "FABRICATED_LOG_MU_REJECTION"):
                continue
            control_results[f"{task_id}|{control_id}"] = {
                str(seed): _failing_control(task_id, seed) for seed in SEEDS
            }
    compiled = compile_s1_gates_v1(
        manifest=MANIFEST,
        positive_results=positive_results,
        control_results=control_results,
        integrity=integrity,
        audits=audits,
    )
    assert compiled["classification"] == CLASSIFICATION_PASS, compiled["contract_violations"]
    assert compiled["contract_violations"] == []
    assert compiled["gates"][
        "CONTROL:HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION|OBJECTIVE_FIREWALL"
    ] is True
    assert compiled["gates"]["CONTROL:OFF_POLICY_VTRACE_CORRECTION|FABRICATED_LOG_MU_REJECTION"] is True
    assert compiled["gates"]["AUDIT:HIGH_BANKRUPTCY_FAILURE_FACT"] is True
