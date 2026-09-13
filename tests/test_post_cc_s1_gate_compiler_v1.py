"""Deterministic S1 gate compiler fixtures and threshold-boundary tests.

Expected verdicts are constructed by hand; the compiler is never asked to
generate its own expected answers.
"""

from __future__ import annotations

import math

import pytest

from cb16_local_opt.post_cc_s1_gate_compiler_v1 import (
    CLASSIFICATION_CONTRACT_MISMATCH,
    CLASSIFICATION_EVIDENCE_INSUFFICIENT,
    CLASSIFICATION_PASS,
    CLASSIFICATION_SCIENTIFIC_FAIL,
    compile_s1_gates_v1,
    evaluate_seed_predicate_v1,
)

SEEDS = [1701, 1702, 1703, 1704, 1705]
MANIFEST_SHA = "a" * 64


def _context_record(return_value: float, probability: float, family_mass: float = 0.0) -> dict:
    return {
        "mean_complete_sample_arithmetic_return": return_value,
        "oracle_direction_probability": probability,
        "higher_ev_family_mass": family_mass,
    }


def _checkpoint(return_value: float, probability: float, contexts=("UP_SIGNAL", "DOWN_SIGNAL"), family_mass=0.0) -> dict:
    return {
        "mean_complete_sample_arithmetic_return": return_value,
        "contexts": {name: _context_record(return_value, probability, family_mass) for name in contexts},
    }


def _unit(index: int, *, restart_ok: bool = True, switch_ok: bool = True) -> dict:
    return {
        "unit_index": index,
        "sequence_ids": [f"seq-{index}"],
        "restart_sentinel": {"restart_verified": restart_ok},
        "generation_switch_receipt": {
            "account_fully_preserved": switch_ok,
            "parent_checkpoint_mutated_in_place": False,
        },
    }


def _result(
    task_id: str = "DELAYED_CONSEQUENCE_CREDIT",
    *,
    initial: float = 0.0,
    final: float = 0.12,
    oracle: float = 0.18,
    initial_probability: float = 0.20,
    final_probability: float = 0.50,
    family_initial: float = 0.0,
    family_final: float = 0.5,
    firewall_ok: bool = True,
    restart_ok: bool = True,
    ratio_above: int = 2,
    ratio_below: int = 3,
    manifest_sha: str = MANIFEST_SHA,
    unit_size: int = 128,
) -> dict:
    contexts = ("UP_SIGNAL", "DOWN_SIGNAL")
    return {
        "schema": "CB16_R11_POST_CC_S1_SEED_RESULT_V1",
        "status": "OK",
        "task_id": task_id,
        "seed": 1701,
        "manifest_sha256": manifest_sha,
        "unit_size": unit_size,
        "oracle": {
            "mean_oracle_return": oracle,
            "contexts": {name: {"oracle_return": oracle, "best_directions": ["LONG"]} for name in contexts},
        },
        "checkpoints": {
            "INITIAL": _checkpoint(initial, initial_probability, contexts, family_initial),
            "FINAL": _checkpoint(final, final_probability, contexts, family_final),
        },
        "ratio_diagnostics": {"max_ratio_above_one_count": ratio_above, "max_ratio_below_one_count": ratio_below},
        "durability": {
            "restart_sentinel_passed": restart_ok,
            "behavior_checkpoint_untouched_every_unit": True,
        },
        "unit_evidence": [_unit(0, restart_ok=restart_ok)],
        "firewall": {
            "FINAL_opened": not firewall_ok,
            "fresh_data_used": False,
            "historical_market_corpus_accessed": False,
            "economic_evidence_claimed": False,
            "transfer_evidence_claimed": False,
        },
        "handcrafted_regime_activation": False,
        "objective_orientation": "COMPLETE_SAMPLE_ARITHMETIC_EQUITY_DELTA",
    }


def _aba_result(*, a1_reduction: float = 0.8, b_reduction: float = 0.8, a2_reduction: float = 0.8) -> dict:
    oracle_a, oracle_b = 0.18, 0.18
    baseline_a, baseline_b = 0.0, 0.0

    def score(oracle, reduction, baseline):
        # gap_reduction = (oracle-baseline - (oracle-score)) / (oracle-baseline)
        initial_gap = oracle - baseline
        return oracle - initial_gap * (1.0 - reduction)

    def checkpoint(a_value: float, b_value: float) -> dict:
        return {
            "mean_complete_sample_arithmetic_return": (a_value + b_value) / 2.0,
            "contexts": {
                "A": _context_record(a_value, 0.8),
                "B": _context_record(b_value, 0.8),
            },
        }

    post_a1_a = score(oracle_a, a1_reduction, baseline_a)
    post_b_b = score(oracle_b, b_reduction, baseline_b)
    post_a2_a = score(oracle_a, a2_reduction, baseline_a)
    units = []
    for index in range(96):
        if index < 32:
            units.append({"unit_index": index, "sequence_ids": [f"A1-{index}"], "restart_sentinel": {"restart_verified": True}, "generation_switch_receipt": {"account_fully_preserved": True, "parent_checkpoint_mutated_in_place": False}})
        elif index < 64:
            units.append({"unit_index": index, "sequence_ids": [f"B-{index}", f"A1-{index % 32}"], "restart_sentinel": {"restart_verified": True}, "generation_switch_receipt": {"account_fully_preserved": True, "parent_checkpoint_mutated_in_place": False}})
        else:
            units.append({"unit_index": index, "sequence_ids": [f"A2-{index}"], "restart_sentinel": {"restart_verified": True}, "generation_switch_receipt": {"account_fully_preserved": True, "parent_checkpoint_mutated_in_place": False}})
    return {
        "task_id": "A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION",
        "manifest_sha256": MANIFEST_SHA,
        "unit_size": 128,
        "retention_evidence": {
            "phase_plan_completed": True,
            "a1_collected_count": 32,
            "a1_durable_sequence_count_after_b": 32,
            "a1_eligible_for_generic_replay_after_b": True,
            "a1_index_rows_still_in_uniform_eligible_pool": 32,
            "no_age_based_expiry": True,
            "eligibility_rule": "ALL_DURABLE_INDEX_ROWS_UNIFORM_NO_EXPIRY",
            "a1_selected_during_b_phase_count": 32,
        },
        "oracle": {
            "mean_oracle_return": (oracle_a + oracle_b) / 2.0,
            "contexts": {"A": {"oracle_return": oracle_a}, "B": {"oracle_return": oracle_b}},
        },
        "checkpoints": {
            "INITIAL": checkpoint(baseline_a, baseline_b),
            "POST_A1": checkpoint(post_a1_a, baseline_b),
            "POST_B": checkpoint(post_a1_a, post_b_b),
            "POST_A2": checkpoint(post_a2_a, post_b_b),
        },
        "durability": {"restart_sentinel_passed": True, "behavior_checkpoint_untouched_every_unit": True},
        "unit_evidence": units,
        "handcrafted_regime_activation": False,
        "firewall": {
            "FINAL_opened": False,
            "fresh_data_used": False,
            "historical_market_corpus_accessed": False,
            "economic_evidence_claimed": False,
            "transfer_evidence_claimed": False,
        },
        "ratio_diagnostics": {"max_ratio_above_one_count": 1, "max_ratio_below_one_count": 1},
    }


def _manifest(controls=("NO_SIGNAL_OR_ZERO_REWARD_CONTROL", "SHUFFLED_CREDIT_CONTROL"), task_id="DELAYED_CONSEQUENCE_CREDIT") -> dict:
    return {
        "schema": "CB16_R11_POST_CC_S1_EXECUTION_MANIFEST_V1",
        "manifest_sha256": MANIFEST_SHA,
        "seeds": SEEDS,
        "minimum_positive_seeds_passing": 4,
        "maximum_false_positive_control_seeds": 1,
        "firewall": {
            "FINAL_opened": False,
            "fresh_data_used": False,
            "historical_market_corpus_accessed": False,
            "economic_evidence_claimed": False,
            "transfer_evidence_claimed": False,
        },
        "tasks": [{"task_id": task_id, "declared_controls": list(controls)}],
    }


def _integrity_ok() -> dict:
    return {"attacks": {"attack-1": {"rejected": True}}, "all_rejected": True}


def _audits_ok() -> dict:
    return {
        "FABRICATED_LOG_MU_REJECTION": {"all_checks_pass": True},
        "OBJECTIVE_FIREWALL": {"all_checks_pass": True},
        "HIGH_BANKRUPTCY_FAILURE_FACT": {"all_checks_pass": True},
    }


def _compile(positive_results, control_results, *, manifest=None, integrity=None, audits=None):
    return compile_s1_gates_v1(
        manifest=manifest or _manifest(),
        positive_results=positive_results,
        control_results=control_results,
        integrity=integrity or _integrity_ok(),
        audits=audits or _audits_ok(),
    )


def _passing_controls(**kwargs) -> dict:
    seeds = SEEDS if kwargs.pop("all_seeds", True) else SEEDS[:1]
    return {
        "DELAYED_CONSEQUENCE_CREDIT|NO_SIGNAL_OR_ZERO_REWARD_CONTROL": {str(seed): _result(final=0.0, final_probability=0.21) for seed in seeds},
        "DELAYED_CONSEQUENCE_CREDIT|SHUFFLED_CREDIT_CONTROL": {str(seed): _result(final=0.02) for seed in seeds},
    }


def test_fixture_01_all_pass_records_compile_pass():
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result() for seed in SEEDS}
    result = _compile(positive, _passing_controls())
    assert result["classification"] == CLASSIFICATION_PASS


def test_fixture_02_exactly_three_of_five_positive_seeds_fails():
    positive = {}
    for index, seed in enumerate(SEEDS):
        positive[f"DELAYED_CONSEQUENCE_CREDIT|{seed}"] = _result(final=0.12 if index < 3 else 0.0)
    result = _compile(positive, _passing_controls())
    assert result["details"]["POSITIVE:DELAYED_CONSEQUENCE_CREDIT"]["passing_count"] == 3
    assert result["classification"] == CLASSIFICATION_SCIENTIFIC_FAIL


def test_fixture_03_exactly_four_of_five_positive_seeds_passes():
    positive = {}
    for index, seed in enumerate(SEEDS):
        positive[f"DELAYED_CONSEQUENCE_CREDIT|{seed}"] = _result(final=0.12 if index < 4 else 0.0)
    result = _compile(positive, _passing_controls())
    assert result["details"]["POSITIVE:DELAYED_CONSEQUENCE_CREDIT"]["passing_count"] == 4
    assert result["classification"] == CLASSIFICATION_PASS


def test_fixture_04_two_of_five_control_false_positives_fails_control():
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result() for seed in SEEDS}
    controls = {
        "DELAYED_CONSEQUENCE_CREDIT|NO_SIGNAL_OR_ZERO_REWARD_CONTROL": {
            str(seed): _result(final=0.12 if index < 2 else 0.0) for index, seed in enumerate(SEEDS)
        },
        "DELAYED_CONSEQUENCE_CREDIT|SHUFFLED_CREDIT_CONTROL": {str(seed): _result(final=0.02) for seed in SEEDS},
    }
    result = _compile(positive, controls)
    assert result["gates"]["CONTROL:DELAYED_CONSEQUENCE_CREDIT|NO_SIGNAL_OR_ZERO_REWARD_CONTROL"] is False
    assert result["classification"] == CLASSIFICATION_SCIENTIFIC_FAIL


def test_fixture_05_one_of_five_control_false_positives_passes_control():
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result() for seed in SEEDS}
    controls = {
        "DELAYED_CONSEQUENCE_CREDIT|NO_SIGNAL_OR_ZERO_REWARD_CONTROL": {
            str(seed): _result(final=0.12 if index < 1 else 0.0) for index, seed in enumerate(SEEDS)
        },
        "DELAYED_CONSEQUENCE_CREDIT|SHUFFLED_CREDIT_CONTROL": {str(seed): _result(final=0.02) for seed in SEEDS},
    }
    result = _compile(positive, controls)
    assert result["gates"]["CONTROL:DELAYED_CONSEQUENCE_CREDIT|NO_SIGNAL_OR_ZERO_REWARD_CONTROL"] is True
    assert result["classification"] == CLASSIFICATION_PASS


def test_fixture_06_equal_shuffled_relation_fails():
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result(final=0.12) for seed in SEEDS}
    controls = {
        "DELAYED_CONSEQUENCE_CREDIT|NO_SIGNAL_OR_ZERO_REWARD_CONTROL": {str(seed): _result(final=0.0) for seed in SEEDS},
        "DELAYED_CONSEQUENCE_CREDIT|SHUFFLED_CREDIT_CONTROL": {str(seed): _result(final=0.12) for seed in SEEDS},
    }
    result = _compile(positive, controls)
    assert result["gates"]["CONTROL:DELAYED_CONSEQUENCE_CREDIT|SHUFFLED_CREDIT_CONTROL"] is False


def test_fixture_07_shuffled_relation_above_tolerance_passes():
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result(final=0.12) for seed in SEEDS}
    controls = {
        "DELAYED_CONSEQUENCE_CREDIT|NO_SIGNAL_OR_ZERO_REWARD_CONTROL": {str(seed): _result(final=0.0) for seed in SEEDS},
        "DELAYED_CONSEQUENCE_CREDIT|SHUFFLED_CREDIT_CONTROL": {str(seed): _result(final=0.1199999990) for seed in SEEDS},
    }
    result = _compile(positive, controls)
    assert result["gates"]["CONTROL:DELAYED_CONSEQUENCE_CREDIT|SHUFFLED_CREDIT_CONTROL"] is True


def test_fixture_08_missing_seed_is_evidence_insufficient_never_pass():
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result() for seed in SEEDS[:-1]}
    result = _compile(positive, _passing_controls())
    assert result["classification"] == CLASSIFICATION_EVIDENCE_INSUFFICIENT
    assert result["classification"] != CLASSIFICATION_PASS
    assert result["gates"]["POSITIVE:DELAYED_CONSEQUENCE_CREDIT"] is False


def test_fixture_09_valid_but_failed_criterion_is_scientific_fail():
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result(final=0.0) for seed in SEEDS}
    result = _compile(positive, _passing_controls())
    assert result["classification"] == CLASSIFICATION_SCIENTIFIC_FAIL
    assert result["contract_violations"] == []


def test_fixture_10_corrupted_provenance_is_contract_mismatch():
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result(restart_ok=False) for seed in SEEDS}
    result = _compile(positive, _passing_controls())
    assert result["classification"] == CLASSIFICATION_CONTRACT_MISMATCH


def test_fixture_11_firewall_violation_fails_closed():
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result(firewall_ok=False) for seed in SEEDS}
    result = _compile(positive, _passing_controls())
    assert result["classification"] == CLASSIFICATION_CONTRACT_MISMATCH


def test_fixture_12_nan_inf_or_zero_denominator_never_passes():
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result() for seed in SEEDS}
    positive["DELAYED_CONSEQUENCE_CREDIT|1705"] = _result()
    positive["DELAYED_CONSEQUENCE_CREDIT|1705"]["checkpoints"]["FINAL"]["mean_complete_sample_arithmetic_return"] = float("nan")
    result = _compile(positive, _passing_controls())
    assert result["classification"] == CLASSIFICATION_EVIDENCE_INSUFFICIENT
    zero_gap = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result(initial=0.18, final=0.18, oracle=0.18) for seed in SEEDS}
    result_zero = _compile(zero_gap, _passing_controls())
    assert result_zero["classification"] == CLASSIFICATION_EVIDENCE_INSUFFICIENT


def test_threshold_boundary_exact_half_passes_and_just_below_fails():
    exact = evaluate_seed_predicate_v1(_result(initial=0.0, final=0.09, oracle=0.18))
    below = evaluate_seed_predicate_v1(_result(initial=0.0, final=0.0899999, oracle=0.18))
    above = evaluate_seed_predicate_v1(_result(initial=0.0, final=0.0900001, oracle=0.18))
    assert exact["gap_reduction"] == pytest.approx(0.5, abs=1e-12)
    assert exact["predicate_pass"] is True
    assert below["gap_reduction"] < 0.5 and below["predicate_pass"] is False
    assert above["gap_reduction"] > 0.5 and above["predicate_pass"] is True


def test_post_score_must_strictly_exceed_initial_beyond_tolerance():
    equal = evaluate_seed_predicate_v1(_result(initial=0.01, final=0.01, oracle=0.18))
    tiny = evaluate_seed_predicate_v1(_result(initial=0.01, final=0.01 + 1e-12, oracle=0.18))
    strict = evaluate_seed_predicate_v1(_result(initial=0.01, final=0.0100001, oracle=0.18))
    assert equal["checks"]["post_exceeds_initial"] is False
    assert tiny["checks"]["post_exceeds_initial"] is False
    assert strict["checks"]["post_exceeds_initial"] is True


def test_shuffled_relation_exactly_one_tolerance_is_not_strictly_greater():
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result(final=0.12) for seed in SEEDS}
    controls = {
        "DELAYED_CONSEQUENCE_CREDIT|NO_SIGNAL_OR_ZERO_REWARD_CONTROL": {str(seed): _result(final=0.0) for seed in SEEDS},
        "DELAYED_CONSEQUENCE_CREDIT|SHUFFLED_CREDIT_CONTROL": {
            str(seed): _result(final=0.12 - 0.18 * 1e-9) for seed in SEEDS
        },
    }
    result = _compile(positive, controls)
    gates = result["gates"]["CONTROL:DELAYED_CONSEQUENCE_CREDIT|SHUFFLED_CREDIT_CONTROL"]
    margins = [
        item["margin"] for item in result["details"]["CONTROL:DELAYED_CONSEQUENCE_CREDIT|SHUFFLED_CREDIT_CONTROL"]["relation_records"].values()
    ]
    assert all(abs(margin - 1e-9) < 1e-12 for margin in margins)
    assert gates is False


def test_aba_predicate_known_answers_and_retention_boundary():
    good = _aba_result()
    predicate = evaluate_seed_predicate_v1(good)
    assert predicate["predicate_pass"] is True
    weak_retention = _aba_result()
    weak_retention["checkpoints"]["POST_B"]["contexts"]["A"]["mean_complete_sample_arithmetic_return"] = 1e-10
    assert evaluate_seed_predicate_v1(weak_retention)["checks"]["RETURN_A_PRE_A2_beats_baseline"] is False
    assert evaluate_seed_predicate_v1(weak_retention)["predicate_pass"] is False
    no_eligible_a1 = _aba_result()
    no_eligible_a1["retention_evidence"]["a1_durable_sequence_count_after_b"] = 0
    no_eligible_a1["retention_evidence"]["a1_eligible_for_generic_replay_after_b"] = False
    predicate_missing = evaluate_seed_predicate_v1(no_eligible_a1)
    assert predicate_missing["checks"]["A1_facts_durable_after_B"] is False
    assert predicate_missing["checks"]["A1_eligible_for_generic_replay_after_B"] is False


def test_manifest_firewall_violation_fails_closed():
    manifest = _manifest()
    manifest["firewall"]["FINAL_opened"] = True
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result() for seed in SEEDS}
    result = _compile(positive, _passing_controls(), manifest=manifest)
    assert result["classification"] == CLASSIFICATION_CONTRACT_MISMATCH


def test_integrity_attack_failure_is_contract_mismatch():
    positive = {f"DELAYED_CONSEQUENCE_CREDIT|{seed}": _result() for seed in SEEDS}
    integrity = {"attacks": {"attack-1": {"rejected": False}}}
    result = _compile(positive, _passing_controls(), integrity=integrity)
    assert result["classification"] == CLASSIFICATION_CONTRACT_MISMATCH
