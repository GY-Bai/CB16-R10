"""CB16 R11 S1 deterministic gate compiler.

The compiler consumes already-produced result records.  It never re-runs
training and never fabricates scientific expectations from the production
runners.  All comparisons and aggregation rules use explicit named helpers so
the below/equal/above boundary semantics can be unit-tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

S1_GATE_COMPILER_ID_V1 = "CB16_R11_POST_CC_S1_DETERMINISTIC_GATE_COMPILER_V1"

CLASSIFICATION_PASS = "PASS"
CLASSIFICATION_SCIENTIFIC_FAIL = "SCIENTIFIC_FAIL"
CLASSIFICATION_CONTRACT_MISMATCH = "CONTRACT_MISMATCH"
CLASSIFICATION_EXECUTION_BLOCKED = "EXECUTION_BLOCKED"
CLASSIFICATION_HARDWARE_LIMIT = "HARDWARE_LIMIT"
CLASSIFICATION_EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"

ALLOWED_CLASSIFICATIONS = (
    CLASSIFICATION_PASS,
    CLASSIFICATION_SCIENTIFIC_FAIL,
    CLASSIFICATION_CONTRACT_MISMATCH,
    CLASSIFICATION_EXECUTION_BLOCKED,
    CLASSIFICATION_HARDWARE_LIMIT,
    CLASSIFICATION_EVIDENCE_INSUFFICIENT,
)

GAP_REDUCTION_THRESHOLD = 0.50
STRICT_TOLERANCE = 1e-9
MINIMUM_POSITIVE_SEEDS_PASSING = 4
MAXIMUM_FALSE_POSITIVE_CONTROL_SEEDS = 1


class GateCompilerError(RuntimeError):
    pass


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _task_gap_v1(result: Mapping[str, Any], *, initial_id: str, final_id: str) -> dict[str, Any]:
    checkpoints = result["checkpoints"]
    initial_score = float(checkpoints[initial_id]["mean_complete_sample_arithmetic_return"])
    final_score = float(checkpoints[final_id]["mean_complete_sample_arithmetic_return"])
    oracle_score = float(result["oracle"]["mean_oracle_return"])
    initial_gap = oracle_score - initial_score
    final_gap = oracle_score - final_score
    if not all(_finite(item) for item in (initial_score, final_score, oracle_score, initial_gap, final_gap)):
        return {
            "initial_score": initial_score,
            "final_score": final_score,
            "oracle_score": oracle_score,
            "initial_gap": initial_gap,
            "final_gap": final_gap,
            "gap_reduction": float("nan"),
            "valid": False,
        }
    initial_gap_positive = initial_gap > STRICT_TOLERANCE
    if initial_gap_positive:
        gap_reduction = (initial_gap - final_gap) / initial_gap
    else:
        gap_reduction = float("nan")
    return {
        "initial_score": initial_score,
        "final_score": final_score,
        "oracle_score": oracle_score,
        "initial_gap": initial_gap,
        "final_gap": final_gap,
        "gap_reduction": gap_reduction,
        "initial_gap_positive": bool(initial_gap_positive),
        "post_exceeds_initial": bool(final_score - initial_score > STRICT_TOLERANCE),
        "meets_gap_threshold": bool(initial_gap_positive and gap_reduction >= GAP_REDUCTION_THRESHOLD),
        "valid": True,
    }


def _context_probability_rise_v1(result: Mapping[str, Any], *, initial_id: str, final_id: str) -> dict[str, bool]:
    initial_contexts = result["checkpoints"][initial_id]["contexts"]
    final_contexts = result["checkpoints"][final_id]["contexts"]
    rises: dict[str, bool] = {}
    for context_id, record in final_contexts.items():
        final_probability = float(record["oracle_direction_probability"])
        initial_probability = float(initial_contexts[context_id]["oracle_direction_probability"])
        rises[context_id] = bool(final_probability - initial_probability > STRICT_TOLERANCE)
    return rises


def _context_family_mass_rise_v1(result: Mapping[str, Any], *, initial_id: str, final_id: str) -> dict[str, bool]:
    initial_contexts = result["checkpoints"][initial_id]["contexts"]
    final_contexts = result["checkpoints"][final_id]["contexts"]
    rises: dict[str, bool] = {}
    for context_id, record in final_contexts.items():
        final_mass = float(record["higher_ev_family_mass"])
        initial_mass = float(initial_contexts[context_id]["higher_ev_family_mass"])
        rises[context_id] = bool(final_mass - initial_mass > STRICT_TOLERANCE)
    return rises


def _aba_predicate_v1(result: Mapping[str, Any]) -> dict[str, Any]:
    checkpoints = result["checkpoints"]
    oracle_contexts = result["oracle"]["contexts"]
    oracle_a = float(oracle_contexts["A"]["oracle_return"])
    oracle_b = float(oracle_contexts["B"]["oracle_return"])
    baseline_a = float(checkpoints["INITIAL"]["contexts"]["A"]["mean_complete_sample_arithmetic_return"])
    post_a1_a = float(checkpoints["POST_A1"]["contexts"]["A"]["mean_complete_sample_arithmetic_return"])
    post_a1_b = float(checkpoints["POST_A1"]["contexts"]["B"]["mean_complete_sample_arithmetic_return"])
    post_b_a = float(checkpoints["POST_B"]["contexts"]["A"]["mean_complete_sample_arithmetic_return"])
    post_b_b = float(checkpoints["POST_B"]["contexts"]["B"]["mean_complete_sample_arithmetic_return"])
    post_a2_a = float(checkpoints["POST_A2"]["contexts"]["A"]["mean_complete_sample_arithmetic_return"])

    a1_initial_gap = oracle_a - baseline_a
    a1_final_gap = oracle_a - post_a1_a
    a1_reduction = (a1_initial_gap - a1_final_gap) / a1_initial_gap if a1_initial_gap > STRICT_TOLERANCE else float("nan")
    b_initial_gap = oracle_b - post_a1_b
    b_final_gap = oracle_b - post_b_b
    b_reduction = (b_initial_gap - b_final_gap) / b_initial_gap if b_initial_gap > STRICT_TOLERANCE else float("nan")
    a2_initial_gap = oracle_a - baseline_a
    a2_final_gap = oracle_a - post_a2_a
    a2_reduction = (a2_initial_gap - a2_final_gap) / a2_initial_gap if a2_initial_gap > STRICT_TOLERANCE else float("nan")
    retention_rise = post_b_a - baseline_a
    a1_unit_records = [
        unit for unit in result["unit_evidence"] if int(unit["unit_index"]) < 4096 // int(result["unit_size"])
    ]
    a1_sequence_ids = {sequence_id for unit in a1_unit_records for sequence_id in unit["sequence_ids"]}
    b_phase_units = [
        unit
        for unit in result["unit_evidence"]
        if int(unit["unit_index"]) >= 4096 // int(result["unit_size"])
        and int(unit["unit_index"]) < 8192 // int(result["unit_size"])
    ]
    a1_samples_used_in_b = any(
        len(a1_sequence_ids.intersection(set(unit["sequence_ids"]))) > 0 for unit in b_phase_units
    )
    checks = {
        "A1_gap_reduction_ge_threshold": bool(a1_initial_gap > STRICT_TOLERANCE and a1_reduction >= GAP_REDUCTION_THRESHOLD),
        "B_gap_reduction_ge_threshold": bool(b_initial_gap > STRICT_TOLERANCE and b_reduction >= GAP_REDUCTION_THRESHOLD),
        "RETURN_A_PRE_A2_beats_baseline": bool(retention_rise > STRICT_TOLERANCE),
        "A2_gap_reduction_ge_threshold": bool(a2_initial_gap > STRICT_TOLERANCE and a2_reduction >= GAP_REDUCTION_THRESHOLD),
        "A1_samples_eligible_after_B": bool(a1_samples_used_in_b),
        "no_handcrafted_regime_activation": bool(result["handcrafted_regime_activation"] is False),
    }
    return {
        "checks": checks,
        "a1_reduction": a1_reduction,
        "b_reduction": b_reduction,
        "a2_reduction": a2_reduction,
        "retention_rise": retention_rise,
        "predicate_pass": all(checks.values()),
        "initial_gap_positive": bool(a1_initial_gap > STRICT_TOLERANCE and b_initial_gap > STRICT_TOLERANCE),
    }


def evaluate_seed_predicate_v1(result: Mapping[str, Any]) -> dict[str, Any]:
    """Frozen per-seed positive predicate; used for positives and matched controls."""
    task_id = str(result["task_id"])
    if task_id == "A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION":
        aba = _aba_predicate_v1(result)
        return {
            **aba,
            "initial_gap": float(aba.get("a1_reduction", float("nan"))),
            "gap_reduction": float(aba.get("a1_reduction", float("nan"))),
            "evidence_insufficient": not aba["initial_gap_positive"],
            "finite": True,
        }
    gap = _task_gap_v1(result, initial_id="INITIAL", final_id="FINAL")
    if not gap["valid"]:
        return {
            "checks": {"finite_scores": False},
            "initial_gap": gap["initial_gap"],
            "gap_reduction": float("nan"),
            "predicate_pass": False,
            "evidence_insufficient": True,
            "initial_gap_positive": False,
            "post_exceeds_initial": False,
            "meets_gap_threshold": False,
        }
    checks: dict[str, bool] = {
        "initial_gap_gt_tolerance": bool(gap["initial_gap_positive"]),
        "gap_reduction_ge_threshold": bool(gap["meets_gap_threshold"]),
        "post_exceeds_initial": bool(gap["post_exceeds_initial"]),
    }
    if task_id == "ACCOUNT_DEPENDENT_ACTION":
        rises = _context_probability_rise_v1(result, initial_id="INITIAL", final_id="FINAL")
        checks["each_context_oracle_direction_probability_rises"] = all(rises.values())
    elif task_id == "DELAYED_CONSEQUENCE_CREDIT":
        rises = _context_probability_rise_v1(result, initial_id="INITIAL", final_id="FINAL")
        checks["each_context_early_direction_probability_rises"] = all(rises.values())
    elif task_id == "HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION":
        rises = _context_family_mass_rise_v1(result, initial_id="INITIAL", final_id="FINAL")
        checks["higher_ev_family_mass_rises"] = all(rises.values())
        checks["no_survivor_filtering_objective"] = bool(
            result.get("objective_orientation") == "COMPLETE_SAMPLE_ARITHMETIC_EQUITY_DELTA"
        )
    elif task_id == "OFF_POLICY_VTRACE_CORRECTION":
        ratios = result["ratio_diagnostics"]
        checks["ratios_above_one_present"] = bool(int(ratios["max_ratio_above_one_count"]) >= 1)
        checks["ratios_below_one_present"] = bool(int(ratios["max_ratio_below_one_count"]) >= 1)
    else:
        raise GateCompilerError(f"UNKNOWN_TASK_ID:{task_id}")
    return {
        "checks": checks,
        "initial_gap": float(gap["initial_gap"]),
        "final_gap": float(gap["final_gap"]),
        "gap_reduction": float(gap["gap_reduction"]),
        "post_exceeds_initial": bool(gap["post_exceeds_initial"]),
        "initial_gap_positive": bool(gap["initial_gap_positive"]),
        "evidence_insufficient": not bool(gap["initial_gap_positive"]),
        "predicate_pass": bool(all(checks.values())),
        "finite": True,
    }


def _seed_key(task_id: str, seed: int) -> str:
    return f"{task_id}|{int(seed)}"


def _result_is_finite(result: Mapping[str, Any]) -> bool:
    try:
        checkpoints = result["checkpoints"]
        for record in checkpoints.values():
            if not _finite(record["mean_complete_sample_arithmetic_return"]):
                return False
            for context in record["contexts"].values():
                if not _finite(context["mean_complete_sample_arithmetic_return"]):
                    return False
        oracle = result["oracle"]
        if not _finite(oracle["mean_oracle_return"]):
            return False
        return True
    except (KeyError, TypeError):
        return False


def _result_firewall_ok(result: Mapping[str, Any]) -> bool:
    firewall = result.get("firewall", {})
    return (
        firewall.get("FINAL_opened") is False
        and firewall.get("fresh_data_used") is False
        and firewall.get("historical_market_corpus_accessed") is False
        and firewall.get("economic_evidence_claimed") is False
        and firewall.get("transfer_evidence_claimed") is False
    )


def _result_provenance_ok(result: Mapping[str, Any], manifest_sha256: str) -> bool:
    if result.get("manifest_sha256") != manifest_sha256:
        return False
    durability = result.get("durability", {})
    if durability.get("restart_sentinel_passed") is not True:
        return False
    if durability.get("behavior_checkpoint_untouched_every_unit") is not True:
        return False
    for unit in result.get("unit_evidence", []):
        if unit.get("restart_sentinel", {}).get("restart_verified") is not True:
            return False
        switch = unit.get("generation_switch_receipt")
        if result["task_id"] != "OFF_POLICY_VTRACE_CORRECTION" and result.get("control_id") is None:
            if not switch or switch.get("account_fully_preserved") is not True:
                return False
            if switch.get("parent_checkpoint_mutated_in_place") is not False:
                return False
    return True


def compile_s1_gates_v1(
    *,
    manifest: Mapping[str, Any],
    positive_results: Mapping[str, Mapping[str, Any]],
    control_results: Mapping[str, Mapping[str, Any]],
    integrity: Mapping[str, Any],
    audits: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile frozen S1 gates from already-produced result records."""
    manifest_sha256 = str(manifest["manifest_sha256"])
    seeds = [int(seed) for seed in manifest["seeds"]]
    task_entries = list(manifest["tasks"])
    gates: dict[str, bool] = {}
    details: dict[str, Any] = {}
    contract_violations: list[str] = []
    evidence_insufficient: list[str] = []
    scientific_failures: list[str] = []

    if not _result_firewall_ok({"firewall": manifest.get("firewall", {})}):
        contract_violations.append("MANIFEST_FIREWALL_VIOLATION")

    for entry in task_entries:
        task_id = str(entry["task_id"])
        seed_predicates: dict[int, Any] = {}
        missing_seeds: list[int] = []
        nonfinite_seeds: list[int] = []
        for seed in seeds:
            key = _seed_key(task_id, seed)
            result = positive_results.get(key)
            if result is None:
                missing_seeds.append(seed)
                continue
            if not _result_is_finite(result):
                nonfinite_seeds.append(seed)
                continue
            if not _result_firewall_ok(result):
                contract_violations.append(f"FIREWALL:{key}")
            if not _result_provenance_ok(result, manifest_sha256):
                contract_violations.append(f"PROVENANCE:{key}")
            seed_predicates[seed] = evaluate_seed_predicate_v1(result)
        passing = [seed for seed, predicate in seed_predicates.items() if predicate["predicate_pass"] is True]
        insufficient = [seed for seed, predicate in seed_predicates.items() if predicate.get("evidence_insufficient") is True]
        positive_gate = (
            not missing_seeds
            and not nonfinite_seeds
            and len(passing) >= int(manifest["minimum_positive_seeds_passing"])
        )
        gates[f"POSITIVE:{task_id}"] = bool(positive_gate)
        details[f"POSITIVE:{task_id}"] = {
            "passing_seeds": sorted(passing),
            "passing_count": len(passing),
            "insufficient_seeds": sorted(insufficient),
            "missing_seeds": sorted(missing_seeds),
            "nonfinite_seeds": sorted(nonfinite_seeds),
            "seed_predicates": {str(seed): predicate for seed, predicate in seed_predicates.items()},
        }
        if missing_seeds or nonfinite_seeds:
            evidence_insufficient.append(f"POSITIVE:{task_id}:MISSING_OR_NONFINITE")
        elif not positive_gate:
            if insufficient:
                evidence_insufficient.append(f"POSITIVE:{task_id}:INSUFFICIENT_SEEDS")
            else:
                scientific_failures.append(f"POSITIVE:{task_id}")

        for control_id in entry.get("declared_controls", []):
            control_key = f"{task_id}|{control_id}"
            control_entry = control_results.get(control_key, {})
            if control_id == "SHUFFLED_CREDIT_CONTROL":
                relation_passes = 0
                relation_records: dict[str, Any] = {}
                relation_missing = False
                for seed in seeds:
                    positive = positive_results.get(_seed_key(task_id, seed))
                    control = control_entry.get(str(seed))
                    if positive is None or control is None:
                        relation_missing = True
                        continue
                    positive_gap = float(evaluate_seed_predicate_v1(positive).get("gap_reduction", float("nan")))
                    control_gap = float(evaluate_seed_predicate_v1(control).get("gap_reduction", float("nan")))
                    margin = positive_gap - control_gap
                    relation_ok = bool(_finite(margin) and margin > STRICT_TOLERANCE)
                    relation_records[str(seed)] = {"positive_gap": positive_gap, "control_gap": control_gap, "margin": margin, "relation_pass": relation_ok}
                    if relation_ok:
                        relation_passes += 1
                gate = not relation_missing and relation_passes >= int(manifest["minimum_positive_seeds_passing"])
                gates[f"CONTROL:{control_key}"] = bool(gate)
                details[f"CONTROL:{control_key}"] = {
                    "relation_passes": relation_passes,
                    "relation_records": relation_records,
                    "relation_missing": relation_missing,
                }
                if relation_missing:
                    evidence_insufficient.append(f"CONTROL:{control_key}:MISSING")
                elif not gate:
                    scientific_failures.append(f"CONTROL:{control_key}")
            elif control_id in ("OBJECTIVE_FIREWALL", "FABRICATED_LOG_MU_REJECTION"):
                audit_key = f"{control_key}"
                audit = audits.get(audit_key, {})
                gate = bool(audit.get("all_checks_pass") is True)
                gates[f"CONTROL:{control_key}"] = gate
                details[f"CONTROL:{control_key}"] = dict(audit)
                if not gate:
                    contract_violations.append(f"CONTROL:{control_key}")
            else:
                false_positive_seeds: list[int] = []
                control_missing: list[int] = []
                for seed in seeds:
                    result = control_entry.get(str(seed))
                    if result is None:
                        control_missing.append(seed)
                        continue
                    if not _result_is_finite(result):
                        control_missing.append(seed)
                        continue
                    predicate = evaluate_seed_predicate_v1(result)
                    if predicate["predicate_pass"] is True:
                        false_positive_seeds.append(seed)
                gate = (
                    not control_missing
                    and len(false_positive_seeds) <= int(manifest["maximum_false_positive_control_seeds"])
                )
                gates[f"CONTROL:{control_key}"] = bool(gate)
                details[f"CONTROL:{control_key}"] = {
                    "false_positive_seeds": sorted(false_positive_seeds),
                    "false_positive_count": len(false_positive_seeds),
                    "missing_seeds": sorted(control_missing),
                }
                if control_missing:
                    evidence_insufficient.append(f"CONTROL:{control_key}:MISSING")
                elif not gate:
                    scientific_failures.append(f"CONTROL:{control_key}")

    attacks = integrity.get("attacks", {})
    integrity_ok = bool(attacks) and all(item.get("rejected") is True for item in attacks.values())
    gates["INTEGRITY_ATTACK_MATRIX"] = integrity_ok
    details["INTEGRITY_ATTACK_MATRIX"] = dict(integrity)
    if not integrity_ok:
        contract_violations.append("INTEGRITY_ATTACK_MATRIX")
    for audit_key in ("FABRICATED_LOG_MU_REJECTION", "OBJECTIVE_FIREWALL"):
        audit = audits.get(audit_key, {})
        gates[f"AUDIT:{audit_key}"] = bool(audit.get("all_checks_pass") is True)
        details[f"AUDIT:{audit_key}"] = dict(audit)
        if audit.get("all_checks_pass") is not True:
            contract_violations.append(f"AUDIT:{audit_key}")

    if contract_violations:
        classification = CLASSIFICATION_CONTRACT_MISMATCH
    elif evidence_insufficient:
        classification = CLASSIFICATION_EVIDENCE_INSUFFICIENT
    elif scientific_failures:
        classification = CLASSIFICATION_SCIENTIFIC_FAIL
    elif all(gates.values()):
        classification = CLASSIFICATION_PASS
    else:
        classification = CLASSIFICATION_SCIENTIFIC_FAIL
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise GateCompilerError("CLASSIFICATION_OUTSIDE_FROZEN_TAXONOMY")
    return {
        "schema": "CB16_R11_POST_CC_S1_RESULT_V1",
        "compiler_id": S1_GATE_COMPILER_ID_V1,
        "manifest_sha256": manifest_sha256,
        "classification": classification,
        "status": classification,
        "gates": gates,
        "details": details,
        "contract_violations": sorted(contract_violations),
        "evidence_insufficient": sorted(evidence_insufficient),
        "scientific_failures": sorted(scientific_failures),
        "classification_taxonomy": list(ALLOWED_CLASSIFICATIONS),
    }
