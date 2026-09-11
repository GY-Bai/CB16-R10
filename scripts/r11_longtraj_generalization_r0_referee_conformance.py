#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_GENERALIZATION_R0_REFEREE_CONFORMANCE_SPEC_V1.json"

G1_FAIL = "SCIENTIFIC_FAIL__FROZEN_SHADOW_CHALLENGER_DID_NOT_GENERALIZE_VS_G0_CHAMPION"
G2_FAIL = "SCIENTIFIC_FAIL__CHALLENGER_IMPROVEMENT_DID_NOT_BEAT_PRIOR_NO_STATE_CLIMATOLOGY"
G3_FAIL = "SCIENTIFIC_FAIL__CHALLENGER_ADVANTAGE_NOT_SPECIFIC_TO_TRUE_STATE_TARGET_CORRESPONDENCE"
ALL_PASS = "GENERALIZATION_TARGET_CORRESPONDENCE_QUALIFIED_FOR_SEPARATE_NEXT_GATE__NO_PROMOTION__NO_MARKET_VERDICT"


class RefereeReject(RuntimeError):
    pass


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RefereeReject(code)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spec() -> Mapping[str, Any]:
    s = json.loads(SPEC.read_text(encoding="utf-8"))
    require(s["status"] == "FROZEN_BEFORE_CONFORMANCE_EXECUTION", "SPEC_NOT_FROZEN")
    require(
        s["classification"]
        == "REFEREE_ONLY_CONFORMANCE_AND_FAULT_INJECTION__NO_NEW_SCIENCE_EVIDENCE__NO_MODEL_EXECUTION",
        "SPEC_CLASSIFICATION_DRIFT",
    )
    return s


def authority_check(x: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    a = spec["authority_canaries"]
    require(x["schema"] == "CB16_R11_LONGTRAJ_GENERALIZATION_R0_EVALUATION_RESULT_V1", "RESULT_SCHEMA")
    require(x.get("executor_revision") == "R2_EXACT_CAMPAIGN_REPRODUCTION", "EXECUTOR_REVISION")
    require(x["cohort"]["dependence_groups"] == int(a["required_new_validation_groups"]), "COHORT_GROUP_COUNT")
    require(x["cohort"]["cohort_resampled"] is False, "COHORT_RESAMPLED")
    require(x["teacher"]["prior_train_evidence_hash"] == a["expected_prior_train_evidence_hash"], "AUTHORITY_REJECT:TRAIN_EVIDENCE_HASH")
    require(x["teacher"]["reproduced_consumed_validation_evidence_hash"] == a["expected_consumed_validation_evidence_hash"], "AUTHORITY_REJECT:CONSUMED_VALIDATION_EVIDENCE_HASH")
    require(x["models"]["champion_policy_hash"] == a["expected_champion_policy_hash"], "AUTHORITY_REJECT:CHAMPION_POLICY_HASH")
    require(x["models"]["challenger_policy_hash"] == a["expected_challenger_policy_hash"], "AUTHORITY_REJECT:CHALLENGER_POLICY_HASH")
    require(x["models"]["challenger_checkpoint_sha256"] == a["expected_challenger_checkpoint_sha256"], "AUTHORITY_REJECT:CHALLENGER_CHECKPOINT_HASH")
    require(x["teacher"]["validation_summary"]["admitted_dependence_groups"] == 48, "VALIDATION_SUPPORT_GROUP_COUNT")
    require(x["teacher"]["validation_summary"]["rejected"] == 0, "VALIDATION_SUPPORT_REJECTION")
    require(x["teacher"]["support_thresholds_relaxed"] is False, "SUPPORT_RELAXED")
    require(x["teacher"]["consumed_validation_scored_in_generalization"] is False, "CONSUMED_VALIDATION_REUSED")
    require(x["teacher"]["rebound_train_outputs_used_for_G2"] is False, "REBOUND_TRAIN_USED_FOR_G2")
    require(x["models"]["training_or_tuning_on_generalization_cohort"] is False, "GENERALIZATION_TUNING")
    shifts = x["G3"]["nonidentity_cyclic_shift_deltas"]
    require(len(shifts) == int(a["required_nonidentity_shifts"]), "GATE_GEOMETRY_REJECT:NONIDENTITY_SHIFT_COUNT")
    f = x["firewalls"]
    require(f["final_holdout_touched"] is False, "FIREWALL_REJECT:FINAL_HOLDOUT")
    require(f["fresh_market_data_downloaded"] is False, "FIREWALL_REJECT:FRESH_DATA")
    require(f["canonical_promotion_authorized"] is False, "FIREWALL_REJECT:PROMOTION")
    require(f["canonical_generation_advance_authorized"] is False, "FIREWALL_REJECT:GENERATION_ADVANCE")
    require(f["scientific_market_verdict"] is None, "FIREWALL_REJECT:MARKET_VERDICT")


def recompute(x: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    authority_check(x, spec)
    g1_point = float(x["G1"]["point"])
    g1_hi = float(x["G1"]["ci_high"])
    g2_point = float(x["G2"]["point"])
    g2_hi = float(x["G2"]["ci_high"])
    identity = float(x["G3"]["identity_delta_challenger_minus_champion"])
    shifts = [float(v) for v in x["G3"]["nonidentity_cyclic_shift_deltas"]]
    g1 = bool(g1_point < 0.0 and g1_hi < 0.0)
    g2 = bool(g2_point < 0.0 and g2_hi < 0.0)
    count_le = sum(v <= identity for v in shifts)
    exact_p = (1 + count_le) / 48.0
    g3 = bool(count_le == 0 and identity < min(shifts) and abs(exact_p - (1.0 / 48.0)) < 1e-15)
    if not g1:
        classification, status = G1_FAIL, "SCIENTIFIC_FAIL"
    elif not g2:
        classification, status = G2_FAIL, "SCIENTIFIC_FAIL"
    elif not g3:
        classification, status = G3_FAIL, "SCIENTIFIC_FAIL"
    else:
        classification, status = ALL_PASS, "PASS"
    return {
        "status": status,
        "classification": classification,
        "G1_pass": g1,
        "G2_pass": g2,
        "G3_pass": g3,
        "G3_count_le": int(count_le),
        "G3_exact_p": float(exact_p),
        "G3_best_nonidentity": float(min(shifts)),
    }


def expect_classification(base: Mapping[str, Any], spec: Mapping[str, Any], mutator, expected: str) -> dict[str, Any]:
    y = copy.deepcopy(base)
    mutator(y)
    r = recompute(y, spec)
    require(r["classification"] == expected, f"FAULT_CASE_CLASSIFICATION_MISMATCH:{r['classification']}!={expected}")
    return r


def expect_reject(base: Mapping[str, Any], spec: Mapping[str, Any], mutator, prefix: str) -> str:
    y = copy.deepcopy(base)
    mutator(y)
    try:
        recompute(y, spec)
    except RefereeReject as e:
        msg = str(e)
        require(msg.startswith(prefix), f"FAULT_CASE_WRONG_REJECTION:{msg}")
        return msg
    raise RefereeReject(f"FAULT_CASE_NOT_REJECTED:{prefix}")


def run_fault_matrix(base: Mapping[str, Any], spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    baseline = recompute(base, spec)
    require(baseline["status"] == base["status"], "BASELINE_STATUS_MISMATCH")
    require(baseline["classification"] == base["classification"], "BASELINE_CLASSIFICATION_MISMATCH")
    require(baseline["G1_pass"] == bool(base["G1"]["pass"]), "BASELINE_G1_MISMATCH")
    require(baseline["G2_pass"] == bool(base["G2"]["pass"]), "BASELINE_G2_MISMATCH")
    require(baseline["G3_pass"] == bool(base["G3"]["pass"]), "BASELINE_G3_MISMATCH")
    require(baseline["G3_count_le"] == int(base["G3"]["nonidentity_shifts_with_delta_le_identity"]), "BASELINE_G3_COUNT_MISMATCH")
    require(abs(baseline["G3_exact_p"] - float(base["G3"]["exact_one_sided_rank_p"])) < 1e-15, "BASELINE_G3_P_MISMATCH")
    out.append({"case": "BASELINE_FROZEN_R2", "pass": True, "observed": baseline})

    r = expect_classification(base, spec, lambda y: y["G1"].update({"point": 0.0}), G1_FAIL)
    out.append({"case": "G1_POINT_NONNEGATIVE", "pass": True, "observed": r["classification"]})
    r = expect_classification(base, spec, lambda y: y["G2"].update({"ci_high": 0.0}), G2_FAIL)
    out.append({"case": "G2_CI_HIGH_NONNEGATIVE", "pass": True, "observed": r["classification"]})

    def make_g3_pass(y):
        identity = float(y["G3"]["identity_delta_challenger_minus_champion"])
        y["G3"]["nonidentity_cyclic_shift_deltas"] = [identity + 0.01 + i * 1e-6 for i in range(47)]

    r = expect_classification(base, spec, make_g3_pass, ALL_PASS)
    out.append({"case": "G3_IDENTITY_STRICT_BEST", "pass": True, "observed": r["classification"]})

    def make_g3_tie(y):
        identity = float(y["G3"]["identity_delta_challenger_minus_champion"])
        y["G3"]["nonidentity_cyclic_shift_deltas"] = [identity] + [identity + 0.01 + i * 1e-6 for i in range(46)]

    r = expect_classification(base, spec, make_g3_tie, G3_FAIL)
    out.append({"case": "G3_TIE_WITH_NONIDENTITY", "pass": True, "observed": r["classification"]})
    msg = expect_reject(base, spec, lambda y: y["teacher"].update({"prior_train_evidence_hash": "0" * 64}), "AUTHORITY_REJECT:TRAIN_EVIDENCE_HASH")
    out.append({"case": "TRAIN_EVIDENCE_HASH_DRIFT", "pass": True, "observed": msg})
    msg = expect_reject(base, spec, lambda y: y["teacher"].update({"reproduced_consumed_validation_evidence_hash": "0" * 64}), "AUTHORITY_REJECT:CONSUMED_VALIDATION_EVIDENCE_HASH")
    out.append({"case": "CONSUMED_VALIDATION_HASH_DRIFT", "pass": True, "observed": msg})
    msg = expect_reject(base, spec, lambda y: y["models"].update({"champion_policy_hash": "0" * 64}), "AUTHORITY_REJECT:CHAMPION_POLICY_HASH")
    out.append({"case": "CHAMPION_HASH_DRIFT", "pass": True, "observed": msg})
    msg = expect_reject(base, spec, lambda y: y["firewalls"].update({"final_holdout_touched": True}), "FIREWALL_REJECT:FINAL_HOLDOUT")
    out.append({"case": "FINAL_HOLDOUT_TRUE", "pass": True, "observed": msg})
    msg = expect_reject(base, spec, lambda y: y["firewalls"].update({"canonical_promotion_authorized": True}), "FIREWALL_REJECT:PROMOTION")
    out.append({"case": "PROMOTION_AUTHORIZED_TRUE", "pass": True, "observed": msg})
    msg = expect_reject(base, spec, lambda y: y["G3"].update({"nonidentity_cyclic_shift_deltas": list(y["G3"]["nonidentity_cyclic_shift_deltas"][:-1])}), "GATE_GEOMETRY_REJECT:NONIDENTITY_SHIFT_COUNT")
    out.append({"case": "NONIDENTITY_SHIFT_COUNT_46", "pass": True, "observed": msg})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    spec = load_spec()
    result_path = Path(args.result).resolve()
    require(sha256_file(result_path) == spec["source_r2"]["generalization_result_sha256"], "SOURCE_RESULT_HASH_DRIFT")
    base = json.loads(result_path.read_text(encoding="utf-8"))
    cases = run_fault_matrix(base, spec)
    require(len(cases) == len(spec["fault_injection_matrix"]), "FAULT_MATRIX_CASE_COUNT_DRIFT")
    require(all(bool(c["pass"]) for c in cases), "FAULT_MATRIX_FAILURE")
    baseline = recompute(base, spec)
    output = {
        "schema": "CB16_R11_LONGTRAJ_GENERALIZATION_R0_REFEREE_CONFORMANCE_RESULT_V1",
        "status": "PASS",
        "classification": "REFEREE_CONFORMANCE_AND_FAULT_INJECTION_PASS__NO_NEW_SCIENCE_VERDICT",
        "source_generalization_result_sha256": sha256_file(result_path),
        "baseline_recomputed": baseline,
        "fault_injection_cases": cases,
        "fault_injection_cases_passed": len(cases),
        "fault_injection_cases_total": len(cases),
        "market_data_read": False,
        "model_loaded_or_scored": False,
        "teacher_recompiled": False,
        "scientific_verdict": None,
        "market_verdict": None,
        "canonical_promotion_authorized": False,
        "final_holdout_touched": False,
        "fresh_market_data_downloaded": False,
    }
    p = Path(args.output).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
