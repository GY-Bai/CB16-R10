from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

EXPECTED_PASS_STATUS = {
    "R10_2": "R10_2_REAL_HISTORICAL_LEARNING_PIPELINE_PASS",
    "R10_3": "R10_3_10ASSET_20GEN_EXPANSION_PASS",
    "R10_4": "R10_4_LONG_100GEN_RESEARCH_PASS",
}


def expected_pass_status(phase: str) -> str:
    try:
        return EXPECTED_PASS_STATUS[phase]
    except KeyError:
        raise ValueError(f"UNKNOWN_R10_PHASE:{phase}") from None


def status_is_pass(result: Mapping[str, Any], phase: str) -> bool:
    return str(result.get("final_status", "")) == expected_pass_status(phase)


def _teacher_groups(result: Mapping[str, Any], split: str) -> int | None:
    x = result.get("teacher_evidence_summary")
    if not isinstance(x, Mapping):
        return None
    row = x.get(split)
    if not isinstance(row, Mapping):
        return None
    value = row.get("admitted_dependence_groups")
    return int(value) if isinstance(value, (int, float)) else None


def scientific_summary(result: Mapping[str, Any], phase: str) -> dict[str, Any]:
    controls = result.get("controls") if isinstance(result.get("controls"), Mapping) else {}
    integrity = result.get("integrity") if isinstance(result.get("integrity"), Mapping) else {}
    bundle = result.get("return_bundle") if isinstance(result.get("return_bundle"), Mapping) else {}
    return {
        "schema": "CB16_R10_CI_SCIENTIFIC_SUMMARY_V1",
        "phase": phase,
        "profile_name": result.get("profile_name"),
        "expected_pass_status": expected_pass_status(phase),
        "final_status": result.get("final_status"),
        "status_driving_pass": status_is_pass(result, phase),
        "mechanistic_pipeline_pass": bool(result.get("mechanistic_pipeline_pass", False)),
        "attempts_requested": result.get("attempts_requested"),
        "attempts_completed": result.get("attempts_completed"),
        "promotions": result.get("promotions"),
        "rejections": result.get("rejections"),
        "final_champion_semantic_sha256": result.get("final_champion_semantic_sha256"),
        "final_holdout_2025_09_accessed": bool(result.get("final_holdout_2025_09_accessed", False)),
        "profitability_claimed": bool(result.get("profitability_claimed", False)),
        "market_alpha_claimed": bool(result.get("market_alpha_claimed", False)),
        "teacher_train_admitted_dependence_groups": _teacher_groups(result, "train"),
        "teacher_validation_admitted_dependence_groups": _teacher_groups(result, "validation"),
        "scientific_controls_status": result.get("scientific_controls_status"),
        "controls_F3_minus_F2": controls.get("F3_minus_F2") if isinstance(controls, Mapping) else None,
        "integrity": {str(k): bool(v) for k, v in sorted(integrity.items()) if isinstance(v, bool)},
        "return_bundle": {
            "sha256": bundle.get("sha256"),
            "size": bundle.get("size"),
        },
    }


def append_scientific_summary(result: Mapping[str, Any], phase: str) -> dict[str, Any]:
    summary = scientific_summary(result, phase)
    ci_out = os.environ.get("CI_OUT")
    if not ci_out:
        return summary
    out = Path(ci_out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scientific_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = out / "REPORT.md"
    with report.open("a", encoding="utf-8") as f:
        f.write("\n## R10 scientific summary\n\n")
        for key in (
            "phase", "profile_name", "expected_pass_status", "final_status", "status_driving_pass",
            "mechanistic_pipeline_pass", "attempts_requested", "attempts_completed", "promotions", "rejections",
            "final_champion_semantic_sha256", "teacher_train_admitted_dependence_groups",
            "teacher_validation_admitted_dependence_groups", "scientific_controls_status", "controls_F3_minus_F2",
            "profitability_claimed", "market_alpha_claimed", "final_holdout_2025_09_accessed",
        ):
            f.write(f"- {key}: {json.dumps(summary.get(key), ensure_ascii=True, allow_nan=False)}\n")
        f.write(f"- integrity: {json.dumps(summary['integrity'], sort_keys=True)}\n")
        f.write(f"- return_bundle_sha256: {json.dumps(summary['return_bundle']['sha256'])}\n")
        f.write(f"- return_bundle_size: {json.dumps(summary['return_bundle']['size'])}\n")
    return summary
