from __future__ import annotations

"""Read-only post-run/snapshot diagnostics for CB16 R10.4."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any

from .model_probe import compare_states, semantic_sha256, _load_state

FROZEN_SOURCE_AUTHORITY = "f056ae6a0722e3e92d71793024a6e6d3fe9af003"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _gen_num(path: Path) -> int | None:
    m = re.fullmatch(r"G(\d+)", path.name)
    return int(m.group(1)) if m else None


def generation_census(run_root: Path, requested: int = 100) -> dict[str, Any]:
    generations_root = run_root / "generations"
    rows = []
    present = set()
    if generations_root.is_dir():
        for gd in sorted(generations_root.iterdir()):
            if not gd.is_dir():
                continue
            g = _gen_num(gd)
            if g is None:
                continue
            present.add(g)
            gr = _read_json(gd / "GENERATION_RESULT.json")
            trace = _read_json(gd / "ON_POLICY_REAL_TRACE_RECEIPT.json")
            train = _read_json(gd / f"CHALLENGER_TRAINING_RECEIPT_G{g}.json")
            decision = None
            if gr and isinstance(gr.get("tournament"), dict):
                decision = gr["tournament"].get("decision")
            rows.append({
                "generation": g,
                "complete": gr is not None,
                "trace_receipt": trace is not None,
                "trace_count": trace.get("trace_count") if trace else None,
                "trace_matured": trace.get("matured") if trace else None,
                "training_receipt": train is not None,
                "challenger_checkpoint": (gd / "challenger.pt").is_file(),
                "champion_checkpoint": (gd / "champion_after.pt").is_file(),
                "decision": decision,
                "parent_semantic_sha256": gr.get("parent_champion_semantic_sha256") if gr else None,
                "challenger_semantic_sha256": (
                    gr.get("challenger", {}).get("semantic_sha256")
                    if gr and isinstance(gr.get("challenger"), dict) else None
                ),
                "champion_after_semantic_sha256": (
                    gr.get("champion_after", {}).get("semantic_sha256")
                    if gr and isinstance(gr.get("champion_after"), dict) else None
                ),
                "parameter_l2_delta_reported": train.get("parameter_l2_delta") if train else None,
                "gradient_group_norms_last_step": train.get("gradient_group_norms_last_step") if train else None,
                "update_group_norms": train.get("update_group_norms") if train else None,
                "validation_loss_before": (
                    gr.get("tournament", {}).get("validation_loss_before")
                    if gr and isinstance(gr.get("tournament"), dict) else None
                ),
                "validation_loss_after": (
                    gr.get("tournament", {}).get("validation_loss_after")
                    if gr and isinstance(gr.get("tournament"), dict) else None
                ),
                "relative_improvement": (
                    gr.get("tournament", {}).get("relative_improvement")
                    if gr and isinstance(gr.get("tournament"), dict) else None
                ),
            })
    missing = [g for g in range(int(requested)) if g not in present]
    complete = [x["generation"] for x in rows if x["complete"]]
    return {
        "schema": "CB16_R10_GENERATION_CENSUS_R0",
        "requested_generations": int(requested),
        "present_generation_count": len(present),
        "complete_generation_count": len(complete),
        "highest_present_generation": max(present) if present else None,
        "highest_complete_generation": max(complete) if complete else None,
        "missing_generation_directories": missing,
        "rows": rows,
        "final_holdout_2025_09_accessed": False,
    }


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def runtime_phase_summary(runtime_diag_dir: Path | None) -> dict[str, Any]:
    if runtime_diag_dir is None:
        return {"schema": "CB16_R10_RUNTIME_PHASE_SUMMARY_R0", "available": False}
    events = _load_events(runtime_diag_dir / "artifact_events.jsonl")
    runtime_summary = _read_json(runtime_diag_dir / "RUNTIME_DIAGNOSTIC_SUMMARY.json")
    by_gen: dict[int, dict[str, float]] = {}
    for e in events:
        rel = str(e.get("relative_path", ""))
        m = re.match(r"generations/G(\d+)/(.+)$", rel)
        if not m:
            continue
        g = int(m.group(1))
        name = m.group(2)
        t = float(e.get("observed_wall_time_unix", 0.0))
        row = by_gen.setdefault(g, {})
        row[name] = min(row.get(name, t), t)
    rows = []
    prev_result_t: float | None = None
    for g in sorted(by_gen):
        e = by_gen[g]
        trace_t = e.get("ON_POLICY_REAL_TRACE_RECEIPT.json")
        train_t = e.get(f"CHALLENGER_TRAINING_RECEIPT_G{g}.json")
        challenger_t = e.get("challenger.pt")
        champion_t = e.get("champion_after.pt")
        result_t = e.get("GENERATION_RESULT.json")
        rows.append({
            "generation": g,
            "generation_wall_s_from_previous_result": (
                result_t - prev_result_t if result_t is not None and prev_result_t is not None else None
            ),
            "prep_plus_on_policy_s_from_previous_result_to_trace": (
                trace_t - prev_result_t if trace_t is not None and prev_result_t is not None else None
            ),
            "trace_to_training_receipt_s": (
                train_t - trace_t if train_t is not None and trace_t is not None else None
            ),
            "training_receipt_to_challenger_saved_s": (
                challenger_t - train_t if challenger_t is not None and train_t is not None else None
            ),
            "challenger_to_champion_saved_s": (
                champion_t - challenger_t if champion_t is not None and challenger_t is not None else None
            ),
            "champion_saved_to_generation_result_s": (
                result_t - champion_t if result_t is not None and champion_t is not None else None
            ),
            "observation_semantics": "SIDECAR_ARTIFACT_APPEARANCE_APPROXIMATION_NOT_INTERNAL_STAGE_TIMER",
        })
        if result_t is not None:
            prev_result_t = result_t
    return {
        "schema": "CB16_R10_RUNTIME_PHASE_SUMMARY_R0",
        "available": bool(events or runtime_summary),
        "runtime_bottleneck": runtime_summary.get("bottleneck") if runtime_summary else None,
        "samples_written": runtime_summary.get("samples_written") if runtime_summary else None,
        "rows": rows,
        "final_holdout_2025_09_accessed": False,
    }


def model_evolution(
    run_root: Path,
    *,
    start_checkpoint: Path | None,
    requested: int = 100,
    max_generations: int | None = None,
) -> dict[str, Any]:
    rows = []
    previous_champion = start_checkpoint
    limit = requested if max_generations is None else min(requested, int(max_generations))
    for g in range(limit):
        gd = run_root / "generations" / f"G{g:02d}"
        gr = _read_json(gd / "GENERATION_RESULT.json")
        champion = gd / "champion_after.pt"
        challenger = gd / "challenger.pt"
        if gr is None or not champion.is_file():
            break
        row: dict[str, Any] = {
            "generation": g,
            "decision": gr.get("tournament", {}).get("decision") if isinstance(gr.get("tournament"), dict) else None,
            "reported_parent_semantic_sha256": gr.get("parent_champion_semantic_sha256"),
            "reported_challenger_semantic_sha256": (
                gr.get("challenger", {}).get("semantic_sha256") if isinstance(gr.get("challenger"), dict) else None
            ),
            "reported_champion_semantic_sha256": (
                gr.get("champion_after", {}).get("semantic_sha256") if isinstance(gr.get("champion_after"), dict) else None
            ),
        }
        if previous_champion is not None and previous_champion.is_file():
            parent_state, _ = _load_state(previous_champion)
            parent_sha = semantic_sha256(parent_state)
            row["computed_parent_semantic_sha256"] = parent_sha
            row["parent_hash_matches_receipt"] = parent_sha == row["reported_parent_semantic_sha256"]
            if challenger.is_file():
                challenger_state, _ = _load_state(challenger)
                challenger_sha = semantic_sha256(challenger_state)
                row["computed_challenger_semantic_sha256"] = challenger_sha
                row["challenger_hash_matches_receipt"] = challenger_sha == row["reported_challenger_semantic_sha256"]
                row["parent_to_challenger"] = compare_states(parent_state, challenger_state)
            champion_state, _ = _load_state(champion)
            champion_sha = semantic_sha256(champion_state)
            row["computed_champion_semantic_sha256"] = champion_sha
            row["champion_hash_matches_receipt"] = champion_sha == row["reported_champion_semantic_sha256"]
            row["parent_to_champion"] = compare_states(parent_state, champion_state)
        rows.append(row)
        previous_champion = champion

    rel = [
        float(x["parent_to_challenger"]["global"]["relative_delta_l2"])
        for x in rows
        if isinstance(x.get("parent_to_challenger"), dict)
        and x["parent_to_challenger"].get("compatible")
        and isinstance(x["parent_to_challenger"].get("global"), dict)
    ]
    promoted = [
        float(x["parent_to_challenger"]["global"]["relative_delta_l2"])
        for x in rows
        if x.get("decision") == "PROMOTE"
        and isinstance(x.get("parent_to_challenger"), dict)
        and x["parent_to_challenger"].get("compatible")
    ]
    rejected = [
        float(x["parent_to_challenger"]["global"]["relative_delta_l2"])
        for x in rows
        if x.get("decision") == "REJECT"
        and isinstance(x.get("parent_to_challenger"), dict)
        and x["parent_to_challenger"].get("compatible")
    ]
    return {
        "schema": "CB16_R10_MODEL_EVOLUTION_R0",
        "generations_probed": len(rows),
        "median_parent_to_challenger_relative_delta_l2": median(rel) if rel else None,
        "min_parent_to_challenger_relative_delta_l2": min(rel) if rel else None,
        "max_parent_to_challenger_relative_delta_l2": max(rel) if rel else None,
        "median_relative_delta_promoted": median(promoted) if promoted else None,
        "median_relative_delta_rejected": median(rejected) if rejected else None,
        "all_parent_hashes_match": all(x.get("parent_hash_matches_receipt") is not False for x in rows),
        "all_challenger_hashes_match": all(x.get("challenger_hash_matches_receipt") is not False for x in rows),
        "all_champion_hashes_match": all(x.get("champion_hash_matches_receipt") is not False for x in rows),
        "rows": rows,
        "interpretation": "DIAGNOSTIC_ONLY_NOT_PIPELINE_PASS_DRIVER",
        "final_holdout_2025_09_accessed": False,
    }


def _numeric_gradient_summary(census: dict[str, Any]) -> dict[str, Any]:
    by_group: dict[str, list[float]] = {}
    updates: dict[str, list[float]] = {}
    for row in census["rows"]:
        grads = row.get("gradient_group_norms_last_step")
        if isinstance(grads, dict):
            for k, v in grads.items():
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    by_group.setdefault(str(k), []).append(float(v))
        ups = row.get("update_group_norms")
        if isinstance(ups, dict):
            for k, v in ups.items():
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    updates.setdefault(str(k), []).append(float(v))
    return {
        "gradient_group_median": {k: median(v) for k, v in sorted(by_group.items()) if v},
        "gradient_group_min": {k: min(v) for k, v in sorted(by_group.items()) if v},
        "update_group_median": {k: median(v) for k, v in sorted(updates.items()) if v},
        "update_group_min": {k: min(v) for k, v in sorted(updates.items()) if v},
    }


def build_diagnostics(
    *,
    run_root: Path,
    out_dir: Path,
    runtime_diag_dir: Path | None,
    start_checkpoint: Path | None,
    source_authority_sha: str,
    requested: int = 100,
    allow_incomplete: bool = False,
    probe_models: bool = True,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    out_dir = out_dir.resolve()
    if _inside(out_dir, run_root):
        raise ValueError("DIAGNOSTIC_OUTPUT_MUST_BE_OUTSIDE_CANONICAL_RUN_ROOT")
    out_dir.mkdir(parents=True, exist_ok=True)

    final_result = _read_json(run_root / "FINAL_RESULT_R102.json")
    census = generation_census(run_root, requested=requested)
    if not allow_incomplete and (
        final_result is None or census["complete_generation_count"] != int(requested)
    ):
        raise RuntimeError(
            f"R10_DIAGNOSTICS_REQUIRE_COMPLETE_CAMPAIGN:final={final_result is not None}:"
            f"complete={census['complete_generation_count']}:requested={requested}"
        )

    runtime = runtime_phase_summary(runtime_diag_dir)
    models = (
        model_evolution(run_root, start_checkpoint=start_checkpoint, requested=requested)
        if probe_models else {
            "schema": "CB16_R10_MODEL_EVOLUTION_R0", "generations_probed": 0,
            "status": "SKIPPED_BY_REQUEST", "final_holdout_2025_09_accessed": False,
        }
    )
    gradient_summary = _numeric_gradient_summary(census)

    decisions = [x.get("decision") for x in census["rows"] if x.get("complete")]
    promotions = sum(x == "PROMOTE" for x in decisions)
    rejections = sum(x == "REJECT" for x in decisions)
    status = "POSTRUN_DIAGNOSTICS_COMPLETE" if final_result is not None and census["complete_generation_count"] == requested else "INCOMPLETE_READ_ONLY_SNAPSHOT"

    summary = {
        "schema": "CB16_R10_POSTRUN_DIAGNOSTIC_SUMMARY_R0",
        "status": status,
        "source_authority_sha": source_authority_sha,
        "source_authority_matches_frozen_r10": source_authority_sha == FROZEN_SOURCE_AUTHORITY,
        "run_root": str(run_root),
        "requested_generations": requested,
        "complete_generations": census["complete_generation_count"],
        "promotions_from_generation_results": promotions,
        "rejections_from_generation_results": rejections,
        "final_result_present": final_result is not None,
        "final_status": final_result.get("final_status") if final_result else None,
        "mechanistic_pipeline_pass": final_result.get("mechanistic_pipeline_pass") if final_result else None,
        "scientific_controls_status": final_result.get("scientific_controls_status") if final_result else None,
        "final_champion_semantic_sha256": final_result.get("final_champion_semantic_sha256") if final_result else None,
        "experience_lake_audit": final_result.get("experience_lake_audit") if final_result else None,
        "integrity": final_result.get("integrity") if final_result else None,
        "runtime_bottleneck": runtime.get("runtime_bottleneck"),
        "model_update_summary": {
            k: v for k, v in models.items()
            if k in {
                "generations_probed",
                "median_parent_to_challenger_relative_delta_l2",
                "min_parent_to_challenger_relative_delta_l2",
                "max_parent_to_challenger_relative_delta_l2",
                "median_relative_delta_promoted",
                "median_relative_delta_rejected",
                "all_parent_hashes_match",
                "all_challenger_hashes_match",
                "all_champion_hashes_match",
            }
        },
        "gradient_update_summary": gradient_summary,
        "scientific_semantics_changed": False,
        "writes_to_canonical_run_root": False,
        "diagnostics_are_status_driving": False,
        "diagnostic_semantics": "OBSERVABILITY_ONLY_NOT_PIPELINE_PASS_DRIVER",
        "final_holdout_2025_09_accessed": False,
    }

    _write_json(out_dir / "GENERATION_CENSUS.json", census)
    _write_json(out_dir / "RUNTIME_PHASE_SUMMARY.json", runtime)
    _write_json(out_dir / "MODEL_EVOLUTION.json", models)
    _write_json(out_dir / "POSTRUN_DIAGNOSTIC_SUMMARY.json", summary)

    files = [
        out_dir / "GENERATION_CENSUS.json",
        out_dir / "RUNTIME_PHASE_SUMMARY.json",
        out_dir / "MODEL_EVOLUTION.json",
        out_dir / "POSTRUN_DIAGNOSTIC_SUMMARY.json",
    ]
    lines = [f"{_sha256_file(p)}  {p.name}" for p in files]
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="CB16 R10.4 read-only post-run diagnostics")
    ap.add_argument("--run-root", default="/data/cb16_hdd/cb16_runtime/R10_4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runtime-diagnostics", default=None)
    ap.add_argument(
        "--start-checkpoint",
        default="/home/bgy/cb16_ssd/runtime/R10_3/generations/G19/champion_after.pt",
    )
    ap.add_argument("--source-authority-sha", default=FROZEN_SOURCE_AUTHORITY)
    ap.add_argument("--requested", type=int, default=100)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--no-model-probe", action="store_true")
    a = ap.parse_args()
    summary = build_diagnostics(
        run_root=Path(a.run_root),
        out_dir=Path(a.out),
        runtime_diag_dir=Path(a.runtime_diagnostics) if a.runtime_diagnostics else None,
        start_checkpoint=Path(a.start_checkpoint) if a.start_checkpoint else None,
        source_authority_sha=a.source_authority_sha,
        requested=a.requested,
        allow_incomplete=a.allow_incomplete,
        probe_models=not a.no_model_probe,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
