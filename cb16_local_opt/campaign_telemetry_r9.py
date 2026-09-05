from __future__ import annotations

"""Post-run telemetry and scientific-boundary audit for R9 short campaigns."""

import dataclasses
import hashlib
import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tree_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except FileNotFoundError:
            pass
    return total


def _json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _safe_cycle_dir(workspace_root: Path, cycle_id: str) -> Path:
    safe = cycle_id.replace("/", "_").replace(":", "_")
    return workspace_root / "cycles" / safe


def _controls_map(obj: dict[str, Any] | None) -> dict[str, float]:
    if not obj:
        return {}
    out = {}
    for x in obj.get("formulations", []):
        f = x.get("formulation")
        q = x.get("qcrps")
        if f is not None and q is not None:
            out[str(f)] = float(q)
    return out


@dataclass(frozen=True)
class GenerationTelemetryR9:
    attempt_index: int
    cycle_id: str
    parent_generation: int
    outcome: str
    resulting_generation: int
    phase_wall_seconds: dict[str, float]
    cycle_wall_seconds: float
    phase_attempt_counts: dict[str, int]
    cycle_directory_bytes: int
    train_dependence_groups: int | None
    validation_dependence_groups: int | None
    h72_farm_groups: int | None
    h72_farm_elapsed_s: float | None
    train_counterfactual_branches: int | None
    validation_counterfactual_branches: int | None
    evidence_total: int | None
    evidence_admitted: int | None
    evidence_admission_fraction: float | None
    validation_controls_status: str | None
    validation_qcrps: dict[str, float]
    f2_minus_f0_delta: float | None
    f3_minus_f2_delta: float | None
    training_examples_seen: int | None
    gradient_steps: int | None
    min_gradient_norm: float | None
    max_gradient_norm: float | None
    validation_delta_utility: float | None
    validation_ci_low: float | None
    validation_ci_high: float | None
    validation_independent_groups: int | None
    adjudication_verdict: str | None
    final_tournament_opened: bool


@dataclass(frozen=True)
class CampaignTelemetryReceiptR9:
    telemetry_version: str
    status: str
    run_id: str
    attempts: int
    promotions: int
    rejections: int
    final_generation: int
    generations: tuple[GenerationTelemetryR9, ...]
    total_cycle_directory_bytes: int
    experience_lake_bytes: int
    checkpoint_bytes: int
    mean_cycle_wall_seconds: float
    mean_evidence_admission_fraction: float | None
    total_training_examples: int
    total_gradient_steps: int
    total_h72_groups: int
    final_tournament_opened_anywhere_in_campaign: bool
    known_final_artifact_paths: tuple[str, ...]

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def collect_campaign_telemetry_r9(
    *,
    run_root: str | Path,
    workspace_root: str | Path,
    experience_lake_root: str | Path,
    checkpoint_root: str | Path,
    generation_state_path: str | Path,
) -> CampaignTelemetryReceiptR9:
    run_root = Path(run_root)
    workspace_root = Path(workspace_root)
    run_db = run_root / "MULTIGEN_RUN.sqlite"
    phase_db = run_root / "PHASES.sqlite"
    if not run_db.is_file() or not phase_db.is_file():
        raise FileNotFoundError("R9 campaign state databases missing")

    rdb = sqlite3.connect(run_db)
    run = rdb.execute(
        "SELECT run_id,state,attempts,promotions,rejections FROM run WHERE singleton=1"
    ).fetchone()
    attempts = rdb.execute(
        """
        SELECT attempt_index,cycle_id,parent_generation,outcome,resulting_generation,
               resulting_weight_hash,created_at,completed_at
        FROM attempts ORDER BY attempt_index
        """
    ).fetchall()
    rdb.close()
    if run is None:
        raise RuntimeError("R9_RUN_STATE_EMPTY")

    pdb = sqlite3.connect(phase_db)
    gens: list[GenerationTelemetryR9] = []
    known_final: list[str] = []
    for row in attempts:
        idx, cycle_id, parent_gen, outcome, result_gen, _, created, completed = row
        phase_rows = pdb.execute(
            """
            SELECT phase,state,attempts,started_at,completed_at
            FROM phases WHERE cycle_id=? ORDER BY ordinal
            """,
            (cycle_id,),
        ).fetchall()
        if any(x[1] != "COMPLETED" for x in phase_rows):
            raise RuntimeError(f"R9_INCOMPLETE_PHASES:{cycle_id}")
        phase_secs = {
            p: max(0.0, float(done) - float(start))
            for p, _, _, start, done in phase_rows
            if start is not None and done is not None
        }
        phase_attempts = {p: int(n) for p, _, n, _, _ in phase_rows}
        cdir = _safe_cycle_dir(workspace_root, cycle_id)
        rollout = _json(cdir / "ROLLOUT_RECEIPT.json")
        teacher = _json(cdir / "TEACHER_CREDIT_RECEIPT.json")
        controls = _json(cdir / "VALIDATION_CONTROLS.json")
        train = _json(cdir / "TRAIN_CHALLENGER_RECEIPT.json")
        tournament = _json(cdir / "VALIDATION_TOURNAMENT_RECEIPT.json")
        adjud = _json(cdir / "ADJUDICATION_RECEIPT.json")

        evidence_total = teacher.get("train_evidence_total") if teacher else None
        admitted = teacher.get("train_evidence_admitted") if teacher else None
        admission_frac = None
        if evidence_total is not None and int(evidence_total) > 0 and admitted is not None:
            admission_frac = float(admitted) / float(evidence_total)

        qmap = _controls_map(controls)
        f2_f0 = None
        f3_f2 = None
        if controls:
            if controls.get("f2_minus_f0"):
                f2_f0 = float(controls["f2_minus_f0"].get("mean_delta"))
            if controls.get("f3_minus_f2"):
                f3_f2 = float(controls["f3_minus_f2"].get("mean_delta"))

        tr = tournament.get("result") if tournament else None
        final_flags = []
        for obj in (rollout, teacher, tournament, adjud):
            if obj:
                for k in (
                    "tournament_market_values_opened",
                    "tournament_data_read",
                    "final_tournament_opened",
                ):
                    if obj.get(k) is True:
                        final_flags.append(k)

        for p in cdir.rglob("*") if cdir.exists() else ():
            if p.is_file() and (
                "FINAL_HOLDOUT" in p.name.upper()
                or "FINAL_TOURNAMENT" in p.name.upper()
                or "FINAL_CONTROLS" in p.name.upper()
            ):
                known_final.append(str(p))

        gens.append(GenerationTelemetryR9(
            attempt_index=int(idx),
            cycle_id=str(cycle_id),
            parent_generation=int(parent_gen),
            outcome=str(outcome),
            resulting_generation=int(result_gen),
            phase_wall_seconds=phase_secs,
            cycle_wall_seconds=(
                max(0.0, float(completed) - float(created))
                if created is not None and completed is not None else sum(phase_secs.values())
            ),
            phase_attempt_counts=phase_attempts,
            cycle_directory_bytes=tree_bytes(cdir),
            train_dependence_groups=(rollout.get("train_dependence_groups") if rollout else None),
            validation_dependence_groups=(rollout.get("validation_dependence_groups") if rollout else None),
            h72_farm_groups=(rollout.get("h72_farm_groups") if rollout else None),
            h72_farm_elapsed_s=(rollout.get("h72_farm_elapsed_s") if rollout else None),
            train_counterfactual_branches=(rollout.get("train_counterfactual_branches") if rollout else None),
            validation_counterfactual_branches=(rollout.get("validation_counterfactual_branches") if rollout else None),
            evidence_total=(int(evidence_total) if evidence_total is not None else None),
            evidence_admitted=(int(admitted) if admitted is not None else None),
            evidence_admission_fraction=admission_frac,
            validation_controls_status=(teacher.get("validation_controls_status") if teacher else None),
            validation_qcrps=qmap,
            f2_minus_f0_delta=f2_f0,
            f3_minus_f2_delta=f3_f2,
            training_examples_seen=(int(train["examples_seen"]) if train and train.get("examples_seen") is not None else None),
            gradient_steps=(int(train["gradient_steps"]) if train and train.get("gradient_steps") is not None else None),
            min_gradient_norm=(float(train["min_gradient_norm"]) if train and train.get("min_gradient_norm") is not None else None),
            max_gradient_norm=(float(train["max_gradient_norm"]) if train and train.get("max_gradient_norm") is not None else None),
            validation_delta_utility=(float(tr["delta_utility"]) if tr else None),
            validation_ci_low=(float(tr["bootstrap_ci_low"]) if tr else None),
            validation_ci_high=(float(tr["bootstrap_ci_high"]) if tr else None),
            validation_independent_groups=(int(tr["independent_groups"]) if tr else None),
            adjudication_verdict=(adjud.get("verdict") if adjud else None),
            final_tournament_opened=bool(final_flags),
        ))
    pdb.close()

    gs = sqlite3.connect(generation_state_path)
    champ = gs.execute("SELECT generation FROM champion WHERE singleton=1").fetchone()
    gs.close()
    final_generation = int(champ[0]) if champ else -1

    fractions = [x.evidence_admission_fraction for x in gens if x.evidence_admission_fraction is not None]
    cycle_secs = [x.cycle_wall_seconds for x in gens]
    receipt = CampaignTelemetryReceiptR9(
        telemetry_version="CB16_SHORT_CAMPAIGN_TELEMETRY_R9",
        status="SHORT_CAMPAIGN_TELEMETRY_COMPLETE",
        run_id=str(run[0]),
        attempts=int(run[2]),
        promotions=int(run[3]),
        rejections=int(run[4]),
        final_generation=final_generation,
        generations=tuple(gens),
        total_cycle_directory_bytes=sum(x.cycle_directory_bytes for x in gens),
        experience_lake_bytes=tree_bytes(experience_lake_root),
        checkpoint_bytes=tree_bytes(checkpoint_root),
        mean_cycle_wall_seconds=(sum(cycle_secs) / len(cycle_secs) if cycle_secs else 0.0),
        mean_evidence_admission_fraction=(sum(fractions) / len(fractions) if fractions else None),
        total_training_examples=sum(x.training_examples_seen or 0 for x in gens),
        total_gradient_steps=sum(x.gradient_steps or 0 for x in gens),
        total_h72_groups=sum(x.h72_farm_groups or 0 for x in gens),
        final_tournament_opened_anywhere_in_campaign=(
            any(x.final_tournament_opened for x in gens) or bool(known_final)
        ),
        known_final_artifact_paths=tuple(sorted(set(known_final))),
    )
    return receipt


def save_campaign_telemetry_r9(receipt: CampaignTelemetryReceiptR9, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**asdict(receipt), "content_hash": receipt.content_hash}, indent=2) + "\n")
    return p
