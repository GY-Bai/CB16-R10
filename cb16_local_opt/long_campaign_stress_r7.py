from __future__ import annotations

"""
Long-campaign crash/restart stress harness.

This is an orchestration/lifecycle conformance fixture, not a market experiment.

It deliberately injects one crash into every attempt, cycling through all seven phases and
alternating BEFORE/AFTER side-effect crash points. The exact same durable
MultiGenerationRunnerR5 + LongRunState + GenerationStateStore path is then restarted until
the campaign completes.

The stress plugin is idempotent:
- Challenger creation can be replayed;
- tournament recording can be replayed;
- adjudication/commit detects already-COMMITTED attempts;
- phase receipts are written only after a successful plugin return.

A deterministic promote/reject schedule tests both Champion-advancing and
Champion-preserving branches.
"""

import dataclasses
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .generation_orchestrator import (
    ChallengerAttempt,
    GenerationStateStore,
    PolicyRecord,
    PromotionRule,
    TournamentResult,
)
from .long_run_controller import LongRunState, PHASES, PhasePlugin
from .multi_generation_runner_r5 import (
    MultiGenerationRunConfigR5,
    MultiGenerationRunnerR5,
    MultiGenerationRunStateR5,
)


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class StressCampaignConfigR7:
    attempts: int = 50
    reject_every: int = 5
    crash_every_attempt: bool = True
    stress_version: str = "CB16_LONG_CAMPAIGN_STRESS_R7"

    def validate(self):
        if self.attempts <= 0:
            raise ValueError("attempts")
        if self.reject_every <= 1:
            raise ValueError("reject_every must be >1")

    @property
    def content_hash(self):
        self.validate()
        return canonical_hash(self)


class StressSideEffectStateR7:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            self.path, isolation_level=None, timeout=30
        )
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS crash_marker(
            cycle_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            stage TEXT NOT NULL,
            crashed INTEGER NOT NULL,
            PRIMARY KEY(cycle_id,phase,stage)
        );
        CREATE TABLE IF NOT EXISTS phase_effect(
            cycle_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            effect_hash TEXT NOT NULL,
            calls INTEGER NOT NULL,
            PRIMARY KEY(cycle_id,phase)
        );
        """)

    def close(self):
        self.db.close()

    def note_effect(self, cycle_id: str, phase: str, effect_hash: str):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                """
                SELECT effect_hash,calls FROM phase_effect
                WHERE cycle_id=? AND phase=?
                """,
                (cycle_id, phase),
            ).fetchone()
            if row is None:
                self.db.execute(
                    "INSERT INTO phase_effect VALUES(?,?,?,1)",
                    (cycle_id, phase, effect_hash),
                )
            else:
                if row[0] != effect_hash:
                    raise RuntimeError("STRESS_PHASE_EFFECT_REWRITE_CONFLICT")
                self.db.execute(
                    """
                    UPDATE phase_effect SET calls=calls+1
                    WHERE cycle_id=? AND phase=?
                    """,
                    (cycle_id, phase),
                )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def should_crash_once(
        self,
        cycle_id: str,
        phase: str,
        stage: str,
    ) -> bool:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                """
                SELECT crashed FROM crash_marker
                WHERE cycle_id=? AND phase=? AND stage=?
                """,
                (cycle_id, phase, stage),
            ).fetchone()
            if row is None:
                self.db.execute(
                    "INSERT INTO crash_marker VALUES(?,?,?,1)",
                    (cycle_id, phase, stage),
                )
                self.db.execute("COMMIT")
                return True
            self.db.execute("COMMIT")
            return False
        except Exception:
            self.db.execute("ROLLBACK")
            raise


def _attempt_index(cycle_id: str) -> int:
    # cycle ids produced by MultiGenerationRunner contain ATTEMPT:000123
    token = ":ATTEMPT:"
    if token not in cycle_id:
        raise RuntimeError("STRESS_UNEXPECTED_CYCLE_ID")
    return int(cycle_id.split(token, 1)[1].split(":", 1)[0])


def _planned_outcome(attempt_index: int, reject_every: int) -> str:
    return (
        "REJECT"
        if (attempt_index + 1) % reject_every == 0
        else "PROMOTE"
    )


def _crash_plan(attempt_index: int):
    phase = PHASES[attempt_index % len(PHASES)]
    stage = "AFTER" if (attempt_index // len(PHASES)) % 2 else "BEFORE"
    return phase, stage


def stress_phase_plugin_r7(
    *,
    phase,
    cycle_spec,
    plugin_config,
    context,
):
    generation_state_path = plugin_config["generation_state_path"]
    stress_state_path = plugin_config["stress_state_path"]
    reject_every = int(plugin_config["reject_every"])
    crash_every = bool(plugin_config.get("crash_every_attempt", True))
    idx = _attempt_index(cycle_spec.cycle_id)
    planned = _planned_outcome(idx, reject_every)
    crash_phase, crash_stage = _crash_plan(idx)

    stress = StressSideEffectStateR7(stress_state_path)
    gs = GenerationStateStore(generation_state_path)
    attempt_id = "STRESS:" + cycle_spec.cycle_id

    def maybe_crash(stage):
        if (
            crash_every
            and phase == crash_phase
            and stage == crash_stage
            and stress.should_crash_once(
                cycle_spec.cycle_id, phase, stage
            )
        ):
            raise RuntimeError(
                f"INJECTED_STRESS_CRASH:{idx}:{phase}:{stage}"
            )

    try:
        maybe_crash("BEFORE")

        payload = {
            "fixture": "TEST_ONLY_LONG_CAMPAIGN_STRESS",
            "attempt_index": idx,
            "phase": phase,
            "planned_outcome": planned,
            "parent_generation": cycle_spec.generation_parent,
            "parent_policy_hash": cycle_spec.parent_policy_hash,
        }

        if phase == "TRAIN_CHALLENGER":
            attempt = ChallengerAttempt(
                attempt_id=attempt_id,
                parent_generation=cycle_spec.generation_parent,
                parent_weight_hash=cycle_spec.parent_policy_hash,
                training_snapshot_hash=canonical_hash({
                    "cycle": cycle_spec.cycle_id,
                    "kind": "STRESS_SNAPSHOT",
                }),
                experiment_version=cycle_spec.experiment_version,
                architecture_hash="STRESS_ARCH",
                tier="TIER_1",
            )
            gs.create_attempt(attempt)
            challenger_hash = canonical_hash({
                "cycle": cycle_spec.cycle_id,
                "parent": cycle_spec.parent_policy_hash,
                "challenger": True,
            })
            challenger = PolicyRecord(
                generation=cycle_spec.generation_parent + 1,
                weight_hash=challenger_hash,
                architecture_hash="STRESS_ARCH",
                tier="TIER_1",
                checkpoint_path=f"{challenger_hash}.pt",
                parent_weight_hash=cycle_spec.parent_policy_hash,
                training_snapshot_hash=attempt.training_snapshot_hash,
            )
            gs.record_challenger(attempt_id, challenger)
            payload["challenger_weight_hash"] = challenger_hash

        elif phase == "TOURNAMENT":
            row = gs.db.execute(
                """
                SELECT challenger_hash,state FROM attempts
                WHERE attempt_id=?
                """,
                (attempt_id,),
            ).fetchone()
            if row is None or not row[0]:
                raise RuntimeError("STRESS_CHALLENGER_MISSING")
            positive = planned == "PROMOTE"
            delta = 0.01 if positive else -0.01
            result = TournamentResult(
                attempt_id=attempt_id,
                champion_weight_hash=cycle_spec.parent_policy_hash,
                challenger_weight_hash=row[0],
                evaluation_dataset_hash=canonical_hash({
                    "cycle": cycle_spec.cycle_id,
                    "lane": "STRESS_VALIDATION",
                }),
                mean_utility_champion=0.0,
                mean_utility_challenger=delta,
                delta_utility=delta,
                bootstrap_ci_low=(0.001 if positive else -0.02),
                bootstrap_ci_high=(0.02 if positive else -0.001),
                independent_groups=10,
                regime_deltas={
                    "R0": delta,
                    "R1": delta,
                },
            )
            gs.record_tournament(result)
            payload["tournament_hash"] = result.content_hash

        elif phase == "ADJUDICATE_COMMIT":
            row = gs.db.execute(
                """
                SELECT state,verdict,verdict_reasons_json
                FROM attempts WHERE attempt_id=?
                """,
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("STRESS_ATTEMPT_MISSING")
            state, verdict, reasons_json = row
            rule = PromotionRule(
                rule_id="STRESS_RULE_R7",
                min_delta_utility=0.0,
                min_ci_lower=0.0,
                min_independent_groups=1,
                max_bad_regime_fraction=1.0,
            )
            if state == "COMMITTED":
                pass
            elif state == "ADJUDICATED":
                gs.commit(attempt_id)
            elif state == "EVALUATED":
                verdict, reasons = gs.adjudicate(attempt_id, rule)
                gs.commit(attempt_id)
                reasons_json = json.dumps(reasons)
            else:
                raise RuntimeError(
                    f"STRESS_BAD_STATE_FOR_COMMIT:{state}"
                )
            row2 = gs.db.execute(
                """
                SELECT verdict,verdict_reasons_json,state
                FROM attempts WHERE attempt_id=?
                """,
                (attempt_id,),
            ).fetchone()
            payload["verdict"] = row2[0]
            payload["reasons"] = json.loads(row2[1])
            payload["attempt_state"] = row2[2]
            champ = gs.current_champion()
            payload["champion_generation_after"] = champ.generation
            payload["champion_weight_hash_after"] = champ.weight_hash
            if row2[0] != planned:
                raise RuntimeError(
                    f"STRESS_PLANNED_OUTCOME_MISMATCH:{planned}:{row2[0]}"
                )

        effect_hash = canonical_hash(payload)
        stress.note_effect(
            cycle_spec.cycle_id, phase, effect_hash
        )
        maybe_crash("AFTER")
        return payload
    finally:
        gs.close()
        stress.close()


@dataclass(frozen=True)
class LongCampaignStressReceiptR7:
    attempts: int
    promotions: int
    rejections: int
    injected_crashes: int
    restart_invocations: int
    final_generation: int
    final_weight_hash: str
    committed_attempts: int
    completed_phase_rows: int
    expected_phase_rows: int
    normalized_state_hash: str
    status: str

    @property
    def content_hash(self):
        return canonical_hash(self)


def run_long_campaign_stress_r7(
    root: str | Path,
    *,
    config: StressCampaignConfigR7 | None = None,
) -> LongCampaignStressReceiptR7:
    config = config or StressCampaignConfigR7()
    config.validate()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generation_path = root / "generation.sqlite"
    run_path = root / "run.sqlite"
    phase_path = root / "phases.sqlite"
    stress_path = root / "stress.sqlite"

    gs = GenerationStateStore(generation_path)
    if gs.current_champion() is None:
        genesis = PolicyRecord(
            generation=0,
            weight_hash="STRESS_GENESIS_W0",
            architecture_hash="STRESS_ARCH",
            tier="TIER_1",
            checkpoint_path="STRESS_GENESIS_W0.pt",
            parent_weight_hash=None,
            training_snapshot_hash=None,
        )
        gs.initialize_champion(genesis)

    plugins = {
        p: PhasePlugin(
            dotted_callable=(
                "cb16_local_opt.long_campaign_stress_r7:"
                "stress_phase_plugin_r7"
            ),
            version="R7_STRESS",
            config={
                "generation_state_path": str(generation_path),
                "stress_state_path": str(stress_path),
                "reject_every": config.reject_every,
                "crash_every_attempt": config.crash_every_attempt,
            },
        )
        for p in PHASES
    }
    mg_config = MultiGenerationRunConfigR5(
        run_id="R7_LONG_CAMPAIGN_STRESS",
        experiment_version=config.stress_version,
        dataset_hash="STRESS_DATASET",
        split_hash="STRESS_SPLIT",
        physics_hash="STRESS_PHYSICS",
        supervisor_hash="STRESS_SUPERVISOR",
        teacher_hash="STRESS_TEACHER",
        promotion_rule_hash="STRESS_RULE",
        phase_plugins=plugins,
        max_attempts=config.attempts,
        max_promotions=config.attempts,
        max_consecutive_rejects=config.attempts,
    )
    rs = MultiGenerationRunStateR5(run_path)
    ps = LongRunState(phase_path)
    runner = MultiGenerationRunnerR5(
        config=mg_config,
        run_state=rs,
        phase_state=ps,
        generation_state=gs,
    )

    restarts = 0
    injected = 0
    max_invocations = config.attempts * 3 + 10
    for _ in range(max_invocations):
        try:
            summary = runner.run()
            break
        except RuntimeError as exc:
            if not str(exc).startswith("INJECTED_STRESS_CRASH:"):
                rs.close(); ps.close(); gs.close()
                raise
            injected += 1
            restarts += 1
            # Simulate process restart: close and reopen every durable store.
            rs.close(); ps.close(); gs.close()
            rs = MultiGenerationRunStateR5(run_path)
            ps = LongRunState(phase_path)
            gs = GenerationStateStore(generation_path)
            runner = MultiGenerationRunnerR5(
                config=mg_config,
                run_state=rs,
                phase_state=ps,
                generation_state=gs,
            )
    else:
        rs.close(); ps.close(); gs.close()
        raise RuntimeError("STRESS_CAMPAIGN_DID_NOT_CONVERGE")

    champ = gs.current_champion()
    committed = gs.db.execute(
        "SELECT COUNT(*) FROM attempts WHERE state='COMMITTED'"
    ).fetchone()[0]
    attempt_rows = gs.db.execute(
        """
        SELECT attempt_id,state,parent_generation,parent_weight_hash,
               challenger_hash,verdict
        FROM attempts ORDER BY created_at,attempt_id
        """
    ).fetchall()
    phase_rows = ps.db.execute(
        """
        SELECT cycle_id,phase,state,attempts,receipt_hash
        FROM phases ORDER BY cycle_id,ordinal
        """
    ).fetchall()
    completed_phases = sum(r[2] == "COMPLETED" for r in phase_rows)
    expected_phases = config.attempts * len(PHASES)

    expected_rejects = config.attempts // config.reject_every
    expected_promotions = config.attempts - expected_rejects
    if summary["attempts"] != config.attempts:
        raise RuntimeError("STRESS_ATTEMPT_COUNT_MISMATCH")
    if summary["promotions"] != expected_promotions:
        raise RuntimeError("STRESS_PROMOTION_COUNT_MISMATCH")
    if summary["rejections"] != expected_rejects:
        raise RuntimeError("STRESS_REJECTION_COUNT_MISMATCH")
    if champ.generation != expected_promotions:
        raise RuntimeError("STRESS_FINAL_GENERATION_MISMATCH")
    if committed != config.attempts:
        raise RuntimeError("STRESS_NOT_ALL_ATTEMPTS_COMMITTED")
    if completed_phases != expected_phases:
        raise RuntimeError("STRESS_PHASE_COMPLETION_MISMATCH")
    if config.crash_every_attempt and injected != config.attempts:
        raise RuntimeError(
            f"STRESS_INJECTED_CRASH_COUNT_MISMATCH:{injected}"
        )

    normalized = {
        "summary": {
            k: summary[k]
            for k in (
                "run_id","state","attempts","promotions",
                "rejections","consecutive_rejects",
            )
        },
        "champion": {
            "generation": champ.generation,
            "weight_hash": champ.weight_hash,
            "parent_weight_hash": champ.parent_weight_hash,
        },
        "attempts": [list(r) for r in attempt_rows],
        "phases": [
            [r[0], r[1], r[2], int(r[3]), r[4]]
            for r in phase_rows
        ],
    }
    receipt = LongCampaignStressReceiptR7(
        attempts=config.attempts,
        promotions=summary["promotions"],
        rejections=summary["rejections"],
        injected_crashes=injected,
        restart_invocations=restarts,
        final_generation=champ.generation,
        final_weight_hash=champ.weight_hash,
        committed_attempts=int(committed),
        completed_phase_rows=int(completed_phases),
        expected_phase_rows=int(expected_phases),
        normalized_state_hash=canonical_hash(normalized),
        status="R7_LONG_CAMPAIGN_CRASH_RESTART_PASS",
    )
    rs.close(); ps.close(); gs.close()
    return receipt
