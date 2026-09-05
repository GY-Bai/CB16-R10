from __future__ import annotations

"""
Durable command-level multi-generation runner.

This module does not hard-code a Teacher or tournament.  Instead it repeatedly executes
the R4 phase controller with versioned real phase plugins and verifies the external
Champion registry after each attempt.

It supplies the part needed for a long unattended run:
- attempt identities;
- restart from an ACTIVE attempt;
- parent Champion hash binding;
- promotion/rejection counts;
- configurable stop conditions;
- no hidden change of phase plugin identities mid-run.
"""

import dataclasses
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .generation_orchestrator import GenerationStateStore
from .long_run_controller import (
    GenerationCycleSpec,
    LongRunGenerationController,
    LongRunState,
    PHASES,
    PhasePlugin,
)


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class MultiGenerationRunConfigR5:
    run_id: str
    experiment_version: str
    dataset_hash: str
    split_hash: str
    physics_hash: str
    supervisor_hash: str
    teacher_hash: str
    promotion_rule_hash: str
    phase_plugins: Mapping[str, PhasePlugin]
    max_attempts: int = 100
    max_promotions: int = 100
    max_consecutive_rejects: int = 20

    def validate(self):
        if not self.run_id:
            raise ValueError("run_id")
        if set(self.phase_plugins) != set(PHASES):
            raise ValueError("phase plugin set must exactly match PHASES")
        if self.max_attempts <= 0 or self.max_promotions <= 0:
            raise ValueError("stop limits")
        if self.max_consecutive_rejects <= 0:
            raise ValueError("max_consecutive_rejects")

    @property
    def content_hash(self) -> str:
        self.validate()
        return canonical_hash({
            "run_id": self.run_id,
            "experiment_version": self.experiment_version,
            "dataset_hash": self.dataset_hash,
            "split_hash": self.split_hash,
            "physics_hash": self.physics_hash,
            "supervisor_hash": self.supervisor_hash,
            "teacher_hash": self.teacher_hash,
            "promotion_rule_hash": self.promotion_rule_hash,
            "phase_plugins": {k: v.identity_hash for k, v in sorted(self.phase_plugins.items())},
            "max_attempts": self.max_attempts,
            "max_promotions": self.max_promotions,
            "max_consecutive_rejects": self.max_consecutive_rejects,
        })


class MultiGenerationRunStateR5:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS run(
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                run_id TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                config_json BLOB NOT NULL,
                state TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                promotions INTEGER NOT NULL,
                rejections INTEGER NOT NULL,
                consecutive_rejects INTEGER NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts(
                attempt_index INTEGER PRIMARY KEY,
                cycle_id TEXT UNIQUE NOT NULL,
                parent_generation INTEGER NOT NULL,
                parent_weight_hash TEXT NOT NULL,
                cycle_hash TEXT NOT NULL,
                state TEXT NOT NULL,
                outcome TEXT,
                resulting_generation INTEGER,
                resulting_weight_hash TEXT,
                created_at REAL NOT NULL,
                completed_at REAL
            );
            """
        )

    def close(self):
        self.db.close()

    def initialize(self, config: MultiGenerationRunConfigR5):
        config.validate()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT run_id,config_hash FROM run WHERE singleton=1").fetchone()
            if row:
                if row[0] != config.run_id or row[1] != config.content_hash:
                    raise RuntimeError("MULTI_GENERATION_RUN_CONFIG_CONFLICT")
                self.db.execute("COMMIT")
                return
            self.db.execute(
                """
                INSERT INTO run(
                    singleton,run_id,config_hash,config_json,state,
                    attempts,promotions,rejections,consecutive_rejects,updated_at
                ) VALUES(1,?,?,?,'ACTIVE',0,0,0,0,?)
                """,
                (
                    config.run_id,
                    config.content_hash,
                    json.dumps(asdict(config), sort_keys=True),
                    time.time(),
                ),
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def summary(self) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT run_id,config_hash,state,attempts,promotions,rejections,consecutive_rejects FROM run WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise RuntimeError("RUN_NOT_INITIALIZED")
        return {
            "run_id": row[0],
            "config_hash": row[1],
            "state": row[2],
            "attempts": int(row[3]),
            "promotions": int(row[4]),
            "rejections": int(row[5]),
            "consecutive_rejects": int(row[6]),
        }

    def active_attempt(self):
        row = self.db.execute(
            """
            SELECT attempt_index,cycle_id,parent_generation,parent_weight_hash,cycle_hash
            FROM attempts WHERE state='ACTIVE' ORDER BY attempt_index DESC LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return {
            "attempt_index": int(row[0]),
            "cycle_id": row[1],
            "parent_generation": int(row[2]),
            "parent_weight_hash": row[3],
            "cycle_hash": row[4],
        }

    def register_attempt(self, cycle: GenerationCycleSpec) -> int:
        active = self.active_attempt()
        if active:
            if active["cycle_id"] != cycle.cycle_id or active["cycle_hash"] != cycle.content_hash:
                raise RuntimeError("ANOTHER_ATTEMPT_ALREADY_ACTIVE")
            return active["attempt_index"]

        summary = self.summary()
        idx = summary["attempts"]
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                """
                INSERT INTO attempts(
                    attempt_index,cycle_id,parent_generation,parent_weight_hash,
                    cycle_hash,state,created_at
                ) VALUES(?,?,?,?,?,'ACTIVE',?)
                """,
                (
                    idx,
                    cycle.cycle_id,
                    cycle.generation_parent,
                    cycle.parent_policy_hash,
                    cycle.content_hash,
                    now,
                ),
            )
            self.db.execute(
                "UPDATE run SET attempts=attempts+1,updated_at=? WHERE singleton=1",
                (now,),
            )
            self.db.execute("COMMIT")
            return idx
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def complete_attempt(
        self,
        *,
        attempt_index: int,
        outcome: str,
        resulting_generation: int,
        resulting_weight_hash: str,
    ):
        if outcome not in {"PROMOTE", "REJECT"}:
            raise ValueError("outcome")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT state,parent_weight_hash FROM attempts WHERE attempt_index=?",
                (attempt_index,),
            ).fetchone()
            if row is None:
                raise RuntimeError("UNKNOWN_ATTEMPT")
            if row[0] == "COMPLETED":
                old = self.db.execute(
                    "SELECT outcome,resulting_generation,resulting_weight_hash FROM attempts WHERE attempt_index=?",
                    (attempt_index,),
                ).fetchone()
                if old != (outcome, resulting_generation, resulting_weight_hash):
                    raise RuntimeError("ATTEMPT_OUTCOME_REWRITE_CONFLICT")
                self.db.execute("COMMIT")
                return
            if row[0] != "ACTIVE":
                raise RuntimeError("BAD_ATTEMPT_STATE")

            now = time.time()
            self.db.execute(
                """
                UPDATE attempts
                SET state='COMPLETED',outcome=?,resulting_generation=?,
                    resulting_weight_hash=?,completed_at=?
                WHERE attempt_index=?
                """,
                (outcome, resulting_generation, resulting_weight_hash, now, attempt_index),
            )
            if outcome == "PROMOTE":
                self.db.execute(
                    """
                    UPDATE run SET promotions=promotions+1,consecutive_rejects=0,updated_at=?
                    WHERE singleton=1
                    """,
                    (now,),
                )
            else:
                self.db.execute(
                    """
                    UPDATE run
                    SET rejections=rejections+1,consecutive_rejects=consecutive_rejects+1,updated_at=?
                    WHERE singleton=1
                    """,
                    (now,),
                )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def set_state(self, state: str):
        self.db.execute(
            "UPDATE run SET state=?,updated_at=? WHERE singleton=1",
            (state, time.time()),
        )


class MultiGenerationRunnerR5:
    def __init__(
        self,
        *,
        config: MultiGenerationRunConfigR5,
        run_state: MultiGenerationRunStateR5,
        phase_state: LongRunState,
        generation_state: GenerationStateStore,
    ):
        config.validate()
        self.config = config
        self.run_state = run_state
        self.phase_controller = LongRunGenerationController(phase_state)
        self.generation_state = generation_state
        self.run_state.initialize(config)

    def _stop_reason(self) -> str | None:
        s = self.run_state.summary()
        if s["attempts"] >= self.config.max_attempts:
            return "MAX_ATTEMPTS"
        if s["promotions"] >= self.config.max_promotions:
            return "MAX_PROMOTIONS"
        if s["consecutive_rejects"] >= self.config.max_consecutive_rejects:
            return "MAX_CONSECUTIVE_REJECTS"
        return None

    def _cycle_for(self, attempt_index: int, parent_generation: int, parent_hash: str) -> GenerationCycleSpec:
        cycle_id = (
            f"{self.config.run_id}:ATTEMPT:{attempt_index:06d}:"
            f"G{parent_generation}:{parent_hash[:12]}"
        )
        return GenerationCycleSpec(
            cycle_id=cycle_id,
            generation_parent=parent_generation,
            parent_policy_hash=parent_hash,
            experiment_version=self.config.experiment_version,
            dataset_hash=self.config.dataset_hash,
            split_hash=self.config.split_hash,
            physics_hash=self.config.physics_hash,
            supervisor_hash=self.config.supervisor_hash,
            teacher_hash=self.config.teacher_hash,
            promotion_rule_hash=self.config.promotion_rule_hash,
            plugins=self.config.phase_plugins,
        )

    def run(self, *, shared_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        while True:
            # Recovery has priority over admission of a new attempt.  An attempt that was
            # already registered before a crash must be allowed to finish even when its
            # registration made attempts == max_attempts.  Stop limits apply only when
            # there is no ACTIVE attempt left to recover.
            active = self.run_state.active_attempt()
            if active is None:
                stop = self._stop_reason()
                if stop:
                    self.run_state.set_state("STOPPED:" + stop)
                    break

            champion = self.generation_state.current_champion()
            if champion is None:
                raise RuntimeError("NO_CHAMPION_FOR_MULTI_GENERATION_RUN")

            if active:
                cycle = self._cycle_for(
                    active["attempt_index"],
                    active["parent_generation"],
                    active["parent_weight_hash"],
                )
                if cycle.content_hash != active["cycle_hash"]:
                    raise RuntimeError("ACTIVE_CYCLE_IDENTITY_MISMATCH")
                attempt_idx = active["attempt_index"]
            else:
                attempt_idx = self.run_state.summary()["attempts"]
                cycle = self._cycle_for(attempt_idx, champion.generation, champion.weight_hash)
                self.run_state.register_attempt(cycle)

            self.phase_controller.run_cycle(cycle, shared_context=shared_context)

            after = self.generation_state.current_champion()
            if after is None:
                raise RuntimeError("CHAMPION_DISAPPEARED")
            if after.weight_hash != cycle.parent_policy_hash:
                if after.generation <= cycle.generation_parent:
                    raise RuntimeError("PROMOTION_DID_NOT_ADVANCE_GENERATION")
                outcome = "PROMOTE"
            else:
                if after.generation != cycle.generation_parent:
                    raise RuntimeError("REJECT_CHANGED_GENERATION_WITH_SAME_HASH")
                outcome = "REJECT"

            self.run_state.complete_attempt(
                attempt_index=attempt_idx,
                outcome=outcome,
                resulting_generation=after.generation,
                resulting_weight_hash=after.weight_hash,
            )

        return self.run_state.summary()


# TEST_ONLY plugin used to prove the unattended multi-generation state machine.
def smoke_promoting_phase_plugin_r5(*, phase, cycle_spec, plugin_config, context):
    from .generation_orchestrator import (
        ChallengerAttempt,
        GenerationStateStore,
        PolicyRecord,
        PromotionRule,
        TournamentResult,
    )
    state_path = plugin_config["generation_state_path"]
    gs = GenerationStateStore(state_path)
    attempt_id = "SMOKE:" + cycle_spec.cycle_id
    try:
        if phase == "TRAIN_CHALLENGER":
            attempt = ChallengerAttempt(
                attempt_id=attempt_id,
                parent_generation=cycle_spec.generation_parent,
                parent_weight_hash=cycle_spec.parent_policy_hash,
                training_snapshot_hash="SMOKE_SNAPSHOT:" + cycle_spec.cycle_id,
                experiment_version=cycle_spec.experiment_version,
                architecture_hash="SMOKE_ARCH",
                tier="TIER_1",
            )
            gs.create_attempt(attempt)
            challenger_hash = canonical_hash({
                "cycle_id": cycle_spec.cycle_id,
                "parent": cycle_spec.parent_policy_hash,
                "kind": "SMOKE_CHALLENGER",
            })
            challenger = PolicyRecord(
                generation=cycle_spec.generation_parent + 1,
                weight_hash=challenger_hash,
                architecture_hash="SMOKE_ARCH",
                tier="TIER_1",
                checkpoint_path=f"{challenger_hash}.pt",
                parent_weight_hash=cycle_spec.parent_policy_hash,
                training_snapshot_hash=attempt.training_snapshot_hash,
            )
            gs.record_challenger(attempt_id, challenger)
            return {"fixture": "TEST_ONLY", "challenger_weight_hash": challenger_hash}

        if phase == "TOURNAMENT":
            row = gs.db.execute(
                "SELECT challenger_hash FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if row is None or not row[0]:
                raise RuntimeError("SMOKE_CHALLENGER_MISSING")
            result = TournamentResult(
                attempt_id=attempt_id,
                champion_weight_hash=cycle_spec.parent_policy_hash,
                challenger_weight_hash=row[0],
                evaluation_dataset_hash="SMOKE_EVAL",
                mean_utility_champion=0.0,
                mean_utility_challenger=0.01,
                delta_utility=0.01,
                bootstrap_ci_low=0.001,
                bootstrap_ci_high=0.02,
                independent_groups=10,
                regime_deltas={"R0": 0.01, "R1": 0.01},
            )
            gs.record_tournament(result)
            return {"fixture": "TEST_ONLY", "tournament_hash": result.content_hash}

        if phase == "ADJUDICATE_COMMIT":
            verdict, reasons = gs.adjudicate(
                attempt_id,
                PromotionRule(
                    rule_id="SMOKE_PROMOTION_R5",
                    min_delta_utility=0.0,
                    min_ci_lower=0.0,
                    min_independent_groups=1,
                    max_bad_regime_fraction=1.0,
                ),
            )
            gs.commit(attempt_id)
            champ = gs.current_champion()
            return {
                "fixture": "TEST_ONLY",
                "verdict": verdict,
                "reasons": list(reasons),
                "champion_generation": champ.generation,
                "champion_weight_hash": champ.weight_hash,
            }

        return {
            "fixture": "TEST_ONLY",
            "phase": phase,
            "parent_generation": cycle_spec.generation_parent,
            "parent_policy_hash": cycle_spec.parent_policy_hash,
        }
    finally:
        gs.close()
