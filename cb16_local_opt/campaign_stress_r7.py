from __future__ import annotations

"""
Long-campaign crash/restart stress harness.

This is infrastructure falsification, not a market experiment.

The harness can inject one synthetic process crash after a phase has performed its durable
side effect but before the phase receipt is committed. On restart the same cycle/phase is
re-entered and must be idempotent.

Critical injected locations:
- TRAIN_CHALLENGER after attempt/challenger persistence;
- TOURNAMENT after TournamentResult persistence;
- ADJUDICATE_COMMIT after Champion CAS/commit.

A crash campaign and a clean campaign with the same deterministic identity must converge to
the same final Champion hash and generation DAG.
"""

import dataclasses
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Any,Mapping,Sequence

from .generation_orchestrator import (
    ChallengerAttempt,
    GenerationStateStore,
    PolicyRecord,
    PromotionRule,
    TournamentResult,
)
from .long_run_controller import PHASES,PhasePlugin
from .multi_generation_runner_r5 import (
    MultiGenerationRunConfigR5,
    MultiGenerationRunnerR5,
    MultiGenerationRunStateR5,
)
from .long_run_controller import LongRunState


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj):obj=asdict(obj)
    return hashlib.sha256(
        json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    ).hexdigest()


class InjectedCrashR7(RuntimeError):
    pass


class CrashLedgerR7:
    def __init__(self,path:str|Path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,isolation_level=None,timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS crashes(
            cycle_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            consumed INTEGER NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(cycle_id,phase)
        )
        """)

    def close(self):self.db.close()

    def consume_once(self,cycle_id:str,phase:str)->bool:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row=self.db.execute(
                "SELECT consumed FROM crashes WHERE cycle_id=? AND phase=?",
                (cycle_id,phase)
            ).fetchone()
            if row is None:
                self.db.execute(
                    "INSERT INTO crashes VALUES(?,?,1,?)",
                    (cycle_id,phase,time.time())
                )
                self.db.execute("COMMIT")
                return True
            self.db.execute("COMMIT")
            return False
        except Exception:
            self.db.execute("ROLLBACK");raise

    def count(self)->int:
        return int(self.db.execute(
            "SELECT COUNT(*) FROM crashes WHERE consumed=1"
        ).fetchone()[0])


def _attempt_index(cycle_id:str)->int:
    # MultiGenerationRunnerR5 cycle format contains ATTEMPT:000123.
    parts=cycle_id.split(":")
    try:
        i=parts.index("ATTEMPT")
        return int(parts[i+1])
    except Exception as exc:
        raise RuntimeError(f"UNEXPECTED_STRESS_CYCLE_ID:{cycle_id}") from exc


def stress_phase_plugin_r7(*,phase,cycle_spec,plugin_config,context):
    gs=GenerationStateStore(plugin_config["generation_state_path"])
    ledger=CrashLedgerR7(plugin_config["crash_ledger_path"])
    crash_phases=set(plugin_config.get("crash_phases",[]))
    crash_every=max(1,int(plugin_config.get("crash_every_attempts",1)))
    idx=_attempt_index(cycle_spec.cycle_id)
    attempt_id="STRESS:"+cycle_spec.cycle_id
    target_gen=cycle_spec.generation_parent+1
    challenger_hash=canonical_hash({
        "stress_identity":plugin_config["stress_identity"],
        "parent":cycle_spec.parent_policy_hash,
        "generation":target_gen,
    })

    def maybe_crash():
        if (
            phase in crash_phases
            and idx % crash_every == 0
            and ledger.consume_once(cycle_spec.cycle_id,phase)
        ):
            raise InjectedCrashR7(
                f"INJECTED_CRASH:{cycle_spec.cycle_id}:{phase}"
            )

    try:
        if phase=="TRAIN_CHALLENGER":
            gs.create_attempt(ChallengerAttempt(
                attempt_id=attempt_id,
                parent_generation=cycle_spec.generation_parent,
                parent_weight_hash=cycle_spec.parent_policy_hash,
                training_snapshot_hash=canonical_hash({
                    "stress_snapshot":cycle_spec.cycle_id
                }),
                experiment_version=cycle_spec.experiment_version,
                architecture_hash="STRESS_ARCH",
                tier="TIER_1",
            ))
            challenger=PolicyRecord(
                generation=target_gen,
                weight_hash=challenger_hash,
                architecture_hash="STRESS_ARCH",
                tier="TIER_1",
                checkpoint_path=f"{challenger_hash}.pt",
                parent_weight_hash=cycle_spec.parent_policy_hash,
                training_snapshot_hash=canonical_hash({
                    "stress_snapshot":cycle_spec.cycle_id
                }),
            )
            gs.record_challenger(attempt_id,challenger)
            maybe_crash()
            return {
                "fixture":"STRESS_ONLY",
                "challenger_hash":challenger_hash,
                "target_generation":target_gen,
            }

        if phase=="TOURNAMENT":
            row=gs.db.execute(
                "SELECT challenger_hash FROM attempts WHERE attempt_id=?",
                (attempt_id,)
            ).fetchone()
            if row is None or row[0]!=challenger_hash:
                raise RuntimeError("STRESS_CHALLENGER_MISSING")
            result=TournamentResult(
                attempt_id=attempt_id,
                champion_weight_hash=cycle_spec.parent_policy_hash,
                challenger_weight_hash=challenger_hash,
                evaluation_dataset_hash=canonical_hash({
                    "stress_eval":target_gen
                }),
                mean_utility_champion=0.0,
                mean_utility_challenger=0.01,
                delta_utility=0.01,
                bootstrap_ci_low=0.001,
                bootstrap_ci_high=0.02,
                independent_groups=16,
                regime_deltas={"R0":0.01,"R1":0.01},
                status="COMPLETE",
            )
            gs.record_tournament(result)
            maybe_crash()
            return {
                "fixture":"STRESS_ONLY",
                "tournament_hash":result.content_hash,
            }

        if phase=="ADJUDICATE_COMMIT":
            row=gs.db.execute(
                "SELECT state,verdict FROM attempts WHERE attempt_id=?",
                (attempt_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError("STRESS_ATTEMPT_MISSING")
            if row[0]=="COMMITTED":
                verdict=row[1]
            else:
                verdict,_=gs.adjudicate(
                    attempt_id,
                    PromotionRule(
                        rule_id="STRESS_ALWAYS_PROMOTE",
                        min_delta_utility=0,
                        min_ci_lower=0,
                        min_independent_groups=1,
                        max_bad_regime_fraction=1,
                    )
                )
                gs.commit(attempt_id)
            champ=gs.current_champion()
            maybe_crash()
            return {
                "fixture":"STRESS_ONLY",
                "verdict":verdict,
                "champion_generation":champ.generation,
                "champion_hash":champ.weight_hash,
            }

        # Remaining phases are immutable no-op receipts for lifecycle stress.
        maybe_crash()
        return {
            "fixture":"STRESS_ONLY",
            "phase":phase,
            "parent_generation":cycle_spec.generation_parent,
            "parent_hash":cycle_spec.parent_policy_hash,
        }
    finally:
        ledger.close()
        gs.close()


@dataclass(frozen=True)
class StressCampaignReceiptR7:
    generations:int
    injected_crashes:int
    runner_restarts:int
    final_generation:int
    final_champion_hash:str
    committed_attempts:int
    unique_challenger_hashes:int
    status:str

    @property
    def content_hash(self):return canonical_hash(self)


def run_stress_campaign_r7(
    *,
    root:str|Path,
    generations:int=30,
    inject_crashes:bool=True,
    crash_phases:Sequence[str]=(
        "TRAIN_CHALLENGER","TOURNAMENT","ADJUDICATE_COMMIT"
    ),
    crash_every_attempts:int=1,
    stress_identity:str="CB16_R7_STRESS",
)->StressCampaignReceiptR7:
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    generation_path=root/"generation.sqlite"
    gs=GenerationStateStore(generation_path)
    genesis=PolicyRecord(
        generation=0,weight_hash=canonical_hash({
            "stress_identity":stress_identity,"genesis":True
        }),
        architecture_hash="STRESS_ARCH",tier="TIER_1",
        checkpoint_path="GENESIS.pt",
        parent_weight_hash=None,training_snapshot_hash=None,
    )
    gs.initialize_champion(genesis)

    plugins={
        p:PhasePlugin(
            dotted_callable="cb16_local_opt.campaign_stress_r7:stress_phase_plugin_r7",
            version="STRESS_R7",
            config={
                "generation_state_path":str(generation_path),
                "crash_ledger_path":str(root/"crash.sqlite"),
                "crash_phases":(
                    list(crash_phases) if inject_crashes else []
                ),
                "crash_every_attempts":crash_every_attempts,
                "stress_identity":stress_identity,
            },
        )
        for p in PHASES
    }
    cfg=MultiGenerationRunConfigR5(
        run_id="STRESS_RUN",
        experiment_version="STRESS_R7",
        dataset_hash="STRESS_DATASET",
        split_hash="STRESS_SPLIT",
        physics_hash="STRESS_PHYSICS",
        supervisor_hash="STRESS_SUPERVISOR",
        teacher_hash="STRESS_TEACHER",
        promotion_rule_hash="STRESS_PROMOTION",
        phase_plugins=plugins,
        max_attempts=generations,
        max_promotions=generations,
        max_consecutive_rejects=generations+1,
    )
    rs=MultiGenerationRunStateR5(root/"run.sqlite")
    ps=LongRunState(root/"phases.sqlite")
    runner=MultiGenerationRunnerR5(
        config=cfg,run_state=rs,phase_state=ps,generation_state=gs
    )
    restarts=0
    while True:
        try:
            summary=runner.run()
            break
        except InjectedCrashR7:
            restarts+=1
            if restarts>generations*len(PHASES)+10:
                raise RuntimeError("STRESS_EXCESSIVE_RESTARTS")

    champ=gs.current_champion()
    dag=gs.generation_dag()
    committed=[x for x in dag if x["state"]=="COMMITTED"]
    hashes=[
        x["challenger_weight_hash"] for x in committed
        if x["challenger_weight_hash"]
    ]
    ledger=CrashLedgerR7(root/"crash.sqlite")
    crashes=ledger.count();ledger.close()

    # A finished runner is idempotent.
    summary2=runner.run()
    if summary2["attempts"]!=summary["attempts"]:
        raise RuntimeError("STRESS_RERUN_CHANGED_ATTEMPT_COUNT")
    final2=gs.current_champion()
    if final2.weight_hash!=champ.weight_hash:
        raise RuntimeError("STRESS_RERUN_CHANGED_CHAMPION")

    receipt=StressCampaignReceiptR7(
        generations=generations,
        injected_crashes=crashes,
        runner_restarts=restarts,
        final_generation=champ.generation,
        final_champion_hash=champ.weight_hash,
        committed_attempts=len(committed),
        unique_challenger_hashes=len(set(hashes)),
        status=(
            "PASS"
            if champ.generation==generations
            and len(committed)==generations
            and len(set(hashes))==generations
            else "FAIL"
        ),
    )
    rs.close();ps.close();gs.close()
    if receipt.status!="PASS":
        raise RuntimeError(f"STRESS_CAMPAIGN_FAIL:{receipt}")
    return receipt
