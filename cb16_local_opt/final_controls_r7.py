from __future__ import annotations

"""
Single-use FINAL TOURNAMENT F0/F1/F2/F3 distributional Examiner.

R6 kept the final TOURNAMENT closed during iterative generations. R7 adds the missing
one-time distributional control report.

Protocol:
- bind exact candidate, immutable constitution, cache, TRAIN sample file and control protocol;
- register OPENED before reading final TOURNAMENT market-value bytes;
- create a fresh account cohort at the first TOURNAMENT decision timestamp;
- run the frozen candidate sequentially through TOURNAMENT;
- at frozen TOURNAMENT anchor groups, capture pre-outcome Market64+Account6 contexts;
- compile same-future H72 action/risk utilities with the vectorized group compiler;
- fit/evaluate F0/F1/F2/F3 using TRAIN dependence groups only;
- paired bootstrap operates on final TOURNAMENT dependence groups;
- complete once; a different candidate/protocol cannot reuse the guard.

This Examiner is diagnostic and does not mutate the Champion registry.
"""

import dataclasses
import hashlib
import json
import math
import os
import sqlite3
import time
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Any,Mapping,Sequence

import numpy as np

from .chronology_constitution_r6 import HistoricalChronologyConstitutionR6
from .generation_orchestrator import PolicyRecord
from .h72_group_compiler_r7 import (
    GroupAnchorBatchR7,
    VectorizedGroupH72CompilerR7,
)
from .historical_campaign_plugins_r6 import (
    LocalSupervisorConfigR6,
    TrainingConfigR6,
    _constitution_from_json,
    _deterministic_actions,
    _load_policy_model,
    _sample_from_json,
    load_campaign_config,
)
from .market_cache_r6 import MarketLatentCacheR6
from .probabilistic_teacher_r6 import DependenceAwareTeacherConfigR6
from .scientific_controls_r6 import (
    DependenceAwareControlSuiteConfigR6,
    DependenceAwareHistoricalControlSuiteR6,
)
from .trajectory_compiler_r6 import EconomicClockR6,default_action_grid_r6
from .vectorized_physics import (
    AccountBatchState,
    MarketBar,
    VectorPhysicsConfig,
    VectorizedPhysics,
)


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj):obj=asdict(obj)
    return hashlib.sha256(
        json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path:str|Path,chunk:int=8<<20)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(chunk),b""):h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class FinalControlProtocolR7:
    protocol_version:str="CB16_FINAL_DISTRIBUTIONAL_CONTROLS_R7"
    account_reset_mode:str="FRESH_AT_TOURNAMENT_START"
    require_all_constitution_groups:bool=True

    def validate(self):
        if self.account_reset_mode!="FRESH_AT_TOURNAMENT_START":
            raise ValueError("unsupported final control account reset mode")

    @property
    def content_hash(self):
        self.validate();return canonical_hash(self)


@dataclass(frozen=True)
class FinalControlIdentityR7:
    dataset_hash:str
    constitution_hash:str
    tournament_group_hash:str
    market_cache_identity:str
    candidate_policy_hash:str
    train_samples_sha256:str
    teacher_protocol_hash:str
    control_protocol_hash:str
    final_protocol_hash:str
    physics_hash:str
    supervisor_hash:str

    @property
    def content_hash(self):return canonical_hash(self)


@dataclass(frozen=True)
class FinalControlReceiptR7:
    identity_hash:str
    candidate_policy_hash:str
    tournament_dependence_groups:int
    tournament_parent_contexts:int
    tournament_counterfactual_branches:int
    control_suite_hash:str
    control_suite_status:str
    f2_minus_f0_mean_delta:float
    f2_minus_f0_ci_low:float
    f2_minus_f0_ci_high:float
    f3_minus_f2_mean_delta:float
    f3_minus_f2_ci_low:float
    f3_minus_f2_ci_high:float
    final_tournament_opened:bool
    result_path:str
    result_sha256:str

    @property
    def content_hash(self):return canonical_hash(self)


class FinalControlGuardR7:
    def __init__(self,path:str|Path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,isolation_level=None,timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS final_controls(
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            identity_hash TEXT NOT NULL,
            identity_json BLOB NOT NULL,
            state TEXT NOT NULL,
            receipt_json BLOB,
            result_hash TEXT,
            opened_at REAL NOT NULL,
            completed_at REAL
        );
        """)

    def close(self):self.db.close()

    def begin(self,identity:FinalControlIdentityR7):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row=self.db.execute(
                "SELECT identity_hash,state FROM final_controls WHERE singleton=1"
            ).fetchone()
            if row is None:
                self.db.execute(
                    "INSERT INTO final_controls VALUES(1,?,?,'OPENED',NULL,NULL,?,NULL)",
                    (
                        identity.content_hash,
                        json.dumps(asdict(identity),sort_keys=True),
                        time.time(),
                    )
                )
                self.db.execute("COMMIT");return "OPENED_NEW"
            if row[0]!=identity.content_hash:
                raise RuntimeError(
                    "FINAL_CONTROLS_ALREADY_BOUND_TO_DIFFERENT_IDENTITY"
                )
            self.db.execute("COMMIT");return row[1]
        except Exception:
            self.db.execute("ROLLBACK");raise

    def completed(self,identity):
        row=self.db.execute(
            "SELECT identity_hash,state,receipt_json FROM final_controls WHERE singleton=1"
        ).fetchone()
        if row is None:return None
        if row[0]!=identity.content_hash:
            raise RuntimeError("FINAL_CONTROLS_IDENTITY_CONFLICT")
        if row[1]!="COMPLETED":return None
        return FinalControlReceiptR7(**json.loads(row[2]))

    def complete(self,identity,receipt):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row=self.db.execute(
                "SELECT identity_hash,state,result_hash FROM final_controls WHERE singleton=1"
            ).fetchone()
            if row is None or row[0]!=identity.content_hash:
                raise RuntimeError("FINAL_CONTROLS_NOT_OPENED")
            if row[1]=="COMPLETED":
                if row[2]!=receipt.result_sha256:
                    raise RuntimeError("FINAL_CONTROLS_RESULT_REWRITE_CONFLICT")
                self.db.execute("COMMIT");return
            self.db.execute(
                """
                UPDATE final_controls SET state='COMPLETED',receipt_json=?,
                    result_hash=?,completed_at=? WHERE singleton=1
                """,
                (
                    json.dumps(asdict(receipt),sort_keys=True),
                    receipt.result_sha256,time.time(),
                )
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK");raise


def _account_group_anchor(
    *,
    state:AccountBatchState,
    account6:np.ndarray,
    market_latent:np.ndarray,
    group_id:str,
    decision_index:int,
    timestamp:int,
    candidate:PolicyRecord,
    cache_identity:str,
)->GroupAnchorBatchR7:
    k=state.n
    parent_ids=tuple(
        f"FINAL:{candidate.weight_hash[:12]}:T{timestamp}:A{i:04d}"
        for i in range(k)
    )
    contexts=tuple(f"FINALCTX:{p}" for p in parent_ids)
    features=np.concatenate([
        np.broadcast_to(
            np.asarray(market_latent,dtype=np.float32),
            (k,64)
        ),
        np.asarray(account6,dtype=np.float32),
    ],axis=1)
    return GroupAnchorBatchR7(
        dependence_group_id=group_id,
        decision_index=decision_index,
        parent_ids=parent_ids,
        student_context_object_ids=contexts,
        context_features=features,
        balance=state.balance.copy(),
        position_qty=state.position_qty.copy(),
        entry_price=state.entry_price.copy(),
        peak_equity=state.peak_equity.copy(),
        realized_pnl=state.realized_pnl.copy(),
        margin_used=state.margin_used.copy(),
        holding_bars=state.holding_bars.copy(),
        risk_budget_remaining=state.risk_budget_remaining.copy(),
        risk_budget_capacity=state.risk_budget_capacity.copy(),
        terminated=state.terminated.copy(),
        last_mark_price=state.last_mark_price.copy(),
        market_lineage_hash=cache_identity,
    )


def run_final_distributional_controls_r7(
    *,
    campaign_config_path:str|Path,
    constitution_path:str|Path,
    train_samples_path:str|Path,
    candidate:PolicyRecord,
    guard:FinalControlGuardR7,
    output_path:str|Path,
    protocol:FinalControlProtocolR7|None=None,
)->FinalControlReceiptR7:
    protocol=protocol or FinalControlProtocolR7()
    protocol.validate()
    cfg=load_campaign_config(campaign_config_path)
    constitution=_constitution_from_json(
        json.loads(Path(constitution_path).read_text())
    )
    cache=MarketLatentCacheR6(cfg.market_cache_root)
    physics_cfg=VectorPhysicsConfig(**dict(cfg.physics))
    supervisor=LocalSupervisorConfigR6(**dict(cfg.supervisor))
    teacher_cfg=DependenceAwareTeacherConfigR6(**dict(cfg.teacher))
    control_kwargs=dict(cfg.controls)
    control_kwargs["teacher"]=teacher_cfg
    control_cfg=DependenceAwareControlSuiteConfigR6(**control_kwargs)

    identity=FinalControlIdentityR7(
        dataset_hash=cache.receipt.dataset_hash,
        constitution_hash=constitution.content_hash,
        tournament_group_hash=constitution.tournament.group_hash,
        market_cache_identity=cache.receipt.scientific_identity_hash,
        candidate_policy_hash=candidate.weight_hash,
        train_samples_sha256=sha256_file(train_samples_path),
        teacher_protocol_hash=teacher_cfg.content_hash,
        control_protocol_hash=control_cfg.content_hash,
        final_protocol_hash=protocol.content_hash,
        physics_hash=physics_cfg.config_hash,
        supervisor_hash=supervisor.content_hash,
    )
    guard.begin(identity)
    old=guard.completed(identity)
    if old is not None:return old

    # FINAL TOURNAMENT is now OPEN for this exact identity.
    ts=cache.arrays["timestamp"]
    start=int(np.searchsorted(
        ts,constitution.tournament.first_decision_timestamp,side="left"
    ))
    last_decision=int(np.searchsorted(
        ts,constitution.tournament.last_decision_timestamp,side="left"
    ))
    group_set=set(constitution.tournament.group_ids)

    model,dev=_load_policy_model(
        candidate,
        device=TrainingConfigR6(**dict(cfg.training)).device,
    )
    engine=VectorizedPhysics(physics_cfg)
    state=AccountBatchState.empty(
        cfg.account_replicas,physics_cfg,account_prefix="FINAL"
    )
    compiler=VectorizedGroupH72CompilerR7(
        physics_cfg,clock=EconomicClockR6(horizon_bars=72)
    )
    candidates=default_action_grid_r6(cfg.risk_levels)
    tournament_samples=[]
    seen_groups=[]

    for t in range(start,last_decision+1):
        if not bool(cache.arrays["latent_valid"][t]):
            continue
        account6=engine.account_observation6(
            state,float(cache.arrays["close"][t])
        )
        market_lat=np.asarray(
            cache.arrays["market_latent"][t],dtype=np.float32
        )
        gid=f"DG:{int(ts[t])}"
        if gid in group_set:
            anchor=_account_group_anchor(
                state=state,
                account6=account6,
                market_latent=market_lat,
                group_id=gid,
                decision_index=t,
                timestamp=int(ts[t]),
                candidate=candidate,
                cache_identity=cache.receipt.scientific_identity_hash,
            )
            compiled=compiler.compile(
                cache.market_path(),anchor,candidates
            )
            tournament_samples.extend(compiled.to_teacher_samples())
            seen_groups.append(gid)

        # Actual candidate account path moves one next bar.
        latent_batch=np.broadcast_to(
            market_lat,(cfg.account_replicas,64)
        )
        d,r=_deterministic_actions(
            model,dev,latent_batch,account6,supervisor
        )
        if t+1>=cache.receipt.rows:break
        engine.step(
            state,
            MarketBar(
                open=float(cache.arrays["open"][t+1]),
                high=float(cache.arrays["high"][t+1]),
                low=float(cache.arrays["low"][t+1]),
                close=float(cache.arrays["close"][t+1]),
                funding_rate=float(cache.arrays["funding_rate"][t+1]),
            ),
            executable_direction=d,
            executable_risk=r,
            requested_direction=d,
            dependence_group_count=1,
        )

    if protocol.require_all_constitution_groups:
        expected=set(constitution.tournament.group_ids)
        if set(seen_groups)!=expected:
            missing=sorted(expected-set(seen_groups))
            raise RuntimeError(
                f"FINAL_CONTROL_TOURNAMENT_GROUPS_MISSING:{len(missing)}"
            )

    train_samples=[
        _sample_from_json(dict(x))
        for x in (
            json.loads(line)
            for line in Path(train_samples_path).read_text().splitlines()
            if line.strip()
        )
    ]
    suite=DependenceAwareHistoricalControlSuiteR6(control_cfg)
    controls=suite.evaluate(
        train_samples+tournament_samples,
        target_parent_ids=sorted({
            s.parent_id for s in tournament_samples
        }),
        eligible_train_dependence_group_ids=constitution.train.group_ids,
    )

    result={
        "schema":"CB16_FINAL_DISTRIBUTIONAL_CONTROLS_RESULT_R7",
        "identity":asdict(identity),
        "protocol":asdict(protocol),
        "candidate":asdict(candidate),
        "tournament_dependence_groups":len(set(
            s.dependence_group_id for s in tournament_samples
        )),
        "tournament_parent_contexts":len(set(
            s.parent_id for s in tournament_samples
        )),
        "tournament_counterfactual_branches":len(tournament_samples),
        "control_suite":asdict(controls),
        "control_suite_hash":controls.content_hash,
        "final_tournament_opened":True,
    }
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_name(out.name+f".{os.getpid()}.partial")
    tmp.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    os.replace(tmp,out)
    result_sha=sha256_file(out)
    receipt=FinalControlReceiptR7(
        identity_hash=identity.content_hash,
        candidate_policy_hash=candidate.weight_hash,
        tournament_dependence_groups=result["tournament_dependence_groups"],
        tournament_parent_contexts=result["tournament_parent_contexts"],
        tournament_counterfactual_branches=len(tournament_samples),
        control_suite_hash=controls.content_hash,
        control_suite_status=controls.status,
        f2_minus_f0_mean_delta=controls.f2_minus_f0.mean_delta,
        f2_minus_f0_ci_low=controls.f2_minus_f0.ci_low,
        f2_minus_f0_ci_high=controls.f2_minus_f0.ci_high,
        f3_minus_f2_mean_delta=controls.f3_minus_f2.mean_delta,
        f3_minus_f2_ci_low=controls.f3_minus_f2.ci_low,
        f3_minus_f2_ci_high=controls.f3_minus_f2.ci_high,
        final_tournament_opened=True,
        result_path=str(out),
        result_sha256=result_sha,
    )
    guard.complete(identity,receipt)
    return receipt
