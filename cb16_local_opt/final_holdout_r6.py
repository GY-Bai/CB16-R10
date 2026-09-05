from __future__ import annotations

"""
Single-use final TOURNAMENT holdout adjudication.

The iterative multi-generation loop uses VALIDATION only. This module is the only intended
historical-R&D path that opens the immutable TOURNAMENT market-value interval.

Safety:
- the holdout identity is registered BEFORE reading tournament market values;
- a completed holdout cannot be rerun under another candidate/baseline/config;
- crash after OPENED may resume only the exact same identity;
- completion writes an immutable result hash;
- this evaluator never mutates the Champion registry.
"""

import dataclasses
import hashlib
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .generation_orchestrator import PolicyRecord
from .historical_campaign_plugins_r6 import (
    HistoricalCampaignConfigR6,
    LocalSupervisorConfigR6,
    TrainingConfigR6,
    _bootstrap_ci,
    _constitution_from_json,
    _simulate_policy_block,
    load_campaign_config,
)
from .market_cache_r6 import MarketLatentCacheR6
from .vectorized_physics import VectorPhysicsConfig


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj=asdict(obj)
    return hashlib.sha256(
        json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class FinalHoldoutProtocolR6:
    protocol_version: str = "CB16_FINAL_HISTORICAL_HOLDOUT_R6"
    block_bars: int = 96
    bootstrap_reps: int = 5000
    min_delta_utility: float = 0.0
    min_ci_lower: float = 0.0
    minimum_independent_blocks: int = 4

    def validate(self):
        if self.block_bars <= 1:
            raise ValueError("block_bars")
        if self.bootstrap_reps <= 0:
            raise ValueError("bootstrap_reps")
        if self.minimum_independent_blocks <= 1:
            raise ValueError("minimum_independent_blocks")

    @property
    def content_hash(self):
        self.validate()
        return canonical_hash(self)


@dataclass(frozen=True)
class FinalHoldoutIdentityR6:
    dataset_hash: str
    constitution_hash: str
    tournament_group_hash: str
    market_cache_identity: str
    baseline_policy_hash: str
    candidate_policy_hash: str
    physics_hash: str
    supervisor_hash: str
    protocol_hash: str

    @property
    def content_hash(self):
        return canonical_hash(self)


@dataclass(frozen=True)
class FinalHoldoutReceiptR6:
    identity_hash: str
    baseline_policy_hash: str
    candidate_policy_hash: str
    independent_blocks: int
    mean_utility_baseline: float
    mean_utility_candidate: float
    mean_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    regime_deltas: Mapping[str,float]
    verdict: str
    tournament_opened: bool
    result_hash: str


class FinalHoldoutGuardR6:
    def __init__(self,path:str|Path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,isolation_level=None,timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS final_holdout(
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            identity_hash TEXT NOT NULL,
            identity_json BLOB NOT NULL,
            state TEXT NOT NULL,
            result_hash TEXT,
            receipt_json BLOB,
            opened_at REAL NOT NULL,
            completed_at REAL
        );
        """)

    def close(self):self.db.close()

    def begin(self,identity:FinalHoldoutIdentityR6)->str:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row=self.db.execute(
                "SELECT identity_hash,state FROM final_holdout WHERE singleton=1"
            ).fetchone()
            if row is None:
                self.db.execute(
                    """
                    INSERT INTO final_holdout(
                        singleton,identity_hash,identity_json,state,opened_at
                    ) VALUES(1,?,?,'OPENED',?)
                    """,
                    (
                        identity.content_hash,
                        json.dumps(asdict(identity),sort_keys=True),
                        time.time(),
                    )
                )
                self.db.execute("COMMIT")
                return "OPENED_NEW"
            if row[0]!=identity.content_hash:
                raise RuntimeError("FINAL_HOLDOUT_ALREADY_BOUND_TO_DIFFERENT_IDENTITY")
            self.db.execute("COMMIT")
            return row[1]
        except Exception:
            self.db.execute("ROLLBACK");raise

    def completed_receipt(self,identity:FinalHoldoutIdentityR6):
        row=self.db.execute(
            "SELECT identity_hash,state,receipt_json FROM final_holdout WHERE singleton=1"
        ).fetchone()
        if row is None:return None
        if row[0]!=identity.content_hash:
            raise RuntimeError("FINAL_HOLDOUT_IDENTITY_CONFLICT")
        if row[1]!="COMPLETED":return None
        return FinalHoldoutReceiptR6(**json.loads(row[2]))

    def complete(self,identity:FinalHoldoutIdentityR6,receipt:FinalHoldoutReceiptR6):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row=self.db.execute(
                "SELECT identity_hash,state,result_hash FROM final_holdout WHERE singleton=1"
            ).fetchone()
            if row is None or row[0]!=identity.content_hash:
                raise RuntimeError("FINAL_HOLDOUT_NOT_OPENED_FOR_IDENTITY")
            if row[1]=="COMPLETED":
                if row[2]!=receipt.result_hash:
                    raise RuntimeError("FINAL_HOLDOUT_RESULT_REWRITE_CONFLICT")
                self.db.execute("COMMIT");return
            self.db.execute(
                """
                UPDATE final_holdout
                SET state='COMPLETED',result_hash=?,receipt_json=?,completed_at=?
                WHERE singleton=1
                """,
                (
                    receipt.result_hash,
                    json.dumps(asdict(receipt),sort_keys=True),
                    time.time(),
                )
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK");raise


def _tournament_blocks(cache,constitution,block_bars):
    ts=cache.arrays["timestamp"]
    start=int(np.searchsorted(
        ts,constitution.tournament.first_decision_timestamp,side="left"
    ))
    end=int(np.searchsorted(
        ts,constitution.tournament.last_maturity_timestamp,side="left"
    ))
    blocks=[]
    i=start
    while i+block_bars<=end:
        blocks.append((i,i+block_bars))
        i+=block_bars
    return blocks


def evaluate_final_holdout_r6(
    *,
    campaign_config:HistoricalCampaignConfigR6,
    constitution_path:str|Path,
    baseline:PolicyRecord,
    candidate:PolicyRecord,
    guard:FinalHoldoutGuardR6,
    protocol:FinalHoldoutProtocolR6|None=None,
)->FinalHoldoutReceiptR6:
    protocol=protocol or FinalHoldoutProtocolR6()
    protocol.validate()

    # Administrative metadata only; no tournament OHLC value access yet.
    constitution=_constitution_from_json(
        json.loads(Path(constitution_path).read_text())
    )
    cache=MarketLatentCacheR6(campaign_config.market_cache_root)
    physics=VectorPhysicsConfig(**dict(campaign_config.physics))
    supervisor=LocalSupervisorConfigR6(**dict(campaign_config.supervisor))
    training=TrainingConfigR6(**dict(campaign_config.training))

    identity=FinalHoldoutIdentityR6(
        dataset_hash=cache.receipt.dataset_hash,
        constitution_hash=constitution.content_hash,
        tournament_group_hash=constitution.tournament.group_hash,
        market_cache_identity=cache.receipt.scientific_identity_hash,
        baseline_policy_hash=baseline.weight_hash,
        candidate_policy_hash=candidate.weight_hash,
        physics_hash=physics.config_hash,
        supervisor_hash=supervisor.content_hash,
        protocol_hash=protocol.content_hash,
    )
    state=guard.begin(identity)
    completed=guard.completed_receipt(identity)
    if completed is not None:
        return completed

    # From this point onward the final holdout is OPENED for this exact identity.
    blocks=_tournament_blocks(cache,constitution,protocol.block_bars)
    if len(blocks)<protocol.minimum_independent_blocks:
        raise RuntimeError(
            f"FINAL_HOLDOUT_INSUFFICIENT_BLOCKS:{len(blocks)}"
        )

    ub=[];uc=[]
    for start,end in blocks:
        ub.append(_simulate_policy_block(
            policy=baseline,cache=cache,start_index=start,end_index=end,
            physics_cfg=physics,supervisor=supervisor,device=training.device
        ))
        uc.append(_simulate_policy_block(
            policy=candidate,cache=cache,start_index=start,end_index=end,
            physics_cfg=physics,supervisor=supervisor,device=training.device
        ))
    if not all(math.isfinite(x) for x in ub+uc):
        raise RuntimeError(
            "FINAL_HOLDOUT_BANKRUPTCY_REQUIRES_EXPLICIT_ADJUDICATION_POLICY"
        )
    delta=np.asarray(uc)-np.asarray(ub)
    lo,hi=_bootstrap_ci(
        delta,protocol.bootstrap_reps,
        seed=int(identity.content_hash[:16],16)
    )
    chunks=np.array_split(np.arange(len(delta)),min(4,len(delta)))
    regimes={
        f"CHRONO_Q{i}":float(np.mean(delta[idx]))
        for i,idx in enumerate(chunks) if len(idx)
    }
    verdict=(
        "FINAL_HOLDOUT_PASS"
        if float(np.mean(delta))>=protocol.min_delta_utility
        and lo>=protocol.min_ci_lower
        else "FINAL_HOLDOUT_FAIL"
    )
    base_payload={
        "identity_hash":identity.content_hash,
        "baseline_policy_hash":baseline.weight_hash,
        "candidate_policy_hash":candidate.weight_hash,
        "independent_blocks":len(blocks),
        "mean_utility_baseline":float(np.mean(ub)),
        "mean_utility_candidate":float(np.mean(uc)),
        "mean_delta":float(np.mean(delta)),
        "bootstrap_ci_low":lo,
        "bootstrap_ci_high":hi,
        "regime_deltas":regimes,
        "verdict":verdict,
        "tournament_opened":True,
    }
    result_hash=canonical_hash(base_payload)
    receipt=FinalHoldoutReceiptR6(
        **base_payload,
        result_hash=result_hash,
    )
    guard.complete(identity,receipt)
    return receipt
