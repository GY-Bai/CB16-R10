from __future__ import annotations

"""
Durable long-run generation controller.

The controller is intentionally orchestration-only. Scientific logic remains in versioned
components. Every phase returns an immutable receipt dictionary whose content hash is
persisted before the next phase may start.

Crash/restart rule:
- COMPLETED phase + matching cycle identity => replay reads receipt, does not rerun side effects.
- RUNNING phase after crash => phase is rerun; underlying components must be idempotent by identity.
- same cycle/phase id with different receipt content => fail closed.
"""

import dataclasses
import hashlib
import importlib
import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PHASES=(
    "ROLLOUT",
    "TEACHER_CREDIT",
    "SEAL_SNAPSHOT",
    "TRAIN_CHALLENGER",
    "TOURNAMENT",
    "ADJUDICATE_COMMIT",
    "RETENTION",
)


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj): obj=asdict(obj)
    return hashlib.sha256(
        json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PhasePlugin:
    dotted_callable:str
    version:str
    config:Mapping[str,Any]

    @property
    def identity_hash(self)->str:
        return canonical_hash({
            "dotted_callable":self.dotted_callable,
            "version":self.version,
            "config":dict(self.config),
        })


@dataclass(frozen=True)
class GenerationCycleSpec:
    cycle_id:str
    generation_parent:int
    parent_policy_hash:str
    experiment_version:str
    dataset_hash:str
    split_hash:str
    physics_hash:str
    supervisor_hash:str
    teacher_hash:str
    promotion_rule_hash:str
    plugins:Mapping[str,PhasePlugin]

    @property
    def content_hash(self)->str:
        return canonical_hash({
            "cycle_id":self.cycle_id,
            "generation_parent":self.generation_parent,
            "parent_policy_hash":self.parent_policy_hash,
            "experiment_version":self.experiment_version,
            "dataset_hash":self.dataset_hash,
            "split_hash":self.split_hash,
            "physics_hash":self.physics_hash,
            "supervisor_hash":self.supervisor_hash,
            "teacher_hash":self.teacher_hash,
            "promotion_rule_hash":self.promotion_rule_hash,
            "plugins":{k:v.identity_hash for k,v in sorted(self.plugins.items())},
        })

    def validate(self):
        missing=set(PHASES)-set(self.plugins)
        if missing: raise ValueError(f"missing phase plugins:{sorted(missing)}")


@dataclass(frozen=True)
class PhaseReceipt:
    cycle_id:str
    phase:str
    cycle_hash:str
    plugin_hash:str
    payload:Mapping[str,Any]

    @property
    def content_hash(self)->str:
        return canonical_hash({
            "cycle_id":self.cycle_id,
            "phase":self.phase,
            "cycle_hash":self.cycle_hash,
            "plugin_hash":self.plugin_hash,
            "payload":dict(self.payload),
        })


def _resolve(path:str):
    module,name=path.split(":",1)
    fn=getattr(importlib.import_module(module),name)
    if not callable(fn): raise TypeError("phase plugin not callable")
    return fn


class LongRunState:
    def __init__(self,path:str|Path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,isolation_level=None,timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS cycles(
            cycle_id TEXT PRIMARY KEY,
            cycle_hash TEXT NOT NULL,
            spec_json BLOB NOT NULL,
            state TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS phases(
            cycle_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            state TEXT NOT NULL,
            plugin_hash TEXT NOT NULL,
            receipt_hash TEXT,
            receipt_json BLOB,
            attempts INTEGER NOT NULL DEFAULT 0,
            started_at REAL,
            completed_at REAL,
            PRIMARY KEY(cycle_id,phase)
        );
        """)
    def close(self): self.db.close()

    def register_cycle(self,spec:GenerationCycleSpec):
        spec.validate(); now=time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old=self.db.execute("SELECT cycle_hash FROM cycles WHERE cycle_id=?",(spec.cycle_id,)).fetchone()
            if old:
                if old[0]!=spec.content_hash: raise RuntimeError("CYCLE_ID_CONTENT_CONFLICT")
                self.db.execute("COMMIT"); return
            self.db.execute(
                "INSERT INTO cycles VALUES(?,?,?,?,?,?)",
                (spec.cycle_id,spec.content_hash,json.dumps(asdict(spec),sort_keys=True),"ACTIVE",now,now)
            )
            for i,p in enumerate(PHASES):
                self.db.execute(
                    "INSERT INTO phases(cycle_id,phase,ordinal,state,plugin_hash) VALUES(?,?,?,?,?)",
                    (spec.cycle_id,p,i,"PENDING",spec.plugins[p].identity_hash)
                )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK"); raise

    def phase_row(self,cycle_id:str,phase:str):
        return self.db.execute(
            "SELECT state,plugin_hash,receipt_hash,receipt_json,attempts FROM phases WHERE cycle_id=? AND phase=?",
            (cycle_id,phase)
        ).fetchone()

    def mark_running(self,cycle_id:str,phase:str):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row=self.phase_row(cycle_id,phase)
            if row is None: raise RuntimeError("UNKNOWN_PHASE")
            if row[0]=="COMPLETED":
                self.db.execute("COMMIT"); return
            # All prior phases must be completed.
            ord_=PHASES.index(phase)
            incomplete=self.db.execute(
                "SELECT phase FROM phases WHERE cycle_id=? AND ordinal<? AND state!='COMPLETED'",
                (cycle_id,ord_)
            ).fetchall()
            if incomplete: raise RuntimeError(f"PRIOR_PHASE_INCOMPLETE:{incomplete}")
            self.db.execute(
                "UPDATE phases SET state='RUNNING',attempts=attempts+1,started_at=? WHERE cycle_id=? AND phase=?",
                (time.time(),cycle_id,phase)
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK"); raise

    def complete(self,receipt:PhaseReceipt):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row=self.phase_row(receipt.cycle_id,receipt.phase)
            if row is None: raise RuntimeError("UNKNOWN_PHASE")
            if row[1]!=receipt.plugin_hash: raise RuntimeError("PHASE_PLUGIN_IDENTITY_MISMATCH")
            if row[0]=="COMPLETED":
                if row[2]!=receipt.content_hash: raise RuntimeError("PHASE_RECEIPT_REWRITE_CONFLICT")
                self.db.execute("COMMIT"); return
            self.db.execute(
                """
                UPDATE phases SET state='COMPLETED',receipt_hash=?,receipt_json=?,completed_at=?
                WHERE cycle_id=? AND phase=?
                """,
                (
                    receipt.content_hash,json.dumps(asdict(receipt),sort_keys=True),time.time(),
                    receipt.cycle_id,receipt.phase
                )
            )
            left=self.db.execute(
                "SELECT COUNT(*) FROM phases WHERE cycle_id=? AND state!='COMPLETED'",
                (receipt.cycle_id,)
            ).fetchone()[0]
            if left==0:
                self.db.execute(
                    "UPDATE cycles SET state='COMPLETED',updated_at=? WHERE cycle_id=?",
                    (time.time(),receipt.cycle_id)
                )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK"); raise

    def receipt(self,cycle_id:str,phase:str)->PhaseReceipt|None:
        row=self.phase_row(cycle_id,phase)
        if not row or row[0]!="COMPLETED": return None
        obj=json.loads(row[3])
        return PhaseReceipt(**obj)

    def status(self,cycle_id:str)->dict[str,Any]:
        cycle=self.db.execute("SELECT cycle_hash,state FROM cycles WHERE cycle_id=?",(cycle_id,)).fetchone()
        if cycle is None: raise RuntimeError("UNKNOWN_CYCLE")
        rows=self.db.execute(
            "SELECT phase,state,attempts,receipt_hash FROM phases WHERE cycle_id=? ORDER BY ordinal",
            (cycle_id,)
        ).fetchall()
        return {
            "cycle_id":cycle_id,"cycle_hash":cycle[0],"state":cycle[1],
            "phases":[{"phase":r[0],"state":r[1],"attempts":int(r[2]),"receipt_hash":r[3]} for r in rows]
        }


class LongRunGenerationController:
    def __init__(self,state:LongRunState):
        self.state=state

    def run_cycle(
        self,
        spec:GenerationCycleSpec,
        *,
        shared_context:Mapping[str,Any]|None=None,
        stop_after_phase:str|None=None,
    )->dict[str,PhaseReceipt]:
        self.state.register_cycle(spec)
        context=dict(shared_context or {})
        outputs={}
        for phase in PHASES:
            prior=self.state.receipt(spec.cycle_id,phase)
            if prior is not None:
                outputs[phase]=prior
                context[phase]=dict(prior.payload)
                if stop_after_phase==phase: break
                continue

            plugin=spec.plugins[phase]
            self.state.mark_running(spec.cycle_id,phase)
            fn=_resolve(plugin.dotted_callable)
            payload=fn(
                phase=phase,
                cycle_spec=spec,
                plugin_config=dict(plugin.config),
                context=dict(context),
            )
            if not isinstance(payload,Mapping):
                raise RuntimeError("PHASE_PLUGIN_MUST_RETURN_MAPPING")
            receipt=PhaseReceipt(
                cycle_id=spec.cycle_id,
                phase=phase,
                cycle_hash=spec.content_hash,
                plugin_hash=plugin.identity_hash,
                payload=dict(payload),
            )
            self.state.complete(receipt)
            outputs[phase]=receipt
            context[phase]=dict(payload)
            if stop_after_phase==phase: break
        return outputs


# Built-in TEST_ONLY lifecycle plugin for conformance/recovery tests.
def smoke_phase_plugin(*,phase,cycle_spec,plugin_config,context):
    seq=PHASES.index(phase)
    return {
        "fixture":"TEST_ONLY",
        "phase":phase,
        "ordinal":seq,
        "parent_policy_hash":cycle_spec.parent_policy_hash,
        "prior_receipt_hashes":{
            p:canonical_hash(context[p]) for p in PHASES[:seq] if p in context
        },
    }
