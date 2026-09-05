from __future__ import annotations

"""
Process-isolated Teacher/Credit/Evidence farm.

Each job is immutable and lineage-bound. Worker plugins are loaded by dotted path,
which keeps spawn compatibility and prevents arbitrary closure state from being
silently captured.

The built-in smoke plugin is TEST_ONLY. Real market experiments should bind a
versioned Teacher plugin and configuration hash.
"""

import concurrent.futures as cf
import dataclasses
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .sharded_experience_lake import ExperienceObject, ShardedExperienceLake


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj=asdict(obj)
    return hashlib.sha256(
        json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class TeacherPluginSpec:
    dotted_callable: str
    teacher_version: str
    config: Mapping[str,Any]

    @property
    def config_hash(self) -> str:
        return canonical_hash(dict(self.config))

    @property
    def identity_hash(self) -> str:
        return canonical_hash({
            "dotted_callable":self.dotted_callable,
            "teacher_version":self.teacher_version,
            "config_hash":self.config_hash,
        })


@dataclass(frozen=True)
class TeacherJob:
    job_id: str
    trajectory_object_id: str
    trajectory_identity_hash: str
    generation: int
    policy_weight_hash: str
    parent_snapshot_hash: str
    plugin: TeacherPluginSpec
    student_context_object_id: str | None = None

    @property
    def identity_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class CreditPackageR4:
    credit_id: str
    trajectory_object_id: str
    trajectory_identity_hash: str
    teacher_identity_hash: str
    generation: int
    direction_credit: tuple[float,float,float] # SHORT/FLAT/LONG soft evidence
    sizing_target: float
    sizing_weight: float
    diagnostic: Mapping[str,Any]

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class EvidencePackageR4:
    evidence_id: str
    credit_hash: str
    trajectory_object_id: str
    teacher_identity_hash: str
    generation: int
    admitted: bool
    lane: str
    payload: Mapping[str,Any]

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class TeacherResult:
    job_id: str
    credit: CreditPackageR4
    evidence: EvidencePackageR4

    @property
    def content_hash(self) -> str:
        return canonical_hash({
            "job_id":self.job_id,
            "credit_hash":self.credit.content_hash,
            "evidence_hash":self.evidence.content_hash,
        })


def _resolve_callable(path: str):
    if ":" not in path:
        raise ValueError("plugin callable must be module:function")
    module,name=path.split(":",1)
    fn=getattr(importlib.import_module(module),name)
    if not callable(fn):
        raise TypeError("plugin target not callable")
    return fn


def _teacher_worker(
    job: TeacherJob,
    trajectory_payload: Mapping[str,Any],
) -> TeacherResult:
    fn=_resolve_callable(job.plugin.dotted_callable)
    result=fn(
        trajectory_payload=dict(trajectory_payload),
        config=dict(job.plugin.config),
        job=job,
    )
    if not isinstance(result,TeacherResult):
        raise RuntimeError("TEACHER_PLUGIN_RETURN_TYPE_INVALID")
    if result.job_id != job.job_id:
        raise RuntimeError("TEACHER_RESULT_JOB_ID_MISMATCH")
    if result.credit.trajectory_identity_hash != job.trajectory_identity_hash:
        raise RuntimeError("CREDIT_TRAJECTORY_HASH_MISMATCH")
    if result.credit.teacher_identity_hash != job.plugin.identity_hash:
        raise RuntimeError("CREDIT_TEACHER_IDENTITY_MISMATCH")
    if result.evidence.credit_hash != result.credit.content_hash:
        raise RuntimeError("EVIDENCE_CREDIT_HASH_MISMATCH")
    return result


def smoke_probabilistic_teacher(
    *,
    trajectory_payload: Mapping[str,Any],
    config: Mapping[str,Any],
    job: TeacherJob,
) -> TeacherResult:
    """TEST_ONLY soft evidence fixture; never a market-authoritative Teacher.

    It intentionally emits a distribution rather than a deterministic BEST_ACTION label.
    """
    total=float(trajectory_payload.get("total_log_equity_reward",0.0))
    scale=float(config.get("scale",10.0))
    import math
    x=max(-20.0,min(20.0,total*scale))
    p_long=1.0/(1.0+math.exp(-x))
    p_short=1.0-p_long
    p_flat=float(config.get("flat_mass",0.10))
    rem=max(1e-12,1.0-p_flat)
    probs=(p_short*rem,p_flat,p_long*rem)
    s=sum(probs); probs=tuple(float(p/s) for p in probs)
    sizing=max(0.0,min(1.0,float(config.get("sizing_target",0.25))))
    cid=f"CREDIT:{job.job_id}"
    credit=CreditPackageR4(
        credit_id=cid,
        trajectory_object_id=job.trajectory_object_id,
        trajectory_identity_hash=job.trajectory_identity_hash,
        teacher_identity_hash=job.plugin.identity_hash,
        generation=job.generation,
        direction_credit=probs,
        sizing_target=sizing,
        sizing_weight=1.0,
        diagnostic={"fixture":"TEST_ONLY","source_total_log_reward":total},
    )
    evidence=EvidencePackageR4(
        evidence_id=f"EVIDENCE:{job.job_id}",
        credit_hash=credit.content_hash,
        trajectory_object_id=job.trajectory_object_id,
        teacher_identity_hash=job.plugin.identity_hash,
        generation=job.generation,
        admitted=True,
        lane="CENTER",
        payload={
            "direction_target_probs":list(probs),
            "requested_risk_target":sizing,
            "direction_weight":1.0,
            "sizing_weight":1.0,
            "fixture":"TEST_ONLY",
            "student_context_object_id": job.student_context_object_id,
        },
    )
    return TeacherResult(job.job_id,credit,evidence)


@dataclass(frozen=True)
class TeacherFarmConfig:
    workers: int=2
    start_method: str="spawn"
    cpu_threads_per_worker: int=1
    max_in_flight: int=4

    def validate(self):
        if self.workers<=0 or self.cpu_threads_per_worker<=0:
            raise ValueError("worker/thread count")
        if self.start_method not in {"spawn","forkserver"}:
            raise ValueError("fork unsupported")
        if self.max_in_flight<self.workers:
            raise ValueError("max_in_flight < workers")


def _worker_init(threads:int):
    os.environ["OMP_NUM_THREADS"]=str(threads)
    os.environ["MKL_NUM_THREADS"]=str(threads)
    os.environ["OPENBLAS_NUM_THREADS"]=str(threads)


class TeacherCreditFarm:
    def __init__(self, config:TeacherFarmConfig):
        config.validate()
        self.config=config

    def run(
        self,
        jobs:Sequence[TeacherJob],
        *,
        lake:ShardedExperienceLake,
        persist:bool=True,
    ) -> list[TeacherResult]:
        if not jobs:return []
        ids=[j.job_id for j in jobs]
        if len(ids)!=len(set(ids)):
            raise ValueError("duplicate teacher job id")

        payloads=[]
        for j in jobs:
            if not j.student_context_object_id:
                raise RuntimeError("TEACHER_JOB_MISSING_STUDENT_CONTEXT_REF")
            cgot=lake.get(j.student_context_object_id)
            if cgot is None or cgot[0].object_type!="STUDENT_CONTEXT":
                raise RuntimeError("TEACHER_JOB_STUDENT_CONTEXT_NOT_FOUND")
            got=lake.get(j.trajectory_object_id)
            if got is None:
                raise RuntimeError(f"TRAJECTORY_NOT_FOUND:{j.trajectory_object_id}")
            ref,payload=got
            if ref.identity_hash != j.trajectory_identity_hash:
                raise RuntimeError("TEACHER_JOB_TRAJECTORY_IDENTITY_MISMATCH")
            payloads.append(payload)

        ctx=mp.get_context(self.config.start_method)
        pending={}
        results={}
        with cf.ProcessPoolExecutor(
            max_workers=self.config.workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(self.config.cpu_threads_per_worker,),
        ) as ex:
            next_i=0
            while next_i<len(jobs) or pending:
                while next_i<len(jobs) and len(pending)<self.config.max_in_flight:
                    f=ex.submit(_teacher_worker,jobs[next_i],payloads[next_i])
                    pending[f]=next_i
                    next_i+=1
                done,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
                for f in done:
                    idx=pending.pop(f)
                    results[idx]=f.result()

        ordered=[results[i] for i in range(len(jobs))]
        if persist:
            for j,res in zip(jobs,ordered):
                credit_obj=ExperienceObject(
                    object_id=res.credit.credit_id,
                    object_type="CREDIT_PACKAGE",
                    generation=j.generation,
                    policy_weight_hash=j.policy_weight_hash,
                    snapshot_hash=j.parent_snapshot_hash,
                    lineage_hash=res.credit.trajectory_identity_hash,
                    payload=asdict(res.credit),
                )
                evidence_obj=ExperienceObject(
                    object_id=res.evidence.evidence_id,
                    object_type="EVIDENCE_PACKAGE",
                    generation=j.generation,
                    policy_weight_hash=j.policy_weight_hash,
                    snapshot_hash=j.parent_snapshot_hash,
                    lineage_hash=res.credit.content_hash,
                    payload=asdict(res.evidence),
                )
                lake.put(credit_obj)
                lake.put(evidence_obj)
        return ordered
