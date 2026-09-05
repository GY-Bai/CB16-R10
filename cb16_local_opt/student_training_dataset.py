from __future__ import annotations

"""
Persistent Student-visible context -> Evidence -> TrainingBatch join.

The Student context is captured at decision time, before outcome maturity. Teacher/FUTURE
bytes are never stored inside this context object. Evidence later points back to the
context by immutable identity.

This closes:
Decision context -> Outcome/Teacher -> Evidence -> Frozen Evidence Snapshot
-> context/evidence join -> ProbabilisticTrainingBatch.
"""

import dataclasses
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from .conformance_guards import guard_student_payload
from .gpu_training_policy import ProbabilisticTrainingBatch
from .sharded_experience_lake import (
    ExperienceObject,
    ExperienceRef,
    ExperienceSnapshot,
    ShardedExperienceLake,
)


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj): obj=asdict(obj)
    return hashlib.sha256(
        json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class StudentContextR4:
    context_id:str
    decision_event_hash:str
    timestamp:int
    symbol:str
    account_id:str
    policy_generation:int
    policy_weight_hash:str
    market_latent:tuple[float,...]
    account_state6:tuple[float,...]
    market_lineage_hash:str
    account_lineage_hash:str

    def validate(self):
        if len(self.market_latent)!=64:
            raise ValueError("market_latent must have 64 dims")
        if len(self.account_state6)!=6:
            raise ValueError("account_state6 must have 6 dims")
        guard_student_payload(asdict(self))
        vals=np.asarray(self.market_latent+self.account_state6,dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            raise ValueError("nonfinite student context")

    @property
    def content_hash(self)->str:
        self.validate()
        return canonical_hash(self)


@dataclass(frozen=True)
class EvidenceTrainingLinkR4:
    evidence_object_id:str
    evidence_identity_hash:str
    student_context_object_id:str
    student_context_identity_hash:str
    generation:int
    policy_weight_hash:str
    teacher_identity_hash:str

    @property
    def content_hash(self)->str:
        return canonical_hash(self)


def store_student_context(
    lake:ShardedExperienceLake,
    context:StudentContextR4,
    *,
    snapshot_hash:str="DECISION_TIME",
)->ExperienceRef:
    context.validate()
    obj=ExperienceObject(
        object_id=context.context_id,
        object_type="STUDENT_CONTEXT",
        generation=context.policy_generation,
        policy_weight_hash=context.policy_weight_hash,
        snapshot_hash=snapshot_hash,
        lineage_hash=canonical_hash({
            "decision_event_hash":context.decision_event_hash,
            "market_lineage_hash":context.market_lineage_hash,
            "account_lineage_hash":context.account_lineage_hash,
        }),
        payload=asdict(context),
    )
    ref,_=lake.put(obj)
    return ref


def build_training_link(
    *,
    lake:ShardedExperienceLake,
    evidence_object_id:str,
)->EvidenceTrainingLinkR4:
    got=lake.get(evidence_object_id)
    if got is None: raise RuntimeError("EVIDENCE_NOT_FOUND")
    eref,epayload=got
    if eref.object_type!="EVIDENCE_PACKAGE":
        raise RuntimeError("OBJECT_IS_NOT_EVIDENCE_PACKAGE")
    # Experience payload is serialized EvidencePackageR4 dataclass.
    evidence_payload=epayload.get("payload",{})
    context_id=evidence_payload.get("student_context_object_id")
    if not context_id:
        raise RuntimeError("EVIDENCE_MISSING_STUDENT_CONTEXT_REF")
    cgot=lake.get(context_id)
    if cgot is None: raise RuntimeError("STUDENT_CONTEXT_NOT_FOUND")
    cref,cpayload=cgot
    if cref.object_type!="STUDENT_CONTEXT":
        raise RuntimeError("CONTEXT_REF_WRONG_OBJECT_TYPE")
    context=StudentContextR4(**cpayload)
    context.validate()
    if int(epayload["generation"])!=context.policy_generation:
        raise RuntimeError("EVIDENCE_CONTEXT_GENERATION_MISMATCH")
    teacher_identity=epayload["teacher_identity_hash"]
    return EvidenceTrainingLinkR4(
        evidence_object_id=evidence_object_id,
        evidence_identity_hash=eref.identity_hash,
        student_context_object_id=context_id,
        student_context_identity_hash=cref.identity_hash,
        generation=context.policy_generation,
        policy_weight_hash=context.policy_weight_hash,
        teacher_identity_hash=teacher_identity,
    )


class FrozenEvidenceTrainingDataset:
    """Read-only lazy join over a sealed ExperienceSnapshot of evidence refs."""

    def __init__(
        self,
        *,
        lake:ShardedExperienceLake,
        snapshot:ExperienceSnapshot,
        expected_parent_generation:int,
        expected_parent_policy_hash:str,
    ):
        self.lake=lake
        self.snapshot=snapshot
        if snapshot.parent_generation!=expected_parent_generation:
            raise RuntimeError("TRAINING_SNAPSHOT_PARENT_GENERATION_MISMATCH")
        if snapshot.parent_policy_hash!=expected_parent_policy_hash:
            raise RuntimeError("TRAINING_SNAPSHOT_PARENT_POLICY_MISMATCH")
        self.links=[]
        for oid,ih in zip(snapshot.object_ids,snapshot.object_identity_hashes):
            got=lake.get(oid)
            if got is None: raise RuntimeError(f"SNAPSHOT_OBJECT_MISSING:{oid}")
            ref,_=got
            if ref.identity_hash!=ih:
                raise RuntimeError(f"SNAPSHOT_OBJECT_IDENTITY_CHANGED:{oid}")
            link=build_training_link(lake=lake,evidence_object_id=oid)
            if link.generation!=expected_parent_generation:
                raise RuntimeError("CROSS_GENERATION_EVIDENCE_IN_SNAPSHOT")
            if link.policy_weight_hash!=expected_parent_policy_hash:
                raise RuntimeError("CROSS_POLICY_EVIDENCE_IN_SNAPSHOT")
            self.links.append(link)

    def __len__(self): return len(self.links)

    def _load_example(self,link:EvidenceTrainingLinkR4):
        eref,ep= self.lake.get(link.evidence_object_id)
        cref,cp= self.lake.get(link.student_context_object_id)
        if eref.identity_hash!=link.evidence_identity_hash:
            raise RuntimeError("EVIDENCE_IDENTITY_CHANGED")
        if cref.identity_hash!=link.student_context_identity_hash:
            raise RuntimeError("CONTEXT_IDENTITY_CHANGED")
        ctx=StudentContextR4(**cp); ctx.validate()
        epkg=ep
        if not bool(epkg.get("admitted",False)):
            raise RuntimeError("NON_ADMITTED_EVIDENCE_IN_TRAINING_SNAPSHOT")
        payload=epkg["payload"]
        guard_student_payload(payload)
        p=np.asarray(payload["direction_target_probs"],dtype=np.float32)
        if p.shape!=(3,) or np.any(p<0) or not np.isclose(p.sum(),1.0,atol=1e-5):
            raise RuntimeError("BAD_DIRECTION_TARGET_DISTRIBUTION")
        risk=float(payload["requested_risk_target"])
        if not 0<=risk<=1: raise RuntimeError("BAD_RISK_TARGET")
        return (
            np.asarray(ctx.market_latent,dtype=np.float32),
            np.asarray(ctx.account_state6,dtype=np.float32),
            p,
            risk,
            float(payload.get("direction_weight",1.0)),
            float(payload.get("sizing_weight",1.0)),
        )

    def iter_batches(
        self,
        *,
        batch_size:int,
        shuffle_seed:int|None=None,
        pin_memory:bool=False,
    )->Iterator[ProbabilisticTrainingBatch]:
        if batch_size<=0: raise ValueError("batch_size")
        order=list(range(len(self.links)))
        if shuffle_seed is not None:
            random.Random(int(shuffle_seed)).shuffle(order)
        for start in range(0,len(order),batch_size):
            idx=order[start:start+batch_size]
            xs=[self._load_example(self.links[i]) for i in idx]
            market=torch.from_numpy(np.stack([x[0] for x in xs]))
            account=torch.from_numpy(np.stack([x[1] for x in xs]))
            p=torch.from_numpy(np.stack([x[2] for x in xs]))
            risk=torch.tensor([x[3] for x in xs],dtype=torch.float32)
            dw=torch.tensor([x[4] for x in xs],dtype=torch.float32)
            sw=torch.tensor([x[5] for x in xs],dtype=torch.float32)
            admitted=torch.ones(len(xs),dtype=torch.bool)
            batch=ProbabilisticTrainingBatch(
                market_latent=market,account_state=account,
                direction_target_probs=p,requested_risk_target=risk,
                direction_weight=dw,sizing_weight=sw,admitted=admitted,
            )
            batch.validate()
            if pin_memory and torch.cuda.is_available():
                batch=batch.pin_memory()
            yield batch
