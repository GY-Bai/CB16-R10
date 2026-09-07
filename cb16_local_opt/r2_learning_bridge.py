from __future__ import annotations

"""Bridge R10.2 probabilistic teacher evidence into R2 immutable storage."""

from dataclasses import asdict
from typing import Mapping, Sequence

from .probabilistic_teacher_r6 import DependenceAwareTeacherEvidenceR6
from .r102_evidence_cache import ParentContextR102
from .r2_evidence_storage import (
    R2EvidenceItem,
    R2EvidenceSetRef,
    R2EvidenceStore,
    R2GenerationSnapshot,
    R2MaterializeReceipt,
)


def build_r2_teacher_items(
    *,
    train_evidence: Sequence[DependenceAwareTeacherEvidenceR6],
    parents: Mapping[str, ParentContextR102],
) -> list[R2EvidenceItem]:
    """Build generation-independent teacher evidence content.

    generation and champion/policy hash are deliberately excluded. They are authority
    and membership facts and belong in the generation snapshot layer, not evidence content.
    """
    out: list[R2EvidenceItem] = []
    for e in train_evidence:
        if not e.admission.admitted:
            continue
        p = parents[e.parent_id]
        payload = {
            "schema": "CB16_R2_EVIDENCE_PACKAGE_V1",
            "parent_id": e.parent_id,
            "dependence_group_id": e.target_dependence_group_id,
            "student_context_object_id": e.student_context_object_id,
            "operator48": list(p.operator48),
            "medium48": list(p.medium48),
            "account6": list(p.account6),
            "direction_target_probs": list(e.direction_target_probs),
            "requested_risk_target": e.requested_risk_target,
            "action_laws": [asdict(x) for x in e.action_laws],
            "admission": asdict(e.admission),
            "teacher_protocol_hash": e.teacher_protocol_hash,
        }
        out.append(
            R2EvidenceItem(
                evidence_id=e.evidence_id,
                parent_snapshot_hash=p.snapshot_sha256,
                lineage_hash=e.content_hash,
                teacher_protocol_hash=e.teacher_protocol_hash,
                payload=payload,
            )
        )
    return out


def materialize_training_evidence_r2(
    *,
    store: R2EvidenceStore,
    train_evidence: Sequence[DependenceAwareTeacherEvidenceR6],
    parents: Mapping[str, ParentContextR102],
    evidence_set_id: str = "R102_TRAINING_EVIDENCE_SET",
) -> tuple[R2EvidenceSetRef, R2MaterializeReceipt]:
    return store.materialize_evidence_set(
        evidence_set_id=evidence_set_id,
        items=build_r2_teacher_items(train_evidence=train_evidence, parents=parents),
    )


def seal_generation_snapshot_r2(
    *,
    store: R2EvidenceStore,
    evidence_set: R2EvidenceSetRef,
    generation: int,
    champion_hash: str,
) -> R2GenerationSnapshot:
    return store.seal_generation_snapshot(
        snapshot_id=f"R102_G{int(generation)}_TRAINING_SNAPSHOT_R2",
        generation=int(generation),
        parent_policy_hash=champion_hash,
        evidence_set=evidence_set,
    )
