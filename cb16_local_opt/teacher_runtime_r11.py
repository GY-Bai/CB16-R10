from __future__ import annotations

"""Canonical CB16 R11 probabilistic Teacher runtime.

R2.1/R2.2 qualified dependence-balanced, exact-replica-invariant normalization.
R2.3 gave those mechanics a distinct R11 Teacher protocol identity.  This runtime
binds the R11 science path to that authority while leaving R10.2 legacy modules
untouched for historical reproduction.

Worker topology remains a scheduling property and is excluded from scientific
identity.  Production legacy config identities are rotated to the R11 candidate
identities before evidence is compiled.
"""

import os
from typing import Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

from .cpu_runtime_r11 import CANONICAL_TEACHER_WORKERS_R11
from .probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from .probabilistic_teacher_r6 import DependenceAwareTeacherConfigR6, DependenceAwareTeacherEvidenceR6
from .r102_evidence_cache import ParentContextR102
from .r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102
from .r11_teacher_authority_candidate import R11_TRAIN_TEACHER_CONFIG, R11_VALIDATION_TEACHER_CONFIG
from .teacher_balanced_runtime_r11 import compile_teacher_evidence_balanced_r11
from .teacher_scheduler_r11 import TeacherWorkerPoolR11, ThreadedTeacherStatsR11

R11_TEACHER_RUNTIME = "CB16_R11_DEPENDENCE_BALANCED_PRODUCTION_RUNTIME_V1"
SUPPORTED_QUALIFICATION_WORKERS_R11 = (1, 4, 8, 12)


def _bind_authoritative_config_r11(
    supplied: DependenceAwareTeacherConfigR6,
    *,
    legacy: DependenceAwareTeacherConfigR6,
    candidate: DependenceAwareTeacherConfigR6,
    lane_name: str,
) -> DependenceAwareTeacherConfigR6:
    """Rotate only the frozen production identity; preserve explicit test/experimental configs."""
    if supplied.content_hash == legacy.content_hash:
        return candidate
    if supplied.content_hash == candidate.content_hash:
        return candidate
    # Reusing an authoritative version string with changed fields would counterfeit
    # provenance.  Custom test/experimental versions remain legal and use balanced geometry.
    if supplied.teacher_version in {legacy.teacher_version, candidate.teacher_version}:
        raise RuntimeError(f"R11_TEACHER_AUTHORITY_CONFIG_DRIFT:{lane_name}")
    return supplied


def compile_teacher_evidence_r11(
    *,
    samples: Sequence[CounterfactualBranchSampleR5],
    parents: Mapping[str, ParentContextR102],
    train_config: DependenceAwareTeacherConfigR6,
    val_config: DependenceAwareTeacherConfigR6,
    workers: int = CANONICAL_TEACHER_WORKERS_R11,
    block_targets: int = 64,
    worker_pool: TeacherWorkerPoolR11 | None = None,
) -> tuple[list[DependenceAwareTeacherEvidenceR6], list[DependenceAwareTeacherEvidenceR6], ThreadedTeacherStatsR11]:
    """Compile R11 Teacher evidence under the dependence-balanced canonical authority."""
    effective_train = _bind_authoritative_config_r11(
        train_config,
        legacy=TRAIN_TEACHER_CONFIG_R102,
        candidate=R11_TRAIN_TEACHER_CONFIG,
        lane_name="TRAIN",
    )
    effective_val = _bind_authoritative_config_r11(
        val_config,
        legacy=VAL_TEACHER_CONFIG_R102,
        candidate=R11_VALIDATION_TEACHER_CONFIG,
        lane_name="VALIDATION",
    )
    return compile_teacher_evidence_balanced_r11(
        samples=samples,
        parents=parents,
        train_config=effective_train,
        val_config=effective_val,
        workers=workers,
        block_targets=block_targets,
        worker_pool=worker_pool,
    )
