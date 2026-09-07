from __future__ import annotations

"""Production entry point for CB16 R11 Teacher V7 evidence compilation.

This module intentionally contains scheduling policy only.  Scientific Teacher semantics
remain in the frozen R11 vectorized kernel/configuration.  Worker count, block size and
completion order are excluded from evidence identity.
"""

import os
from typing import Mapping, Sequence

# Best-effort pre-import protection for BLAS libraries not loaded yet.  The scheduler also
# applies threadpoolctl at execution time, which is the authoritative runtime guard for
# already-loaded NumPy/BLAS runtimes.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

from .probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from .probabilistic_teacher_r6 import DependenceAwareTeacherConfigR6, DependenceAwareTeacherEvidenceR6
from .r102_evidence_cache import ParentContextR102
from .teacher_scheduler_r11 import ThreadedTeacherStatsR11, compile_teacher_evidence_threaded_r11


R11_TEACHER_RUNTIME = "CB16_R11_TEACHER_V7_PRODUCTION_RUNTIME_V1"
SUPPORTED_QUALIFICATION_WORKERS_R11 = (1, 4, 8, 12)


def compile_teacher_evidence_r11(
    *,
    samples: Sequence[CounterfactualBranchSampleR5],
    parents: Mapping[str, ParentContextR102],
    train_config: DependenceAwareTeacherConfigR6,
    val_config: DependenceAwareTeacherConfigR6,
    workers: int = 12,
    block_targets: int = 64,
) -> tuple[
    list[DependenceAwareTeacherEvidenceR6],
    list[DependenceAwareTeacherEvidenceR6],
    ThreadedTeacherStatsR11,
]:
    """Production Teacher V7 adapter, integration-ready for a later campaign owner.

    No legacy Teacher is called here.  The worker topology is deliberately not passed to
    Teacher config and therefore cannot enter protocol/evidence identity.
    """

    return compile_teacher_evidence_threaded_r11(
        samples=samples,
        parents=parents,
        train_config=train_config,
        val_config=val_config,
        workers=workers,
        block_targets=block_targets,
    )
