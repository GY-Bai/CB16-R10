from __future__ import annotations

"""R11 dependence-balanced Teacher authority candidate.

This module does not bind production callers.  It gives the R2.2-qualified
mechanics a distinct R11 protocol identity so cutover can be qualified without
reusing the legacy R10.2 Teacher hash.
"""

from dataclasses import replace

from .r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102
from .r102_teacher_incremental import ExactIncrementalTeacherR6
from .science_teacher_requalification_r11 import (
    DependenceBalancedProbabilisticTeacherShadowR11,
    SHADOW_GEOMETRY_VERSION,
)

R11_TEACHER_AUTHORITY_VERSION = "CB16_R11_DEPENDENCE_BALANCED_TEACHER_AUTHORITY_V1"
R11_TRAIN_TEACHER_VERSION = "CB16_R11_DEPENDENCE_BALANCED_BLOCKED_CROSSFIT_TEACHER_V1"
R11_VALIDATION_TEACHER_VERSION = "CB16_R11_DEPENDENCE_BALANCED_PREQUENTIAL_VALIDATION_TEACHER_V1"

R11_TRAIN_TEACHER_CONFIG = replace(
    TRAIN_TEACHER_CONFIG_R102,
    teacher_version=R11_TRAIN_TEACHER_VERSION,
)
R11_VALIDATION_TEACHER_CONFIG = replace(
    VAL_TEACHER_CONFIG_R102,
    teacher_version=R11_VALIDATION_TEACHER_VERSION,
)


class DependenceBalancedProbabilisticTeacherR11(
    DependenceBalancedProbabilisticTeacherShadowR11
):
    """Candidate production class for the R2.2-qualified geometry."""

    authority_version = R11_TEACHER_AUTHORITY_VERSION
    geometry_version = SHADOW_GEOMETRY_VERSION


class DependenceBalancedExactIncrementalTeacherR11(
    ExactIncrementalTeacherR6,
    DependenceBalancedProbabilisticTeacherShadowR11,
):
    """Exact incremental compiler using the same R11 normalization geometry."""

    authority_version = R11_TEACHER_AUTHORITY_VERSION
    geometry_version = SHADOW_GEOMETRY_VERSION


__all__ = [
    "R11_TEACHER_AUTHORITY_VERSION",
    "R11_TRAIN_TEACHER_VERSION",
    "R11_VALIDATION_TEACHER_VERSION",
    "R11_TRAIN_TEACHER_CONFIG",
    "R11_VALIDATION_TEACHER_CONFIG",
    "DependenceBalancedProbabilisticTeacherR11",
    "DependenceBalancedExactIncrementalTeacherR11",
]
