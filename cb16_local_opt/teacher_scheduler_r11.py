from __future__ import annotations

"""Shared-memory CPU scheduler for the R11 vectorized Teacher.

The old process farm replicated an object-heavy Python Teacher index in every process.
R11 keeps one immutable columnar index and one support-regime matrix in the process, then
parallelizes independent target blocks with threads.  NumPy's heavy matrix/sort kernels
release the GIL, so this can use Ryzen SMT threads without multiplying the scientific data
or Python object graph by worker count.

Nested BLAS parallelism must be disabled externally (OMP/MKL/OPENBLAS=1) when workers>1.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Sequence

from .probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from .probabilistic_teacher_r6 import DependenceAwareTeacherConfigR6, DependenceAwareTeacherEvidenceR6
from .r102_evidence_cache import ParentContextR102
from .teacher_vectorized_r11 import (
    VectorizedTeacherStatsR11,
    _compile_block_r11,
    build_columnar_teacher_index_r11,
    group_targets_by_support_r11,
    prepare_support_regime_r11,
)


R11_THREADED_SCHEDULER = "CB16_R11_SHARED_COLUMNAR_THREAD_SCHEDULER_V1"


@dataclass(frozen=True)
class ThreadedTeacherStatsR11:
    core: VectorizedTeacherStatsR11
    workers: int
    scheduler: str = R11_THREADED_SCHEDULER
    memory_model: str = "ONE_SHARED_COLUMNAR_INDEX__THREAD_LOCAL_TARGET_BLOCK_SCRATCH"
    nested_blas_threads_required: int = 1


def compile_teacher_evidence_threaded_r11(
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
    workers = int(workers)
    block_targets = int(block_targets)
    if workers <= 0:
        raise ValueError("workers")
    if block_targets <= 0:
        raise ValueError("block_targets")
    train_config.validate()
    val_config.validate()

    index = build_columnar_teacher_index_r11(samples)
    train_parent_ids = sorted(
        p.parent_id for p in parents.values()
        if p.split == "TRAIN" and p.parent_id in index.parent_row_by_id
    )
    val_parent_ids = sorted(
        p.parent_id for p in parents.values()
        if p.split == "VALIDATION" and p.parent_id in index.parent_row_by_id
    )
    eligible_train_groups = {
        p.dependence_group_id for p in parents.values() if p.split == "TRAIN"
    }

    compiled: dict[str, DependenceAwareTeacherEvidenceR6] = {}
    regime_count = 0
    block_count = 0

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cb16-r11-teacher") if workers > 1 else None
    try:
        for target_ids, config in (
            (train_parent_ids, train_config),
            (val_parent_ids, val_config),
        ):
            support_groups = group_targets_by_support_r11(
                target_parent_ids=target_ids,
                index=index,
                config=config,
                eligible_train_dependence_groups=eligible_train_groups,
            )
            for dep_rows, regime_targets in support_groups:
                regime = prepare_support_regime_r11(dep_rows=dep_rows, index=index)
                regime_count += 1
                chunks = [
                    regime_targets[start : start + block_targets]
                    for start in range(0, len(regime_targets), block_targets)
                ]
                block_count += len(chunks)

                if executor is None:
                    block_results = [
                        _compile_block_r11(
                            target_parent_ids=chunk,
                            index=index,
                            regime=regime,
                            config=config,
                        )
                        for chunk in chunks
                    ]
                else:
                    futures = [
                        executor.submit(
                            _compile_block_r11,
                            target_parent_ids=chunk,
                            index=index,
                            regime=regime,
                            config=config,
                        )
                        for chunk in chunks
                    ]
                    # Resolve in canonical chunk order.  Scheduling order therefore has no
                    # effect on returned evidence order or floating-point reduction order.
                    block_results = [future.result() for future in futures]

                for rows in block_results:
                    for evidence in rows:
                        if evidence.parent_id in compiled:
                            raise RuntimeError(f"R11_DUPLICATE_COMPILED_TARGET:{evidence.parent_id}")
                        compiled[evidence.parent_id] = evidence
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    train = [compiled[p] for p in train_parent_ids]
    val = [compiled[p] for p in val_parent_ids]
    core_stats = VectorizedTeacherStatsR11(
        targets=len(train) + len(val),
        support_regimes=regime_count,
        geometry_blocks=block_count,
        max_targets_per_block=block_targets,
        feature_dim=index.feature_dim,
        actions=index.action_count,
    )
    return train, val, ThreadedTeacherStatsR11(core=core_stats, workers=workers)
