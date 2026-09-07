from __future__ import annotations

"""CPU execution budgeting for CB16 R11 compute-plane stages.

This module is runtime policy only.  It never reads market data and it never changes
Teacher, Supervisor, Physics, probabilistic-learning, or Champion/Challenger semantics.
Worker topology is explicitly excluded from scientific identity.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from threading import RLock
from typing import Iterator, Literal


R11_CPU_RUNTIME = "CB16_R11_STAGE2_CPU_EXECUTION_BUDGET_V1"
CANONICAL_TEACHER_WORKERS_R11 = 8
CANONICAL_TRACE_WORKERS_R11 = 8
CANONICAL_BLAS_THREADS_R11 = 1
CANONICAL_STORAGE_BACKGROUND_THREADS_R11 = 1

CpuStageR11 = Literal["IDLE", "TEACHER", "TRACE", "STORAGE"]
_VALID_STAGES_R11 = frozenset(("IDLE", "TEACHER", "TRACE", "STORAGE"))


@dataclass(frozen=True)
class CpuTopologyR11:
    logical_cpus: int
    physical_cores_estimate: int
    affinity_cpus: tuple[int, ...]
    detection: str
    topology_in_scientific_identity: bool = False


@dataclass(frozen=True)
class CpuStageAllocationR11:
    stage: CpuStageR11
    logical_cpus: int
    physical_cores_estimate: int
    teacher_workers: int
    trace_workers: int
    storage_background_threads: int
    blas_threads: int
    compute_worker_threads: int
    total_scheduled_threads: int
    oversubscribed: bool
    topology_in_scientific_identity: bool = False

    def assert_valid(self) -> None:
        if self.blas_threads != 1:
            raise RuntimeError("R11_CPU_BLAS_THREADS_MUST_BE_ONE")
        if self.teacher_workers and self.trace_workers:
            raise RuntimeError("R11_CPU_COMPUTE_STAGES_OVERLAP")
        if self.compute_worker_threads > self.physical_cores_estimate:
            raise RuntimeError("R11_CPU_PHYSICAL_CORE_BUDGET_EXCEEDED")
        if self.total_scheduled_threads > self.logical_cpus or self.oversubscribed:
            raise RuntimeError("R11_CPU_LOGICAL_BUDGET_OVERSUBSCRIBED")


def _affinity_cpus_r11() -> tuple[int, ...]:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        try:
            cpus = tuple(sorted(int(x) for x in getter(0)))
            if cpus:
                return cpus
        except (OSError, ValueError):
            pass
    n = int(os.cpu_count() or 1)
    return tuple(range(max(1, n)))


def _linux_physical_cores_r11(affinity: tuple[int, ...]) -> int | None:
    topology_root = Path("/sys/devices/system/cpu")
    pairs: set[tuple[int, int]] = set()
    for cpu in affinity:
        base = topology_root / f"cpu{cpu}" / "topology"
        try:
            package_id = int((base / "physical_package_id").read_text(encoding="ascii").strip())
            core_id = int((base / "core_id").read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return None
        pairs.add((package_id, core_id))
    return len(pairs) or None


def detect_cpu_topology_r11(
    *,
    logical_cpus: int | None = None,
    physical_cores_estimate: int | None = None,
) -> CpuTopologyR11:
    """Detect an affinity-aware topology with a conservative physical-core fallback.

    Explicit values exist for deterministic hosted tests and deployment configuration.
    They are runtime-only inputs and are not scientific identity.
    """

    affinity = _affinity_cpus_r11()
    if logical_cpus is None:
        logical = len(affinity)
        detection = "AFFINITY"
    else:
        logical = int(logical_cpus)
        detection = "EXPLICIT"
        if logical <= 0:
            raise ValueError("R11_CPU_LOGICAL_CPUS_MUST_BE_POSITIVE")
        affinity = tuple(range(logical))

    if physical_cores_estimate is None:
        physical = _linux_physical_cores_r11(affinity)
        if physical is not None:
            detection += "_SYSFS_TOPOLOGY"
        else:
            physical = max(1, (logical + 1) // 2)
            detection += "_SMT2_ESTIMATE"
    else:
        physical = int(physical_cores_estimate)
        detection += "_PHYSICAL_EXPLICIT"

    if physical <= 0:
        raise ValueError("R11_CPU_PHYSICAL_CORES_MUST_BE_POSITIVE")
    if physical > logical:
        raise ValueError("R11_CPU_PHYSICAL_CORES_EXCEED_LOGICAL_CPUS")
    return CpuTopologyR11(
        logical_cpus=logical,
        physical_cores_estimate=physical,
        affinity_cpus=affinity,
        detection=detection,
    )


class CpuExecutionBudgetR11:
    """Exclusive compute-stage budget with dynamic Teacher/Trace CPU borrowing.

    Teacher and Trace are intentionally not allowed to consume their full worker counts
    simultaneously.  Each active compute stage may use at most the estimated physical
    cores, while a small storage lane may use spare logical CPUs.  BLAS is fixed to one
    thread so outer worker counts cannot multiply into nested BLAS oversubscription.
    """

    def __init__(
        self,
        *,
        logical_cpus: int | None = None,
        physical_cores_estimate: int | None = None,
        teacher_workers: int = CANONICAL_TEACHER_WORKERS_R11,
        trace_workers: int = CANONICAL_TRACE_WORKERS_R11,
        storage_background_threads: int = CANONICAL_STORAGE_BACKGROUND_THREADS_R11,
        blas_threads: int = CANONICAL_BLAS_THREADS_R11,
    ) -> None:
        self.topology = detect_cpu_topology_r11(
            logical_cpus=logical_cpus,
            physical_cores_estimate=physical_cores_estimate,
        )
        self.requested_teacher_workers = int(teacher_workers)
        self.requested_trace_workers = int(trace_workers)
        self.requested_storage_background_threads = int(storage_background_threads)
        self.blas_threads = int(blas_threads)
        if self.requested_teacher_workers <= 0:
            raise ValueError("R11_CPU_TEACHER_WORKERS_MUST_BE_POSITIVE")
        if self.requested_trace_workers <= 0:
            raise ValueError("R11_CPU_TRACE_WORKERS_MUST_BE_POSITIVE")
        if self.requested_storage_background_threads < 0:
            raise ValueError("R11_CPU_STORAGE_THREADS_MUST_BE_NONNEGATIVE")
        if self.blas_threads != CANONICAL_BLAS_THREADS_R11:
            raise RuntimeError("R11_CPU_BLAS_THREADS_MUST_BE_ONE")

        compute_ceiling = min(
            self.topology.physical_cores_estimate,
            self.topology.logical_cpus,
        )
        self.teacher_workers = min(self.requested_teacher_workers, compute_ceiling)
        self.trace_workers = min(self.requested_trace_workers, compute_ceiling)
        self._stage: CpuStageR11 = "IDLE"
        self._lock = RLock()
        self._transitions = 0

    @property
    def stage(self) -> CpuStageR11:
        with self._lock:
            return self._stage

    @property
    def transitions(self) -> int:
        with self._lock:
            return int(self._transitions)

    def allocation_for(self, stage: CpuStageR11) -> CpuStageAllocationR11:
        stage = str(stage).upper()  # type: ignore[assignment]
        if stage not in _VALID_STAGES_R11:
            raise ValueError(f"R11_CPU_STAGE_INVALID:{stage}")

        teacher = self.teacher_workers if stage == "TEACHER" else 0
        trace = self.trace_workers if stage == "TRACE" else 0
        compute = teacher + trace
        spare_logical = max(0, self.topology.logical_cpus - compute * self.blas_threads)
        storage = min(self.requested_storage_background_threads, spare_logical)
        if stage == "IDLE":
            storage = 0
        elif stage == "STORAGE":
            storage = min(
                self.requested_storage_background_threads,
                self.topology.logical_cpus,
            )
        total = compute * self.blas_threads + storage
        allocation = CpuStageAllocationR11(
            stage=stage,  # type: ignore[arg-type]
            logical_cpus=self.topology.logical_cpus,
            physical_cores_estimate=self.topology.physical_cores_estimate,
            teacher_workers=teacher,
            trace_workers=trace,
            storage_background_threads=storage,
            blas_threads=self.blas_threads,
            compute_worker_threads=compute,
            total_scheduled_threads=total,
            oversubscribed=total > self.topology.logical_cpus,
        )
        allocation.assert_valid()
        return allocation

    def snapshot(self) -> CpuStageAllocationR11:
        with self._lock:
            return self.allocation_for(self._stage)

    @contextmanager
    def stage_lease(self, stage: CpuStageR11) -> Iterator[CpuStageAllocationR11]:
        stage = str(stage).upper()  # type: ignore[assignment]
        if stage == "IDLE":
            raise ValueError("R11_CPU_IDLE_STAGE_CANNOT_BE_LEASED")
        allocation = self.allocation_for(stage)
        with self._lock:
            if self._stage != "IDLE":
                raise RuntimeError(
                    f"R11_CPU_STAGE_ALREADY_ACTIVE:{self._stage}:REQUESTED={stage}"
                )
            self._stage = stage  # type: ignore[assignment]
            self._transitions += 1
        try:
            yield allocation
        finally:
            with self._lock:
                if self._stage != stage:
                    raise RuntimeError(
                        f"R11_CPU_STAGE_RELEASE_MISMATCH:{self._stage}:EXPECTED={stage}"
                    )
                self._stage = "IDLE"
                self._transitions += 1

    def runtime_receipt(self) -> dict[str, object]:
        return {
            "schema": R11_CPU_RUNTIME,
            "scientific_semantics_changed": False,
            "topology_in_scientific_identity": False,
            "logical_cpus": self.topology.logical_cpus,
            "physical_cores_estimate": self.topology.physical_cores_estimate,
            "teacher_workers": self.teacher_workers,
            "trace_workers": self.trace_workers,
            "storage_background_threads": self.requested_storage_background_threads,
            "blas_threads": self.blas_threads,
            "active_stage": self.stage,
        }
