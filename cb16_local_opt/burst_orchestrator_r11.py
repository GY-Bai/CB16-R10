from __future__ import annotations

"""Short-burst heterogeneous overlap scheduler for CB16 R11.

This module changes *when/where* already-admitted R11 work executes.  Scientific
identity, generation/champion ownership, snapshot sealing, tournament commit, and
recovery authority remain in :mod:`orchestrator_r11` and the injected engines.

The important scheduling distinction is between admission and execution.  A
``WorkItem`` must be admitted by ``R11Orchestrator.enqueue`` while its scientific
stage is legal.  Once durably admitted, it may execute later on its compatible
resource lane even if the mainline state has advanced.  This is what permits
current-Champion CPU trace and storage work to overlap Challenger GPU training
without weakening any lineage barrier.
"""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os
import time
from typing import Callable, Mapping, Protocol

from .orchestrator_r11 import (
    BackpressureRequired,
    LineageViolation,
    R11Orchestrator,
)
from .runtime_events_r11 import (
    CompletionDisposition,
    EvidenceRef,
    GenerationState,
    PolicyResult,
    TournamentResult,
    TrainingResult,
    ValidationResult,
    WorkerPool,
    WorkCompletion,
    WorkItem,
    WorkKind,
)


class BurstOrchestrationError(RuntimeError):
    pass


class BurstBackpressure(BurstOrchestrationError):
    pass


class BurstBarrierViolation(BurstOrchestrationError):
    pass


class BurstWorkerFailure(BurstOrchestrationError):
    pass


class BurstSemanticFailure(BurstOrchestrationError):
    pass


class BurstPhase(str, Enum):
    NEW = "NEW"
    WARMUP = "WARMUP"
    BURST = "BURST"
    DRAINING = "DRAINING"
    SEALED = "SEALED"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class BurstLimits:
    """Resource and backpressure envelope for a short R11 burst.

    ``max_in_flight_generations`` is intentionally fixed to one by this Task-D
    implementation.  A next-generation R11 orchestrator may only be constructed
    from ``release_next_generation`` after the current tournament/checkpoint
    barrier has committed.
    """

    cpu_workers: int = 8
    gpu_workers: int = 1
    storage_workers: int = 1
    max_pending_work: int = 24
    max_in_flight_generations: int = 1
    evidence_high_bytes: int = 512 * 1024 * 1024
    evidence_low_bytes: int = 384 * 1024 * 1024
    storage_high_depth: int = 8
    storage_low_depth: int = 4
    ram_high_fraction: float = 0.88
    ram_low_fraction: float = 0.78
    max_retries: int = 1

    def __post_init__(self) -> None:
        ints = {
            "cpu_workers": self.cpu_workers,
            "gpu_workers": self.gpu_workers,
            "storage_workers": self.storage_workers,
            "max_pending_work": self.max_pending_work,
            "max_in_flight_generations": self.max_in_flight_generations,
            "evidence_high_bytes": self.evidence_high_bytes,
            "evidence_low_bytes": self.evidence_low_bytes,
            "storage_high_depth": self.storage_high_depth,
            "storage_low_depth": self.storage_low_depth,
        }
        if any(v < 1 for v in ints.values()):
            raise ValueError(f"R11_BURST_LIMIT_MUST_BE_POSITIVE:{ints}")
        if self.max_in_flight_generations != 1:
            raise ValueError("R11_BURST_PARENT_BARRIER_REQUIRES_ONE_SCIENTIFIC_GENERATION")
        if self.gpu_workers != 1:
            raise ValueError("R11_BURST_SINGLE_GTX1060_REQUIRES_ONE_GPU_LANE")
        if self.evidence_low_bytes > self.evidence_high_bytes:
            raise ValueError("R11_BURST_EVIDENCE_LOW_GT_HIGH")
        if self.storage_low_depth > self.storage_high_depth:
            raise ValueError("R11_BURST_STORAGE_LOW_GT_HIGH")
        if not 0.0 < self.ram_low_fraction <= self.ram_high_fraction < 1.0:
            raise ValueError("R11_BURST_INVALID_RAM_WATERMARKS")


@dataclass(frozen=True)
class RamSample:
    used_bytes: int
    limit_bytes: int

    @property
    def fraction(self) -> float:
        if self.limit_bytes <= 0:
            return 0.0
        return min(1.0, max(0.0, self.used_bytes / self.limit_bytes))


class RamProbe(Protocol):
    def __call__(self) -> RamSample: ...


def linux_ram_probe() -> RamSample:
    """Best-effort Linux/cgroup RAM sample without third-party dependencies."""

    cgroup_current = Path("/sys/fs/cgroup/memory.current")
    cgroup_max = Path("/sys/fs/cgroup/memory.max")
    try:
        if cgroup_current.exists() and cgroup_max.exists():
            raw_max = cgroup_max.read_text().strip()
            if raw_max != "max":
                limit = int(raw_max)
                used = int(cgroup_current.read_text().strip())
                if limit > 0:
                    return RamSample(used, limit)
    except (OSError, ValueError):
        pass

    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
        total = values["MemTotal"]
        available = values.get("MemAvailable", values.get("MemFree", 0))
        return RamSample(max(0, total - available), total)
    except (OSError, KeyError, ValueError):
        return RamSample(0, max(1, os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")))


Runner = Callable[[WorkItem], WorkCompletion]
CompletionApplier = Callable[[R11Orchestrator, WorkItem, WorkCompletion], object]


@dataclass
class _Reservation:
    work: WorkItem
    evidence_bytes: int


@dataclass
class _Running:
    work: WorkItem
    future: Future[WorkCompletion]


@dataclass(frozen=True)
class BurstStatus:
    phase: BurstPhase
    generation: int
    queued_or_pending: int
    running: int
    cpu_running: int
    gpu_running: int
    storage_running: int
    evidence_bytes_reserved: int
    storage_depth: int
    ram_fraction: float
    admission_paused: bool
    stop_requested: bool


_POOL_KIND_COMPATIBILITY: dict[WorkKind, WorkerPool] = {
    WorkKind.TEACHER_ACQUIRE: WorkerPool.CPU_TEACHER,
    WorkKind.POLICY_INFER: WorkerPool.GPU_BRAIN,
    WorkKind.TRACE_EXECUTE: WorkerPool.CPU_TRACE,
    WorkKind.MATERIALIZE_EVIDENCE: WorkerPool.STORAGE,
    WorkKind.TRAIN_CHALLENGER: WorkerPool.GPU_BRAIN,
    WorkKind.VALIDATE: WorkerPool.GPU_BRAIN,
    WorkKind.TOURNAMENT: WorkerPool.GPU_BRAIN,
    WorkKind.CHECKPOINT_SEAL: WorkerPool.STORAGE,
    WorkKind.PREFETCH_IMMUTABLE: WorkerPool.IO_PREFETCH,
}

_SCIENTIFIC_EXECUTABLE = {
    WorkKind.TEACHER_ACQUIRE,
    WorkKind.POLICY_INFER,
    WorkKind.TRACE_EXECUTE,
    WorkKind.MATERIALIZE_EVIDENCE,
    WorkKind.TRAIN_CHALLENGER,
    WorkKind.VALIDATE,
    WorkKind.TOURNAMENT,
}

_CPU_POOLS = (WorkerPool.CPU_TEACHER, WorkerPool.CPU_TRACE)
_GPU_POOLS = (WorkerPool.GPU_BRAIN,)
_STORAGE_POOLS = (WorkerPool.STORAGE,)


class BurstOrchestratorR11:
    """Resource-aware short-burst dispatcher around one R11 generation.

    All scientific work is first admitted through ``core.enqueue``.  The wrapper
    then executes that durable work on a bounded CPU/GPU/storage executor and
    applies completion through the same R11 semantic methods used by serial
    execution.  No next generation can be admitted here.
    """

    def __init__(
        self,
        core: R11Orchestrator,
        *,
        limits: BurstLimits = BurstLimits(),
        runners: Mapping[WorkKind, Runner] | None = None,
        completion_appliers: Mapping[WorkKind, CompletionApplier] | None = None,
        ram_probe: RamProbe = linux_ram_probe,
    ) -> None:
        self.core = core
        self.limits = limits
        self._runners = dict(runners or {})
        self._appliers = dict(completion_appliers or {})
        self._ram_probe = ram_probe
        self.phase = BurstPhase.NEW
        self.stop_requested = False
        self._pressure_latched = False
        self._reservations: dict[str, _Reservation] = {}
        self._running: dict[str, _Running] = {}
        self._extra_completed: set[str] = set()
        self._fatal: BaseException | None = None
        self._rr_cpu = 0
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._adopt_recovered_pending()

    @property
    def generation(self) -> int:
        return self.core.record.generation

    def start(self) -> None:
        if self.phase is not BurstPhase.NEW:
            raise BurstBarrierViolation(f"R11_BURST_START_FROM:{self.phase.value}")
        self._executors = {
            "cpu": ThreadPoolExecutor(max_workers=self.limits.cpu_workers, thread_name_prefix="r11-cpu"),
            "gpu": ThreadPoolExecutor(max_workers=1, thread_name_prefix="r11-gpu"),
            "storage": ThreadPoolExecutor(
                max_workers=self.limits.storage_workers, thread_name_prefix="r11-storage"
            ),
        }
        self.phase = BurstPhase.WARMUP

    def warmup(self, jobs: tuple[Callable[[], object], ...] = ()) -> tuple[object, ...]:
        """Run ephemeral cache/prefetch warmup only; no scientific work is admitted."""

        self._require_phase(BurstPhase.WARMUP)
        futures = [self._executors["cpu"].submit(job) for job in jobs]
        return tuple(f.result() for f in futures)

    def begin_burst(self) -> None:
        self._require_phase(BurstPhase.WARMUP)
        self.phase = BurstPhase.BURST

    def request_stop(self) -> None:
        if self.phase in {BurstPhase.CLOSED, BurstPhase.SEALED, BurstPhase.FAILED}:
            return
        self.stop_requested = True
        if self.phase in {BurstPhase.WARMUP, BurstPhase.BURST}:
            self.phase = BurstPhase.DRAINING

    def schedule(self, work: WorkItem, *, evidence_bytes: int = 0) -> bool:
        """Durably admit current-generation work and reserve bounded resources.

        Returns ``True`` when the base per-pool mailbox accepted the item
        immediately.  ``False`` means the item is still durably scheduled by the
        R11 journal and will be pumped after queue pressure clears.
        """

        if self.phase not in {BurstPhase.BURST, BurstPhase.DRAINING}:
            raise BurstBarrierViolation(f"R11_BURST_SCHEDULE_FROM:{self.phase.value}")
        if work.kind not in _SCIENTIFIC_EXECUTABLE:
            raise BurstBarrierViolation(f"R11_BURST_UNSUPPORTED_SCIENTIFIC_KIND:{work.kind.value}")
        self._validate_identity(work)
        if evidence_bytes < 0:
            raise ValueError("R11_BURST_NEGATIVE_EVIDENCE_RESERVATION")

        prior = self._reservations.get(work.work_id)
        if prior is not None:
            if prior.work != work or prior.evidence_bytes != evidence_bytes:
                raise BurstSemanticFailure(f"R11_BURST_DUPLICATE_WORK_CONFLICT:{work.work_id}")
            return work.work_id not in self.core.pending_work_ids()

        self._refresh_pressure()
        outstanding = self._outstanding_ids()
        if len(outstanding) >= self.limits.max_pending_work:
            self._pressure_latched = True
            raise BurstBackpressure("R11_BURST_GLOBAL_PENDING_HIGH_WATERMARK")
        if self._pressure_latched:
            raise BurstBackpressure("R11_BURST_ADMISSION_PAUSED_BY_WATERMARK")
        if self._evidence_reserved() + evidence_bytes > self.limits.evidence_high_bytes:
            self._pressure_latched = True
            raise BurstBackpressure("R11_BURST_EVIDENCE_HIGH_WATERMARK")
        if work.pool is WorkerPool.STORAGE and self._storage_depth() >= self.limits.storage_high_depth:
            self._pressure_latched = True
            raise BurstBackpressure("R11_BURST_STORAGE_HIGH_WATERMARK")

        self._reservations[work.work_id] = _Reservation(work, evidence_bytes)
        queued = True
        try:
            self.core.enqueue(work, block=False)
        except BackpressureRequired:
            queued = False
        return queued

    def seal_training_snapshot(self, seal) -> None:
        pending_teacher = [
            wid
            for wid in self._outstanding_ids()
            if self._work_for_id(wid).kind is WorkKind.TEACHER_ACQUIRE
        ]
        if pending_teacher:
            raise BurstBarrierViolation(
                "R11_BURST_SNAPSHOT_WITH_PENDING_TEACHER:" + ",".join(sorted(pending_teacher))
            )
        self.core.seal_training_snapshot(seal)

    def start_challenger_training(self, work: WorkItem, *, evidence_bytes: int = 0) -> bool:
        if work.kind is not WorkKind.TRAIN_CHALLENGER:
            raise BurstBarrierViolation("R11_BURST_START_TRAINING_REQUIRES_TRAIN_WORK")
        queued = self.schedule(work, evidence_bytes=evidence_bytes)
        self.core.start_challenger_training(work_id=work.work_id)
        return queued

    def pump_once(self) -> bool:
        self._raise_if_fatal()
        if self.phase not in {BurstPhase.BURST, BurstPhase.DRAINING}:
            return False
        progressed = self._reap_finished()
        self._raise_if_fatal()
        self.core.pump_pending_work()

        while self._count_running(_CPU_POOLS) < self.limits.cpu_workers:
            work = self._claim_cpu_round_robin()
            if work is None:
                break
            self._launch(work, "cpu")
            progressed = True

        if self._count_running(_GPU_POOLS) < 1:
            work = self.core.claim_work(WorkerPool.GPU_BRAIN)
            if work is not None:
                self._launch(work, "gpu")
                progressed = True

        while self._count_running(_STORAGE_POOLS) < self.limits.storage_workers:
            work = self.core.claim_work(WorkerPool.STORAGE)
            if work is None:
                break
            self._launch(work, "storage")
            progressed = True
        return progressed

    def wait_for(self, work_ids: tuple[str, ...], *, timeout: float | None = None) -> None:
        wanted = set(work_ids)
        unknown = wanted - set(self.core.record.scheduled_work)
        if unknown:
            raise BurstBarrierViolation("R11_BURST_WAIT_UNKNOWN_WORK:" + ",".join(sorted(unknown)))
        deadline = None if timeout is None else time.monotonic() + timeout
        while not wanted.issubset(self._completed_ids()):
            self._raise_if_fatal()
            self.pump_once()
            if wanted.issubset(self._completed_ids()):
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("R11_BURST_WAIT_TIMEOUT")
            futures = [x.future for x in self._running.values()]
            if futures:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                wait(futures, timeout=min(0.02, remaining) if remaining is not None else 0.02, return_when=FIRST_COMPLETED)
            else:
                time.sleep(0.001)

    def drain(self, *, timeout: float | None = None) -> None:
        if self.phase is BurstPhase.BURST:
            self.request_stop()
        if self.phase is BurstPhase.WARMUP:
            self.request_stop()
        if self.phase not in {BurstPhase.DRAINING, BurstPhase.FAILED}:
            raise BurstBarrierViolation(f"R11_BURST_DRAIN_FROM:{self.phase.value}")
        self._raise_if_fatal()
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._outstanding_ids() or self._running:
            self.pump_once()
            self._raise_if_fatal()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("R11_BURST_DRAIN_TIMEOUT")
            if self._running:
                futures = [x.future for x in self._running.values()]
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                wait(futures, timeout=min(0.02, remaining) if remaining is not None else 0.02, return_when=FIRST_COMPLETED)
            else:
                time.sleep(0.001)
        self.core.pump_pending_work()

    def seal(self) -> None:
        if self.phase is BurstPhase.BURST:
            self.request_stop()
        self._require_phase(BurstPhase.DRAINING)
        self._raise_if_fatal()
        if self._running:
            raise BurstBarrierViolation("R11_BURST_SEAL_WITH_RUNNING_WORK")
        if self._outstanding_ids():
            raise BurstBarrierViolation("R11_BURST_SEAL_WITH_PENDING_WORK")
        if self._count_running(_GPU_POOLS):
            raise BurstBarrierViolation("R11_BURST_ORPHAN_GPU_JOB")
        if self.core.record.state is not GenerationState.COMMITTED:
            raise BurstBarrierViolation(
                f"R11_BURST_HALF_GENERATION:{self.core.record.state.value}"
            )
        if self.core.record.commit is None or self.core.record.checkpoint is None:
            raise BurstBarrierViolation("R11_BURST_COMMIT_CHECKPOINT_REQUIRED")
        self.phase = BurstPhase.SEALED

    def close(self) -> None:
        self._require_phase(BurstPhase.SEALED)
        for executor in self._executors.values():
            executor.shutdown(wait=True, cancel_futures=False)
        self._executors.clear()
        self.phase = BurstPhase.CLOSED

    def release_next_generation(self):
        if self._running or self._outstanding_ids():
            raise BurstBarrierViolation("R11_BURST_RELEASE_WITH_OUTSTANDING_WORK")
        if self.core.record.state is not GenerationState.COMMITTED:
            raise BurstBarrierViolation("R11_BURST_RELEASE_BEFORE_TOURNAMENT_CHECKPOINT")
        return self.core.release_next_generation()

    def status(self) -> BurstStatus:
        ram = self._ram_probe()
        return BurstStatus(
            phase=self.phase,
            generation=self.generation,
            queued_or_pending=len(self._outstanding_ids()),
            running=len(self._running),
            cpu_running=self._count_running(_CPU_POOLS),
            gpu_running=self._count_running(_GPU_POOLS),
            storage_running=self._count_running(_STORAGE_POOLS),
            evidence_bytes_reserved=self._evidence_reserved(),
            storage_depth=self._storage_depth(),
            ram_fraction=ram.fraction,
            admission_paused=self._pressure_latched,
            stop_requested=self.stop_requested,
        )

    def _require_phase(self, *phases: BurstPhase) -> None:
        if self.phase not in phases:
            names = ",".join(p.value for p in phases)
            raise BurstBarrierViolation(f"R11_BURST_PHASE_REQUIRED:{names}:GOT:{self.phase.value}")

    def _validate_identity(self, work: WorkItem) -> None:
        if work.generation != self.generation:
            raise LineageViolation(
                f"R11_BURST_GENERATION_MIXING:EXPECTED:{self.generation}:GOT:{work.generation}"
            )
        if work.parent_champion_id != self.core.record.authority.champion_id:
            raise LineageViolation(
                "R11_BURST_STALE_CHAMPION_WORK:"
                f"EXPECTED:{self.core.record.authority.champion_id}:GOT:{work.parent_champion_id}"
            )
        expected_pool = _POOL_KIND_COMPATIBILITY[work.kind]
        if work.pool is not expected_pool:
            raise LineageViolation(
                f"R11_BURST_POOL_MISMATCH:{work.kind.value}:{work.pool.value}:{expected_pool.value}"
            )
        if work.snapshot_id is not None:
            snapshot = self.core.record.snapshot
            if snapshot is None or snapshot.snapshot_id != work.snapshot_id:
                raise LineageViolation("R11_BURST_WORK_SNAPSHOT_MISMATCH")

    def _adopt_recovered_pending(self) -> None:
        for work_id in self.core.pending_work_ids():
            work = self.core.record.scheduled_work[work_id]
            if work.kind in _SCIENTIFIC_EXECUTABLE:
                self._reservations.setdefault(work_id, _Reservation(work, 0))

        for work_id, reservation in list(self._reservations.items()):
            work = reservation.work
            if (
                work.kind is WorkKind.TEACHER_ACQUIRE
                and work.payload_ref
                and work.payload_ref in self.core.record.accepted_evidence
            ):
                evidence = self.core.record.accepted_evidence[work.payload_ref]
                self._mark_extra_completion(work_id, evidence.payload_hash)

    def _refresh_pressure(self) -> None:
        ram_fraction = self._ram_probe().fraction
        evidence = self._evidence_reserved()
        storage = self._storage_depth()
        if not self._pressure_latched:
            if (
                ram_fraction >= self.limits.ram_high_fraction
                or evidence >= self.limits.evidence_high_bytes
                or storage >= self.limits.storage_high_depth
            ):
                self._pressure_latched = True
        else:
            if (
                ram_fraction <= self.limits.ram_low_fraction
                and evidence <= self.limits.evidence_low_bytes
                and storage <= self.limits.storage_low_depth
                and len(self._outstanding_ids()) < self.limits.max_pending_work
            ):
                self._pressure_latched = False

    def _claim_cpu_round_robin(self) -> WorkItem | None:
        pools = _CPU_POOLS
        for offset in range(len(pools)):
            idx = (self._rr_cpu + offset) % len(pools)
            work = self.core.claim_work(pools[idx])
            if work is not None:
                self._rr_cpu = (idx + 1) % len(pools)
                return work
        return None

    def _launch(self, work: WorkItem, executor_name: str) -> None:
        self._validate_identity(work)
        scheduled = self.core.record.scheduled_work.get(work.work_id)
        if scheduled != work:
            raise BurstSemanticFailure(f"R11_BURST_EXECUTE_NOT_DURABLY_ADMITTED:{work.work_id}")
        future = self._executors[executor_name].submit(self._execute_adapter, work)
        self._running[work.work_id] = _Running(work, future)

    def _execute_adapter(self, work: WorkItem) -> WorkCompletion:
        runner = self._runners.get(work.kind)
        if runner is None:
            engine = self._engine_for(work.kind)
            if engine is None:
                raise BurstWorkerFailure(f"R11_BURST_ENGINE_UNAVAILABLE:{work.kind.value}")
            completion = engine.execute(work)
        else:
            completion = runner(work)
        if completion.work_id != work.work_id or completion.kind is not work.kind:
            raise BurstSemanticFailure("R11_BURST_COMPLETION_WORK_IDENTITY_MISMATCH")
        if completion.generation != work.generation:
            raise LineageViolation("R11_BURST_COMPLETION_GENERATION_MIXING")
        if completion.poison_bits:
            raise LineageViolation(
                f"R11_BURST_COMPLETION_POISON:{work.work_id}:{'|'.join(completion.poison_bits)}"
            )
        result_hash = getattr(completion.result, "payload_hash", completion.payload_hash)
        if result_hash != completion.payload_hash:
            raise BurstSemanticFailure(f"R11_BURST_COMPLETION_PAYLOAD_HASH_MISMATCH:{work.work_id}")
        return completion

    def _engine_for(self, kind: WorkKind):
        engines = self.core.engines
        if kind is WorkKind.TEACHER_ACQUIRE:
            return engines.teacher
        if kind is WorkKind.TRACE_EXECUTE:
            return engines.trace
        if kind is WorkKind.POLICY_INFER:
            return engines.inference
        if kind is WorkKind.TRAIN_CHALLENGER:
            return engines.training
        if kind is WorkKind.VALIDATE:
            return engines.validation
        if kind is WorkKind.TOURNAMENT:
            return engines.tournament
        return None

    def _reap_finished(self) -> bool:
        progressed = False
        for work_id, running in list(self._running.items()):
            if not running.future.done():
                continue
            progressed = True
            del self._running[work_id]
            try:
                completion = running.future.result()
            except BaseException as exc:
                self._task_done(running.work)
                if running.work.attempt < self.limits.max_retries:
                    retried = self.core.worker_crashed(running.work)
                    reservation = self._reservations.get(work_id)
                    if reservation is not None:
                        reservation.work = retried
                    continue
                self.phase = BurstPhase.FAILED
                self._fatal = BurstWorkerFailure(
                    f"R11_BURST_WORKER_FAILED:{running.work.work_id}:ATTEMPT:{running.work.attempt}"
                )
                self._fatal.__cause__ = exc
                continue

            try:
                self._apply_completion(running.work, completion)
            except BaseException as exc:
                self._task_done(running.work)
                self.phase = BurstPhase.FAILED
                self._fatal = BurstSemanticFailure(
                    f"R11_BURST_SEMANTIC_COMPLETION_FAILED:{running.work.work_id}"
                )
                self._fatal.__cause__ = exc
                continue
            self._task_done(running.work)
            self._reservations.pop(work_id, None)
        self._refresh_pressure()
        return progressed

    def _apply_completion(self, work: WorkItem, completion: WorkCompletion) -> None:
        applier = self._appliers.get(work.kind)
        if applier is not None:
            result = applier(self.core, work, completion)
            self._require_accepted(result, work.work_id)
            return

        result = completion.result
        if work.kind is WorkKind.TEACHER_ACQUIRE:
            if not isinstance(result, EvidenceRef):
                raise BurstSemanticFailure("R11_BURST_TEACHER_RESULT_TYPE")
            if work.payload_ref is not None and work.payload_ref != result.evidence_id:
                raise LineageViolation("R11_BURST_TEACHER_PAYLOAD_REF_MISMATCH")
            self.core.accept_teacher_evidence(result)
            self._mark_extra_completion(work.work_id, result.payload_hash)
            return
        if work.kind is WorkKind.POLICY_INFER:
            if not isinstance(result, PolicyResult):
                raise BurstSemanticFailure("R11_BURST_POLICY_RESULT_TYPE")
            self._require_accepted(self.core.accept_policy_result(result), work.work_id)
            return
        if work.kind in {WorkKind.TRACE_EXECUTE, WorkKind.MATERIALIZE_EVIDENCE}:
            if not isinstance(result, EvidenceRef):
                raise BurstSemanticFailure("R11_BURST_TRACE_RESULT_TYPE")
            self._require_accepted(
                self.core.materialize_trace_evidence(result, work_id=work.work_id), work.work_id
            )
            return
        if work.kind is WorkKind.TRAIN_CHALLENGER:
            if not isinstance(result, TrainingResult):
                raise BurstSemanticFailure("R11_BURST_TRAINING_RESULT_TYPE")
            self._require_accepted(self.core.complete_training(result), work.work_id)
            return
        if work.kind is WorkKind.VALIDATE:
            if not isinstance(result, ValidationResult):
                raise BurstSemanticFailure("R11_BURST_VALIDATION_RESULT_TYPE")
            self._require_accepted(self.core.complete_validation(result), work.work_id)
            return
        if work.kind is WorkKind.TOURNAMENT:
            if not isinstance(result, TournamentResult):
                raise BurstSemanticFailure("R11_BURST_TOURNAMENT_RESULT_TYPE")
            self._require_accepted(self.core.decide_tournament(result), work.work_id)
            return
        raise BurstSemanticFailure(f"R11_BURST_NO_COMPLETION_APPLIER:{work.kind.value}")

    @staticmethod
    def _require_accepted(result: object, work_id: str) -> None:
        if isinstance(result, CompletionDisposition) and not result.accepted:
            raise BurstSemanticFailure(f"R11_BURST_COMPLETION_NOT_ACCEPTED:{work_id}")

    def _mark_extra_completion(self, work_id: str, payload_hash: str) -> None:
        marker = getattr(self.core, "_mark_completion", None)
        if marker is not None:
            marker(work_id, payload_hash)
        self._extra_completed.add(work_id)

    def _task_done(self, work: WorkItem) -> None:
        try:
            self.core.queues.task_done(work.pool)
        except (AttributeError, ValueError):
            pass

    def _completed_ids(self) -> set[str]:
        return set(self.core.record.completed_work_ids) | set(self._extra_completed)

    def _outstanding_ids(self) -> set[str]:
        scheduled = set(self._reservations)
        return scheduled - self._completed_ids()

    def _work_for_id(self, work_id: str) -> WorkItem:
        reservation = self._reservations.get(work_id)
        if reservation is not None:
            return reservation.work
        return self.core.record.scheduled_work[work_id]

    def _evidence_reserved(self) -> int:
        completed = self._completed_ids()
        return sum(
            reservation.evidence_bytes
            for work_id, reservation in self._reservations.items()
            if work_id not in completed
        )

    def _storage_depth(self) -> int:
        completed = self._completed_ids()
        return sum(
            1
            for work_id, reservation in self._reservations.items()
            if work_id not in completed and reservation.work.pool is WorkerPool.STORAGE
        )

    def _count_running(self, pools: tuple[WorkerPool, ...]) -> int:
        return sum(1 for item in self._running.values() if item.work.pool in pools)

    def _raise_if_fatal(self) -> None:
        if self._fatal is not None:
            raise self._fatal


STATE_DAG_R11_TASK_D = (
    "PREPARING: frozen-input verification + bounded Teacher acquisition",
    "SNAPSHOT_BARRIER: all admitted Teacher work durable -> immutable snapshot seal",
    "GPU_POLICY: current Champion policy inference",
    "TRACE_ADMISSION: current Champion trace work durably admitted",
    "OVERLAP_REGION_A: admitted CPU trace || GPU Challenger training",
    "OVERLAP_REGION_B: admitted CPU trace/storage || GPU validation/tournament",
    "TOURNAMENT_BARRIER: PROMOTE/REJECT atomic commit",
    "CHECKPOINT_BARRIER: durable checkpoint seal",
    "DRAIN: no queued/running work and no orphan GPU job",
    "NEXT_GENERATION_RELEASE: only after committed parent barrier",
)

INTEGRATION_REQUIREMENTS_R11_TASK_D = (
    "Teacher WorkItem.payload_ref should equal EvidenceRef.evidence_id for crash-resume reconciliation",
    "all trace work intended to overlap training must be enqueue-admitted while TRACE_RUNNING",
    "training WorkItem must be admitted before start_challenger_training advances the state",
    "storage adapters must return WorkCompletion with exact generation/work identity",
    "Task-F owns the full ~10 minute integrated burst and may tune BurstLimits from Task-A/C budgets",
)
