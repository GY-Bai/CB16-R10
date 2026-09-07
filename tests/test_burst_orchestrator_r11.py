from __future__ import annotations

from collections import defaultdict
from threading import Event

import pytest

from cb16_local_opt.burst_orchestrator_r11 import (
    BurstBackpressure,
    BurstBarrierViolation,
    BurstLimits,
    BurstOrchestratorR11,
    BurstPhase,
    BurstSemanticFailure,
    RamSample,
)
from cb16_local_opt.orchestrator_r11 import (
    BoundedWorkQueues,
    LineageViolation,
    QueueLimits,
    R11Orchestrator,
)
from cb16_local_opt.runtime_events_r11 import (
    AuthorityStamp,
    CommitReceipt,
    EvidenceRef,
    FrozenInputReceipt,
    GenerationState,
    JournalReceipt,
    PolicyResult,
    SnapshotSeal,
    StoreReceipt,
    TournamentDecision,
    TournamentResult,
    TrainingResult,
    ValidationResult,
    WorkerPool,
    WorkCompletion,
    WorkItem,
    WorkKind,
)


AUTH = AuthorityStamp(3, "champion-c3", "hash-c3", "teacher-v7", "physics-frozen-v1")
RAM_OK = lambda: RamSample(1, 100)


class FakeEvidenceStore:
    def __init__(self):
        self.items = {}
        self.snapshots = {}

    def put_once(self, evidence):
        old = self.items.get(evidence.evidence_id)
        if old is None:
            self.items[evidence.evidence_id] = evidence
            return StoreReceipt(evidence.evidence_id, evidence.payload_hash, True, True)
        if old != evidence:
            return StoreReceipt(evidence.evidence_id, old.payload_hash, True, False)
        return StoreReceipt(evidence.evidence_id, evidence.payload_hash, True, False)

    def get(self, evidence_id):
        return self.items.get(evidence_id)

    def seal_snapshot(self, seal):
        old = self.snapshots.get(seal.snapshot_id)
        if old is not None and old != seal:
            raise RuntimeError("snapshot conflict")
        self.snapshots[seal.snapshot_id] = seal
        return seal

    def get_snapshot(self, snapshot_id):
        return self.snapshots.get(snapshot_id)


class FakeJournal:
    def __init__(self):
        self.events = []
        self.ids = {}

    def append_once(self, event):
        if event.event_id in self.ids:
            return JournalReceipt(event.event_id, self.ids[event.event_id], True)
        sequence = len(self.events) + 1
        self.events.append(event)
        self.ids[event.event_id] = sequence
        return JournalReceipt(event.event_id, sequence, True)

    def read_generation(self, generation):
        return [event for event in self.events if event.generation == generation]


class FakeCheckpointStore:
    def __init__(self):
        self.commits = {}
        self.checkpoints = {}

    def atomic_commit(self, proposal):
        if proposal.generation in self.commits:
            return self.commits[proposal.generation]
        if proposal.decision is TournamentDecision.PROMOTE:
            next_id, next_hash = proposal.challenger_id, proposal.challenger_hash
        else:
            next_id, next_hash = proposal.parent_champion_id, AUTH.champion_hash
        receipt = CommitReceipt(
            proposal.generation,
            proposal.decision,
            proposal.parent_champion_id,
            next_id,
            next_hash,
            f"commit-{proposal.generation}-{proposal.decision.value.lower()}",
        )
        self.commits[proposal.generation] = receipt
        return receipt

    def read_commit(self, generation):
        return self.commits.get(generation)

    def seal_checkpoint(self, seal):
        self.checkpoints[seal.generation] = seal
        return seal

    def read_checkpoint(self, generation):
        return self.checkpoints.get(generation)


def old_evidence(eid="e-old"):
    return EvidenceRef(
        eid,
        f"hash-{eid}",
        2,
        "champion-c2",
        AUTH.teacher_authority_id,
        AUTH.physics_authority_id,
        2,
    )


def trace_evidence(eid):
    return EvidenceRef(
        eid,
        f"hash-{eid}",
        3,
        AUTH.champion_id,
        AUTH.teacher_authority_id,
        AUTH.physics_authority_id,
        3,
    )


def completion(work, result):
    return WorkCompletion(
        work_id=work.work_id,
        kind=work.kind,
        generation=work.generation,
        payload_hash=result.payload_hash,
        result=result,
    )


def new_core(*, evidence=None, journal=None, checkpoints=None, queues=None, recover=False):
    return R11Orchestrator(
        authority=AUTH,
        evidence_store=evidence or FakeEvidenceStore(),
        journal=journal or FakeJournal(),
        checkpoint_store=checkpoints or FakeCheckpointStore(),
        queues=queues,
        recover=recover,
    )


def to_snapshot(core):
    core.verify_frozen_inputs(FrozenInputReceipt("inputs", "seal"))
    core.accept_teacher_evidence(old_evidence())
    seal = SnapshotSeal("snapshot-g3", "snapshot-hash", 3, AUTH.champion_id, ("e-old",))
    core.seal_training_snapshot(seal)
    return seal


def to_trace(core):
    seal = to_snapshot(core)
    core.accept_policy_result(PolicyResult("policy-direct", 3, AUTH.champion_id, "policy-hash"))
    core.start_trace("trace-start")
    return seal


def direct_complete_training(core, seal=None):
    seal = seal or core.record.snapshot
    core.start_challenger_training(work_id="train-direct")
    core.complete_training(
        TrainingResult(
            "train-direct",
            3,
            "challenger-x",
            "hash-x",
            AUTH.champion_id,
            seal.snapshot_id,
            "train-hash",
        )
    )


def start_burst(core, *, runners=None, limits=None):
    burst = BurstOrchestratorR11(
        core,
        limits=limits or BurstLimits(cpu_workers=2, max_pending_work=12),
        runners=runners or {},
        ram_probe=RAM_OK,
    )
    burst.start()
    burst.warmup(())
    burst.begin_burst()
    return burst


def trace_work(work_id):
    return WorkItem(work_id, WorkKind.TRACE_EXECUTE, 3, WorkerPool.CPU_TRACE, AUTH.champion_id)


def train_work(work_id="train-w"):
    return WorkItem(
        work_id,
        WorkKind.TRAIN_CHALLENGER,
        3,
        WorkerPool.GPU_BRAIN,
        AUTH.champion_id,
        snapshot_id="snapshot-g3",
    )


def validate_work(work_id="validate-w"):
    return WorkItem(
        work_id,
        WorkKind.VALIDATE,
        3,
        WorkerPool.GPU_BRAIN,
        AUTH.champion_id,
        snapshot_id="snapshot-g3",
    )


def tournament_work(work_id="tournament-w"):
    return WorkItem(
        work_id,
        WorkKind.TOURNAMENT,
        3,
        WorkerPool.GPU_BRAIN,
        AUTH.champion_id,
        snapshot_id="snapshot-g3",
    )


def materialize_work(work_id):
    return WorkItem(work_id, WorkKind.MATERIALIZE_EVIDENCE, 3, WorkerPool.STORAGE, AUTH.champion_id)


def test_cpu_slow_gpu_fast_overlap_and_late_trace_acceptance():
    core = new_core()
    to_trace(core)
    trace_started, release_trace, gpu_started = Event(), Event(), Event()

    def run_trace(work):
        trace_started.set()
        assert release_trace.wait(2)
        return completion(work, trace_evidence("trace-slow"))

    def run_train(work):
        gpu_started.set()
        return completion(
            work,
            TrainingResult(
                work.work_id, 3, "challenger-x", "hash-x", AUTH.champion_id,
                "snapshot-g3", "train-hash"
            ),
        )

    burst = start_burst(core, runners={WorkKind.TRACE_EXECUTE: run_trace, WorkKind.TRAIN_CHALLENGER: run_train})
    burst.schedule(trace_work("trace-slow-w"), evidence_bytes=1024)
    burst.start_challenger_training(train_work())
    burst.pump_once()
    assert trace_started.wait(1)
    assert gpu_started.wait(1)
    burst.wait_for(("train-w",), timeout=2)
    assert core.record.state is GenerationState.VALIDATING
    assert "trace-slow-w" not in core.record.completed_work_ids
    release_trace.set()
    burst.wait_for(("trace-slow-w",), timeout=2)
    assert "trace-slow" in core.record.trace_evidence_ids


def test_gpu_slow_cpu_fast_overlap():
    core = new_core()
    to_trace(core)
    release_gpu, gpu_started = Event(), Event()

    def run_trace(work):
        return completion(work, trace_evidence("trace-fast"))

    def run_train(work):
        gpu_started.set()
        assert release_gpu.wait(2)
        return completion(
            work,
            TrainingResult(
                work.work_id, 3, "challenger-x", "hash-x", AUTH.champion_id,
                "snapshot-g3", "train-hash"
            ),
        )

    burst = start_burst(core, runners={WorkKind.TRACE_EXECUTE: run_trace, WorkKind.TRAIN_CHALLENGER: run_train})
    burst.schedule(trace_work("trace-fast-w"))
    burst.start_challenger_training(train_work())
    burst.pump_once()
    assert gpu_started.wait(1)
    burst.wait_for(("trace-fast-w",), timeout=2)
    assert core.record.state is GenerationState.CHALLENGER_TRAINING
    release_gpu.set()
    burst.wait_for(("train-w",), timeout=2)
    assert core.record.state is GenerationState.VALIDATING


def test_hdd_slow_storage_overlaps_gpu_validation():
    core = new_core()
    seal = to_trace(core)
    direct_complete_training(core, seal)
    release_storage, storage_started = Event(), Event()

    def run_storage(work):
        storage_started.set()
        assert release_storage.wait(2)
        return completion(work, trace_evidence("storage-late"))

    def run_validate(work):
        return completion(
            work,
            ValidationResult(
                work.work_id, 3, "challenger-x", AUTH.champion_id,
                "snapshot-g3", "validation-v", "validation-hash"
            ),
        )

    burst = start_burst(core, runners={WorkKind.MATERIALIZE_EVIDENCE: run_storage, WorkKind.VALIDATE: run_validate})
    burst.schedule(materialize_work("storage-w"), evidence_bytes=4096)
    burst.schedule(validate_work())
    burst.pump_once()
    assert storage_started.wait(1)
    burst.wait_for(("validate-w",), timeout=2)
    assert core.record.state is GenerationState.TOURNAMENT_PENDING
    release_storage.set()
    burst.wait_for(("storage-w",), timeout=2)
    assert "storage-late" in core.record.trace_evidence_ids


def test_queue_full_is_durably_accounted_and_pumped_later():
    queues = BoundedWorkQueues(QueueLimits(cpu_trace=1))
    core = new_core(queues=queues)
    to_trace(core)

    def run_trace(work):
        return completion(work, trace_evidence(work.work_id + "-e"))

    burst = start_burst(core, runners={WorkKind.TRACE_EXECUTE: run_trace})
    assert burst.schedule(trace_work("q1")) is True
    assert burst.schedule(trace_work("q2")) is False
    assert set(core.record.scheduled_work) >= {"q1", "q2"}
    burst.wait_for(("q1", "q2"), timeout=2)
    assert {"q1", "q2"}.issubset(core.record.completed_work_ids)


def test_global_high_watermark_refuses_unbounded_pending_queue():
    core = new_core()
    to_trace(core)
    burst = start_burst(core, runners={}, limits=BurstLimits(cpu_workers=1, max_pending_work=1))
    burst.schedule(trace_work("only"))
    with pytest.raises(BurstBackpressure, match="GLOBAL_PENDING"):
        burst.schedule(trace_work("refused"))
    assert "refused" not in core.record.scheduled_work


def test_out_of_order_trace_completion_is_deterministic():
    core = new_core()
    to_trace(core)
    release_slow, slow_started = Event(), Event()

    def run_trace(work):
        if work.work_id == "slow":
            slow_started.set()
            assert release_slow.wait(2)
        return completion(work, trace_evidence("e-" + work.work_id))

    burst = start_burst(core, runners={WorkKind.TRACE_EXECUTE: run_trace})
    burst.schedule(trace_work("slow"))
    burst.schedule(trace_work("fast"))
    burst.pump_once()
    assert slow_started.wait(1)
    burst.wait_for(("fast",), timeout=2)
    assert "slow" not in core.record.completed_work_ids
    release_slow.set()
    burst.wait_for(("slow",), timeout=2)
    assert core.record.trace_evidence_ids == {"e-fast", "e-slow"}


def test_stale_champion_result_fails_closed():
    core = new_core()
    seal = to_snapshot(core)

    def stale_policy(work):
        return completion(work, PolicyResult(work.work_id, 3, "champion-stale", "policy-stale"))

    burst = start_burst(core, runners={WorkKind.POLICY_INFER: stale_policy})
    work = WorkItem(
        "policy-w", WorkKind.POLICY_INFER, 3, WorkerPool.GPU_BRAIN,
        AUTH.champion_id, snapshot_id=seal.snapshot_id
    )
    burst.schedule(work)
    with pytest.raises(BurstSemanticFailure) as caught:
        burst.wait_for(("policy-w",), timeout=2)
    assert isinstance(caught.value.__cause__, LineageViolation)
    assert burst.phase is BurstPhase.FAILED


def test_worker_exception_requeues_same_scientific_identity_then_succeeds():
    core = new_core()
    to_trace(core)
    calls = defaultdict(int)

    def flaky(work):
        calls[work.work_id] += 1
        if calls[work.work_id] == 1:
            raise RuntimeError("synthetic worker failure")
        assert work.attempt == 1
        return completion(work, trace_evidence("retry-e"))

    burst = start_burst(
        core,
        runners={WorkKind.TRACE_EXECUTE: flaky},
        limits=BurstLimits(cpu_workers=1, max_pending_work=4, max_retries=1),
    )
    burst.schedule(trace_work("retry-w"))
    burst.wait_for(("retry-w",), timeout=2)
    assert calls["retry-w"] == 2
    assert core.record.scheduled_work["retry-w"].attempt == 1
    assert "retry-e" in core.record.trace_evidence_ids


def _drive_generation_with_burst(decision):
    core = new_core()
    to_trace(core)

    def run_train(work):
        return completion(
            work,
            TrainingResult(
                work.work_id, 3, "challenger-x", "hash-x", AUTH.champion_id,
                "snapshot-g3", "train-hash"
            ),
        )

    def run_validate(work):
        return completion(
            work,
            ValidationResult(
                work.work_id, 3, "challenger-x", AUTH.champion_id,
                "snapshot-g3", "validation-v", "validation-hash"
            ),
        )

    def run_tournament(work):
        return completion(
            work,
            TournamentResult(
                work.work_id, 3, "challenger-x", AUTH.champion_id,
                "validation-v", decision, f"tournament-{decision.value.lower()}"
            ),
        )

    burst = start_burst(
        core,
        runners={
            WorkKind.TRAIN_CHALLENGER: run_train,
            WorkKind.VALIDATE: run_validate,
            WorkKind.TOURNAMENT: run_tournament,
        },
    )
    burst.start_challenger_training(train_work())
    burst.wait_for(("train-w",), timeout=2)
    burst.schedule(validate_work())
    burst.wait_for(("validate-w",), timeout=2)
    burst.schedule(tournament_work())
    burst.wait_for(("tournament-w",), timeout=2)
    core.commit_tournament()
    core.seal_checkpoint("checkpoint-g3")
    return core, burst


@pytest.mark.parametrize(
    "decision, expected_id, expected_hash",
    [
        (TournamentDecision.REJECT, AUTH.champion_id, AUTH.champion_hash),
        (TournamentDecision.PROMOTE, "challenger-x", "hash-x"),
    ],
)
def test_reject_and_promote_parent_barrier(decision, expected_id, expected_hash):
    core, burst = _drive_generation_with_burst(decision)
    nxt = burst.release_next_generation()
    assert nxt.champion_id == expected_id
    assert nxt.champion_hash == expected_hash
    assert nxt.generation == 4


def test_stop_requested_mid_generation_drains_current_work_but_never_releases_future_parent():
    core = new_core()
    to_trace(core)
    release, started = Event(), Event()

    def slow_trace(work):
        started.set()
        assert release.wait(2)
        return completion(work, trace_evidence("stop-e"))

    burst = start_burst(core, runners={WorkKind.TRACE_EXECUTE: slow_trace})
    burst.schedule(trace_work("stop-w"))
    burst.pump_once()
    assert started.wait(1)
    burst.request_stop()
    assert burst.phase is BurstPhase.DRAINING
    with pytest.raises(LineageViolation, match="GENERATION_MIXING"):
        burst.schedule(WorkItem("future", WorkKind.TEACHER_ACQUIRE, 4, WorkerPool.CPU_TEACHER, AUTH.champion_id))
    release.set()
    burst.drain(timeout=2)
    with pytest.raises(BurstBarrierViolation, match="HALF_GENERATION"):
        burst.seal()


def test_graceful_drain_seal_and_clean_exit_has_no_orphan_work():
    core, burst = _drive_generation_with_burst(TournamentDecision.REJECT)
    burst.request_stop()
    burst.drain(timeout=2)
    burst.seal()
    assert burst.status().running == 0
    assert burst.status().queued_or_pending == 0
    assert burst.status().gpu_running == 0
    burst.close()
    assert burst.phase is BurstPhase.CLOSED


def test_forced_crash_resume_rebuilds_durable_pending_work():
    evidence = FakeEvidenceStore()
    journal = FakeJournal()
    checkpoints = FakeCheckpointStore()
    core1 = new_core(evidence=evidence, journal=journal, checkpoints=checkpoints)
    to_trace(core1)
    burst1 = start_burst(core1, runners={})
    burst1.schedule(trace_work("crash-pending"), evidence_bytes=1234)

    core2 = new_core(
        evidence=evidence,
        journal=journal,
        checkpoints=checkpoints,
        queues=BoundedWorkQueues(),
        recover=True,
    )

    def recovered_trace(work):
        return completion(work, trace_evidence("recovered-e"))

    burst2 = start_burst(core2, runners={WorkKind.TRACE_EXECUTE: recovered_trace})
    burst2.wait_for(("crash-pending",), timeout=2)
    assert "recovered-e" in core2.record.trace_evidence_ids
    assert "crash-pending" in core2.record.completed_work_ids


def test_snapshot_barrier_refuses_pending_teacher_and_accepts_after_completion():
    core = new_core()
    core.verify_frozen_inputs(FrozenInputReceipt("inputs", "seal"))
    release = Event()

    def teacher(work):
        assert release.wait(2)
        return completion(work, old_evidence("teacher-new"))

    burst = start_burst(core, runners={WorkKind.TEACHER_ACQUIRE: teacher})
    work = WorkItem(
        "teacher-w", WorkKind.TEACHER_ACQUIRE, 3, WorkerPool.CPU_TEACHER,
        AUTH.champion_id, payload_ref="teacher-new"
    )
    burst.schedule(work, evidence_bytes=256)
    seal = SnapshotSeal("snapshot-g3", "h", 3, AUTH.champion_id, ("teacher-new",))
    with pytest.raises(BurstBarrierViolation, match="PENDING_TEACHER"):
        burst.seal_training_snapshot(seal)
    burst.pump_once()
    release.set()
    burst.wait_for(("teacher-w",), timeout=2)
    burst.seal_training_snapshot(seal)
    assert core.record.snapshot == seal
