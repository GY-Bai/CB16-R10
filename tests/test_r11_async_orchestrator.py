from __future__ import annotations

from dataclasses import replace

import pytest

from cb16_local_opt.orchestrator_r11 import (
    BackpressureRequired,
    BoundedWorkQueues,
    DuplicateConflict,
    IllegalTransition,
    LineageViolation,
    QueueLimits,
    R11Orchestrator,
    WorkerPoolUnavailable,
)
from cb16_local_opt.runtime_events_r11 import (
    AuthorityStamp,
    CheckpointSeal,
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
    WorkItem,
    WorkKind,
)
from cb16_local_opt.runtime_protocols_r11 import EngineBundle


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
        seq = len(self.events) + 1
        self.events.append(event)
        self.ids[event.event_id] = seq
        return JournalReceipt(event.event_id, seq, True)

    def read_generation(self, generation):
        return [e for e in self.events if e.generation == generation]


class FakeCheckpointStore:
    def __init__(self):
        self.commits = {}
        self.checkpoints = {}

    def atomic_commit(self, proposal):
        old = self.commits.get(proposal.generation)
        if old is not None:
            return old
        if proposal.decision is TournamentDecision.PROMOTE:
            next_id = proposal.challenger_id
            next_hash = proposal.challenger_hash
        else:
            next_id = proposal.parent_champion_id
            next_hash = AUTH.champion_hash
        receipt = CommitReceipt(
            generation=proposal.generation,
            decision=proposal.decision,
            parent_champion_id=proposal.parent_champion_id,
            next_champion_id=next_id,
            next_champion_hash=next_hash,
            commit_id=f"commit-g{proposal.generation}-{proposal.decision.value.lower()}",
        )
        self.commits[proposal.generation] = receipt
        return receipt

    def read_commit(self, generation):
        return self.commits.get(generation)

    def seal_checkpoint(self, seal):
        old = self.checkpoints.get(seal.generation)
        if old is not None and old != seal:
            raise RuntimeError("checkpoint conflict")
        self.checkpoints[seal.generation] = seal
        return seal

    def read_checkpoint(self, generation):
        return self.checkpoints.get(generation)


AUTH = AuthorityStamp(3, "champion-c3", "hash-c3", "teacher-v7", "physics-frozen-v1")


def old_evidence(eid="e-old", *, poison=(), detached=True, source=2, teacher_generation=2):
    return EvidenceRef(
        evidence_id=eid,
        payload_hash=f"hash-{eid}",
        source_generation=source,
        producer_champion_id="champion-c2",
        teacher_authority_id=AUTH.teacher_authority_id,
        physics_authority_id=AUTH.physics_authority_id,
        teacher_generation=teacher_generation,
        detached_from_autograd=detached,
        poison_bits=tuple(poison),
    )


def trace_evidence(eid="e-trace", *, poison=(), source=3, producer=None, teacher_generation=3):
    return EvidenceRef(
        evidence_id=eid,
        payload_hash=f"hash-{eid}",
        source_generation=source,
        producer_champion_id=producer or AUTH.champion_id,
        teacher_authority_id=AUTH.teacher_authority_id,
        physics_authority_id=AUTH.physics_authority_id,
        teacher_generation=teacher_generation,
        poison_bits=tuple(poison),
    )


def new_runtime(*, queues=None, journal=None, evidence=None, checkpoints=None, recover=False):
    return R11Orchestrator(
        authority=AUTH,
        evidence_store=evidence or FakeEvidenceStore(),
        journal=journal or FakeJournal(),
        checkpoint_store=checkpoints or FakeCheckpointStore(),
        queues=queues,
        recover=recover,
    )


def to_snapshot(rt, evidence_ids=("e-old",)):
    rt.verify_frozen_inputs(FrozenInputReceipt("inputs", "seal"))
    for eid in evidence_ids:
        rt.accept_teacher_evidence(old_evidence(eid))
    seal = SnapshotSeal(
        "snapshot-g3",
        "snapshot-hash",
        3,
        AUTH.champion_id,
        tuple(evidence_ids),
    )
    rt.seal_training_snapshot(seal)
    return seal


def to_trace(rt):
    seal = to_snapshot(rt)
    rt.accept_policy_result(PolicyResult("policy-w", 3, AUTH.champion_id, "policy-hash"))
    rt.start_trace("trace-w")
    return seal


def to_training(rt):
    seal = to_trace(rt)
    rt.start_challenger_training(work_id="train-w")
    return seal


def complete_train(rt, seal=None):
    seal = seal or rt.record.snapshot
    result = TrainingResult(
        "train-w", 3, "challenger-x", "hash-x", AUTH.champion_id, seal.snapshot_id, "train-hash"
    )
    rt.complete_training(result)
    return result


def complete_validate(rt):
    result = ValidationResult(
        "validate-w",
        3,
        "challenger-x",
        AUTH.champion_id,
        rt.record.snapshot.snapshot_id,
        "validation-v",
        "validation-hash",
    )
    rt.complete_validation(result)
    return result


def complete_tournament(rt, decision):
    result = TournamentResult(
        "tournament-w",
        3,
        "challenger-x",
        AUTH.champion_id,
        "validation-v",
        decision,
        f"tournament-{decision.value.lower()}-hash",
    )
    rt.decide_tournament(result)
    return result


def to_committing(rt, decision=TournamentDecision.REJECT):
    seal = to_training(rt)
    complete_train(rt, seal)
    complete_validate(rt)
    complete_tournament(rt, decision)


def test_teacher_completions_out_of_order_and_snapshot_is_hard_barrier():
    rt = new_runtime()
    rt.verify_frozen_inputs(FrozenInputReceipt("inputs", "seal"))
    rt.accept_teacher_evidence(old_evidence("e2"))
    rt.accept_teacher_evidence(old_evidence("e1"))
    rt.seal_training_snapshot(SnapshotSeal("s", "h", 3, AUTH.champion_id, ("e1", "e2")))
    assert rt.record.state is GenerationState.SNAPSHOT_SEALED
    with pytest.raises(IllegalTransition):
        rt.accept_teacher_evidence(old_evidence("late"))


def test_future_evidence_and_future_teacher_are_fail_closed():
    rt = new_runtime()
    rt.verify_frozen_inputs(FrozenInputReceipt("inputs", "seal"))
    with pytest.raises(LineageViolation, match="FUTURE_EVIDENCE"):
        rt.accept_teacher_evidence(old_evidence("future", source=3, teacher_generation=3))
    with pytest.raises(LineageViolation, match="FUTURE_TEACHER"):
        rt.accept_teacher_evidence(old_evidence("teacher-future", source=2, teacher_generation=3))


def test_teacher_student_autograd_and_poison_bits_fail_closed():
    rt = new_runtime()
    rt.verify_frozen_inputs(FrozenInputReceipt("inputs", "seal"))
    with pytest.raises(LineageViolation, match="AUTOGRAD"):
        rt.accept_teacher_evidence(old_evidence("grad", detached=False))
    with pytest.raises(LineageViolation, match="POISON_OR_FUTURE_TAINT"):
        rt.accept_teacher_evidence(old_evidence("poison", poison=("FUTURE_TAINT",)))


def test_stale_policy_result_arriving_late_is_rejected():
    rt = new_runtime()
    to_snapshot(rt)
    with pytest.raises(LineageViolation, match="GENERATION_MIXING"):
        rt.accept_policy_result(PolicyResult("old-policy", 2, "champion-c2", "old"))


def test_trace_evidence_is_current_champion_owned_and_not_in_sealed_snapshot():
    rt = new_runtime()
    seal = to_trace(rt)
    rt.materialize_trace_evidence(trace_evidence(), work_id="trace-completion")
    assert "e-trace" in rt.record.trace_evidence_ids
    assert "e-trace" not in seal.evidence_ids
    with pytest.raises(LineageViolation, match="STALE_POLICY_LINEAGE"):
        rt.materialize_trace_evidence(trace_evidence("wrong", producer="challenger-x"), work_id="wrong")


def test_duplicate_completion_is_idempotent_but_conflict_is_rejected():
    rt = new_runtime()
    seal = to_training(rt)
    result = complete_train(rt, seal)
    dup = rt.complete_training(result)
    assert dup.idempotent_duplicate is True
    with pytest.raises(DuplicateConflict):
        rt.complete_training(replace(result, payload_hash="different"))


def test_worker_crash_requeues_same_scientific_work_identity():
    rt = new_runtime()
    work = WorkItem("w1", WorkKind.TEACHER_ACQUIRE, 3, WorkerPool.CPU_TEACHER, AUTH.champion_id)
    retried = rt.worker_crashed(work)
    assert retried.work_id == work.work_id
    assert retried.attempt == 1
    claimed = rt.queues.claim(WorkerPool.CPU_TEACHER)
    assert claimed == retried


def test_storage_backpressure_never_drops_work():
    queues = BoundedWorkQueues(QueueLimits(storage=1))
    rt = new_runtime(queues=queues)
    to_trace(rt)
    a = WorkItem("a", WorkKind.MATERIALIZE_EVIDENCE, 3, WorkerPool.STORAGE, AUTH.champion_id)
    b = WorkItem("b", WorkKind.MATERIALIZE_EVIDENCE, 3, WorkerPool.STORAGE, AUTH.champion_id)
    rt.enqueue(a)
    with pytest.raises(BackpressureRequired):
        rt.enqueue(b)
    assert queues.depth(WorkerPool.STORAGE) == 1
    assert queues.claim(WorkerPool.STORAGE).work_id == "a"


def test_gpu_unavailable_retains_queued_work():
    queues = BoundedWorkQueues()
    rt = new_runtime(queues=queues)
    seal = to_snapshot(rt)
    work = WorkItem("gpu", WorkKind.POLICY_INFER, 3, WorkerPool.GPU_BRAIN, AUTH.champion_id, snapshot_id=seal.snapshot_id)
    rt.enqueue(work)
    queues.set_available(WorkerPool.GPU_BRAIN, False)
    with pytest.raises(WorkerPoolUnavailable):
        queues.claim(WorkerPool.GPU_BRAIN)
    assert queues.depth(WorkerPool.GPU_BRAIN) == 1
    queues.set_available(WorkerPool.GPU_BRAIN, True)
    assert queues.claim(WorkerPool.GPU_BRAIN) == work


def test_work_stealing_is_explicitly_limited_to_compatible_pools():
    queues = BoundedWorkQueues()
    teacher = WorkItem("teacher", WorkKind.TEACHER_ACQUIRE, 3, WorkerPool.CPU_TEACHER, AUTH.champion_id)
    trace = WorkItem("trace", WorkKind.TRACE_EXECUTE, 3, WorkerPool.CPU_TRACE, AUTH.champion_id)
    queues.enqueue(trace)
    queues.enqueue(teacher)
    stolen = queues.claim_any((WorkerPool.CPU_TEACHER, WorkerPool.CPU_TRACE))
    assert stolen == teacher
    assert queues.claim_any((WorkerPool.CPU_TEACHER, WorkerPool.CPU_TRACE)) == trace


@pytest.mark.parametrize(
    "decision, expected_id, expected_hash",
    [
        (TournamentDecision.REJECT, AUTH.champion_id, AUTH.champion_hash),
        (TournamentDecision.PROMOTE, "challenger-x", "hash-x"),
    ],
)
def test_tournament_commit_barrier_reject_and_promote(decision, expected_id, expected_hash):
    rt = new_runtime()
    to_committing(rt, decision)
    with pytest.raises(IllegalTransition):
        rt.release_next_generation()
    rt.commit_tournament()
    with pytest.raises(IllegalTransition):
        rt.release_next_generation()
    rt.seal_checkpoint("checkpoint-g3")
    nxt = rt.release_next_generation()
    assert nxt.generation == 4
    assert nxt.champion_id == expected_id
    assert nxt.champion_hash == expected_hash


def test_future_generation_result_injected_into_current_generation_fails_closed():
    rt = new_runtime()
    seal = to_training(rt)
    future = TrainingResult(
        "future-train", 4, "bad", "bad-h", AUTH.champion_id, seal.snapshot_id, "bad-payload"
    )
    with pytest.raises(LineageViolation, match="GENERATION_MIXING"):
        rt.complete_training(future)


def test_trace_completion_can_arrive_after_gpu_training_and_validation_started():
    rt = new_runtime()
    seal = to_training(rt)
    complete_train(rt, seal)
    # Current Champion trace is an asynchronous side lane and may materialize late.
    disp = rt.materialize_trace_evidence(trace_evidence("late-trace"), work_id="late-trace-work")
    assert disp.accepted
    assert rt.record.state is GenerationState.VALIDATING


def build_to_state(target):
    evidence = FakeEvidenceStore()
    journal = FakeJournal()
    checkpoints = FakeCheckpointStore()
    rt = new_runtime(evidence=evidence, journal=journal, checkpoints=checkpoints)
    if target is GenerationState.PREPARING:
        return rt, evidence, journal, checkpoints
    to_snapshot(rt)
    if target is GenerationState.SNAPSHOT_SEALED:
        return rt, evidence, journal, checkpoints
    rt.accept_policy_result(PolicyResult("policy-w", 3, AUTH.champion_id, "policy-hash"))
    rt.start_trace("trace-w")
    if target is GenerationState.TRACE_RUNNING:
        return rt, evidence, journal, checkpoints
    rt.start_challenger_training(work_id="train-w")
    if target is GenerationState.CHALLENGER_TRAINING:
        return rt, evidence, journal, checkpoints
    complete_train(rt)
    if target is GenerationState.VALIDATING:
        return rt, evidence, journal, checkpoints
    complete_validate(rt)
    if target is GenerationState.TOURNAMENT_PENDING:
        return rt, evidence, journal, checkpoints
    complete_tournament(rt, TournamentDecision.REJECT)
    if target is GenerationState.COMMITTING:
        return rt, evidence, journal, checkpoints
    rt.commit_tournament()
    rt.seal_checkpoint("cp")
    return rt, evidence, journal, checkpoints


@pytest.mark.parametrize("state", list(GenerationState))
def test_restart_in_every_generation_state_is_deterministic(state):
    original, evidence, journal, checkpoints = build_to_state(state)
    recovered = new_runtime(
        evidence=evidence, journal=journal, checkpoints=checkpoints, recover=True
    )
    assert recovered.record.state is state
    assert recovered.record.authority == original.record.authority
    assert recovered.record.snapshot == original.record.snapshot
    assert recovered.record.challenger_id == original.record.challenger_id
    assert recovered.record.validation_id == original.record.validation_id
    if state is GenerationState.COMMITTED:
        assert recovered.record.commit == original.record.commit
        assert recovered.record.checkpoint == original.record.checkpoint


def test_recovery_reconciles_crash_after_atomic_commit_before_journal_ack():
    rt, evidence, journal, checkpoints = build_to_state(GenerationState.COMMITTING)
    proposal = rt.record.tournament
    # Simulate store-side atomic commit completing before the orchestrator journals it.
    receipt = checkpoints.atomic_commit(
        __import__("cb16_local_opt.runtime_events_r11", fromlist=["TournamentCommitProposal"]).TournamentCommitProposal(
            3,
            proposal.decision,
            AUTH.champion_id,
            rt.record.challenger_id,
            rt.record.challenger_hash,
        )
    )
    recovered = new_runtime(evidence=evidence, journal=journal, checkpoints=checkpoints, recover=True)
    assert recovered.record.state is GenerationState.COMMITTING
    assert recovered.record.commit == receipt


def test_snapshot_rejects_duplicate_evidence_ids():
    rt = new_runtime()
    rt.verify_frozen_inputs(FrozenInputReceipt("inputs", "seal"))
    rt.accept_teacher_evidence(old_evidence("e1"))
    with pytest.raises((DuplicateConflict, LineageViolation)):
        rt.seal_training_snapshot(SnapshotSeal("s", "h", 3, AUTH.champion_id, ("e1", "e1")))


def test_backpressured_work_remains_durable_and_can_be_pumped_later():
    queues = BoundedWorkQueues(QueueLimits(storage=1))
    journal = FakeJournal()
    rt = new_runtime(queues=queues, journal=journal)
    to_trace(rt)
    a = WorkItem("bp-a", WorkKind.MATERIALIZE_EVIDENCE, 3, WorkerPool.STORAGE, AUTH.champion_id)
    b = WorkItem("bp-b", WorkKind.MATERIALIZE_EVIDENCE, 3, WorkerPool.STORAGE, AUTH.champion_id)
    rt.enqueue(a)
    with pytest.raises(BackpressureRequired):
        rt.enqueue(b)
    assert "bp-b" in rt.pending_work_ids()
    rt.claim_work(WorkerPool.STORAGE)
    assert rt.pump_pending_work() == 1
    assert rt.claim_work(WorkerPool.STORAGE) == b


def test_restart_requeues_durable_incomplete_work_but_not_completed_work():
    evidence = FakeEvidenceStore()
    journal = FakeJournal()
    checkpoints = FakeCheckpointStore()
    queues1 = BoundedWorkQueues()
    rt = new_runtime(evidence=evidence, journal=journal, checkpoints=checkpoints, queues=queues1)
    pending = WorkItem("pending", WorkKind.TEACHER_ACQUIRE, 3, WorkerPool.CPU_TEACHER, AUTH.champion_id)
    rt.enqueue(pending)

    queues2 = BoundedWorkQueues()
    recovered = new_runtime(
        evidence=evidence,
        journal=journal,
        checkpoints=checkpoints,
        queues=queues2,
        recover=True,
    )
    assert recovered.claim_work(WorkerPool.CPU_TEACHER) == pending


def test_missing_gpu_engine_fails_closed_without_consuming_work():
    rt = new_runtime()
    seal = to_snapshot(rt)
    work = WorkItem("gpu-engine", WorkKind.POLICY_INFER, 3, WorkerPool.GPU_BRAIN, AUTH.champion_id, snapshot_id=seal.snapshot_id)
    with pytest.raises(WorkerPoolUnavailable, match="ENGINE_UNAVAILABLE"):
        rt.execute_claimed(work)


def test_reject_receipt_cannot_self_launder_challenger():
    class BadCheckpointStore(FakeCheckpointStore):
        def atomic_commit(self, proposal):
            return CommitReceipt(
                generation=proposal.generation,
                decision=TournamentDecision.REJECT,
                parent_champion_id=proposal.parent_champion_id,
                next_champion_id=proposal.challenger_id,
                next_champion_hash=proposal.challenger_hash,
                commit_id="bad-self-launder",
            )

    rt = new_runtime(checkpoints=BadCheckpointStore())
    to_committing(rt, TournamentDecision.REJECT)
    with pytest.raises(LineageViolation, match="SELF_LAUNDERING"):
        rt.commit_tournament()


def test_work_stage_gating_rejects_training_before_trace_barrier():
    rt = new_runtime()
    work = WorkItem(
        "early-train",
        WorkKind.TRAIN_CHALLENGER,
        3,
        WorkerPool.GPU_BRAIN,
        AUTH.champion_id,
    )
    with pytest.raises(IllegalTransition, match="WORK_STAGE_VIOLATION"):
        rt.enqueue(work)


def test_work_kind_cannot_be_routed_to_wrong_worker_pool():
    rt = new_runtime()
    work = WorkItem(
        "misrouted-teacher",
        WorkKind.TEACHER_ACQUIRE,
        3,
        WorkerPool.GPU_BRAIN,
        AUTH.champion_id,
    )
    with pytest.raises(LineageViolation, match="WORK_POOL_MISMATCH"):
        rt.enqueue(work)
