from __future__ import annotations

from dataclasses import dataclass

import pytest

from cb16_local_opt.runtime_events_r11 import (
    AuthorityStamp,
    PolicyResult,
    TournamentCommitProposal,
    TournamentDecision,
    TournamentResult,
    TrainingResult,
    ValidationResult,
    WorkCompletion,
    WorkItem,
    WorkKind,
    WorkerPool,
)
from cb16_local_opt.runtime_protocols_r11 import EngineBundle
from cb16_local_opt.stage4_authoritative_engines_r11 import (
    CanonicalEngineAuthorityContextR11,
    EngineCompletionRejected,
    EngineIdentityBindingR11,
    EngineLineageMismatch,
    InvalidEngineAuthorityContext,
    TrainingNumericIdentityUnproven,
    build_authoritative_engine_bundle_r11,
)
from cb16_local_opt.stage4_canonical_runtime_r11 import (
    AuthorityBinding,
    AuthorityLease,
    RecoveredRuntimeState,
    RuntimeContext,
)
from cb16_local_opt.stage4_legacy_retirement_r11 import (
    Capability,
    LegacyRetirementGuard,
    RuntimeRole,
)


KEY = b"stage4-intc-test-key-material-0000000000000000"
GENERATION = 7
AUTHORITY_ID = "AUTHORITY:G7"
CHAMPION_ID = "CHAMPION:G6"
CHAMPION_HASH = "champion-hash-g6"


@dataclass(frozen=True)
class _Config:
    amp_enabled: bool = False
    dtype: str = "torch.float32"


class FakeEngine:
    def __init__(self, result, *, config=None):
        self.result = result
        self.calls = 0
        self.last_work = None
        self.promote_calls = 0
        self.gradient_owners = frozenset(
            {
                "Operator Brain Stem",
                "Medium Brain Stem",
                "Account Brain Stem",
                "Shared Decision Core",
                "Direction Head",
                "Requested-Risk Head",
            }
        )
        if config is not None:
            self.config = config

    def execute(self, work: WorkItem) -> WorkCompletion:
        self.calls += 1
        self.last_work = work
        return WorkCompletion(
            work_id=work.work_id,
            kind=work.kind,
            generation=work.generation,
            payload_hash="completion-payload-hash",
            result=self.result,
        )

    def promote(self):
        self.promote_calls += 1


class FakeFence:
    def __init__(self, expected_lease, *, fail_on_call: int | None = None):
        self.expected_lease = expected_lease
        self.fail_on_call = fail_on_call
        self.calls = 0

    def assert_current(self, lease: AuthorityLease) -> None:
        self.calls += 1
        assert lease is self.expected_lease
        if self.fail_on_call == self.calls:
            raise RuntimeError("STALE_FENCE")


def _guard() -> LegacyRetirementGuard:
    return LegacyRetirementGuard(KEY, issuer_id="INTC_TEST_ISSUER")


def _identity_binding(guard: LegacyRetirementGuard, subject: str, role=RuntimeRole.R11_CANONICAL_RUNTIME):
    identity = guard.issue_identity(subject, role)
    if role is RuntimeRole.R11_CANONICAL_RUNTIME:
        token = guard.issue_token(identity, Capability.ACQUIRE_CANONICAL_RUNTIME_AUTHORITY)
    else:
        # A safe token is deliberately insufficient for authoritative execution.
        safe_cap = (
            Capability.ENGINEERING_REPLAY
            if role in {RuntimeRole.LEGACY_REPLAY, RuntimeRole.TEST}
            else Capability.RUN_DIAGNOSTICS
        )
        token = guard.issue_token(identity, safe_cap)
    return EngineIdentityBindingR11(identity=identity, admission_token=token)


def _authority(guard: LegacyRetirementGuard, *, runtime_role=RuntimeRole.R11_CANONICAL_RUNTIME):
    binding = AuthorityBinding(
        authority_id=AUTHORITY_ID,
        generation=GENERATION,
        semantic_freeze_id="CB16_SEMANTIC_FREEZE_V1",
    )
    recovered = RecoveredRuntimeState(
        authority_id=AUTHORITY_ID,
        generation=GENERATION,
        state_id="STATE:G7",
        scientific_history_id="HISTORY:FROZEN",
    )
    lease = AuthorityLease(authority_id=AUTHORITY_ID, fencing_token="opaque-live-fence")
    runtime = RuntimeContext(binding=binding, recovered=recovered, lease=lease)
    runtime_identity = guard.issue_identity(AUTHORITY_ID, runtime_role)
    if runtime_role is RuntimeRole.R11_CANONICAL_RUNTIME:
        runtime_token = guard.issue_token(
            runtime_identity, Capability.ACQUIRE_CANONICAL_RUNTIME_AUTHORITY
        )
    else:
        runtime_token = guard.issue_token(runtime_identity, Capability.ENGINEERING_REPLAY)
    context = CanonicalEngineAuthorityContextR11(
        runtime_context=runtime,
        authority_stamp=AuthorityStamp(
            generation=GENERATION,
            champion_id=CHAMPION_ID,
            champion_hash=CHAMPION_HASH,
            teacher_authority_id="TEACHER:FROZEN",
            physics_authority_id="PHYSICS:FROZEN",
        ),
        runtime_identity=runtime_identity,
        runtime_admission_token=runtime_token,
    )
    return context, lease


def _work(kind: WorkKind, *, generation=GENERATION, champion=CHAMPION_ID, snapshot="SNAP:G7"):
    pool = {
        WorkKind.TEACHER_ACQUIRE: WorkerPool.CPU_TEACHER,
        WorkKind.TRACE_EXECUTE: WorkerPool.CPU_TRACE,
        WorkKind.POLICY_INFER: WorkerPool.GPU_BRAIN,
        WorkKind.TRAIN_CHALLENGER: WorkerPool.GPU_BRAIN,
        WorkKind.VALIDATE: WorkerPool.GPU_BRAIN,
        WorkKind.TOURNAMENT: WorkerPool.GPU_BRAIN,
    }[kind]
    return WorkItem(
        work_id=f"WORK:{kind.value}",
        kind=kind,
        generation=generation,
        pool=pool,
        parent_champion_id=champion,
        snapshot_id=snapshot,
    )


def _bundle_for(kind: WorkKind, engine: FakeEngine) -> EngineBundle:
    kwargs = {
        "teacher": None,
        "trace": None,
        "inference": None,
        "training": None,
        "validation": None,
        "tournament": None,
    }
    kwargs[
        {
            WorkKind.TEACHER_ACQUIRE: "teacher",
            WorkKind.TRACE_EXECUTE: "trace",
            WorkKind.POLICY_INFER: "inference",
            WorkKind.TRAIN_CHALLENGER: "training",
            WorkKind.VALIDATE: "validation",
            WorkKind.TOURNAMENT: "tournament",
        }[kind]
    ] = engine
    return EngineBundle(**kwargs)


def _wrapped(kind: WorkKind, engine: FakeEngine, *, guard=None, authority=None, identity=None, fence=None):
    guard = guard or _guard()
    authority, lease = authority or _authority(guard)
    identity = identity or _identity_binding(guard, f"ENGINE:{kind.value}")
    fence = fence or FakeFence(lease)
    wrapped = build_authoritative_engine_bundle_r11(
        engines=_bundle_for(kind, engine),
        engine_identities={kind: identity},
        authority_context=authority,
        legacy_guard=guard,
        fence=fence,
    )
    return getattr(
        wrapped,
        {
            WorkKind.TEACHER_ACQUIRE: "teacher",
            WorkKind.TRACE_EXECUTE: "trace",
            WorkKind.POLICY_INFER: "inference",
            WorkKind.TRAIN_CHALLENGER: "training",
            WorkKind.VALIDATE: "validation",
            WorkKind.TOURNAMENT: "tournament",
        }[kind],
    ), fence


def test_teacher_output_and_meaning_are_exact_passthrough():
    result = {"teacher_label_object": object(), "meaning": "UNCHANGED"}
    engine = FakeEngine(result)
    wrapped, fence = _wrapped(WorkKind.TEACHER_ACQUIRE, engine)
    completion = engine.execute(_work(WorkKind.TEACHER_ACQUIRE))
    engine.calls = 0
    engine.result = completion.result
    work = _work(WorkKind.TEACHER_ACQUIRE)
    returned = wrapped.execute(work)
    assert engine.last_work is work
    assert returned.result is result
    assert engine.calls == 1
    assert fence.calls == 2


def test_trace_requested_risk_payload_is_untouched():
    requested_risk = object()
    result = {"requested_risk": requested_risk, "outcome_is_label": False}
    engine = FakeEngine(result)
    wrapped, _ = _wrapped(WorkKind.TRACE_EXECUTE, engine)
    returned = wrapped.execute(_work(WorkKind.TRACE_EXECUTE))
    assert returned.result is result
    assert returned.result["requested_risk"] is requested_risk
    assert returned.result["outcome_is_label"] is False


def test_valid_s4g_admission_without_live_fence_does_not_execute():
    guard = _guard()
    authority, lease = _authority(guard)
    engine = FakeEngine({"teacher": "same"})
    fence = FakeFence(lease, fail_on_call=1)
    wrapped, _ = _wrapped(
        WorkKind.TEACHER_ACQUIRE, engine, guard=guard, authority=(authority, lease), fence=fence
    )
    with pytest.raises(RuntimeError, match="STALE_FENCE"):
        wrapped.execute(_work(WorkKind.TEACHER_ACQUIRE))
    assert engine.calls == 0


def test_stale_fence_after_computation_refuses_output_acceptance():
    guard = _guard()
    authority, lease = _authority(guard)
    engine = FakeEngine({"teacher": "computed"})
    fence = FakeFence(lease, fail_on_call=2)
    wrapped, _ = _wrapped(
        WorkKind.TEACHER_ACQUIRE, engine, guard=guard, authority=(authority, lease), fence=fence
    )
    with pytest.raises(RuntimeError, match="STALE_FENCE"):
        wrapped.execute(_work(WorkKind.TEACHER_ACQUIRE))
    assert engine.calls == 1


def test_wrong_generation_fails_before_engine():
    engine = FakeEngine({})
    wrapped, _ = _wrapped(WorkKind.TRACE_EXECUTE, engine)
    with pytest.raises(EngineLineageMismatch, match="GENERATION"):
        wrapped.execute(_work(WorkKind.TRACE_EXECUTE, generation=GENERATION + 1))
    assert engine.calls == 0


def test_wrong_champion_lineage_fails_before_engine():
    engine = FakeEngine({})
    wrapped, _ = _wrapped(WorkKind.TRACE_EXECUTE, engine)
    with pytest.raises(EngineLineageMismatch, match="CHAMPION_LINEAGE"):
        wrapped.execute(_work(WorkKind.TRACE_EXECUTE, champion="STALE_CHAMPION"))
    assert engine.calls == 0


@pytest.mark.parametrize("role", [RuntimeRole.LEGACY_REPLAY, RuntimeRole.TEST, RuntimeRole.DIAGNOSTIC])
def test_legacy_replay_test_or_diagnostic_engine_identity_cannot_escalate(role):
    guard = _guard()
    authority = _authority(guard)
    engine = FakeEngine({})
    identity = _identity_binding(guard, f"NONCANONICAL:{role.value}", role=role)
    wrapped, _ = _wrapped(
        WorkKind.TRACE_EXECUTE,
        engine,
        guard=guard,
        authority=authority,
        identity=identity,
    )
    with pytest.raises(Exception):
        wrapped.execute(_work(WorkKind.TRACE_EXECUTE))
    assert engine.calls == 0


def test_replay_runtime_identity_cannot_create_authoritative_computation():
    guard = _guard()
    authority, lease = _authority(guard, runtime_role=RuntimeRole.LEGACY_REPLAY)
    engine = FakeEngine({"would_be_evidence": True})
    with pytest.raises(Exception):
        build_authoritative_engine_bundle_r11(
            engines=_bundle_for(WorkKind.TEACHER_ACQUIRE, engine),
            engine_identities={
                WorkKind.TEACHER_ACQUIRE: _identity_binding(guard, "ENGINE:TEACHER")
            },
            authority_context=authority,
            legacy_guard=guard,
            fence=FakeFence(lease),
        )
    assert engine.calls == 0


def test_missing_engine_identity_binding_fails_closed():
    guard = _guard()
    authority, lease = _authority(guard)
    with pytest.raises(InvalidEngineAuthorityContext, match="BINDING_REQUIRED"):
        build_authoritative_engine_bundle_r11(
            engines=_bundle_for(WorkKind.TEACHER_ACQUIRE, FakeEngine({})),
            engine_identities={},
            authority_context=authority,
            legacy_guard=guard,
            fence=FakeFence(lease),
        )


def test_policy_result_wrong_champion_is_rejected_after_compute():
    result = PolicyResult(
        work_id="WORK:POLICY_INFER",
        generation=GENERATION,
        champion_id="WRONG",
        payload_hash="p",
    )
    engine = FakeEngine(result)
    wrapped, _ = _wrapped(WorkKind.POLICY_INFER, engine)
    with pytest.raises(EngineCompletionRejected, match="POLICY_RESULT_LINEAGE"):
        wrapped.execute(_work(WorkKind.POLICY_INFER))
    assert engine.calls == 1


def test_training_wrapper_preserves_gradient_ownership_and_fp32_amp_false():
    result = TrainingResult(
        work_id="WORK:TRAIN_CHALLENGER",
        generation=GENERATION,
        challenger_id="CHALLENGER:G7",
        challenger_hash="challenger-hash",
        parent_champion_id=CHAMPION_ID,
        snapshot_id="SNAP:G7",
        payload_hash="training",
    )
    engine = FakeEngine(result, config=_Config())
    owners_before = engine.gradient_owners
    wrapped, _ = _wrapped(WorkKind.TRAIN_CHALLENGER, engine)
    returned = wrapped.execute(_work(WorkKind.TRAIN_CHALLENGER))
    assert returned.result is result
    assert engine.gradient_owners == owners_before
    assert engine.config.amp_enabled is False
    assert engine.config.dtype == "torch.float32"


def test_training_amp_or_non_fp32_is_rejected_before_engine():
    result = TrainingResult(
        work_id="WORK:TRAIN_CHALLENGER",
        generation=GENERATION,
        challenger_id="C",
        challenger_hash="H",
        parent_champion_id=CHAMPION_ID,
        snapshot_id="SNAP:G7",
        payload_hash="P",
    )
    for config in (_Config(amp_enabled=True), _Config(dtype="torch.float16")):
        engine = FakeEngine(result, config=config)
        wrapped, _ = _wrapped(WorkKind.TRAIN_CHALLENGER, engine)
        with pytest.raises(TrainingNumericIdentityUnproven):
            wrapped.execute(_work(WorkKind.TRAIN_CHALLENGER))
        assert engine.calls == 0


def test_validation_wrong_parent_champion_is_rejected():
    result = ValidationResult(
        work_id="WORK:VALIDATE",
        generation=GENERATION,
        challenger_id="C",
        parent_champion_id="STALE",
        snapshot_id="SNAP:G7",
        validation_id="V",
        payload_hash="P",
    )
    engine = FakeEngine(result)
    wrapped, _ = _wrapped(WorkKind.VALIDATE, engine)
    with pytest.raises(EngineCompletionRejected, match="VALIDATION_RESULT_LINEAGE"):
        wrapped.execute(_work(WorkKind.VALIDATE))


def test_tournament_computes_result_but_cannot_promote_champion():
    result = TournamentResult(
        work_id="WORK:TOURNAMENT",
        generation=GENERATION,
        challenger_id="C",
        parent_champion_id=CHAMPION_ID,
        validation_id="V",
        decision=TournamentDecision.PROMOTE,
        payload_hash="P",
    )
    engine = FakeEngine(result)
    wrapped, _ = _wrapped(WorkKind.TOURNAMENT, engine)
    returned = wrapped.execute(_work(WorkKind.TOURNAMENT))
    assert returned.result is result
    assert engine.promote_calls == 0
    assert not hasattr(wrapped, "promote")


def test_tournament_commit_proposal_is_not_an_engine_result():
    forged = TournamentCommitProposal(
        generation=GENERATION,
        decision=TournamentDecision.PROMOTE,
        parent_champion_id=CHAMPION_ID,
        challenger_id="C",
        challenger_hash="H",
    )
    engine = FakeEngine(forged)
    wrapped, _ = _wrapped(WorkKind.TOURNAMENT, engine)
    with pytest.raises(EngineCompletionRejected, match="TOURNAMENT_RESULT_TYPE"):
        wrapped.execute(_work(WorkKind.TOURNAMENT))
    assert engine.promote_calls == 0


def test_wrong_work_kind_cannot_reach_same_python_method():
    engine = FakeEngine({})
    wrapped, _ = _wrapped(WorkKind.TEACHER_ACQUIRE, engine)
    with pytest.raises(EngineLineageMismatch, match="WORK_KIND"):
        wrapped.execute(_work(WorkKind.TRACE_EXECUTE))
    assert engine.calls == 0
