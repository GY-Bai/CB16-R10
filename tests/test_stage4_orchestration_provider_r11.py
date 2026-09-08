from __future__ import annotations

import threading

import pytest

import cb16_local_opt.stage4_canonical_runtime_r11 as s4b
import cb16_local_opt.stage4_orchestration_provider_r11 as intf


BINDING = s4b.AuthorityBinding("r11-authority", 17, "freeze-v1")
RECOVERED = s4b.RecoveredRuntimeState("r11-authority", 17, "state-17", "history-17")
LEASE = s4b.AuthorityLease("r11-authority", "opaque-fence")
CONTEXT = s4b.RuntimeContext(BINDING, RECOVERED, LEASE)


class FakeAuthority:
    def __init__(self) -> None:
        self.live = True
        self.calls = 0

    def assert_current(self, context: s4b.RuntimeContext) -> None:
        self.calls += 1
        if not self.live:
            raise RuntimeError("STALE_FENCE")


class FakeComponent:
    def __init__(self, name: str, *, fail_start: bool = False):
        self._name = name
        self.fail_start = fail_start
        self.live = False
        self.events: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def start(self, context, config):
        self.events.append("start")
        self.live = True
        if self.fail_start:
            raise RuntimeError(f"synthetic-start:{self.name}")

    def begin_drain(self, context):
        self.events.append("begin_drain")

    def await_drained(self, context):
        self.events.append("await_drained")

    def stop(self, context):
        self.events.append("stop")
        self.live = False

    def has_live_workers(self):
        return self.live


class RecordingSeal:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def observe(self, context, config, component_names):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic-seal")
        return "engineering-only-seal"


def provider(*, authority=None, components=(), seal=None):
    return intf.Stage4OrchestrationProviderR11(
        live_authority=authority or FakeAuthority(),
        components=components,
        seal_observer=seal,
    )


def clean_shutdown(p):
    p.drain_workers(CONTEXT)
    observation = p.seal_runtime_state(CONTEXT)
    p.stop_workers(CONTEXT)
    return observation


def test_provider_conforms_to_s4b_orchestration_protocol_and_canonical_defaults():
    p = provider()
    assert isinstance(p, s4b.OrchestrationProvider)
    assert p.config.teacher_workers == 8
    assert p.config.trace_workers == 8
    assert p.config.queue_depth == 8
    assert p.config.buffer_mib == 64
    assert p.config.nested_numeric_threads == 1
    assert p.config.numeric_mode == "FP32"
    assert p.config.amp is False


def test_worker_start_without_live_authority_is_rejected():
    authority = FakeAuthority()
    authority.live = False
    c = FakeComponent("teacher")
    p = provider(authority=authority, components=(c,))
    with pytest.raises(intf.WorkerAuthorityError, match="LIVE_AUTHORITY"):
        p.start_workers(CONTEXT)
    assert p.phase is intf.WorkerPhase.FAILED
    assert c.live is False
    p.assert_no_orphan_workers()


def test_stale_authority_before_worker_start_rejected():
    authority = FakeAuthority()
    authority.live = False
    p = provider(authority=authority, components=(FakeComponent("teacher"),))
    with pytest.raises(intf.WorkerAuthorityError):
        p.start_workers(CONTEXT)
    assert p.phase is intf.WorkerPhase.FAILED


def test_partial_startup_cleans_every_registered_component_in_reverse_order():
    a = FakeComponent("a")
    b = FakeComponent("b", fail_start=True)
    p = provider(components=(a, b))
    with pytest.raises(RuntimeError, match="synthetic-start:b"):
        p.start_workers(CONTEXT)
    assert a.live is False
    assert b.live is False
    assert a.events == ["start", "stop"]
    assert b.events == ["start", "stop"]
    p.assert_no_orphan_workers()


def test_double_start_rejected():
    p = provider()
    p.start_workers(CONTEXT)
    with pytest.raises(intf.WorkerLifecycleTransitionError, match="DOUBLE_OR_ILLEGAL_START"):
        p.start_workers(CONTEXT)
    clean_shutdown(p)


def test_drain_closes_new_admission_and_waits_for_already_owned_bounded_work():
    c = FakeComponent("teacher")
    p = provider(components=(c,))
    p.start_workers(CONTEXT)
    ticket = p.admit_authoritative_work(CONTEXT)

    drained = threading.Event()
    failure: list[BaseException] = []

    def run_drain():
        try:
            p.drain_workers(CONTEXT)
        except BaseException as exc:
            failure.append(exc)
        finally:
            drained.set()

    t = threading.Thread(target=run_drain, daemon=True)
    t.start()

    for _ in range(1000):
        if p.phase is intf.WorkerPhase.DRAINING:
            break
        threading.Event().wait(0.001)

    assert p.phase is intf.WorkerPhase.DRAINING
    assert p.accepting_authoritative_work is False
    assert drained.is_set() is False

    with pytest.raises(intf.WorkerLifecycleTransitionError, match="ADMISSION_CLOSED"):
        p.admit_authoritative_work(CONTEXT)

    p.complete_authoritative_work(ticket)
    t.join(timeout=2)
    assert not failure
    assert drained.is_set()
    assert p.phase is intf.WorkerPhase.DRAINED
    assert c.events[:3] == ["start", "begin_drain", "await_drained"]

    p.seal_runtime_state(CONTEXT)
    p.stop_workers(CONTEXT)


def test_seal_rejected_before_drain():
    p = provider()
    p.start_workers(CONTEXT)
    with pytest.raises(intf.WorkerLifecycleTransitionError, match="SEAL_REQUIRES_DRAIN"):
        p.seal_runtime_state(CONTEXT)
    clean_shutdown(p)


def test_successful_stop_requires_successful_seal():
    c = FakeComponent("teacher")
    p = provider(components=(c,))
    p.start_workers(CONTEXT)
    with pytest.raises(intf.WorkerLifecycleTransitionError, match="STOP_REQUIRES_SEAL"):
        p.stop_workers(CONTEXT)
    p.drain_workers(CONTEXT)
    with pytest.raises(intf.WorkerLifecycleTransitionError, match="STOP_REQUIRES_SEAL"):
        p.stop_workers(CONTEXT)
    p.seal_runtime_state(CONTEXT)
    p.stop_workers(CONTEXT)
    assert p.phase is intf.WorkerPhase.STOPPED
    assert c.live is False


def test_failed_seal_cleans_workers_and_s4b_retry_stop_is_idempotent():
    c = FakeComponent("teacher")
    seal = RecordingSeal(fail=True)
    p = provider(components=(c,), seal=seal)
    p.start_workers(CONTEXT)
    p.drain_workers(CONTEXT)
    with pytest.raises(RuntimeError, match="synthetic-seal"):
        p.seal_runtime_state(CONTEXT)
    assert p.phase is intf.WorkerPhase.FAILED
    assert c.live is False
    p.stop_workers(CONTEXT)
    p.assert_no_orphan_workers()


def test_stale_authority_after_start_cannot_admit_or_accept_seal():
    authority = FakeAuthority()
    c = FakeComponent("teacher")
    p = provider(authority=authority, components=(c,))
    p.start_workers(CONTEXT)
    authority.live = False
    with pytest.raises(intf.WorkerAuthorityError):
        p.admit_authoritative_work(CONTEXT)
    with pytest.raises(intf.WorkerAuthorityError):
        p.drain_workers(CONTEXT)
    assert p.phase is intf.WorkerPhase.FAILED
    assert c.live is False


def test_wrong_generation_or_runtime_context_fails_closed():
    p = provider()
    p.start_workers(CONTEXT)
    wrong = s4b.RuntimeContext(
        BINDING,
        s4b.RecoveredRuntimeState("r11-authority", 18, "state-17", "history-17"),
        LEASE,
    )
    with pytest.raises(intf.WorkerAuthorityError, match="GENERATION_MISMATCH"):
        p.admit_authoritative_work(wrong)
    clean_shutdown(p)


def test_seal_observation_preserves_authority_generation_and_scientific_history():
    p = provider(seal=RecordingSeal())
    p.start_workers(CONTEXT)
    observation = clean_shutdown(p)
    assert observation == s4b.RuntimeSealObservation(
        "r11-authority", 17, "history-17", "engineering-only-seal"
    )


def test_restart_does_not_reinterpret_replay_or_change_scientific_identity():
    first = provider()
    first.start_workers(CONTEXT)
    seal1 = clean_shutdown(first)

    second = provider()
    second.start_workers(CONTEXT)
    seal2 = clean_shutdown(second)

    assert intf.REPLAY_SEMANTICS_R11 == "ENGINEERING_RECOVERY_ONLY__NOT_NEW_SCIENTIFIC_EVIDENCE"
    assert seal1.authority_id == seal2.authority_id == BINDING.authority_id
    assert seal1.generation == seal2.generation == BINDING.generation
    assert seal1.scientific_history_id == seal2.scientific_history_id == RECOVERED.scientific_history_id
    assert seal1.seal_id == seal2.seal_id


def test_fp32_amp_identity_is_fail_closed_not_tunable_here():
    assert intf.NUMERIC_MODE_R11 == "FP32"
    assert intf.AMP_ENABLED_R11 is False
    with pytest.raises(ValueError, match="CANONICAL_RUNTIME_DEFAULTS_REQUIRED"):
        intf.Stage4WorkerRuntimeConfigR11(amp=True)
    with pytest.raises(ValueError, match="CANONICAL_RUNTIME_DEFAULTS_REQUIRED"):
        intf.Stage4WorkerRuntimeConfigR11(numeric_mode="BF16")
    with pytest.raises(ValueError, match="CANONICAL_RUNTIME_DEFAULTS_REQUIRED"):
        intf.Stage4WorkerRuntimeConfigR11(teacher_workers=16)


def test_managed_closeable_adapter_closes_resource_and_leaves_no_orphan():
    closed: list[bool] = []

    class Resource:
        def close(self):
            closed.append(True)

    component = intf.ManagedCloseableWorkerComponentR11(
        name="qualified-runtime",
        factory=lambda context, config: Resource(),
    )
    p = provider(components=(component,))
    p.start_workers(CONTEXT)
    assert component.has_live_workers()
    clean_shutdown(p)
    assert closed == [True]
    assert component.has_live_workers() is False
    p.assert_no_orphan_workers()


def test_teacher_and_trace_builders_reject_topology_tuning():
    with pytest.raises(ValueError, match="TEACHER_WORKERS_MUST_REMAIN_8"):
        intf.build_teacher_worker_component_r11(max_workers=4)
    with pytest.raises(ValueError, match="TRACE_WORKERS_MUST_REMAIN_8"):
        intf.build_fork_trace_worker_component_r11(
            physics=object(), market_cache=object(), max_workers=4
        )


def test_scientific_status_is_preserved_and_no_cutover_verdict_is_emitted():
    assert (
        intf.SCIENTIFIC_STATUS_R11
        == "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
    )
    source = open(intf.__file__, "r", encoding="utf-8").read()
    assert "R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED" not in source
    assert "final_holdout" not in source.lower()
