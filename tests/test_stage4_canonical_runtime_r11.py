from __future__ import annotations

import ast
import inspect
from dataclasses import replace

import pytest

import cb16_local_opt.stage4_canonical_runtime_r11 as s4b


BINDING = s4b.AuthorityBinding("r11-authority", 17, "freeze-v1")
RECOVERED = s4b.RecoveredRuntimeState("r11-authority", 17, "state-17", "history-17")
LEASE = s4b.AuthorityLease("r11-authority", "fence-23")
SEAL = s4b.RuntimeSealObservation("r11-authority", 17, "history-17", "runtime-seal-17")


class FakeProviders:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.binding = BINDING
        self.recovered = RECOVERED
        self.lease = LEASE
        self.seal = SEAL
        self.fail_at: str | None = None

    def _event(self, name: str) -> None:
        self.events.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"synthetic:{name}")

    def adopt_or_verify_existing_authority(self):
        self._event("authority.verify")
        return self.binding

    def verify_state_roots(self, binding):
        self._event("roots.verify")

    def assert_canonical_only(self, binding):
        self._event("legacy.guard")

    def recover_existing_state(self, binding):
        self._event("recovery.recover")
        return self.recovered

    def acquire_authority(self, binding):
        self._event("lease.acquire")
        return self.lease

    def assert_current(self, lease):
        self._event("lease.assert")

    def release_authority(self, lease):
        self._event("lease.release")

    def start_workers(self, context):
        self._event("workers.start")

    def drain_workers(self, context):
        self._event("workers.drain")

    def seal_runtime_state(self, context):
        self._event("runtime.seal")
        return self.seal

    def stop_workers(self, context):
        self._event("workers.stop")


def controller(fake: FakeProviders) -> s4b.CanonicalRuntimeControllerR11:
    return s4b.CanonicalRuntimeControllerR11(
        authority_adoption=fake,
        state_roots=fake,
        legacy_guard=fake,
        recovery=fake,
        authority_lease=fake,
        orchestration=fake,
    )


def test_happy_lifecycle_requires_all_authority_gates_and_drain_before_seal_stop():
    fake = FakeProviders()
    runtime = controller(fake)

    context = runtime.start()
    assert runtime.phase is s4b.LifecyclePhase.RUNNING
    assert context.binding == BINDING
    assert fake.events == [
        "authority.verify",
        "roots.verify",
        "legacy.guard",
        "recovery.recover",
        "lease.acquire",
        "lease.assert",
        "workers.start",
        "lease.assert",
    ]
    assert fake.events.index("lease.acquire") < fake.events.index("workers.start")

    observation = runtime.shutdown()
    assert observation == SEAL
    assert runtime.phase is s4b.LifecyclePhase.STOPPED
    assert fake.events[-6:] == [
        "lease.assert",
        "workers.drain",
        "lease.assert",
        "runtime.seal",
        "workers.stop",
        "lease.release",
    ]


def test_no_running_state_when_any_pre_worker_authority_gate_fails():
    for failure in ("authority.verify", "roots.verify", "legacy.guard", "recovery.recover", "lease.acquire"):
        fake = FakeProviders()
        fake.fail_at = failure
        runtime = controller(fake)
        with pytest.raises(RuntimeError, match="synthetic"):
            runtime.start()
        assert runtime.phase is s4b.LifecyclePhase.FAILED
        assert "workers.start" not in fake.events


def test_partial_worker_startup_unwinds_and_releases_authority():
    fake = FakeProviders()
    fake.fail_at = "workers.start"
    runtime = controller(fake)
    with pytest.raises(RuntimeError, match="synthetic:workers.start"):
        runtime.start()
    assert runtime.phase is s4b.LifecyclePhase.FAILED
    assert fake.events.index("lease.acquire") < fake.events.index("workers.start")
    assert fake.events[-2:] == ["workers.stop", "lease.release"]


def test_recovery_cannot_manufacture_new_generation():
    fake = FakeProviders()
    fake.recovered = replace(RECOVERED, generation=18)
    runtime = controller(fake)
    with pytest.raises(s4b.StartupInvariantError, match="GENERATION_MANUFACTURE_REFUSED"):
        runtime.start()
    assert runtime.phase is s4b.LifecyclePhase.FAILED
    assert "lease.acquire" not in fake.events
    assert "workers.start" not in fake.events


def test_recovery_authority_identity_must_match_adopted_authority():
    fake = FakeProviders()
    fake.recovered = replace(RECOVERED, authority_id="other-authority")
    with pytest.raises(s4b.StartupInvariantError, match="RECOVERY_AUTHORITY_MISMATCH"):
        controller(fake).start()
    assert "lease.acquire" not in fake.events


def test_worker_start_rejected_when_acquired_lease_does_not_match_authority():
    fake = FakeProviders()
    fake.lease = replace(LEASE, authority_id="other-authority")
    runtime = controller(fake)
    with pytest.raises(s4b.StartupInvariantError, match="LEASE_AUTHORITY_MISMATCH"):
        runtime.start()
    assert runtime.phase is s4b.LifecyclePhase.FAILED
    assert "workers.start" not in fake.events


def test_double_start_and_illegal_shutdown_are_rejected():
    fake = FakeProviders()
    runtime = controller(fake)
    with pytest.raises(s4b.LifecycleTransitionError, match="ILLEGAL_SHUTDOWN"):
        runtime.shutdown()
    runtime.start()
    with pytest.raises(s4b.LifecycleTransitionError, match="DOUBLE_OR_ILLEGAL_START"):
        runtime.start()
    runtime.shutdown()
    with pytest.raises(s4b.LifecycleTransitionError, match="DOUBLE_OR_ILLEGAL_START"):
        runtime.start()


def test_shutdown_refuses_generation_or_history_rewrite_and_fails_closed():
    for bad_seal in (
        replace(SEAL, generation=18),
        replace(SEAL, scientific_history_id="rewritten-history"),
        replace(SEAL, authority_id="other-authority"),
    ):
        fake = FakeProviders()
        fake.seal = bad_seal
        runtime = controller(fake)
        runtime.start()
        with pytest.raises(s4b.ShutdownInvariantError):
            runtime.shutdown()
        assert runtime.phase is s4b.LifecyclePhase.FAILED
        assert fake.events[-2:] == ["workers.stop", "lease.release"]


def test_drain_failure_never_reaches_seal_and_still_unwinds():
    fake = FakeProviders()
    fake.fail_at = "workers.drain"
    runtime = controller(fake)
    runtime.start()
    with pytest.raises(RuntimeError, match="synthetic:workers.drain"):
        runtime.shutdown()
    assert runtime.phase is s4b.LifecyclePhase.FAILED
    assert "runtime.seal" not in fake.events
    assert fake.events[-2:] == ["workers.stop", "lease.release"]


def test_fail_closed_default_has_no_legacy_fallback():
    runtime = s4b.build_fail_closed_runtime()
    with pytest.raises(s4b.IntegrationRequiredError, match="authority_adoption"):
        runtime.start()
    assert runtime.phase is s4b.LifecyclePhase.FAILED


def test_s4b_has_no_sibling_stage4_module_imports_or_scientific_adjudication_dependency():
    source = inspect.getsource(s4b)
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert not [name for name in imports if "stage4_" in name]
    assert "final_holdout" not in source.lower()
    assert "DISTRIBUTIONAL_MARKET_INFORMATION" not in source


def test_lifecycle_description_is_production_lifecycle_not_qualification_harness():
    assert s4b.lifecycle_description() == (
        "BOOT -> VERIFY_AUTHORITY -> RECOVER_STATE -> ACQUIRE_AUTHORITY -> "
        "START_WORKERS -> RUNNING -> DRAIN -> SEAL -> STOPPED"
    )
