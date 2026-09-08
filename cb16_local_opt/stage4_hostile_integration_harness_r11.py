from __future__ import annotations

"""Production-facing Stage-4 INTG hostile-cutover integration harness.

INTG preserves the qualified S4H H01-H20 meanings while defining the adapter
and fault-boundary contract that final consolidation can bind to real runtime
components.  This module does not import sibling Integration implementations
and never claims final integrated-runtime qualification.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Protocol

from cb16_local_opt.stage4_hostile_cutover_r11 import (
    FailClosed,
    INVARIANT_KEYS,
    MATRIX_SCHEMA,
    REPORT_SCHEMA,
    REQUIRED_CASES,
    SCIENTIFIC_STATUS,
    SEMANTIC_FREEZE_BLOB,
    HostileCaseSpec,
    ReferenceHostileCutoverAdapter,
)

INTEGRATION_SEED_SHA = "86a4ac8a9080cd8382600cb998059e9585495a11"
INTEGRATION_MANIFEST_SCHEMA = "CB16_R11_STAGE4_INTG_BINDING_MANIFEST_V1"

_EXPECTED_CASES = (
    ("H01", "duplicate canonical runtime startup", "runtime_singleton"),
    ("H02", "simultaneous ownership attempt", "authority_ownership"),
    ("H03", "stale fencing token", "fencing"),
    ("H04", "conflicting authority adoption", "adoption"),
    ("H05", "duplicate identical adoption", "adoption"),
    ("H06", "legacy authoritative writer attempt", "legacy_retirement"),
    ("H07", "crash immediately before authority acquisition", "crash_recovery"),
    ("H08", "crash immediately after authority acquisition", "crash_recovery"),
    ("H09", "crash around adoption seal", "crash_recovery"),
    ("H10", "crash around authoritative journal transition", "crash_recovery"),
    ("H11", "duplicate commit", "idempotency"),
    ("H12", "duplicate generation release", "idempotency"),
    ("H13", "stale Champion writer", "lineage"),
    ("H14", "forged Permission", "permission"),
    ("H15", "direct Physics bypass attempt", "permission"),
    ("H16", "replay presented as new Evidence", "evidence"),
    ("H17", "orphan/incomplete state object", "storage"),
    ("H18", "restart after successful cutover", "restart"),
    ("H19", "second restart with stale owner", "restart_fencing"),
    ("H20", "malformed/corrupt authority metadata", "metadata"),
)

_EXPECTED_INVARIANTS = (
    "exactly_one_authoritative_generation",
    "exactly_one_authoritative_champion",
    "no_dual_writer",
    "no_stale_writer_mutation",
    "no_new_scientific_evidence_from_replay",
    "no_permission_bypass",
    "final_holdout_untouched",
    "semantic_freeze_unchanged",
    "ambiguous_authority_fails_closed",
)


class HostileBoundary(str, Enum):
    H01_DUPLICATE_RUNTIME_STARTUP = "H01_DUPLICATE_RUNTIME_STARTUP"
    H02_SIMULTANEOUS_OWNERSHIP = "H02_SIMULTANEOUS_OWNERSHIP"
    H03_STALE_FENCING_TOKEN = "H03_STALE_FENCING_TOKEN"
    H04_CONFLICTING_AUTHORITY_ADOPTION = "H04_CONFLICTING_AUTHORITY_ADOPTION"
    H05_DUPLICATE_IDENTICAL_ADOPTION = "H05_DUPLICATE_IDENTICAL_ADOPTION"
    H06_LEGACY_AUTHORITATIVE_WRITER = "H06_LEGACY_AUTHORITATIVE_WRITER"
    H07_CRASH_BEFORE_AUTHORITY_ACQUISITION = "H07_CRASH_BEFORE_AUTHORITY_ACQUISITION"
    H08_CRASH_AFTER_AUTHORITY_ACQUISITION = "H08_CRASH_AFTER_AUTHORITY_ACQUISITION"
    H09_CRASH_AROUND_ADOPTION_SEAL = "H09_CRASH_AROUND_ADOPTION_SEAL"
    H10_CRASH_AROUND_JOURNAL_TRANSITION = "H10_CRASH_AROUND_JOURNAL_TRANSITION"
    H11_DUPLICATE_COMMIT = "H11_DUPLICATE_COMMIT"
    H12_DUPLICATE_GENERATION_RELEASE = "H12_DUPLICATE_GENERATION_RELEASE"
    H13_STALE_CHAMPION_WRITER = "H13_STALE_CHAMPION_WRITER"
    H14_FORGED_PERMISSION = "H14_FORGED_PERMISSION"
    H15_DIRECT_PHYSICS_BYPASS = "H15_DIRECT_PHYSICS_BYPASS"
    H16_REPLAY_AS_NEW_EVIDENCE = "H16_REPLAY_AS_NEW_EVIDENCE"
    H17_ORPHAN_INCOMPLETE_STATE_OBJECT = "H17_ORPHAN_INCOMPLETE_STATE_OBJECT"
    H18_RESTART_AFTER_CUTOVER = "H18_RESTART_AFTER_CUTOVER"
    H19_SECOND_RESTART_STALE_OWNER = "H19_SECOND_RESTART_STALE_OWNER"
    H20_CORRUPT_AUTHORITY_METADATA = "H20_CORRUPT_AUTHORITY_METADATA"


@dataclass(frozen=True)
class HostileBoundarySpec:
    case_id: str
    boundary: HostileBoundary
    surface: str
    production_fault_requirement: str


FAULT_BOUNDARIES = (
    HostileBoundarySpec("H01", HostileBoundary.H01_DUPLICATE_RUNTIME_STARTUP, "process/runtime", "attempt a real duplicate canonical runtime start"),
    HostileBoundarySpec("H02", HostileBoundary.H02_SIMULTANEOUS_OWNERSHIP, "lease/fence", "attempt concurrent authoritative ownership"),
    HostileBoundarySpec("H03", HostileBoundary.H03_STALE_FENCING_TOKEN, "writer/fence", "present a stale native fence at an authoritative mutation"),
    HostileBoundarySpec("H04", HostileBoundary.H04_CONFLICTING_AUTHORITY_ADOPTION, "adoption/persistence", "publish a conflicting adopted source identity"),
    HostileBoundarySpec("H05", HostileBoundary.H05_DUPLICATE_IDENTICAL_ADOPTION, "adoption/persistence", "repeat the byte-identical adopted source identity"),
    HostileBoundarySpec("H06", HostileBoundary.H06_LEGACY_AUTHORITATIVE_WRITER, "writer/admission", "attempt authoritative write admission from a legacy identity"),
    HostileBoundarySpec("H07", HostileBoundary.H07_CRASH_BEFORE_AUTHORITY_ACQUISITION, "process/lease", "kill the runtime immediately before authority acquisition"),
    HostileBoundarySpec("H08", HostileBoundary.H08_CRASH_AFTER_AUTHORITY_ACQUISITION, "process/lease", "kill the runtime immediately after authority acquisition"),
    HostileBoundarySpec("H09", HostileBoundary.H09_CRASH_AROUND_ADOPTION_SEAL, "process/filesystem/adoption", "kill between adoption prepare and seal publication"),
    HostileBoundarySpec("H10", HostileBoundary.H10_CRASH_AROUND_JOURNAL_TRANSITION, "process/filesystem/journal", "kill between journal prepare and authoritative commit"),
    HostileBoundarySpec("H11", HostileBoundary.H11_DUPLICATE_COMMIT, "filesystem/checkpoint", "repeat identical commit and conflict the same commit identity"),
    HostileBoundarySpec("H12", HostileBoundary.H12_DUPLICATE_GENERATION_RELEASE, "control-plane/generation", "repeat the same generation release without a second advance"),
    HostileBoundarySpec("H13", HostileBoundary.H13_STALE_CHAMPION_WRITER, "writer/lineage", "write using a non-current Champion identity"),
    HostileBoundarySpec("H14", HostileBoundary.H14_FORGED_PERMISSION, "permission/execution", "present a forged Permission to the execution boundary"),
    HostileBoundarySpec("H15", HostileBoundary.H15_DIRECT_PHYSICS_BYPASS, "permission/physics", "invoke Physics without Supervisor-issued Permission"),
    HostileBoundarySpec("H16", HostileBoundary.H16_REPLAY_AS_NEW_EVIDENCE, "evidence/admission", "present replay material to scientific Evidence admission"),
    HostileBoundarySpec("H17", HostileBoundary.H17_ORPHAN_INCOMPLETE_STATE_OBJECT, "filesystem/state-root", "surface orphan and incomplete state objects"),
    HostileBoundarySpec("H18", HostileBoundary.H18_RESTART_AFTER_CUTOVER, "process/recovery", "restart after successful cutover while preserving authority history"),
    HostileBoundarySpec("H19", HostileBoundary.H19_SECOND_RESTART_STALE_OWNER, "process/recovery/fence", "restart twice and attack with both stale owners"),
    HostileBoundarySpec("H20", HostileBoundary.H20_CORRUPT_AUTHORITY_METADATA, "filesystem/authority-metadata", "corrupt durable authority metadata before acquisition"),
)
_BOUNDARY_BY_CASE = {row.case_id: row for row in FAULT_BOUNDARIES}


@dataclass(frozen=True)
class IntegratedAuthoritySnapshot:
    generation: int
    champion_id: str
    authoritative_generations: tuple[int, ...]
    authoritative_champions: tuple[str, ...]
    runtime_id: str | None
    writer_owner_id: str | None
    live_writer_count: int
    fence_epoch: int
    adoption_identity: str | None
    scientific_evidence_count: int
    replay_evidence_admitted: int
    stale_writer_mutations: int
    permission_bypass_mutations: int
    final_holdout_accesses: int
    semantic_freeze_blob: str
    scientific_status: str
    metadata_valid: bool
    max_writer_count: int
    ambiguous_authority_writes: int
    journal_event_count: int


class ProductionHostileAdapter(Protocol):
    """Low-level production binding. Native tokens may remain opaque objects."""

    def snapshot(self) -> IntegratedAuthoritySnapshot: ...
    def start_runtime(self, runtime_id: str) -> None: ...
    def stop_runtime(self, runtime_id: str) -> None: ...
    def acquire_authority(self, runtime_id: str, *, role: str = "CANONICAL") -> object: ...
    def release_authority(self, token: object) -> None: ...
    def crash(self, runtime_id: str) -> None: ...
    def recover(self, runtime_id: str) -> None: ...
    def fence_strictly_newer(self, newer: object, older: object) -> bool: ...
    def adopt_authority(self, token: object, identity: str) -> str: ...
    def begin_adoption(self, token: object, identity: str) -> None: ...
    def seal_pending_adoption(self, token: object) -> str: ...
    def begin_journal_transition(self, token: object, event_id: str, payload_hash: str, champion_id: str) -> None: ...
    def commit_pending_journal(self, token: object, event_id: str) -> str: ...
    def append_journal(self, token: object, event_id: str, payload_hash: str, champion_id: str) -> str: ...
    def commit_transition(self, token: object, commit_id: str, next_champion_id: str) -> str: ...
    def release_generation(self, token: object, release_id: str, next_generation: int) -> str: ...
    def issue_permission(self, token: object, intent_id: str, *, source: str) -> str: ...
    def physics_transition(self, token: object, intent_id: str, permission_token: str | None) -> str: ...
    def admit_evidence(self, token: object, evidence_id: str, *, source: str) -> str: ...
    def place_state_object(self, object_id: str, state: str) -> None: ...
    def adopt_state_object(self, token: object, object_id: str) -> str: ...
    def corrupt_authority_metadata(self) -> None: ...


class HostileFaultHooks(Protocol):
    """Arm/observe real process and filesystem fault boundaries for each H01-H20 case."""

    def before_boundary(self, spec: HostileBoundarySpec, adapter: ProductionHostileAdapter) -> None: ...
    def after_boundary(self, spec: HostileBoundarySpec, adapter: ProductionHostileAdapter) -> None: ...


class NoopHostileFaultHooks:
    """Reference-only hooks. Final consolidated qualification must replace when real faults need arming."""

    def before_boundary(self, spec: HostileBoundarySpec, adapter: ProductionHostileAdapter) -> None:
        del spec, adapter

    def after_boundary(self, spec: HostileBoundarySpec, adapter: ProductionHostileAdapter) -> None:
        del spec, adapter


class ReferenceProductionHostileAdapter:
    """Bridge the already-qualified S4H deterministic adapter into the INTG contract."""

    def __init__(self) -> None:
        self._inner = ReferenceHostileCutoverAdapter()

    def snapshot(self) -> IntegratedAuthoritySnapshot:
        snap = self._inner.snapshot()
        return IntegratedAuthoritySnapshot(
            **asdict(snap),
            max_writer_count=self._inner.max_writer_count,
            ambiguous_authority_writes=self._inner.ambiguous_authority_writes,
            journal_event_count=self._inner.journal_event_count,
        )

    def fence_strictly_newer(self, newer: object, older: object) -> bool:
        return bool(
            getattr(newer, "epoch", None) is not None
            and getattr(older, "epoch", None) is not None
            and getattr(newer, "epoch") > getattr(older, "epoch")
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def assert_s4h_contract_exact() -> None:
    observed_cases = tuple((x.case_id, x.title, x.category) for x in REQUIRED_CASES)
    if observed_cases != _EXPECTED_CASES:
        raise FailClosed("S4H_CASE_MEANING_DRIFT")
    if tuple(INVARIANT_KEYS) != _EXPECTED_INVARIANTS:
        raise FailClosed("S4H_COMMON_INVARIANT_DRIFT")
    ids = tuple(x.case_id for x in REQUIRED_CASES)
    if ids != tuple(f"H{i:02d}" for i in range(1, 21)):
        raise FailClosed("S4H_CASE_ID_SET_DRIFT")
    if tuple(row.case_id for row in FAULT_BOUNDARIES) != ids:
        raise FailClosed("INTG_FAULT_BOUNDARY_COVERAGE_DRIFT")


def _expect_fail(observed: list[str], expected_code: str, fn: Callable[[], Any]) -> None:
    try:
        fn()
    except FailClosed as exc:
        observed.append(exc.code)
        if exc.code != expected_code:
            raise AssertionError(f"expected {expected_code}, got {exc.code}") from exc
        return
    raise AssertionError(f"expected fail-closed code {expected_code}")


def _bootstrap(a: ProductionHostileAdapter, runtime_id: str = "runtime-a") -> object:
    a.start_runtime(runtime_id)
    return a.acquire_authority(runtime_id)


def _common_invariants(a: ProductionHostileAdapter, before: IntegratedAuthoritySnapshot) -> dict[str, bool]:
    after = a.snapshot()
    return {
        "exactly_one_authoritative_generation": (
            len(after.authoritative_generations) == 1
            and after.authoritative_generations[0] == after.generation
        ),
        "exactly_one_authoritative_champion": (
            len(after.authoritative_champions) == 1
            and after.authoritative_champions[0] == after.champion_id
        ),
        "no_dual_writer": after.live_writer_count <= 1 and after.max_writer_count <= 1,
        "no_stale_writer_mutation": after.stale_writer_mutations == 0,
        "no_new_scientific_evidence_from_replay": after.replay_evidence_admitted == 0,
        "no_permission_bypass": after.permission_bypass_mutations == 0,
        "final_holdout_untouched": after.final_holdout_accesses == 0,
        "semantic_freeze_unchanged": (
            after.semantic_freeze_blob == before.semantic_freeze_blob == SEMANTIC_FREEZE_BLOB
        ),
        "ambiguous_authority_fails_closed": after.ambiguous_authority_writes == 0,
    }


def _h01(a: ProductionHostileAdapter, o: list[str]) -> None:
    a.start_runtime("runtime-a")
    _expect_fail(o, "DUPLICATE_CANONICAL_RUNTIME", lambda: a.start_runtime("runtime-b"))


def _h02(a: ProductionHostileAdapter, o: list[str]) -> None:
    _bootstrap(a)
    _expect_fail(o, "RUNTIME_NOT_ACTIVE", lambda: a.acquire_authority("runtime-b"))


def _h03(a: ProductionHostileAdapter, o: list[str]) -> None:
    t1 = _bootstrap(a)
    a.release_authority(t1)
    a.stop_runtime("runtime-a")
    a.start_runtime("runtime-b")
    t2 = a.acquire_authority("runtime-b")
    assert a.fence_strictly_newer(t2, t1)
    _expect_fail(o, "STALE_OR_INVALID_FENCE", lambda: a.append_journal(t1, "e-stale", "p", a.snapshot().champion_id))


def _h04(a: ProductionHostileAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    assert a.adopt_authority(t, "source-authority-a") == "ADOPTED"
    _expect_fail(o, "CONFLICTING_AUTHORITY_ADOPTION", lambda: a.adopt_authority(t, "source-authority-b"))


def _h05(a: ProductionHostileAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    g = a.snapshot().generation
    assert a.adopt_authority(t, "source-authority-a") == "ADOPTED"
    assert a.adopt_authority(t, "source-authority-a") == "ALREADY_ADOPTED"
    assert a.snapshot().generation == g


def _h06(a: ProductionHostileAdapter, o: list[str]) -> None:
    a.start_runtime("legacy-runtime")
    _expect_fail(o, "LEGACY_OR_NONCANONICAL_AUTHORITY_DENIED", lambda: a.acquire_authority("legacy-runtime", role="LEGACY"))


def _h07(a: ProductionHostileAdapter, o: list[str]) -> None:
    a.start_runtime("runtime-a")
    a.crash("runtime-a")
    a.recover("runtime-b")
    token = a.acquire_authority("runtime-b")
    assert token is not None


def _h08(a: ProductionHostileAdapter, o: list[str]) -> None:
    t1 = _bootstrap(a)
    a.crash("runtime-a")
    a.recover("runtime-b")
    t2 = a.acquire_authority("runtime-b")
    assert a.fence_strictly_newer(t2, t1)
    _expect_fail(o, "STALE_OR_INVALID_FENCE", lambda: a.append_journal(t1, "stale", "p", a.snapshot().champion_id))


def _h09(a: ProductionHostileAdapter, o: list[str]) -> None:
    t1 = _bootstrap(a)
    a.begin_adoption(t1, "source-authority-a")
    a.crash("runtime-a")
    a.recover("runtime-b")
    t2 = a.acquire_authority("runtime-b")
    _expect_fail(o, "NO_PENDING_ADOPTION", lambda: a.seal_pending_adoption(t2))
    assert a.adopt_authority(t2, "source-authority-a") == "ADOPTED"


def _h10(a: ProductionHostileAdapter, o: list[str]) -> None:
    t1 = _bootstrap(a)
    champion = a.snapshot().champion_id
    a.begin_journal_transition(t1, "event-1", "payload-1", champion)
    a.crash("runtime-a")
    a.recover("runtime-b")
    t2 = a.acquire_authority("runtime-b")
    _expect_fail(o, "NO_PENDING_JOURNAL_TRANSITION", lambda: a.commit_pending_journal(t2, "event-1"))
    assert a.append_journal(t2, "event-1", "payload-1", champion) == "APPENDED"
    assert a.snapshot().journal_event_count == 1


def _h11(a: ProductionHostileAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    assert a.commit_transition(t, "commit-1", "challenger-g42") == "COMMITTED"
    assert a.commit_transition(t, "commit-1", "challenger-g42") == "ALREADY_COMMITTED"
    _expect_fail(o, "DUPLICATE_COMMIT_CONFLICT", lambda: a.commit_transition(t, "commit-1", "other-challenger"))


def _h12(a: ProductionHostileAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    g = a.snapshot().generation
    a.commit_transition(t, "commit-1", "challenger-g42")
    assert a.release_generation(t, "release-1", g + 1) == "RELEASED"
    assert a.release_generation(t, "release-1", g + 1) == "ALREADY_RELEASED"
    assert a.snapshot().generation == g + 1


def _h13(a: ProductionHostileAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    _expect_fail(o, "STALE_CHAMPION_WRITER", lambda: a.append_journal(t, "event-stale-champion", "p", "old-champion"))
    assert a.snapshot().journal_event_count == 0


def _h14(a: ProductionHostileAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    _expect_fail(o, "PHYSICS_PERMISSION_REQUIRED", lambda: a.physics_transition(t, "intent-1", "forged-permission"))


def _h15(a: ProductionHostileAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    _expect_fail(o, "PHYSICS_PERMISSION_REQUIRED", lambda: a.physics_transition(t, "intent-1", None))
    p = a.issue_permission(t, "intent-1", source="FROZEN_SUPERVISOR")
    assert a.physics_transition(t, "intent-1", p)


def _h16(a: ProductionHostileAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    count = a.snapshot().scientific_evidence_count
    _expect_fail(o, "REPLAY_IS_ENGINEERING_ONLY_NOT_NEW_EVIDENCE", lambda: a.admit_evidence(t, "replay-e1", source="REPLAY"))
    assert a.snapshot().scientific_evidence_count == count


def _h17(a: ProductionHostileAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    a.place_state_object("orphan-1", "ORPHAN")
    a.place_state_object("incomplete-1", "INCOMPLETE")
    _expect_fail(o, "STATE_OBJECT_NOT_SEALED_AUTHORITATIVE", lambda: a.adopt_state_object(t, "orphan-1"))
    _expect_fail(o, "STATE_OBJECT_NOT_SEALED_AUTHORITATIVE", lambda: a.adopt_state_object(t, "incomplete-1"))


def _h18(a: ProductionHostileAdapter, o: list[str]) -> None:
    del o
    t1 = _bootstrap(a)
    a.adopt_authority(t1, "source-authority-a")
    g = a.snapshot().generation
    a.commit_transition(t1, "commit-1", "challenger-g42")
    a.release_generation(t1, "release-1", g + 1)
    after_cutover = a.snapshot()
    a.crash("runtime-a")
    a.recover("runtime-b")
    a.acquire_authority("runtime-b")
    restarted = a.snapshot()
    assert restarted.generation == after_cutover.generation
    assert restarted.champion_id == after_cutover.champion_id
    assert restarted.adoption_identity == after_cutover.adoption_identity


def _h19(a: ProductionHostileAdapter, o: list[str]) -> None:
    t1 = _bootstrap(a)
    a.crash("runtime-a")
    a.recover("runtime-b")
    t2 = a.acquire_authority("runtime-b")
    a.crash("runtime-b")
    a.recover("runtime-c")
    t3 = a.acquire_authority("runtime-c")
    assert a.fence_strictly_newer(t2, t1) and a.fence_strictly_newer(t3, t2)
    _expect_fail(o, "STALE_OR_INVALID_FENCE", lambda: a.append_journal(t1, "old-owner", "p", a.snapshot().champion_id))
    _expect_fail(o, "STALE_OR_INVALID_FENCE", lambda: a.append_journal(t2, "middle-owner", "p", a.snapshot().champion_id))


def _h20(a: ProductionHostileAdapter, o: list[str]) -> None:
    a.start_runtime("runtime-a")
    a.corrupt_authority_metadata()
    _expect_fail(o, "AUTHORITY_METADATA_CORRUPT", lambda: a.acquire_authority("runtime-a"))


_CASE_RUNNERS: dict[str, Callable[[ProductionHostileAdapter, list[str]], None]] = {
    f"H{i:02d}": fn
    for i, fn in enumerate(
        (_h01, _h02, _h03, _h04, _h05, _h06, _h07, _h08, _h09, _h10,
         _h11, _h12, _h13, _h14, _h15, _h16, _h17, _h18, _h19, _h20),
        start=1,
    )
}


def run_integrated_case(
    spec: HostileCaseSpec,
    adapter_factory: Callable[[], ProductionHostileAdapter],
    hooks_factory: Callable[[], HostileFaultHooks] = NoopHostileFaultHooks,
) -> dict[str, Any]:
    assert_s4h_contract_exact()
    adapter = adapter_factory()
    hooks = hooks_factory()
    before = adapter.snapshot()
    observed: list[str] = []
    error: str | None = None
    boundary = _BOUNDARY_BY_CASE[spec.case_id]
    try:
        hooks.before_boundary(boundary, adapter)
        _CASE_RUNNERS[spec.case_id](adapter, observed)
        hooks.after_boundary(boundary, adapter)
    except BaseException as exc:
        error = f"{type(exc).__name__}:{exc}"
        observed.append(error)
    try:
        invariant_map = _common_invariants(adapter, before)
        after = adapter.snapshot()
    except BaseException as exc:
        invariant_map = {key: False for key in INVARIANT_KEYS}
        after = before
        observed.append(f"OBSERVATION_FAILURE:{type(exc).__name__}:{exc}")
        error = error or observed[-1]
    return {
        "case_id": spec.case_id,
        "title": spec.title,
        "category": spec.category,
        "passed": error is None and all(invariant_map.values()),
        "observed": observed,
        "invariants": invariant_map,
        "before": asdict(before),
        "after": asdict(after),
    }


def run_integrated_hostile_matrix(
    adapter_factory: Callable[[], ProductionHostileAdapter] = ReferenceProductionHostileAdapter,
    hooks_factory: Callable[[], HostileFaultHooks] = NoopHostileFaultHooks,
) -> dict[str, Any]:
    """Run exact H01-H20 semantics while permanently withholding final qualification."""

    assert_s4h_contract_exact()
    rows = [run_integrated_case(spec, adapter_factory, hooks_factory) for spec in REQUIRED_CASES]
    passed = sum(1 for row in rows if row["passed"])
    return {
        "schema": REPORT_SCHEMA,
        "matrix_schema": MATRIX_SCHEMA,
        "status": "PASS" if passed == len(rows) else "FAIL",
        "harness_qualified": passed == len(rows),
        "integrated_runtime_qualified": False,
        "scientific_status": SCIENTIFIC_STATUS,
        "semantic_freeze_blob": SEMANTIC_FREEZE_BLOB,
        "case_count": len(rows),
        "passed_cases": passed,
        "failed_cases": len(rows) - passed,
        "invariants": list(INVARIANT_KEYS),
        "cases": rows,
    }


def integration_binding_manifest() -> dict[str, Any]:
    assert_s4h_contract_exact()
    return {
        "schema": INTEGRATION_MANIFEST_SCHEMA,
        "integration_seed_sha": INTEGRATION_SEED_SHA,
        "scientific_status": SCIENTIFIC_STATUS,
        "semantic_freeze_blob": SEMANTIC_FREEZE_BLOB,
        "integrated_runtime_qualification_claimed": False,
        "sibling_integration_dependency_used": False,
        "adapter_contract": "ProductionHostileAdapter",
        "fault_hook_contract": "HostileFaultHooks",
        "required_adapter_methods": [
            "snapshot", "start_runtime", "stop_runtime", "acquire_authority", "release_authority",
            "crash", "recover", "fence_strictly_newer", "adopt_authority", "begin_adoption",
            "seal_pending_adoption", "begin_journal_transition", "commit_pending_journal",
            "append_journal", "commit_transition", "release_generation", "issue_permission",
            "physics_transition", "admit_evidence", "place_state_object", "adopt_state_object",
            "corrupt_authority_metadata",
        ],
        "required_snapshot_fields": list(IntegratedAuthoritySnapshot.__dataclass_fields__),
        "fault_boundaries": [
            {
                "case_id": row.case_id,
                "boundary": row.boundary.value,
                "surface": row.surface,
                "production_fault_requirement": row.production_fault_requirement,
            }
            for row in FAULT_BOUNDARIES
        ],
        "final_consolidation_requirements": [
            "Bind ProductionHostileAdapter to the consolidated canonical runtime, not the reference adapter.",
            "Bind HostileFaultHooks to real process/filesystem fault controls where the boundary requires crash, torn-state, orphan-state, or corruption injection.",
            "Translate native fail-closed exceptions into the qualified S4H FailClosed adjudication codes without weakening case meaning.",
            "Populate IntegratedAuthoritySnapshot from real canonical authority, Champion/generation, writer, Evidence, Permission, holdout, and Semantic-Freeze observations.",
            "Rerun all H01-H20 against the consolidated runtime; INTG itself never emits R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED.",
        ],
    }
