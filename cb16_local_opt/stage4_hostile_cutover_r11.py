from __future__ import annotations

"""Independent Stage-4 S4H hostile cutover / split-brain qualification harness.

This module is deliberately self-contained. It defines a narrow adapter Protocol plus a
reference in-memory adapter used only to qualify the adversarial harness. It does not
import or implement sibling Stage-4 production modules and it does not claim that the
integrated R11 runtime has passed these attacks.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Callable, Protocol, runtime_checkable

REPORT_SCHEMA = "CB16_R11_STAGE4_HOSTILE_CUTOVER_REPORT_V1"
MATRIX_SCHEMA = "CB16_R11_STAGE4_HOSTILE_CUTOVER_MATRIX_V1"
SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
SEMANTIC_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"
INITIAL_GENERATION = 41
INITIAL_CHAMPION = "stage4h-synthetic-champion-g41"

INVARIANT_KEYS = (
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


class FailClosed(RuntimeError):
    """Deterministic rejection used by the harness and reference adapter."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class FenceToken:
    owner_id: str
    epoch: int


@dataclass(frozen=True)
class AuthoritySnapshot:
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


@runtime_checkable
class HostileCutoverAdapter(Protocol):
    """Boundary final Stage-4 integration can implement without S4H importing siblings."""

    def snapshot(self) -> AuthoritySnapshot: ...
    def start_runtime(self, runtime_id: str) -> None: ...
    def stop_runtime(self, runtime_id: str) -> None: ...
    def acquire_authority(self, runtime_id: str, *, role: str = "CANONICAL") -> FenceToken: ...
    def release_authority(self, token: FenceToken) -> None: ...
    def crash(self, runtime_id: str) -> None: ...
    def recover(self, runtime_id: str) -> None: ...
    def adopt_authority(self, token: FenceToken, identity: str) -> str: ...
    def begin_adoption(self, token: FenceToken, identity: str) -> None: ...
    def seal_pending_adoption(self, token: FenceToken) -> str: ...
    def begin_journal_transition(
        self, token: FenceToken, event_id: str, payload_hash: str, champion_id: str
    ) -> None: ...
    def commit_pending_journal(self, token: FenceToken, event_id: str) -> str: ...
    def append_journal(
        self, token: FenceToken, event_id: str, payload_hash: str, champion_id: str
    ) -> str: ...
    def commit_transition(self, token: FenceToken, commit_id: str, next_champion_id: str) -> str: ...
    def release_generation(self, token: FenceToken, release_id: str, next_generation: int) -> str: ...
    def issue_permission(self, token: FenceToken, intent_id: str, *, source: str) -> str: ...
    def physics_transition(
        self, token: FenceToken, intent_id: str, permission_token: str | None
    ) -> str: ...
    def admit_evidence(self, token: FenceToken, evidence_id: str, *, source: str) -> str: ...
    def place_state_object(self, object_id: str, state: str) -> None: ...
    def adopt_state_object(self, token: FenceToken, object_id: str) -> str: ...
    def corrupt_authority_metadata(self) -> None: ...


class ReferenceHostileCutoverAdapter:
    """Deterministic fake authority model. It is not a production runtime."""

    def __init__(self) -> None:
        self.generation = INITIAL_GENERATION
        self.champion_id = INITIAL_CHAMPION
        self.runtime_id: str | None = None
        self.owner_id: str | None = None
        self.fence_epoch = 0
        self._live_token: FenceToken | None = None
        self._stale_tokens: set[FenceToken] = set()
        self._max_writer_count = 0
        self._ambiguous_authority_writes = 0
        self._stale_writer_mutations = 0
        self._permission_bypass_mutations = 0
        self._final_holdout_accesses = 0
        self._scientific_evidence_ids: set[str] = set()
        self._replay_evidence_admitted = 0
        self._adoption_identity: str | None = None
        self._pending_adoption: str | None = None
        self._journal: dict[str, str] = {}
        self._pending_journal: dict[str, tuple[str, str]] = {}
        self._commit: tuple[str, str] | None = None
        self._release: tuple[str, int] | None = None
        self._permissions: dict[str, str] = {}
        self._physics_transitions: set[str] = set()
        self._state_objects: dict[str, str] = {}
        self._adopted_state_objects: set[str] = set()
        self._metadata_valid = True
        self._freeze_blob = SEMANTIC_FREEZE_BLOB
        self._scientific_status = SCIENTIFIC_STATUS

    def snapshot(self) -> AuthoritySnapshot:
        return AuthoritySnapshot(
            generation=self.generation,
            champion_id=self.champion_id,
            authoritative_generations=(self.generation,),
            authoritative_champions=(self.champion_id,),
            runtime_id=self.runtime_id,
            writer_owner_id=self.owner_id,
            live_writer_count=1 if self.owner_id is not None else 0,
            fence_epoch=self.fence_epoch,
            adoption_identity=self._adoption_identity,
            scientific_evidence_count=len(self._scientific_evidence_ids),
            replay_evidence_admitted=self._replay_evidence_admitted,
            stale_writer_mutations=self._stale_writer_mutations,
            permission_bypass_mutations=self._permission_bypass_mutations,
            final_holdout_accesses=self._final_holdout_accesses,
            semantic_freeze_blob=self._freeze_blob,
            scientific_status=self._scientific_status,
            metadata_valid=self._metadata_valid,
        )

    def _require_metadata(self) -> None:
        if not self._metadata_valid:
            raise FailClosed("AUTHORITY_METADATA_CORRUPT")

    def _assert_fence(self, token: FenceToken) -> None:
        self._require_metadata()
        if self._live_token != token or self.owner_id != token.owner_id:
            raise FailClosed("STALE_OR_INVALID_FENCE")

    def start_runtime(self, runtime_id: str) -> None:
        self._require_metadata()
        if self.runtime_id is not None:
            raise FailClosed("DUPLICATE_CANONICAL_RUNTIME")
        self.runtime_id = runtime_id

    def stop_runtime(self, runtime_id: str) -> None:
        self._require_metadata()
        if self.runtime_id != runtime_id:
            raise FailClosed("RUNTIME_ID_MISMATCH")
        if self.owner_id is not None:
            raise FailClosed("AUTHORITATIVE_OWNER_MUST_RELEASE_BEFORE_STOP")
        self.runtime_id = None

    def acquire_authority(self, runtime_id: str, *, role: str = "CANONICAL") -> FenceToken:
        self._require_metadata()
        if role != "CANONICAL":
            raise FailClosed("LEGACY_OR_NONCANONICAL_AUTHORITY_DENIED")
        if self.runtime_id != runtime_id:
            raise FailClosed("RUNTIME_NOT_ACTIVE")
        if self.owner_id is not None:
            raise FailClosed("AUTHORITATIVE_OWNER_ALREADY_PRESENT")
        self.fence_epoch += 1
        token = FenceToken(runtime_id, self.fence_epoch)
        self.owner_id = runtime_id
        self._live_token = token
        self._max_writer_count = max(self._max_writer_count, 1)
        return token

    def release_authority(self, token: FenceToken) -> None:
        self._assert_fence(token)
        self._stale_tokens.add(token)
        self.owner_id = None
        self._live_token = None

    def crash(self, runtime_id: str) -> None:
        if self.runtime_id != runtime_id:
            raise FailClosed("CRASH_TARGET_NOT_ACTIVE")
        if self._live_token is not None:
            self._stale_tokens.add(self._live_token)
        self.owner_id = None
        self._live_token = None
        self.runtime_id = None
        self._pending_adoption = None
        self._pending_journal.clear()

    def recover(self, runtime_id: str) -> None:
        self._require_metadata()
        if self.runtime_id is not None:
            raise FailClosed("RECOVERY_WITH_ACTIVE_RUNTIME")
        self.runtime_id = runtime_id
        self._pending_adoption = None
        self._pending_journal.clear()
        self._adopted_state_objects = {
            x for x in self._adopted_state_objects if self._state_objects.get(x) == "SEALED"
        }

    def adopt_authority(self, token: FenceToken, identity: str) -> str:
        self._assert_fence(token)
        if self._adoption_identity is None:
            self._adoption_identity = identity
            return "ADOPTED"
        if self._adoption_identity == identity:
            return "ALREADY_ADOPTED"
        raise FailClosed("CONFLICTING_AUTHORITY_ADOPTION")

    def begin_adoption(self, token: FenceToken, identity: str) -> None:
        self._assert_fence(token)
        if self._adoption_identity is not None and self._adoption_identity != identity:
            raise FailClosed("CONFLICTING_AUTHORITY_ADOPTION")
        self._pending_adoption = identity

    def seal_pending_adoption(self, token: FenceToken) -> str:
        self._assert_fence(token)
        if self._pending_adoption is None:
            raise FailClosed("NO_PENDING_ADOPTION")
        identity = self._pending_adoption
        self._pending_adoption = None
        return self.adopt_authority(token, identity)

    def begin_journal_transition(
        self, token: FenceToken, event_id: str, payload_hash: str, champion_id: str
    ) -> None:
        self._assert_fence(token)
        if champion_id != self.champion_id:
            raise FailClosed("STALE_CHAMPION_WRITER")
        old = self._journal.get(event_id)
        if old is not None and old != payload_hash:
            raise FailClosed("JOURNAL_EVENT_CONFLICT")
        self._pending_journal[event_id] = (payload_hash, champion_id)

    def commit_pending_journal(self, token: FenceToken, event_id: str) -> str:
        self._assert_fence(token)
        pending = self._pending_journal.pop(event_id, None)
        if pending is None:
            raise FailClosed("NO_PENDING_JOURNAL_TRANSITION")
        return self.append_journal(token, event_id, pending[0], pending[1])

    def append_journal(
        self, token: FenceToken, event_id: str, payload_hash: str, champion_id: str
    ) -> str:
        self._assert_fence(token)
        if champion_id != self.champion_id:
            raise FailClosed("STALE_CHAMPION_WRITER")
        old = self._journal.get(event_id)
        if old is None:
            self._journal[event_id] = payload_hash
            return "APPENDED"
        if old == payload_hash:
            return "ALREADY_APPENDED"
        raise FailClosed("JOURNAL_EVENT_CONFLICT")

    def commit_transition(self, token: FenceToken, commit_id: str, next_champion_id: str) -> str:
        self._assert_fence(token)
        proposed = (commit_id, next_champion_id)
        if self._commit is None:
            self._commit = proposed
            return "COMMITTED"
        if self._commit == proposed:
            return "ALREADY_COMMITTED"
        raise FailClosed("DUPLICATE_COMMIT_CONFLICT")

    def release_generation(self, token: FenceToken, release_id: str, next_generation: int) -> str:
        self._assert_fence(token)
        if self._commit is None:
            raise FailClosed("GENERATION_RELEASE_WITHOUT_COMMIT")
        if self._release is not None:
            if self._release == (release_id, next_generation):
                return "ALREADY_RELEASED"
            raise FailClosed("DUPLICATE_GENERATION_RELEASE_CONFLICT")
        if next_generation != self.generation + 1:
            raise FailClosed("GENERATION_RELEASE_SEQUENCE_INVALID")
        self.generation = next_generation
        self.champion_id = self._commit[1]
        self._release = (release_id, next_generation)
        return "RELEASED"

    def issue_permission(self, token: FenceToken, intent_id: str, *, source: str) -> str:
        self._assert_fence(token)
        if source != "FROZEN_SUPERVISOR":
            raise FailClosed("PERMISSION_SOURCE_NOT_SUPERVISOR")
        permission = stable_hash({"intent_id": intent_id, "fence": asdict(token), "source": source})
        self._permissions[permission] = intent_id
        return permission

    def physics_transition(
        self, token: FenceToken, intent_id: str, permission_token: str | None
    ) -> str:
        self._assert_fence(token)
        if permission_token is None or self._permissions.get(permission_token) != intent_id:
            raise FailClosed("PHYSICS_PERMISSION_REQUIRED")
        transition_id = stable_hash(
            {"intent_id": intent_id, "permission": permission_token, "generation": self.generation}
        )
        self._physics_transitions.add(transition_id)
        return transition_id

    def admit_evidence(self, token: FenceToken, evidence_id: str, *, source: str) -> str:
        self._assert_fence(token)
        if source == "REPLAY":
            raise FailClosed("REPLAY_IS_ENGINEERING_ONLY_NOT_NEW_EVIDENCE")
        if source != "SCIENTIFIC_EVIDENCE_PROVIDER":
            raise FailClosed("EVIDENCE_SOURCE_NOT_AUTHORIZED")
        if evidence_id in self._scientific_evidence_ids:
            return "ALREADY_ADMITTED"
        self._scientific_evidence_ids.add(evidence_id)
        return "ADMITTED"

    def place_state_object(self, object_id: str, state: str) -> None:
        if state not in {"SEALED", "ORPHAN", "INCOMPLETE"}:
            raise ValueError("invalid state object state")
        self._state_objects[object_id] = state

    def adopt_state_object(self, token: FenceToken, object_id: str) -> str:
        self._assert_fence(token)
        if self._state_objects.get(object_id) != "SEALED":
            raise FailClosed("STATE_OBJECT_NOT_SEALED_AUTHORITATIVE")
        self._adopted_state_objects.add(object_id)
        return "STATE_OBJECT_ADOPTED"

    def corrupt_authority_metadata(self) -> None:
        self._metadata_valid = False

    @property
    def journal_event_count(self) -> int:
        return len(self._journal)

    @property
    def max_writer_count(self) -> int:
        return self._max_writer_count

    @property
    def ambiguous_authority_writes(self) -> int:
        return self._ambiguous_authority_writes


@dataclass(frozen=True)
class HostileCaseSpec:
    case_id: str
    title: str
    category: str


@dataclass
class HostileCaseResult:
    case_id: str
    title: str
    category: str
    passed: bool
    observed: list[str]
    invariants: dict[str, bool]
    before: dict[str, Any]
    after: dict[str, Any]


@dataclass
class HostileMatrixReport:
    schema: str
    matrix_schema: str
    status: str
    harness_qualified: bool
    integrated_runtime_qualified: bool
    scientific_status: str
    semantic_freeze_blob: str
    case_count: int
    passed_cases: int
    failed_cases: int
    invariants: list[str]
    cases: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_CASES = (
    HostileCaseSpec("H01", "duplicate canonical runtime startup", "runtime_singleton"),
    HostileCaseSpec("H02", "simultaneous ownership attempt", "authority_ownership"),
    HostileCaseSpec("H03", "stale fencing token", "fencing"),
    HostileCaseSpec("H04", "conflicting authority adoption", "adoption"),
    HostileCaseSpec("H05", "duplicate identical adoption", "adoption"),
    HostileCaseSpec("H06", "legacy authoritative writer attempt", "legacy_retirement"),
    HostileCaseSpec("H07", "crash immediately before authority acquisition", "crash_recovery"),
    HostileCaseSpec("H08", "crash immediately after authority acquisition", "crash_recovery"),
    HostileCaseSpec("H09", "crash around adoption seal", "crash_recovery"),
    HostileCaseSpec("H10", "crash around authoritative journal transition", "crash_recovery"),
    HostileCaseSpec("H11", "duplicate commit", "idempotency"),
    HostileCaseSpec("H12", "duplicate generation release", "idempotency"),
    HostileCaseSpec("H13", "stale Champion writer", "lineage"),
    HostileCaseSpec("H14", "forged Permission", "permission"),
    HostileCaseSpec("H15", "direct Physics bypass attempt", "permission"),
    HostileCaseSpec("H16", "replay presented as new Evidence", "evidence"),
    HostileCaseSpec("H17", "orphan/incomplete state object", "storage"),
    HostileCaseSpec("H18", "restart after successful cutover", "restart"),
    HostileCaseSpec("H19", "second restart with stale owner", "restart_fencing"),
    HostileCaseSpec("H20", "malformed/corrupt authority metadata", "metadata"),
)


def _expect_fail(observed: list[str], expected_code: str, fn: Callable[[], Any]) -> None:
    try:
        fn()
    except FailClosed as exc:
        observed.append(exc.code)
        if exc.code != expected_code:
            raise AssertionError(f"expected {expected_code}, got {exc.code}") from exc
        return
    raise AssertionError(f"expected fail-closed code {expected_code}")


def _bootstrap(adapter: HostileCutoverAdapter, runtime_id: str = "runtime-a") -> FenceToken:
    adapter.start_runtime(runtime_id)
    return adapter.acquire_authority(runtime_id)


def _invariants(
    adapter: ReferenceHostileCutoverAdapter, before: AuthoritySnapshot
) -> dict[str, bool]:
    after = adapter.snapshot()
    return {
        "exactly_one_authoritative_generation": (
            len(after.authoritative_generations) == 1
            and after.authoritative_generations[0] == after.generation
        ),
        "exactly_one_authoritative_champion": (
            len(after.authoritative_champions) == 1
            and after.authoritative_champions[0] == after.champion_id
        ),
        "no_dual_writer": after.live_writer_count <= 1 and adapter.max_writer_count <= 1,
        "no_stale_writer_mutation": after.stale_writer_mutations == 0,
        "no_new_scientific_evidence_from_replay": after.replay_evidence_admitted == 0,
        "no_permission_bypass": after.permission_bypass_mutations == 0,
        "final_holdout_untouched": after.final_holdout_accesses == 0,
        "semantic_freeze_unchanged": (
            after.semantic_freeze_blob == before.semantic_freeze_blob == SEMANTIC_FREEZE_BLOB
        ),
        "ambiguous_authority_fails_closed": adapter.ambiguous_authority_writes == 0,
    }


def _case_h01(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    a.start_runtime("runtime-a")
    _expect_fail(o, "DUPLICATE_CANONICAL_RUNTIME", lambda: a.start_runtime("runtime-b"))


def _case_h02(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    _bootstrap(a)
    _expect_fail(o, "RUNTIME_NOT_ACTIVE", lambda: a.acquire_authority("runtime-b"))


def _case_h03(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t1 = _bootstrap(a)
    a.release_authority(t1)
    a.stop_runtime("runtime-a")
    a.start_runtime("runtime-b")
    t2 = a.acquire_authority("runtime-b")
    assert t2.epoch > t1.epoch
    _expect_fail(
        o,
        "STALE_OR_INVALID_FENCE",
        lambda: a.append_journal(t1, "e-stale", "p", INITIAL_CHAMPION),
    )


def _case_h04(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    assert a.adopt_authority(t, "source-authority-a") == "ADOPTED"
    _expect_fail(
        o,
        "CONFLICTING_AUTHORITY_ADOPTION",
        lambda: a.adopt_authority(t, "source-authority-b"),
    )


def _case_h05(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    g = a.snapshot().generation
    assert a.adopt_authority(t, "source-authority-a") == "ADOPTED"
    assert a.adopt_authority(t, "source-authority-a") == "ALREADY_ADOPTED"
    assert a.snapshot().generation == g


def _case_h06(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    a.start_runtime("legacy-runtime")
    _expect_fail(
        o,
        "LEGACY_OR_NONCANONICAL_AUTHORITY_DENIED",
        lambda: a.acquire_authority("legacy-runtime", role="LEGACY"),
    )


def _case_h07(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    a.start_runtime("runtime-a")
    a.crash("runtime-a")
    a.recover("runtime-b")
    t = a.acquire_authority("runtime-b")
    assert t.owner_id == "runtime-b"


def _case_h08(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t1 = _bootstrap(a)
    a.crash("runtime-a")
    a.recover("runtime-b")
    t2 = a.acquire_authority("runtime-b")
    assert t2.epoch > t1.epoch
    _expect_fail(
        o,
        "STALE_OR_INVALID_FENCE",
        lambda: a.append_journal(t1, "stale", "p", INITIAL_CHAMPION),
    )


def _case_h09(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t1 = _bootstrap(a)
    a.begin_adoption(t1, "source-authority-a")
    a.crash("runtime-a")
    a.recover("runtime-b")
    t2 = a.acquire_authority("runtime-b")
    _expect_fail(o, "NO_PENDING_ADOPTION", lambda: a.seal_pending_adoption(t2))
    assert a.adopt_authority(t2, "source-authority-a") == "ADOPTED"


def _case_h10(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t1 = _bootstrap(a)
    a.begin_journal_transition(t1, "event-1", "payload-1", INITIAL_CHAMPION)
    a.crash("runtime-a")
    a.recover("runtime-b")
    t2 = a.acquire_authority("runtime-b")
    _expect_fail(
        o,
        "NO_PENDING_JOURNAL_TRANSITION",
        lambda: a.commit_pending_journal(t2, "event-1"),
    )
    assert a.append_journal(t2, "event-1", "payload-1", INITIAL_CHAMPION) == "APPENDED"
    assert a.journal_event_count == 1


def _case_h11(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    assert a.commit_transition(t, "commit-1", "challenger-g42") == "COMMITTED"
    assert a.commit_transition(t, "commit-1", "challenger-g42") == "ALREADY_COMMITTED"
    _expect_fail(
        o,
        "DUPLICATE_COMMIT_CONFLICT",
        lambda: a.commit_transition(t, "commit-1", "other-challenger"),
    )


def _case_h12(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    g = a.snapshot().generation
    a.commit_transition(t, "commit-1", "challenger-g42")
    assert a.release_generation(t, "release-1", g + 1) == "RELEASED"
    assert a.release_generation(t, "release-1", g + 1) == "ALREADY_RELEASED"
    assert a.snapshot().generation == g + 1


def _case_h13(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    _expect_fail(
        o,
        "STALE_CHAMPION_WRITER",
        lambda: a.append_journal(t, "event-stale-champion", "p", "old-champion"),
    )
    assert a.journal_event_count == 0


def _case_h14(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    _expect_fail(
        o,
        "PHYSICS_PERMISSION_REQUIRED",
        lambda: a.physics_transition(t, "intent-1", "forged-permission"),
    )


def _case_h15(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    _expect_fail(
        o,
        "PHYSICS_PERMISSION_REQUIRED",
        lambda: a.physics_transition(t, "intent-1", None),
    )
    p = a.issue_permission(t, "intent-1", source="FROZEN_SUPERVISOR")
    assert a.physics_transition(t, "intent-1", p)


def _case_h16(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    count = a.snapshot().scientific_evidence_count
    _expect_fail(
        o,
        "REPLAY_IS_ENGINEERING_ONLY_NOT_NEW_EVIDENCE",
        lambda: a.admit_evidence(t, "replay-e1", source="REPLAY"),
    )
    assert a.snapshot().scientific_evidence_count == count


def _case_h17(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t = _bootstrap(a)
    a.place_state_object("orphan-1", "ORPHAN")
    a.place_state_object("incomplete-1", "INCOMPLETE")
    _expect_fail(
        o,
        "STATE_OBJECT_NOT_SEALED_AUTHORITATIVE",
        lambda: a.adopt_state_object(t, "orphan-1"),
    )
    _expect_fail(
        o,
        "STATE_OBJECT_NOT_SEALED_AUTHORITATIVE",
        lambda: a.adopt_state_object(t, "incomplete-1"),
    )


def _case_h18(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
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


def _case_h19(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    t1 = _bootstrap(a)
    a.crash("runtime-a")
    a.recover("runtime-b")
    t2 = a.acquire_authority("runtime-b")
    a.crash("runtime-b")
    a.recover("runtime-c")
    t3 = a.acquire_authority("runtime-c")
    assert t1.epoch < t2.epoch < t3.epoch
    _expect_fail(
        o,
        "STALE_OR_INVALID_FENCE",
        lambda: a.append_journal(t1, "old-owner", "p", INITIAL_CHAMPION),
    )
    _expect_fail(
        o,
        "STALE_OR_INVALID_FENCE",
        lambda: a.append_journal(t2, "middle-owner", "p", INITIAL_CHAMPION),
    )


def _case_h20(a: ReferenceHostileCutoverAdapter, o: list[str]) -> None:
    a.start_runtime("runtime-a")
    a.corrupt_authority_metadata()
    _expect_fail(o, "AUTHORITY_METADATA_CORRUPT", lambda: a.acquire_authority("runtime-a"))


_CASE_RUNNERS: dict[str, Callable[[ReferenceHostileCutoverAdapter, list[str]], None]] = {
    "H01": _case_h01,
    "H02": _case_h02,
    "H03": _case_h03,
    "H04": _case_h04,
    "H05": _case_h05,
    "H06": _case_h06,
    "H07": _case_h07,
    "H08": _case_h08,
    "H09": _case_h09,
    "H10": _case_h10,
    "H11": _case_h11,
    "H12": _case_h12,
    "H13": _case_h13,
    "H14": _case_h14,
    "H15": _case_h15,
    "H16": _case_h16,
    "H17": _case_h17,
    "H18": _case_h18,
    "H19": _case_h19,
    "H20": _case_h20,
}


def run_case(
    spec: HostileCaseSpec,
    adapter_factory: Callable[[], ReferenceHostileCutoverAdapter] = ReferenceHostileCutoverAdapter,
) -> HostileCaseResult:
    adapter = adapter_factory()
    before_obj = adapter.snapshot()
    observed: list[str] = []
    error: str | None = None
    try:
        _CASE_RUNNERS[spec.case_id](adapter, observed)
    except BaseException as exc:
        error = f"{type(exc).__name__}:{exc}"
        observed.append(error)
    invariant_map = _invariants(adapter, before_obj)
    passed = error is None and all(invariant_map.values())
    return HostileCaseResult(
        case_id=spec.case_id,
        title=spec.title,
        category=spec.category,
        passed=passed,
        observed=observed,
        invariants=invariant_map,
        before=asdict(before_obj),
        after=asdict(adapter.snapshot()),
    )


def run_hostile_matrix(
    adapter_factory: Callable[[], ReferenceHostileCutoverAdapter] = ReferenceHostileCutoverAdapter,
) -> dict[str, Any]:
    results = [run_case(spec, adapter_factory) for spec in REQUIRED_CASES]
    passed = sum(1 for result in results if result.passed)
    report = HostileMatrixReport(
        schema=REPORT_SCHEMA,
        matrix_schema=MATRIX_SCHEMA,
        status="PASS" if passed == len(results) else "FAIL",
        harness_qualified=passed == len(results),
        integrated_runtime_qualified=False,
        scientific_status=SCIENTIFIC_STATUS,
        semantic_freeze_blob=SEMANTIC_FREEZE_BLOB,
        case_count=len(results),
        passed_cases=passed,
        failed_cases=len(results) - passed,
        invariants=list(INVARIANT_KEYS),
        cases=[asdict(result) for result in results],
    )
    return report.to_dict()


def matrix_manifest() -> dict[str, Any]:
    return {
        "schema": MATRIX_SCHEMA,
        "scientific_status": SCIENTIFIC_STATUS,
        "semantic_freeze_blob": SEMANTIC_FREEZE_BLOB,
        "integrated_runtime_qualification_claimed": False,
        "invariants": list(INVARIANT_KEYS),
        "cases": [asdict(case) for case in REQUIRED_CASES],
    }
