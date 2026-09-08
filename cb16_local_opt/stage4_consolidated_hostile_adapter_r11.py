from __future__ import annotations

"""Final Stage-4 hostile adapter bound to consolidated R11 authority primitives.

This is qualification plumbing, not a second runtime implementation.  Positive
process authority is always the real S4D flock/fencing primitive; authority
adoption is persisted through the real S4C atomic receipt; legacy/replay
admission is checked through the real S4G policy.  A small durable observation
ledger records hostile-case state so the INTG H01-H20 harness can inspect the
consolidated authority invariants without granting authority itself.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
from typing import Any

from .event_journal_r11 import EventItemR11, EventJournalR11
from .stage4_authority_adoption_r11 import (
    AuthorityAdoptionConflict,
    AuthorityAdoptionContract,
    SourceAuthorityIdentity,
    SourceAuthorityMismatch,
    adopt_authority as publish_authority_adoption,
)
from .stage4_authority_lease_r11 import (
    AuthorityLeaseError,
    CorruptAuthorityLeaseError,
    FencingTokenR11,
    Stage4AuthorityLeaseR11,
    StaleFencingTokenError,
)
from .stage4_hostile_cutover_r11 import (
    FailClosed,
    INITIAL_CHAMPION,
    INITIAL_GENERATION,
    SCIENTIFIC_STATUS,
    SEMANTIC_FREEZE_BLOB,
)
from .stage4_hostile_integration_harness_r11 import IntegratedAuthoritySnapshot
from .stage4_legacy_retirement_r11 import (
    Capability,
    LegacyAuthorityDenied,
    LegacyRetirementGuard,
    RuntimeRole,
)


CONTROL_SCHEMA = "CB16_R11_STAGE4_CONSOLIDATED_HOSTILE_CONTROL_V1"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConsolidatedHostileFence:
    runtime_id: str
    lease: Stage4AuthorityLeaseR11
    native: FencingTokenR11

    @property
    def epoch(self) -> int:
        return int(self.native.epoch)


class ConsolidatedProductionHostileAdapterR11:
    """Production-facing INTG adapter using real Stage-4 authority primitives."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="cb16-stage4-hostile-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.lease_root = self.root / "authority_lease"
        Stage4AuthorityLeaseR11.initialize(self.lease_root)
        self.adoption_path = self.root / "authority_adoption.json"
        self.control_path = self.root / "hostile_control.json"
        self.state_objects = self.root / "state_objects"
        self.state_objects.mkdir(exist_ok=True)
        self.journal = EventJournalR11(self.root / "event_journal")
        self.guard = LegacyRetirementGuard(b"stage4-final-hostile-guard-key-0000000000000000")

        self.runtime_id: str | None = None
        self._lease: Stage4AuthorityLeaseR11 | None = None
        self._active: ConsolidatedHostileFence | None = None
        self._recovery_requested = False
        self._pending_adoption: str | None = None
        self._pending_journal: tuple[str, str, str] | None = None
        self._permissions: dict[str, str] = {}
        self._max_writer_count = 0
        self._metadata_valid = True
        self._load_or_initialize_control()

    # ---------- durable observation ledger; never grants authority ----------
    def _default_control(self) -> dict[str, Any]:
        return {
            "schema": CONTROL_SCHEMA,
            "generation": INITIAL_GENERATION,
            "champion_id": INITIAL_CHAMPION,
            "adoption_identity": None,
            "scientific_evidence_count": 0,
            "replay_evidence_admitted": 0,
            "stale_writer_mutations": 0,
            "permission_bypass_mutations": 0,
            "final_holdout_accesses": 0,
            "ambiguous_authority_writes": 0,
            "commits": {},
            "releases": {},
        }

    def _load_or_initialize_control(self) -> None:
        if not self.control_path.exists():
            self._control = self._default_control()
            self._persist_control()
            return
        try:
            obj = json.loads(self.control_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise FailClosed("AUTHORITY_METADATA_CORRUPT") from exc
        if not isinstance(obj, dict) or obj.get("schema") != CONTROL_SCHEMA:
            raise FailClosed("AUTHORITY_METADATA_CORRUPT")
        self._control = obj

    def _persist_control(self) -> None:
        tmp = self.control_path.with_suffix(f".{os.getpid()}.{secrets.token_hex(4)}.tmp")
        raw = json.dumps(self._control, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        with open(tmp, "x", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.control_path)
        fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    # ---------- authority helpers ----------
    def _require_runtime(self, runtime_id: str) -> None:
        if self.runtime_id != runtime_id:
            raise FailClosed("RUNTIME_NOT_ACTIVE")

    def _require_token(self, token: object) -> ConsolidatedHostileFence:
        if not isinstance(token, ConsolidatedHostileFence):
            raise FailClosed("STALE_OR_INVALID_FENCE")
        if self._active is None or token != self._active or token.runtime_id != self.runtime_id:
            raise FailClosed("STALE_OR_INVALID_FENCE")
        try:
            token.lease.assert_fencing_token(token.native)
        except (StaleFencingTokenError, AuthorityLeaseError) as exc:
            raise FailClosed("STALE_OR_INVALID_FENCE") from exc
        return token

    def _source(self, identity: str) -> SourceAuthorityIdentity:
        champion = str(self._control["champion_id"])
        return SourceAuthorityIdentity(
            source_repo="GY-Bai/CB16-R10",
            source_sha=_sha256("hostile-source:" + identity)[:40],
            semantic_freeze_identity=SEMANTIC_FREEZE_BLOB,
            source_generation=int(self._control["generation"]),
            champion_identity=champion,
            champion_hash=_sha256(champion),
            checkpoint_identity="hostile-checkpoint:" + identity,
            checkpoint_hash=_sha256("checkpoint:" + identity),
            evidence_root_identity="hostile-evidence-root",
            journal_head_identity="hostile-journal-head",
            checkpoint_root_identity="hostile-checkpoint-root",
        )

    # ---------- INTG adapter surface ----------
    def snapshot(self) -> IntegratedAuthoritySnapshot:
        epoch = 0
        metadata_valid = self._metadata_valid
        try:
            snap = Stage4AuthorityLeaseR11(self.lease_root, owner_id="snapshot-only").inspect_current_authority()
            epoch = int(snap.epoch)
        except CorruptAuthorityLeaseError:
            metadata_valid = False
        return IntegratedAuthoritySnapshot(
            generation=int(self._control["generation"]),
            champion_id=str(self._control["champion_id"]),
            authoritative_generations=(int(self._control["generation"]),),
            authoritative_champions=(str(self._control["champion_id"]),),
            runtime_id=self.runtime_id,
            writer_owner_id=None if self._active is None else self._active.runtime_id,
            live_writer_count=0 if self._active is None else 1,
            fence_epoch=epoch,
            adoption_identity=self._control.get("adoption_identity"),
            scientific_evidence_count=int(self._control["scientific_evidence_count"]),
            replay_evidence_admitted=int(self._control["replay_evidence_admitted"]),
            stale_writer_mutations=int(self._control["stale_writer_mutations"]),
            permission_bypass_mutations=int(self._control["permission_bypass_mutations"]),
            final_holdout_accesses=int(self._control["final_holdout_accesses"]),
            semantic_freeze_blob=SEMANTIC_FREEZE_BLOB,
            scientific_status=SCIENTIFIC_STATUS,
            metadata_valid=metadata_valid,
            max_writer_count=self._max_writer_count,
            ambiguous_authority_writes=int(self._control["ambiguous_authority_writes"]),
            journal_event_count=int(self.journal.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
        )

    def start_runtime(self, runtime_id: str) -> None:
        if self.runtime_id is not None:
            raise FailClosed("DUPLICATE_CANONICAL_RUNTIME")
        self.runtime_id = str(runtime_id)
        self._lease = Stage4AuthorityLeaseR11(self.lease_root, owner_id=self.runtime_id)

    def stop_runtime(self, runtime_id: str) -> None:
        self._require_runtime(runtime_id)
        if self._active is not None:
            raise FailClosed("AUTHORITATIVE_RUNTIME_STILL_OWNS_FENCE")
        self.runtime_id = None
        self._lease = None

    def acquire_authority(self, runtime_id: str, *, role: str = "CANONICAL") -> object:
        self._require_runtime(runtime_id)
        if role != "CANONICAL":
            legacy = self.guard.issue_identity(runtime_id, RuntimeRole.LEGACY_REFERENCE)
            try:
                self.guard.issue_token(legacy, Capability.ACQUIRE_CANONICAL_RUNTIME_AUTHORITY)
            except LegacyAuthorityDenied as exc:
                raise FailClosed("LEGACY_OR_NONCANONICAL_AUTHORITY_DENIED") from exc
            raise FailClosed("LEGACY_OR_NONCANONICAL_AUTHORITY_DENIED")
        if self._active is not None:
            raise FailClosed("RUNTIME_ALREADY_OWNS_AUTHORITY")
        assert self._lease is not None
        try:
            native = (
                self._lease.recover_after_dead_owner()
                if self._recovery_requested
                else self._lease.acquire_authority()
            )
        except CorruptAuthorityLeaseError as exc:
            self._metadata_valid = False
            raise FailClosed("AUTHORITY_METADATA_CORRUPT") from exc
        except AuthorityLeaseError as exc:
            raise FailClosed("AUTHORITY_ACQUISITION_FAILED") from exc
        self._recovery_requested = False
        self._active = ConsolidatedHostileFence(runtime_id, self._lease, native)
        self._max_writer_count = max(self._max_writer_count, 1)
        return self._active

    def release_authority(self, token: object) -> None:
        t = self._require_token(token)
        t.lease.release_authority(t.native)
        self._active = None

    def crash(self, runtime_id: str) -> None:
        self._require_runtime(runtime_id)
        if self._active is not None:
            lease = self._active.lease
            fd = getattr(lease, "_fd", None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            # Hostile fault injection: emulate process death after kernel closes fd,
            # while intentionally leaving durable ACTIVE metadata untouched.
            lease._fd = None
            lease._token = None
            lease._owner_process_pid = None
        self._active = None
        self.runtime_id = None
        self._lease = None
        self._pending_adoption = None
        self._pending_journal = None
        self._permissions.clear()

    def recover(self, runtime_id: str) -> None:
        if self.runtime_id is not None:
            raise FailClosed("DUPLICATE_CANONICAL_RUNTIME")
        self._load_or_initialize_control()
        self.runtime_id = str(runtime_id)
        self._lease = Stage4AuthorityLeaseR11(self.lease_root, owner_id=self.runtime_id)
        try:
            state = self._lease.inspect_current_authority()
            self._recovery_requested = state.state == "ACTIVE"
        except CorruptAuthorityLeaseError as exc:
            self._metadata_valid = False
            raise FailClosed("AUTHORITY_METADATA_CORRUPT") from exc

    def fence_strictly_newer(self, newer: object, older: object) -> bool:
        return isinstance(newer, ConsolidatedHostileFence) and isinstance(older, ConsolidatedHostileFence) and newer.epoch > older.epoch

    def adopt_authority(self, token: object, identity: str) -> str:
        self._require_token(token)
        source = self._source(identity)
        contract = AuthorityAdoptionContract(source, "CB16-R11-CANONICAL-RUNTIME")
        try:
            result = publish_authority_adoption(source, contract=contract, receipt_path=self.adoption_path)
        except (AuthorityAdoptionConflict, SourceAuthorityMismatch) as exc:
            raise FailClosed("CONFLICTING_AUTHORITY_ADOPTION") from exc
        self._control["adoption_identity"] = identity
        self._persist_control()
        return "ADOPTED" if result.status == "ADOPTED" else "ALREADY_ADOPTED"

    def begin_adoption(self, token: object, identity: str) -> None:
        self._require_token(token)
        self._source(identity).validate()
        self._pending_adoption = identity

    def seal_pending_adoption(self, token: object) -> str:
        self._require_token(token)
        if self._pending_adoption is None:
            raise FailClosed("NO_PENDING_ADOPTION")
        value = self._pending_adoption
        self._pending_adoption = None
        return self.adopt_authority(token, value)

    def begin_journal_transition(self, token: object, event_id: str, payload_hash: str, champion_id: str) -> None:
        self._require_token(token)
        if champion_id != self._control["champion_id"]:
            raise FailClosed("STALE_CHAMPION_WRITER")
        self._pending_journal = (event_id, payload_hash, champion_id)

    def commit_pending_journal(self, token: object, event_id: str) -> str:
        self._require_token(token)
        if self._pending_journal is None or self._pending_journal[0] != event_id:
            raise FailClosed("NO_PENDING_JOURNAL_TRANSITION")
        row = self._pending_journal
        self._pending_journal = None
        return self.append_journal(token, *row)

    def append_journal(self, token: object, event_id: str, payload_hash: str, champion_id: str) -> str:
        self._require_token(token)
        if champion_id != self._control["champion_id"]:
            raise FailClosed("STALE_CHAMPION_WRITER")
        item = EventItemR11(
            event_id=event_id,
            event_type="STAGE4_HOSTILE_CONTROL",
            generation=int(self._control["generation"]),
            policy_weight_hash=champion_id,
            snapshot_hash="STAGE4_HOSTILE_SNAPSHOT",
            lineage_hash=payload_hash,
            payload={"event_id": event_id, "payload_hash": payload_hash, "champion_id": champion_id},
        )
        self.journal.seal_trace_batch([item], trace_batch_id="HOSTILE:" + event_id)
        return "APPENDED"

    def commit_transition(self, token: object, commit_id: str, next_champion_id: str) -> str:
        self._require_token(token)
        commits = self._control["commits"]
        if commit_id in commits:
            if commits[commit_id] == next_champion_id:
                return "ALREADY_COMMITTED"
            raise FailClosed("DUPLICATE_COMMIT_CONFLICT")
        commits[commit_id] = next_champion_id
        self._persist_control()
        return "COMMITTED"

    def release_generation(self, token: object, release_id: str, next_generation: int) -> str:
        self._require_token(token)
        releases = self._control["releases"]
        if release_id in releases:
            if int(releases[release_id]) == int(next_generation):
                return "ALREADY_RELEASED"
            raise FailClosed("DUPLICATE_GENERATION_RELEASE_CONFLICT")
        if int(next_generation) != int(self._control["generation"]) + 1:
            raise FailClosed("GENERATION_RELEASE_NOT_STRICTLY_NEXT")
        if not self._control["commits"]:
            raise FailClosed("GENERATION_RELEASE_REQUIRES_COMMIT")
        next_champion = list(self._control["commits"].values())[-1]
        releases[release_id] = int(next_generation)
        self._control["generation"] = int(next_generation)
        self._control["champion_id"] = next_champion
        self._persist_control()
        return "RELEASED"

    def issue_permission(self, token: object, intent_id: str, *, source: str) -> str:
        t = self._require_token(token)
        if source != "FROZEN_SUPERVISOR":
            raise FailClosed("PHYSICS_PERMISSION_REQUIRED")
        permission = _sha256(f"permission:{t.epoch}:{intent_id}:{source}")
        self._permissions[intent_id] = permission
        return permission

    def physics_transition(self, token: object, intent_id: str, permission_token: str | None) -> str:
        self._require_token(token)
        if permission_token is None or self._permissions.get(intent_id) != permission_token:
            raise FailClosed("PHYSICS_PERMISSION_REQUIRED")
        return "EXECUTED"

    def admit_evidence(self, token: object, evidence_id: str, *, source: str) -> str:
        self._require_token(token)
        if source == "REPLAY":
            replay = self.guard.issue_identity("hostile-replay", RuntimeRole.LEGACY_REPLAY)
            try:
                self.guard.issue_token(replay, Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE)
            except LegacyAuthorityDenied as exc:
                raise FailClosed("REPLAY_IS_ENGINEERING_ONLY_NOT_NEW_EVIDENCE") from exc
            raise FailClosed("REPLAY_IS_ENGINEERING_ONLY_NOT_NEW_EVIDENCE")
        self._control["scientific_evidence_count"] = int(self._control["scientific_evidence_count"]) + 1
        self._persist_control()
        return evidence_id

    def place_state_object(self, object_id: str, state: str) -> None:
        path = self.state_objects / (object_id + ".json")
        path.write_text(json.dumps({"object_id": object_id, "state": state}, sort_keys=True) + "\n", encoding="utf-8")

    def adopt_state_object(self, token: object, object_id: str) -> str:
        self._require_token(token)
        path = self.state_objects / (object_id + ".json")
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise FailClosed("STATE_OBJECT_NOT_SEALED_AUTHORITATIVE") from exc
        if obj.get("state") != "SEALED":
            raise FailClosed("STATE_OBJECT_NOT_SEALED_AUTHORITATIVE")
        return "ADOPTED_STATE_OBJECT"

    def corrupt_authority_metadata(self) -> None:
        (self.lease_root / "stage4_authority_lease_state.json").write_text("{corrupt\n", encoding="utf-8")
        self._metadata_valid = False
