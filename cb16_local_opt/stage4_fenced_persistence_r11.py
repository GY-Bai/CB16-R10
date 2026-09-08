from __future__ import annotations

"""Canonical Stage-4 persistence fencing integration for CB16 R11.

This module composes already-qualified S4D fencing, S4F state-root ownership, and
S4G legacy-retirement eligibility around the existing R11 persistence protocol
adapters. It changes no scientific meaning and grants no authority by itself.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .checkpoint_store_r11 import CheckpointStoreR11
from .evidence_store_r11 import EvidenceStoreR11
from .event_journal_r11 import EventJournalR11
from .integration_adapters_r11 import (
    CheckpointStoreProtocolAdapterR11,
    EvidenceStoreProtocolAdapterR11,
    EventJournalProtocolAdapterR11,
)
from .stage4_authority_lease_r11 import FencingTokenR11, Stage4AuthorityLeaseR11
from .stage4_legacy_retirement_r11 import (
    Capability,
    LegacyRetirementGuard,
    RetirementCapabilityToken,
    RuntimeIdentity,
    RuntimeRole,
)
from .stage4_state_roots_r11 import Stage4StartupReceiptR11, Stage4StateRootsR11


SCIENTIFIC_STATUS = (
    "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
)


class Stage4CanonicalPersistenceError(RuntimeError):
    pass


class Stage4CanonicalRoleRequired(Stage4CanonicalPersistenceError):
    pass


class Stage4PersistenceRootBindingError(Stage4CanonicalPersistenceError):
    pass


_REQUIRED_CAPABILITIES = (
    Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE,
    Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL,
    Capability.SEAL_TRAINING_SNAPSHOT,
    Capability.TRANSITION_CHALLENGER_CHAMPION,
    Capability.SEAL_CHECKPOINT,
    Capability.RELEASE_GENERATION,
)


def _is_under(path: Path, root: Path) -> bool:
    path = path.resolve(strict=False)
    root = root.resolve(strict=False)
    return path == root or root in path.parents


@dataclass(frozen=True)
class CanonicalWriteAuthorityR11:
    """Typed, process-local mutation authority binding.

    S4G admission is negative eligibility only. Positive write authority remains
    the currently-live S4D lease/fence, asserted after root and role validation and
    immediately before every authoritative mutation boundary.
    """

    roots: Stage4StateRootsR11
    root_receipt: Stage4StartupReceiptR11
    lease: Stage4AuthorityLeaseR11
    fencing_token: FencingTokenR11
    retirement_guard: LegacyRetirementGuard
    runtime_identity: RuntimeIdentity
    capability_tokens: Mapping[Capability, RetirementCapabilityToken]

    @classmethod
    def bind(
        cls,
        *,
        roots: Stage4StateRootsR11,
        root_receipt: Stage4StartupReceiptR11,
        lease: Stage4AuthorityLeaseR11,
        fencing_token: FencingTokenR11,
        retirement_guard: LegacyRetirementGuard,
        runtime_identity: RuntimeIdentity,
    ) -> "CanonicalWriteAuthorityR11":
        role = retirement_guard.validate_identity(runtime_identity)
        if role is not RuntimeRole.R11_CANONICAL_RUNTIME:
            raise Stage4CanonicalRoleRequired(
                f"INTB_CANONICAL_RUNTIME_ROLE_REQUIRED:{role.value}"
            )
        if not isinstance(fencing_token, FencingTokenR11):
            raise TypeError("INTB_TYPED_FENCING_TOKEN_REQUIRED")
        receipt = roots.verify_startup(
            expected_control_id=root_receipt.control_root_id,
            expected_data_id=root_receipt.data_root_id,
        )
        if receipt.frozen_raw_identity != root_receipt.frozen_raw_identity:
            raise Stage4PersistenceRootBindingError("INTB_FROZEN_RAW_IDENTITY_DRIFT")
        lease_namespace = roots.control_path("runtime_lease_fencing_state")
        if not _is_under(Path(lease.root), lease_namespace):
            raise Stage4PersistenceRootBindingError(
                f"INTB_LEASE_OUTSIDE_S4F_CONTROL_NAMESPACE:{lease.root}"
            )
        tokens = {
            capability: retirement_guard.issue_token(runtime_identity, capability)
            for capability in _REQUIRED_CAPABILITIES
        }
        authority = cls(
            roots=roots,
            root_receipt=root_receipt,
            lease=lease,
            fencing_token=fencing_token,
            retirement_guard=retirement_guard,
            runtime_identity=runtime_identity,
            capability_tokens=tokens,
        )
        authority.lease.assert_fencing_token(authority.fencing_token)
        return authority

    def verify_roots(self) -> Stage4StartupReceiptR11:
        receipt = self.roots.verify_startup(
            expected_control_id=self.root_receipt.control_root_id,
            expected_data_id=self.root_receipt.data_root_id,
        )
        if receipt.frozen_raw_identity != self.root_receipt.frozen_raw_identity:
            raise Stage4PersistenceRootBindingError("INTB_FROZEN_RAW_IDENTITY_DRIFT")
        return receipt

    def assert_mutation(self, capability: Capability, *, mutation: str) -> None:
        """Require exact S4F roots + S4G eligibility + current live S4D fence.

        The S4D assertion is deliberately last. S4G never grants positive authority.
        """
        self.verify_roots()
        token = self.capability_tokens.get(capability)
        if token is None:
            raise Stage4CanonicalPersistenceError(
                f"INTB_CAPABILITY_TOKEN_MISSING:{capability.value}:{mutation}"
            )
        decision = self.retirement_guard.assert_token(
            self.runtime_identity, capability, token
        )
        if decision.authority_granted:
            raise Stage4CanonicalPersistenceError("INTB_S4G_MUST_NOT_GRANT_AUTHORITY")
        if not decision.eligible_under_legacy_retirement_policy:
            raise Stage4CanonicalPersistenceError("INTB_S4G_WRITE_NOT_ELIGIBLE")
        if not decision.requires_independent_runtime_authority:
            raise Stage4CanonicalPersistenceError("INTB_AUTHORITATIVE_CAPABILITY_EXPECTED")
        self.lease.assert_fencing_token(self.fencing_token)


class Stage4FencedEvidenceStoreR11(EvidenceStoreR11):
    """Existing immutable Evidence payload store with canonical live-fence admission."""

    def __init__(self, *args: Any, authority: CanonicalWriteAuthorityR11, **kwargs: Any):
        self._write_authority = authority
        authority.assert_mutation(
            Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE,
            mutation="evidence.backend_open",
        )
        super().__init__(*args, **kwargs)

    def put_evidence(self, *args: Any, **kwargs: Any) -> Any:
        self._write_authority.assert_mutation(
            Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE,
            mutation="evidence.put_evidence",
        )
        return super().put_evidence(*args, **kwargs)

    def seal_evidence_set(self, *args: Any, **kwargs: Any) -> Any:
        self._write_authority.assert_mutation(
            Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE,
            mutation="evidence.seal_evidence_set",
        )
        return super().seal_evidence_set(*args, **kwargs)

    def seal_all_active_segments(self) -> None:
        self._write_authority.assert_mutation(
            Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE,
            mutation="evidence.seal_all_active_segments",
        )
        return super().seal_all_active_segments()

    def checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        self._write_authority.assert_mutation(
            Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE,
            mutation=f"evidence.wal_checkpoint:{mode}",
        )
        return super().checkpoint(mode)


class Stage4FencedEventJournalR11(EventJournalR11):
    """Existing Event Journal with live-fence admission on all exposed mutations."""

    def __init__(self, *args: Any, authority: CanonicalWriteAuthorityR11, **kwargs: Any):
        self._write_authority = authority
        authority.assert_mutation(
            Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL,
            mutation="journal.backend_open",
        )
        super().__init__(*args, **kwargs)

    def seal_trace_batch(self, *args: Any, **kwargs: Any) -> Any:
        self._write_authority.assert_mutation(
            Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL,
            mutation="journal.seal_trace_batch",
        )
        return super().seal_trace_batch(*args, **kwargs)

    def seal_generation_outcome(self, *args: Any, **kwargs: Any) -> Any:
        self._write_authority.assert_mutation(
            Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL,
            mutation="journal.seal_generation_outcome",
        )
        return super().seal_generation_outcome(*args, **kwargs)

    def checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        self._write_authority.assert_mutation(
            Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL,
            mutation=f"journal.wal_checkpoint:{mode}",
        )
        return super().checkpoint(mode)


class Stage4RoutedCheckpointStoreR11(CheckpointStoreR11):
    """Existing checkpoint semantics with S4F split control/data placement and fencing.

    SQLite, object metadata, generation snapshots and generation pointers stay under the
    S4F checkpoint control namespace. Immutable tensor object bytes are redirected into
    the S4F cold-content data namespace. Tensor semantic identity is unchanged.
    """

    def __init__(
        self,
        *,
        control_root: str | Path,
        object_data_root: str | Path,
        authority: CanonicalWriteAuthorityR11,
        synchronous: str = "FULL",
    ):
        self._write_authority = authority
        authority.assert_mutation(
            Capability.SEAL_CHECKPOINT, mutation="checkpoint.backend_open"
        )
        super().__init__(control_root, synchronous=synchronous)
        self.object_root = Path(object_data_root).resolve()
        authority.assert_mutation(
            Capability.SEAL_CHECKPOINT, mutation="checkpoint.object_data_root_open"
        )
        self.object_root.mkdir(parents=True, exist_ok=True)

    def put_state_dict(self, state: Any) -> Any:
        self._write_authority.assert_mutation(
            Capability.SEAL_CHECKPOINT, mutation="checkpoint.put_state_dict"
        )
        return super().put_state_dict(state)

    def seal_generation_checkpoint(self, **kwargs: Any) -> Any:
        self._write_authority.assert_mutation(
            Capability.SEAL_CHECKPOINT,
            mutation=f"checkpoint.seal_generation_checkpoint:G{kwargs.get('generation', '')}",
        )
        return super().seal_generation_checkpoint(**kwargs)

    def checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        self._write_authority.assert_mutation(
            Capability.SEAL_CHECKPOINT, mutation=f"checkpoint.wal_checkpoint:{mode}"
        )
        return super().checkpoint(mode)


@dataclass
class CanonicalPersistenceBundleR11:
    evidence_store: EvidenceStoreProtocolAdapterR11
    journal: EventJournalProtocolAdapterR11
    checkpoint_store: CheckpointStoreProtocolAdapterR11
    evidence_payloads: Stage4FencedEvidenceStoreR11
    checkpoint_objects: Stage4RoutedCheckpointStoreR11
    _journal_backend: Stage4FencedEventJournalR11

    def close(self) -> None:
        errors: list[BaseException] = []
        for backend in (self.evidence_payloads, self._journal_backend, self.checkpoint_objects):
            try:
                backend.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise Stage4CanonicalPersistenceError(
                "INTB_CANONICAL_PERSISTENCE_CLOSE_FAILED:" + ";".join(repr(x) for x in errors)
            )


def _evidence_adapter_guard(authority: CanonicalWriteAuthorityR11, operation: str) -> None:
    capability = (
        Capability.SEAL_TRAINING_SNAPSHOT
        if ".seal_snapshot" in operation
        else Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE
    )
    authority.assert_mutation(capability, mutation=operation)



def _journal_adapter_guard(authority: CanonicalWriteAuthorityR11, operation: str) -> None:
    authority.assert_mutation(
        Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL, mutation=operation
    )
    if ".append_once:NEXT_GENERATION_RELEASED:" in operation:
        authority.assert_mutation(
            Capability.RELEASE_GENERATION, mutation=operation + ":release_generation"
        )

def _checkpoint_adapter_guard(authority: CanonicalWriteAuthorityR11, operation: str) -> None:
    capability = (
        Capability.TRANSITION_CHALLENGER_CHAMPION
        if ".atomic_commit" in operation
        else Capability.SEAL_CHECKPOINT
    )
    authority.assert_mutation(capability, mutation=operation)


def open_canonical_persistence_r11(
    authority: CanonicalWriteAuthorityR11,
    *,
    evidence_segment_target_bytes: int = 8 * 1024 * 1024,
    evidence_codec: str = "zlib",
) -> CanonicalPersistenceBundleR11:
    """Open the INTB canonical writer bundle after exact S4F verification.

    Existing persistence semantics and protocol types are reused. No legacy root is
    auto-adopted; frozen/raw market authority remains external and read-only.
    """

    authority.verify_roots()
    roots = authority.roots

    evidence_metadata = roots.control_path("hot_indexes/evidence_store")
    evidence_payload = roots.data_path("evidence_payloads/r11_evidence_store/lane0")
    journal_root = roots.control_path("authoritative_journal_metadata/event_journal")
    checkpoint_control = roots.control_path("checkpoint_metadata/checkpoint_store")
    checkpoint_objects = roots.data_path("cold_content_artifacts/checkpoint_objects")

    evidence_backend: Stage4FencedEvidenceStoreR11 | None = None
    journal_backend: Stage4FencedEventJournalR11 | None = None
    checkpoint_backend: Stage4RoutedCheckpointStoreR11 | None = None
    try:
        evidence_backend = Stage4FencedEvidenceStoreR11(
            metadata_root=evidence_metadata,
            payload_roots=(evidence_payload,),
            segment_target_bytes=int(evidence_segment_target_bytes),
            codec=evidence_codec,
            read_only_source_roots=(roots.frozen_raw_root,),
            authority=authority,
        )
        evidence_inner = EvidenceStoreProtocolAdapterR11(
            evidence_backend,
            mutation_guard=lambda operation: _evidence_adapter_guard(authority, operation),
        )

        journal_backend = Stage4FencedEventJournalR11(
            journal_root, authority=authority
        )
        journal_inner = EventJournalProtocolAdapterR11(
            journal_backend,
            mutation_guard=lambda operation: _journal_adapter_guard(authority, operation),
        )

        checkpoint_backend = Stage4RoutedCheckpointStoreR11(
            control_root=checkpoint_control,
            object_data_root=checkpoint_objects,
            authority=authority,
        )
        checkpoint_inner = CheckpointStoreProtocolAdapterR11(
            checkpoint_backend,
            mutation_guard=lambda operation: _checkpoint_adapter_guard(authority, operation),
        )

        return CanonicalPersistenceBundleR11(
            evidence_store=evidence_inner,
            journal=journal_inner,
            checkpoint_store=checkpoint_inner,
            evidence_payloads=evidence_backend,
            checkpoint_objects=checkpoint_backend,
            _journal_backend=journal_backend,
        )
    except BaseException:
        for backend in (checkpoint_backend, journal_backend, evidence_backend):
            if backend is not None:
                try:
                    backend.close()
                except BaseException:
                    pass
        raise


__all__ = [
    "SCIENTIFIC_STATUS",
    "CanonicalPersistenceBundleR11",
    "CanonicalWriteAuthorityR11",
    "Stage4CanonicalPersistenceError",
    "Stage4CanonicalRoleRequired",
    "Stage4FencedEvidenceStoreR11",
    "Stage4FencedEventJournalR11",
    "Stage4PersistenceRootBindingError",
    "Stage4RoutedCheckpointStoreR11",
    "open_canonical_persistence_r11",
]
