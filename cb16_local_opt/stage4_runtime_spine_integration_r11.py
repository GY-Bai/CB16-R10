from __future__ import annotations

"""Concrete Stage-4 Integration Task INTA process-authority spine.

This module is integration-only. It adapts already-qualified S4C/S4D/S4F/S4G
contracts to the S4B CanonicalRuntimeControllerR11 provider protocols. Recovery
and orchestration remain injected boundaries owned by later integration tasks.

A retirement-policy token is never positive authority. An S4B opaque lease
identity is never sufficient by itself: every current-authority assertion is
resolved back to the live S4D FencingTokenR11 and calls S4D validation.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import secrets
from threading import RLock

from .stage4_authority_adoption_r11 import (
    AuthorityAdoptionContract,
    load_adoption_receipt,
)
from .stage4_authority_lease_r11 import FencingTokenR11, Stage4AuthorityLeaseR11
from .stage4_canonical_runtime_r11 import (
    AuthorityBinding,
    AuthorityLease,
    CanonicalRuntimeControllerR11,
    OrchestrationProvider,
    RecoveryProvider,
)
from .stage4_legacy_retirement_r11 import (
    Capability,
    LegacyRetirementGuard as S4GLegacyRetirementGuard,
    RetirementCapabilityToken,
    RuntimeIdentity,
    RuntimeRole,
)
from .stage4_state_roots_r11 import (
    SEMANTIC_FREEZE_BLOB_SHA,
    Stage4StartupReceiptR11,
    Stage4StateRootsR11,
)


SCIENTIFIC_STATUS = (
    "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
)
CANONICAL_RUNTIME_CAPABILITY = Capability.ACQUIRE_CANONICAL_RUNTIME_AUTHORITY


class RuntimeSpineIntegrationError(RuntimeError):
    """Fail-closed INTA adapter/configuration error."""


class LeaseAcquisitionMode(str, Enum):
    ORDINARY = "ORDINARY"
    EXPLICIT_DEAD_OWNER_RECOVERY = "EXPLICIT_DEAD_OWNER_RECOVERY"


@dataclass(frozen=True)
class _LiveLeaseRecord:
    lease: AuthorityLease
    binding: AuthorityBinding
    fencing_token: FencingTokenR11


class Stage4AuthorityAdoptionProviderR11:
    """Adapt one verified S4C adoption receipt into an S4B AuthorityBinding."""

    def __init__(
        self,
        *,
        contract: AuthorityAdoptionContract,
        receipt_path: str | Path,
    ) -> None:
        self._contract = contract
        self._receipt_path = Path(receipt_path)

    @property
    def contract(self) -> AuthorityAdoptionContract:
        return self._contract

    def adopt_or_verify_existing_authority(self) -> AuthorityBinding:
        # S4C validates every accepted source identity, target identity, generation
        # equality, semantic guards and canonical content hash.
        receipt = load_adoption_receipt(self._receipt_path, contract=self._contract)
        source = self._contract.accepted_source
        target = self._contract.target_r11_authority_identity

        # Keep exact identities explicit at this boundary rather than trusting only
        # that the receipt was parseable.
        if receipt["source"]["source_sha"] != source.source_sha:
            raise RuntimeSpineIntegrationError("INTA_ADOPTION_SOURCE_SHA_MISMATCH")
        if receipt["source"]["semantic_freeze_identity"] != source.semantic_freeze_identity:
            raise RuntimeSpineIntegrationError("INTA_ADOPTION_SEMANTIC_FREEZE_MISMATCH")
        if int(receipt["source"]["source_generation"]) != int(source.source_generation):
            raise RuntimeSpineIntegrationError("INTA_ADOPTION_SOURCE_GENERATION_MISMATCH")
        if receipt["target"]["r11_authority_identity"] != target:
            raise RuntimeSpineIntegrationError("INTA_ADOPTION_TARGET_IDENTITY_MISMATCH")
        if int(receipt["target"]["adoption_generation"]) != int(source.source_generation):
            raise RuntimeSpineIntegrationError("INTA_ADOPTION_GENERATION_REWRITE_REFUSED")

        return AuthorityBinding(
            authority_id=target,
            generation=int(source.source_generation),
            semantic_freeze_id=source.semantic_freeze_identity,
        )


class Stage4StateRootProviderR11:
    """Bind S4F startup verification to the exact canonical root identities."""

    def __init__(
        self,
        *,
        state_roots: Stage4StateRootsR11,
        expected_control_root_id: str,
        expected_data_root_id: str,
        expected_frozen_raw_identity: str,
        expected_authority_id: str,
        expected_generation: int,
    ) -> None:
        self._state_roots = state_roots
        self._expected_control_root_id = str(expected_control_root_id)
        self._expected_data_root_id = str(expected_data_root_id)
        self._expected_frozen_raw_identity = str(expected_frozen_raw_identity)
        self._expected_authority_id = str(expected_authority_id)
        self._expected_generation = int(expected_generation)
        self._last_verified: Stage4StartupReceiptR11 | None = None

    @property
    def last_verified(self) -> Stage4StartupReceiptR11 | None:
        return self._last_verified

    def verify_state_roots(self, binding: AuthorityBinding) -> None:
        if binding.authority_id != self._expected_authority_id:
            raise RuntimeSpineIntegrationError("INTA_ROOT_BINDING_AUTHORITY_MISMATCH")
        if binding.generation != self._expected_generation:
            raise RuntimeSpineIntegrationError("INTA_ROOT_BINDING_GENERATION_MISMATCH")
        if binding.semantic_freeze_id != SEMANTIC_FREEZE_BLOB_SHA:
            raise RuntimeSpineIntegrationError("INTA_ROOT_BINDING_SEMANTIC_FREEZE_MISMATCH")

        verified = self._state_roots.verify_startup(
            expected_control_id=self._expected_control_root_id,
            expected_data_id=self._expected_data_root_id,
        )
        if verified.frozen_raw_identity != self._expected_frozen_raw_identity:
            raise RuntimeSpineIntegrationError("INTA_FROZEN_RAW_IDENTITY_MISMATCH")
        self._last_verified = verified


class Stage4LegacyRetirementProviderR11:
    """Use S4G solely as a negative canonical-eligibility guard."""

    def __init__(
        self,
        *,
        guard: S4GLegacyRetirementGuard,
        identity: RuntimeIdentity,
        token: RetirementCapabilityToken,
    ) -> None:
        self._guard = guard
        self._identity = identity
        self._token = token

    def assert_canonical_only(self, binding: AuthorityBinding) -> None:
        if self._identity.subject_id != binding.authority_id:
            raise RuntimeSpineIntegrationError("INTA_S4G_SUBJECT_AUTHORITY_MISMATCH")
        if self._identity.role != RuntimeRole.R11_CANONICAL_RUNTIME.value:
            raise RuntimeSpineIntegrationError("INTA_NONCANONICAL_RUNTIME_ROLE_REFUSED")

        decision = self._guard.assert_token(
            self._identity,
            CANONICAL_RUNTIME_CAPABILITY,
            self._token,
        )
        if not decision.eligible_under_legacy_retirement_policy:
            raise RuntimeSpineIntegrationError("INTA_S4G_CANONICAL_ELIGIBILITY_REFUSED")
        # This is a critical negative assertion: S4G must never become positive
        # runtime authority merely because it admitted the canonical capability.
        if decision.authority_granted is not False:
            raise RuntimeSpineIntegrationError("INTA_S4G_TOKEN_MUST_NOT_GRANT_AUTHORITY")
        if decision.requires_independent_runtime_authority is not True:
            raise RuntimeSpineIntegrationError(
                "INTA_S4G_MUST_REQUIRE_INDEPENDENT_RUNTIME_AUTHORITY"
            )


class Stage4AuthorityLeaseProviderR11:
    """Adapt S4D live fencing into S4B's opaque AuthorityLease protocol.

    The S4B token string is a handle into this provider's process-local binding.
    The actual FencingTokenR11 object is retained and is always revalidated by
    Stage4AuthorityLeaseR11.assert_fencing_token before authoritative use.
    """

    def __init__(
        self,
        *,
        lease: Stage4AuthorityLeaseR11,
        expected_authority_id: str,
        expected_generation: int,
        acquisition_mode: LeaseAcquisitionMode = LeaseAcquisitionMode.ORDINARY,
    ) -> None:
        self._lease = lease
        self._expected_authority_id = str(expected_authority_id)
        self._expected_generation = int(expected_generation)
        self._acquisition_mode = LeaseAcquisitionMode(acquisition_mode)
        self._records: dict[str, _LiveLeaseRecord] = {}
        self._lock = RLock()

    @property
    def acquisition_mode(self) -> LeaseAcquisitionMode:
        return self._acquisition_mode

    def _assert_binding(self, binding: AuthorityBinding) -> None:
        if binding.authority_id != self._expected_authority_id:
            raise RuntimeSpineIntegrationError("INTA_LEASE_AUTHORITY_MISMATCH")
        if binding.generation != self._expected_generation:
            raise RuntimeSpineIntegrationError("INTA_LEASE_GENERATION_MISMATCH")

    def acquire_authority(self, binding: AuthorityBinding) -> AuthorityLease:
        self._assert_binding(binding)
        with self._lock:
            if self._records:
                raise RuntimeSpineIntegrationError("INTA_PROVIDER_ALREADY_HAS_LIVE_LEASE")
            if self._acquisition_mode is LeaseAcquisitionMode.ORDINARY:
                token = self._lease.acquire_authority()
            else:
                # This mode is intentionally explicit. INTA does not decide that a
                # dead-owner recovery is legitimate; INTE/final consolidation must
                # choose this provider mode only after its qualified recovery checks.
                token = self._lease.recover_after_dead_owner()

            try:
                opaque = "INTA-S4D-" + secrets.token_hex(24)
                result = AuthorityLease(
                    authority_id=binding.authority_id,
                    fencing_token=opaque,
                )
                self._records[opaque] = _LiveLeaseRecord(result, binding, token)
                return result
            except BaseException:
                # If adapter construction fails after S4D acquisition, do not leak
                # live authority.
                self._lease.release_authority(token)
                raise

    def _record_for(self, lease: AuthorityLease) -> _LiveLeaseRecord:
        if not isinstance(lease, AuthorityLease):
            raise RuntimeSpineIntegrationError("INTA_LEASE_TYPE_INVALID")
        if lease.authority_id != self._expected_authority_id:
            raise RuntimeSpineIntegrationError("INTA_LEASE_HANDLE_AUTHORITY_MISMATCH")
        with self._lock:
            record = self._records.get(lease.fencing_token)
        if record is None or record.lease != lease:
            raise RuntimeSpineIntegrationError("INTA_OPAQUE_LEASE_NOT_LIVE_IN_PROVIDER")
        return record

    def assert_current(self, lease: AuthorityLease) -> None:
        record = self._record_for(lease)
        self._assert_binding(record.binding)
        # Mandatory S4D live-fence validation. A PID, timestamp, copied string or
        # persisted epoch without the current kernel-lock-owning S4D handle is
        # insufficient.
        self._lease.assert_fencing_token(record.fencing_token)

    def release_authority(self, lease: AuthorityLease) -> None:
        record = self._record_for(lease)
        self._lease.release_authority(record.fencing_token)
        with self._lock:
            self._records.pop(lease.fencing_token, None)


@dataclass(frozen=True)
class CanonicalRuntimeAuthoritySpineR11:
    """Concrete process-authority providers; recovery/orchestration stay injected."""

    authority_adoption: Stage4AuthorityAdoptionProviderR11
    state_roots: Stage4StateRootProviderR11
    legacy_guard: Stage4LegacyRetirementProviderR11
    authority_lease: Stage4AuthorityLeaseProviderR11

    def build_controller(
        self,
        *,
        recovery: RecoveryProvider,
        orchestration: OrchestrationProvider,
    ) -> CanonicalRuntimeControllerR11:
        return CanonicalRuntimeControllerR11(
            authority_adoption=self.authority_adoption,
            state_roots=self.state_roots,
            legacy_guard=self.legacy_guard,
            recovery=recovery,
            authority_lease=self.authority_lease,
            orchestration=orchestration,
        )
