from __future__ import annotations

"""Stage-4 adoption/restart/recovery identity integration for CB16 R11.

INTE binds already-qualified S4C adoption, S4D fencing recovery, S4F canonical
roots, and a read-only Stage-3 recovery inspection boundary.  It does not start
workers, open authoritative writers, train, run tournaments, grant Permission,
or reinterpret replay as scientific Evidence.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from .stage4_authority_adoption_r11 import (
    AuthorityAdoptionContract,
    SourceAuthorityIdentity,
    load_adoption_receipt,
)
from .stage4_authority_lease_r11 import Stage4AuthorityLeaseR11
from .stage4_canonical_runtime_r11 import AuthorityBinding, RecoveredRuntimeState
from .stage4_state_roots_r11 import Stage4StateRootsR11


SCHEMA = "CB16_R11_STAGE4_RECOVERY_INTEGRATION_V1"
SCIENTIFIC_STATUS = (
    "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
)


class Stage4RecoveryIntegrationError(RuntimeError):
    """Base fail-closed INTE error."""


class RecoveryAuthorityMismatch(Stage4RecoveryIntegrationError):
    """Recovered or supplied identity does not equal accepted authority."""


class RecoveryStateIncomplete(Stage4RecoveryIntegrationError):
    """A recovery candidate is not fully sealed and authoritative."""


class RecoverySemanticViolation(Stage4RecoveryIntegrationError):
    """Recovery attempted to manufacture scientific meaning or Evidence."""


class ExplicitDeadOwnerRecoveryRequired(Stage4RecoveryIntegrationError):
    """Durable ACTIVE residue cannot be silently converted to new ownership."""


class DeadOwnerRecoveryMode(str, Enum):
    ORDINARY_RESTART = "ORDINARY_RESTART"
    EXPLICIT_DEAD_OWNER_RECOVERY = "EXPLICIT_DEAD_OWNER_RECOVERY"


@dataclass(frozen=True)
class CanonicalRootExpectationR11:
    """Independently accepted target S4F root identities."""

    control_root_id: str
    data_root_id: str
    frozen_raw_identity: str

    def validate(self) -> None:
        for field in ("control_root_id", "data_root_id", "frozen_raw_identity"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise RecoveryAuthorityMismatch(f"STAGE4_RECOVERY_{field.upper()}_MISSING")


@dataclass(frozen=True)
class Stage3RecoveryObservationR11:
    """Read-only observation emitted by the qualified Stage-3 recovery boundary.

    ``observed_source`` is compared as a complete S4C SourceAuthorityIdentity.  The
    booleans are negative-authority gates: they may reject recovery but never grant
    authority on their own.
    """

    observed_source: SourceAuthorityIdentity
    fully_sealed: bool
    journal_transition_complete: bool
    checkpoint_transition_complete: bool
    journal_audit_passed: bool
    checkpoint_audit_passed: bool
    replay_engineering_only: bool = True
    new_evidence_created: bool = False
    scientific_history_rewritten: bool = False
    new_scientific_verdict: bool = False


@runtime_checkable
class Stage3RecoveryInspectorR11(Protocol):
    """Read-only bridge to already-qualified Stage-3 recovery identities.

    Final consolidation must bind this protocol to the canonical journal/checkpoint
    readers supplied by the persistence integration.  INTE intentionally owns no
    writer opening or writer mutation path.
    """

    def inspect_recovery_state(
        self, accepted_source: SourceAuthorityIdentity
    ) -> Stage3RecoveryObservationR11: ...


class Stage4RecoveryProviderR11:
    """Concrete S4B RecoveryProvider with exact identity and fencing recovery gates."""

    def __init__(
        self,
        *,
        adoption_contract: AuthorityAdoptionContract,
        adoption_receipt_path: str | Path,
        state_roots: Stage4StateRootsR11,
        root_expectation: CanonicalRootExpectationR11,
        stage3_recovery: Stage3RecoveryInspectorR11,
        lease_root: str | Path,
        recovery_owner_id: str,
        dead_owner_mode: DeadOwnerRecoveryMode = DeadOwnerRecoveryMode.ORDINARY_RESTART,
    ) -> None:
        adoption_contract.validate()
        root_expectation.validate()
        if not isinstance(stage3_recovery, Stage3RecoveryInspectorR11):
            raise TypeError("STAGE4_RECOVERY_STAGE3_INSPECTOR_PROTOCOL_REQUIRED")
        if not isinstance(recovery_owner_id, str) or not recovery_owner_id.strip():
            raise ValueError("STAGE4_RECOVERY_OWNER_ID_REQUIRED")
        if not isinstance(dead_owner_mode, DeadOwnerRecoveryMode):
            raise ValueError("STAGE4_RECOVERY_DEAD_OWNER_MODE_INVALID")

        self._contract = adoption_contract
        self._receipt_path = Path(adoption_receipt_path)
        self._roots = state_roots
        self._root_expectation = root_expectation
        self._stage3 = stage3_recovery
        self._lease_root = Path(lease_root)
        self._owner_id = recovery_owner_id.strip()
        self._mode = dead_owner_mode
        self._assert_control_namespace_placement()

    def _assert_control_namespace_placement(self) -> None:
        expected_adoption_parent = self._roots.control_path("adoption_control_metadata")
        expected_lease = self._roots.control_path("runtime_lease_fencing_state")
        if self._receipt_path.resolve(strict=False).parent != expected_adoption_parent.resolve(
            strict=False
        ):
            raise RecoveryAuthorityMismatch(
                "STAGE4_RECOVERY_ADOPTION_RECEIPT_OUTSIDE_CANONICAL_CONTROL_NAMESPACE"
            )
        if self._lease_root.resolve(strict=False) != expected_lease.resolve(strict=False):
            raise RecoveryAuthorityMismatch(
                "STAGE4_RECOVERY_LEASE_OUTSIDE_CANONICAL_CONTROL_NAMESPACE"
            )

    def _verify_binding(self, binding: AuthorityBinding) -> None:
        if not isinstance(binding, AuthorityBinding):
            raise RecoveryAuthorityMismatch("STAGE4_RECOVERY_AUTHORITY_BINDING_TYPE_INVALID")
        source = self._contract.accepted_source
        if binding.authority_id != self._contract.target_r11_authority_identity:
            raise RecoveryAuthorityMismatch("STAGE4_RECOVERY_TARGET_AUTHORITY_IDENTITY_MISMATCH")
        if int(binding.generation) != int(source.source_generation):
            raise RecoveryAuthorityMismatch("STAGE4_RECOVERY_BINDING_GENERATION_MISMATCH")
        if binding.semantic_freeze_id != source.semantic_freeze_identity:
            raise RecoveryAuthorityMismatch("STAGE4_RECOVERY_SEMANTIC_FREEZE_MISMATCH")

    def _verify_adoption_and_roots(self, binding: AuthorityBinding) -> None:
        receipt = load_adoption_receipt(self._receipt_path, contract=self._contract)
        if int(receipt["target"]["adoption_generation"]) != int(binding.generation):
            raise RecoveryAuthorityMismatch("STAGE4_RECOVERY_ADOPTION_GENERATION_MISMATCH")

        startup = self._roots.verify_startup(
            expected_control_id=self._root_expectation.control_root_id,
            expected_data_id=self._root_expectation.data_root_id,
        )
        if startup.frozen_raw_identity != self._root_expectation.frozen_raw_identity:
            raise RecoveryAuthorityMismatch("STAGE4_RECOVERY_FROZEN_RAW_IDENTITY_MISMATCH")

    def _verify_stage3_observation(self) -> Stage3RecoveryObservationR11:
        accepted = self._contract.accepted_source
        observation = self._stage3.inspect_recovery_state(accepted)
        if not isinstance(observation, Stage3RecoveryObservationR11):
            raise RecoveryStateIncomplete("STAGE4_RECOVERY_STAGE3_OBSERVATION_TYPE_INVALID")
        try:
            observation.observed_source.validate()
        except Exception as exc:
            raise RecoveryAuthorityMismatch("STAGE4_RECOVERY_OBSERVED_SOURCE_INVALID") from exc
        if observation.observed_source != accepted:
            expected = accepted.__dict__
            observed = observation.observed_source.__dict__
            fields = sorted(k for k in expected if expected[k] != observed[k])
            raise RecoveryAuthorityMismatch(
                "STAGE4_RECOVERY_ACCEPTED_SOURCE_MISMATCH:" + ",".join(fields)
            )
        if not observation.fully_sealed:
            raise RecoveryStateIncomplete("STAGE4_RECOVERY_UNSEALED_STATE_REFUSED")
        if not observation.journal_transition_complete:
            raise RecoveryStateIncomplete("STAGE4_RECOVERY_JOURNAL_TRANSITION_INCOMPLETE")
        if not observation.checkpoint_transition_complete:
            raise RecoveryStateIncomplete("STAGE4_RECOVERY_CHECKPOINT_TRANSITION_INCOMPLETE")
        if not observation.journal_audit_passed:
            raise RecoveryStateIncomplete("STAGE4_RECOVERY_JOURNAL_AUDIT_FAILED")
        if not observation.checkpoint_audit_passed:
            raise RecoveryStateIncomplete("STAGE4_RECOVERY_CHECKPOINT_AUDIT_FAILED")
        if observation.replay_engineering_only is not True:
            raise RecoverySemanticViolation("STAGE4_RECOVERY_REPLAY_EVIDENCE_REINTERPRETATION_REFUSED")
        if observation.new_evidence_created:
            raise RecoverySemanticViolation("STAGE4_RECOVERY_NEW_EVIDENCE_REFUSED")
        if observation.scientific_history_rewritten:
            raise RecoverySemanticViolation("STAGE4_RECOVERY_HISTORY_REWRITE_REFUSED")
        if observation.new_scientific_verdict:
            raise RecoverySemanticViolation("STAGE4_RECOVERY_NEW_SCIENTIFIC_VERDICT_REFUSED")
        return observation

    def _settle_dead_owner_residue(self) -> None:
        lease = Stage4AuthorityLeaseR11(self._lease_root, owner_id=self._owner_id)
        before = lease.inspect_current_authority()
        if before.state != "ACTIVE":
            return
        if self._mode is not DeadOwnerRecoveryMode.EXPLICIT_DEAD_OWNER_RECOVERY:
            raise ExplicitDeadOwnerRecoveryRequired(
                "STAGE4_RECOVERY_EXPLICIT_DEAD_OWNER_RECOVERY_REQUIRED"
            )

        token = lease.recover_after_dead_owner()
        try:
            lease.assert_fencing_token(token)
            during = lease.inspect_current_authority()
            if during.state != "ACTIVE" or during.epoch <= before.epoch:
                raise RecoveryAuthorityMismatch(
                    "STAGE4_RECOVERY_FENCING_EPOCH_DID_NOT_ADVANCE"
                )
        finally:
            try:
                lease.release_authority(token)
            except Exception:
                raise

        after = lease.inspect_current_authority()
        if after.state != "RELEASED" or after.epoch <= before.epoch:
            raise RecoveryAuthorityMismatch("STAGE4_RECOVERY_FENCE_SETTLEMENT_INVALID")

    def recover_existing_state(self, binding: AuthorityBinding) -> RecoveredRuntimeState:
        """Verify and reconstruct exactly the already-accepted authority state."""

        self._verify_binding(binding)
        self._verify_adoption_and_roots(binding)
        self._verify_stage3_observation()
        self._settle_dead_owner_residue()

        accepted = self._contract.accepted_source
        return RecoveredRuntimeState(
            authority_id=binding.authority_id,
            generation=int(accepted.source_generation),
            state_id=accepted.checkpoint_identity,
            scientific_history_id=accepted.journal_head_identity,
        )


__all__ = [
    "SCHEMA",
    "SCIENTIFIC_STATUS",
    "CanonicalRootExpectationR11",
    "DeadOwnerRecoveryMode",
    "ExplicitDeadOwnerRecoveryRequired",
    "RecoveryAuthorityMismatch",
    "RecoverySemanticViolation",
    "RecoveryStateIncomplete",
    "Stage3RecoveryInspectorR11",
    "Stage3RecoveryObservationR11",
    "Stage4RecoveryIntegrationError",
    "Stage4RecoveryProviderR11",
]
