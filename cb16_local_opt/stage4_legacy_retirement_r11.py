from __future__ import annotations

"""Stage-4 legacy-retirement policy for CB16 R11.

This module is deliberately a *negative* authority guard.  It can prove that a
legacy/reference/replay/diagnostic/test identity is ineligible for canonical
writer capabilities.  A token issued to an R11 canonical identity is only a
retirement-policy admission token; it is never sufficient to acquire runtime
authority, satisfy a lease/fence, grant Permission, or perform a Physics/account
transition.

SEMANTIC CONTRACTS ARE AUTHORITY.  LEGACY PYTHON IMPLEMENTATION IS NOT AUTHORITY.
"""

import hashlib
import hmac
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping


SCHEMA = "CB16_R11_STAGE4_LEGACY_RETIREMENT_POLICY_V1"
TOKEN_SCHEMA = "CB16_R11_STAGE4_RETIREMENT_ADMISSION_TOKEN_V1"
SCIENTIFIC_STATUS = (
    "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
)


class LegacyRetirementError(RuntimeError):
    """Base class for fail-closed S4G policy failures."""


class UnknownRuntimeRole(LegacyRetirementError):
    pass


class UnknownCapability(LegacyRetirementError):
    pass


class InvalidRuntimeIdentity(LegacyRetirementError):
    pass


class InvalidRetirementToken(LegacyRetirementError):
    pass


class LegacyAuthorityDenied(LegacyRetirementError):
    pass


class CapabilityDenied(LegacyRetirementError):
    pass


class LegacyFallbackDenied(LegacyRetirementError):
    pass


class ActiveLegacyWriterDetected(LegacyRetirementError):
    pass


class RuntimeRole(str, Enum):
    R11_CANONICAL_RUNTIME = "R11_CANONICAL_RUNTIME"
    LEGACY_REFERENCE = "LEGACY_REFERENCE"
    LEGACY_ORACLE = "LEGACY_ORACLE"
    LEGACY_REPLAY = "LEGACY_REPLAY"
    LEGACY_COMPATIBILITY_READER = "LEGACY_COMPATIBILITY_READER"
    DIAGNOSTIC = "DIAGNOSTIC"
    TEST = "TEST"


class Capability(str, Enum):
    # Canonical authoritative-write capabilities.  S4G may deny these but does
    # not itself establish the positive authority needed to exercise them.
    ACQUIRE_CANONICAL_RUNTIME_AUTHORITY = "ACQUIRE_CANONICAL_RUNTIME_AUTHORITY"
    MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE = "MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE"
    APPEND_AUTHORITATIVE_EVENT_JOURNAL = "APPEND_AUTHORITATIVE_EVENT_JOURNAL"
    SEAL_TRAINING_SNAPSHOT = "SEAL_TRAINING_SNAPSHOT"
    TRANSITION_CHALLENGER_CHAMPION = "TRANSITION_CHALLENGER_CHAMPION"
    SEAL_CHECKPOINT = "SEAL_CHECKPOINT"
    RELEASE_GENERATION = "RELEASE_GENERATION"
    GRANT_PERMISSION = "GRANT_PERMISSION"
    EXECUTE_ACCOUNT_PHYSICS_TRANSITION = "EXECUTE_ACCOUNT_PHYSICS_TRANSITION"

    # Non-authoritative capabilities intentionally left available to legacy
    # code where appropriate.
    READ_LEGACY_REFERENCE = "READ_LEGACY_REFERENCE"
    COMPARE_LEGACY_ORACLE = "COMPARE_LEGACY_ORACLE"
    ENGINEERING_REPLAY = "ENGINEERING_REPLAY"
    READ_HISTORICAL_COMPATIBILITY = "READ_HISTORICAL_COMPATIBILITY"
    RUN_DIAGNOSTICS = "RUN_DIAGNOSTICS"


AUTHORITATIVE_CAPABILITIES = frozenset(
    {
        Capability.ACQUIRE_CANONICAL_RUNTIME_AUTHORITY,
        Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE,
        Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL,
        Capability.SEAL_TRAINING_SNAPSHOT,
        Capability.TRANSITION_CHALLENGER_CHAMPION,
        Capability.SEAL_CHECKPOINT,
        Capability.RELEASE_GENERATION,
        Capability.GRANT_PERMISSION,
        Capability.EXECUTE_ACCOUNT_PHYSICS_TRANSITION,
    }
)

NON_AUTHORITATIVE_ROLES = frozenset(
    {
        RuntimeRole.LEGACY_REFERENCE,
        RuntimeRole.LEGACY_ORACLE,
        RuntimeRole.LEGACY_REPLAY,
        RuntimeRole.LEGACY_COMPATIBILITY_READER,
        RuntimeRole.DIAGNOSTIC,
        RuntimeRole.TEST,
    }
)

_SAFE_CAPABILITIES = frozenset(set(Capability) - set(AUTHORITATIVE_CAPABILITIES))

_ALLOWED_BY_ROLE: Mapping[RuntimeRole, frozenset[Capability]] = {
    RuntimeRole.R11_CANONICAL_RUNTIME: frozenset(set(Capability)),
    RuntimeRole.LEGACY_REFERENCE: frozenset({Capability.READ_LEGACY_REFERENCE}),
    RuntimeRole.LEGACY_ORACLE: frozenset(
        {Capability.READ_LEGACY_REFERENCE, Capability.COMPARE_LEGACY_ORACLE}
    ),
    RuntimeRole.LEGACY_REPLAY: frozenset(
        {Capability.READ_LEGACY_REFERENCE, Capability.ENGINEERING_REPLAY}
    ),
    RuntimeRole.LEGACY_COMPATIBILITY_READER: frozenset(
        {
            Capability.READ_LEGACY_REFERENCE,
            Capability.READ_HISTORICAL_COMPATIBILITY,
        }
    ),
    RuntimeRole.DIAGNOSTIC: frozenset(
        {Capability.READ_LEGACY_REFERENCE, Capability.RUN_DIAGNOSTICS}
    ),
    RuntimeRole.TEST: frozenset(
        {
            Capability.READ_LEGACY_REFERENCE,
            Capability.ENGINEERING_REPLAY,
            Capability.RUN_DIAGNOSTICS,
        }
    ),
}


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _parse_role(value: RuntimeRole | str) -> RuntimeRole:
    try:
        return value if isinstance(value, RuntimeRole) else RuntimeRole(str(value))
    except ValueError as exc:
        raise UnknownRuntimeRole(f"S4G_UNKNOWN_RUNTIME_ROLE:{value}") from exc


def _parse_capability(value: Capability | str) -> Capability:
    try:
        return value if isinstance(value, Capability) else Capability(str(value))
    except ValueError as exc:
        raise UnknownCapability(f"S4G_UNKNOWN_CAPABILITY:{value}") from exc


def is_authoritative_capability(value: Capability | str) -> bool:
    return _parse_capability(value) in AUTHORITATIVE_CAPABILITIES


def is_non_authoritative_role(value: RuntimeRole | str) -> bool:
    return _parse_role(value) in NON_AUTHORITATIVE_ROLES


@dataclass(frozen=True)
class RuntimeIdentity:
    subject_id: str
    role: str
    issuer_id: str
    signature: str


@dataclass(frozen=True)
class RetirementCapabilityToken:
    schema: str
    subject_id: str
    role: str
    capability: str
    issuer_id: str
    identity_signature: str
    admission_only: bool
    authority_granted: bool
    signature: str


@dataclass(frozen=True)
class RetirementDecision:
    subject_id: str
    role: str
    capability: str
    eligible_under_legacy_retirement_policy: bool
    authority_granted: bool
    requires_independent_runtime_authority: bool


@dataclass(frozen=True)
class ActiveCapabilityClaim:
    identity: RuntimeIdentity
    capability: str
    token: RetirementCapabilityToken | None
    active: bool = True


@dataclass(frozen=True)
class DeclaredCapabilityClaim:
    subject_id: str
    role: str
    capability: str
    active: bool = True


class LegacyRetirementGuard:
    """Fail-closed retirement eligibility guard.

    The HMAC is an integrity binding for identities/tokens supplied to this
    policy.  It is not a distributed identity system and it is not a lease or
    fencing primitive.  Final integration must keep the key in canonical
    control and combine this guard with the independent positive authority
    mechanisms required by Stage-4.
    """

    def __init__(self, signing_key: bytes, *, issuer_id: str = SCHEMA):
        if not isinstance(signing_key, (bytes, bytearray)) or len(signing_key) < 32:
            raise ValueError("S4G_SIGNING_KEY_MUST_BE_AT_LEAST_32_BYTES")
        if not str(issuer_id).strip():
            raise ValueError("S4G_ISSUER_ID_REQUIRED")
        self._key = bytes(signing_key)
        self.issuer_id = str(issuer_id)

    def _mac(self, body: Mapping[str, Any]) -> str:
        return hmac.new(self._key, _canonical_json_bytes(body), hashlib.sha256).hexdigest()

    @staticmethod
    def _subject(subject_id: str) -> str:
        subject = str(subject_id)
        if not subject or subject != subject.strip():
            raise InvalidRuntimeIdentity("S4G_INVALID_SUBJECT_ID")
        return subject

    def _identity_body(self, subject_id: str, role: RuntimeRole) -> dict[str, str]:
        return {
            "schema": "CB16_R11_STAGE4_RUNTIME_IDENTITY_V1",
            "subject_id": self._subject(subject_id),
            "role": role.value,
            "issuer_id": self.issuer_id,
        }

    def issue_identity(
        self, subject_id: str, role: RuntimeRole | str
    ) -> RuntimeIdentity:
        parsed = _parse_role(role)
        body = self._identity_body(subject_id, parsed)
        return RuntimeIdentity(
            subject_id=body["subject_id"],
            role=parsed.value,
            issuer_id=self.issuer_id,
            signature=self._mac(body),
        )

    def validate_identity(self, identity: RuntimeIdentity) -> RuntimeRole:
        if not isinstance(identity, RuntimeIdentity):
            raise InvalidRuntimeIdentity("S4G_IDENTITY_TYPE_INVALID")
        role = _parse_role(identity.role)
        if identity.issuer_id != self.issuer_id:
            raise InvalidRuntimeIdentity("S4G_IDENTITY_ISSUER_MISMATCH")
        body = self._identity_body(identity.subject_id, role)
        expected = self._mac(body)
        if not hmac.compare_digest(expected, str(identity.signature)):
            raise InvalidRuntimeIdentity("S4G_IDENTITY_SIGNATURE_INVALID")
        return role

    @staticmethod
    def _require_allowed(role: RuntimeRole, capability: Capability) -> None:
        if capability in AUTHORITATIVE_CAPABILITIES and role in NON_AUTHORITATIVE_ROLES:
            raise LegacyAuthorityDenied(
                f"S4G_NONCANONICAL_AUTHORITATIVE_CAPABILITY_DENIED:{role.value}:{capability.value}"
            )
        if capability not in _ALLOWED_BY_ROLE[role]:
            raise CapabilityDenied(
                f"S4G_ROLE_CAPABILITY_DENIED:{role.value}:{capability.value}"
            )

    def _token_body(
        self,
        identity: RuntimeIdentity,
        role: RuntimeRole,
        capability: Capability,
    ) -> dict[str, Any]:
        return {
            "schema": TOKEN_SCHEMA,
            "subject_id": identity.subject_id,
            "role": role.value,
            "capability": capability.value,
            "issuer_id": self.issuer_id,
            "identity_signature": identity.signature,
            "admission_only": True,
            "authority_granted": False,
        }

    def issue_token(
        self,
        identity: RuntimeIdentity,
        capability: Capability | str,
    ) -> RetirementCapabilityToken:
        role = self.validate_identity(identity)
        cap = _parse_capability(capability)
        self._require_allowed(role, cap)
        body = self._token_body(identity, role, cap)
        return RetirementCapabilityToken(**body, signature=self._mac(body))

    def assert_token(
        self,
        identity: RuntimeIdentity,
        capability: Capability | str,
        token: RetirementCapabilityToken,
    ) -> RetirementDecision:
        role = self.validate_identity(identity)
        cap = _parse_capability(capability)
        self._require_allowed(role, cap)
        if not isinstance(token, RetirementCapabilityToken):
            raise InvalidRetirementToken("S4G_TOKEN_TYPE_INVALID")
        body = self._token_body(identity, role, cap)
        for field, expected in body.items():
            if getattr(token, field, None) != expected:
                raise InvalidRetirementToken(f"S4G_TOKEN_BINDING_MISMATCH:{field}")
        if not hmac.compare_digest(self._mac(body), str(token.signature)):
            raise InvalidRetirementToken("S4G_TOKEN_SIGNATURE_INVALID")
        return RetirementDecision(
            subject_id=identity.subject_id,
            role=role.value,
            capability=cap.value,
            eligible_under_legacy_retirement_policy=True,
            authority_granted=False,
            requires_independent_runtime_authority=cap in AUTHORITATIVE_CAPABILITIES,
        )

    def require_canonical_startup(
        self,
        *,
        canonical_startup_succeeded: bool,
        proposed_fallback: RuntimeIdentity | None = None,
    ) -> str:
        if not canonical_startup_succeeded:
            raise LegacyFallbackDenied("S4G_R11_STARTUP_FAILED_NO_LEGACY_AUTHORITY_FALLBACK")
        if proposed_fallback is not None:
            role = self.validate_identity(proposed_fallback)
            if role is not RuntimeRole.R11_CANONICAL_RUNTIME:
                raise LegacyFallbackDenied(
                    f"S4G_LEGACY_FALLBACK_FORBIDDEN:{role.value}"
                )
        return "R11_CANONICAL_STARTUP_CONTINUES_WITH_INDEPENDENT_AUTHORITY_GATES"

    def audit_active_claims(
        self, claims: Iterable[ActiveCapabilityClaim]
    ) -> dict[str, Any]:
        active_count = 0
        canonical_authoritative = 0
        safe_count = 0
        inactive_count = 0
        for claim in claims:
            if not isinstance(claim, ActiveCapabilityClaim):
                raise LegacyRetirementError("S4G_ACTIVE_CLAIM_TYPE_INVALID")
            role = self.validate_identity(claim.identity)
            cap = _parse_capability(claim.capability)
            if not claim.active:
                inactive_count += 1
                continue
            active_count += 1
            if cap in AUTHORITATIVE_CAPABILITIES and role in NON_AUTHORITATIVE_ROLES:
                raise ActiveLegacyWriterDetected(
                    f"S4G_ACTIVE_LEGACY_WRITER_DETECTED:{claim.identity.subject_id}:{role.value}:{cap.value}"
                )
            if claim.token is None:
                raise InvalidRetirementToken(
                    f"S4G_ACTIVE_CLAIM_MISSING_TOKEN:{claim.identity.subject_id}:{cap.value}"
                )
            decision = self.assert_token(claim.identity, cap, claim.token)
            if decision.authority_granted:
                raise LegacyRetirementError("S4G_RETIREMENT_TOKEN_MUST_NEVER_GRANT_AUTHORITY")
            if cap in AUTHORITATIVE_CAPABILITIES:
                canonical_authoritative += 1
            else:
                safe_count += 1
        return {
            "schema": "CB16_R11_STAGE4_ACTIVE_CAPABILITY_AUDIT_V1",
            "status": "PASS",
            "active_claim_count": active_count,
            "inactive_claim_count": inactive_count,
            "canonical_authoritative_claim_count": canonical_authoritative,
            "non_authoritative_claim_count": safe_count,
            "active_legacy_authoritative_writer_count": 0,
            "canonical_claims_require_independent_runtime_authority": True,
        }


def audit_declared_claims(
    claims: Iterable[DeclaredCapabilityClaim],
) -> dict[str, Any]:
    """Static fail-closed audit for discovered claims that do not carry tokens.

    This function never converts a declaration into authority.  It only rejects
    unknown declarations and active non-canonical authoritative capabilities.
    Canonical declarations are reported as requiring independent runtime proof.
    """

    active = 0
    inactive = 0
    canonical_authoritative = 0
    safe = 0
    for claim in claims:
        if not isinstance(claim, DeclaredCapabilityClaim):
            raise LegacyRetirementError("S4G_DECLARED_CLAIM_TYPE_INVALID")
        role = _parse_role(claim.role)
        cap = _parse_capability(claim.capability)
        if not str(claim.subject_id).strip():
            raise InvalidRuntimeIdentity("S4G_DECLARED_SUBJECT_ID_REQUIRED")
        if not claim.active:
            inactive += 1
            continue
        active += 1
        if cap in AUTHORITATIVE_CAPABILITIES and role in NON_AUTHORITATIVE_ROLES:
            raise ActiveLegacyWriterDetected(
                f"S4G_ACTIVE_LEGACY_WRITER_DETECTED:{claim.subject_id}:{role.value}:{cap.value}"
            )
        if cap in AUTHORITATIVE_CAPABILITIES:
            canonical_authoritative += 1
        else:
            if cap not in _ALLOWED_BY_ROLE[role]:
                raise CapabilityDenied(
                    f"S4G_DECLARED_ROLE_CAPABILITY_DENIED:{role.value}:{cap.value}"
                )
            safe += 1
    return {
        "schema": "CB16_R11_STAGE4_DECLARED_CAPABILITY_AUDIT_V1",
        "status": "PASS",
        "active_claim_count": active,
        "inactive_claim_count": inactive,
        "canonical_authoritative_claim_count": canonical_authoritative,
        "non_authoritative_claim_count": safe,
        "active_legacy_authoritative_writer_count": 0,
        "declarations_never_establish_authority": True,
    }


def policy_manifest() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "ACTIVE_FAIL_CLOSED_POLICY",
        "scientific_status": SCIENTIFIC_STATUS,
        "authoritative_capabilities": sorted(x.value for x in AUTHORITATIVE_CAPABILITIES),
        "authoritative_capability_count": len(AUTHORITATIVE_CAPABILITIES),
        "non_authoritative_roles": sorted(x.value for x in NON_AUTHORITATIVE_ROLES),
        "safe_capabilities": sorted(x.value for x in _SAFE_CAPABILITIES),
        "allowed_by_role": {
            role.value: sorted(cap.value for cap in caps)
            for role, caps in sorted(_ALLOWED_BY_ROLE.items(), key=lambda item: item[0].value)
        },
        "retirement_token_is_positive_runtime_authority": False,
        "legacy_python_is_runtime_authority": False,
        "legacy_oracle_comparison_allowed": True,
        "legacy_replay_is_engineering_only": True,
        "legacy_replay_may_mint_new_evidence": False,
        "legacy_fallback_on_r11_startup_failure": False,
        "scientific_semantics_changed": False,
        "new_scientific_verdict": False,
        "final_holdout_accessed": False,
        "fresh_market_data_downloaded": False,
        "historical_market_data_mutated": False,
    }


def self_audit_policy() -> dict[str, Any]:
    """Deterministic unit-style matrix audit with synthetic identities only."""

    guard = LegacyRetirementGuard(b"S4G_SELF_AUDIT_ONLY_NOT_RUNTIME_AUTHORITY!!")
    denied = 0
    for role in sorted(NON_AUTHORITATIVE_ROLES, key=lambda x: x.value):
        identity = guard.issue_identity(f"self-audit-{role.value.lower()}", role)
        for cap in sorted(AUTHORITATIVE_CAPABILITIES, key=lambda x: x.value):
            try:
                guard.issue_token(identity, cap)
            except LegacyAuthorityDenied:
                denied += 1
            else:
                raise LegacyRetirementError(
                    f"S4G_SELF_AUDIT_ESCALATION_NOT_REJECTED:{role.value}:{cap.value}"
                )

    canonical = guard.issue_identity("self-audit-r11", RuntimeRole.R11_CANONICAL_RUNTIME)
    canonical_token = guard.issue_token(
        canonical, Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL
    )
    decision = guard.assert_token(
        canonical, Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL, canonical_token
    )
    if decision.authority_granted or not decision.requires_independent_runtime_authority:
        raise LegacyRetirementError("S4G_CANONICAL_ADMISSION_TOKEN_BECAME_AUTHORITY")

    expected_denied = len(NON_AUTHORITATIVE_ROLES) * len(AUTHORITATIVE_CAPABILITIES)
    if denied != expected_denied:
        raise LegacyRetirementError(
            f"S4G_SELF_AUDIT_DENIAL_COUNT_MISMATCH:{denied}:{expected_denied}"
        )

    return {
        **policy_manifest(),
        "self_audit": {
            "status": "PASS",
            "noncanonical_authoritative_escalations_tested": expected_denied,
            "noncanonical_authoritative_escalations_denied": denied,
            "canonical_admission_token_authority_granted": False,
        },
    }
