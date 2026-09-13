"""Exact named-identity binding checks for CB16 qualification.

The stage adapter is responsible for obtaining authoritative expected and
observed identities. This module only checks exact named bindings; it does not
infer Git ancestry, authorization validity, semantic equivalence, or whether a
stage supplied every identity that its own contract requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class QualificationIdentityError(ValueError):
    pass


def _require_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise QualificationIdentityError(f"{field_name.upper()}_NOT_STRING")
    if not value.strip():
        raise QualificationIdentityError(f"EMPTY_{field_name.upper()}")
    return value


@dataclass(frozen=True)
class ExactIdentityBindingV1:
    identity_id: str
    expected: str
    observed: str
    expected_source_ref: str
    observed_source_ref: str
    required: bool = True

    def validate(self) -> "ExactIdentityBindingV1":
        for field_name in (
            "identity_id",
            "expected",
            "observed",
            "expected_source_ref",
            "observed_source_ref",
        ):
            _require_nonempty_string(getattr(self, field_name), field_name)
        if type(self.required) is not bool:
            raise QualificationIdentityError("REQUIRED_FLAG_NOT_BOOL")
        return self


def audit_exact_identity_bindings_v1(
    bindings: Sequence[ExactIdentityBindingV1 | Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not isinstance(bindings, (list, tuple)):
        raise QualificationIdentityError("IDENTITY_BINDINGS_NOT_SEQUENCE")

    normalized: list[ExactIdentityBindingV1] = []
    for item in bindings:
        if isinstance(item, ExactIdentityBindingV1):
            normalized.append(item.validate())
        elif isinstance(item, Mapping):
            normalized.append(ExactIdentityBindingV1(**dict(item)).validate())
        else:
            raise QualificationIdentityError("IDENTITY_BINDING_NOT_MAPPING")

    ids = [item.identity_id for item in normalized]
    if len(set(ids)) != len(ids):
        raise QualificationIdentityError("DUPLICATE_IDENTITY_BINDING")

    matched: list[str] = []
    optional_mismatches: list[str] = []
    violations: list[str] = []
    for item in normalized:
        if item.observed == item.expected:
            matched.append(item.identity_id)
        elif item.required:
            violations.append(f"EXACT_IDENTITY_MISMATCH:{item.identity_id}")
        else:
            optional_mismatches.append(item.identity_id)

    return {
        "schema": "CB16_QUALIFICATION_EXACT_IDENTITY_AUDIT_V1",
        "binding_count": len(normalized),
        "required_binding_count": sum(1 for item in normalized if item.required),
        "matched_identity_ids": sorted(matched),
        "optional_mismatch_identity_ids": sorted(optional_mismatches),
        "contract_violations": sorted(violations),
        "all_required_exact_bindings_match": not violations,
    }
