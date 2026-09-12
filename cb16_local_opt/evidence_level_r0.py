from __future__ import annotations

"""Fail-closed evidence-level contract for CB16 R11 BC Round 2.

Evidence levels are ordered claims, not interchangeable labels. A receipt records
the strongest level actually justified by its source artifacts and may only be
serialized at that level or below. This module deliberately does not infer a
stronger claim from workflow success, component tests, or naming conventions.
"""

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from typing import Iterable, Mapping


EVIDENCE_SCHEMA_R0 = "CB16_R11_BC_EVIDENCE_LEVEL_V1_R0"


class EvidenceLevel(IntEnum):
    CONTRACT = 1
    COMPONENT = 2
    CLOSED_LOOP = 3
    KNOWN_ANSWER = 4
    ECONOMIC = 5
    TRANSFER = 6


EVIDENCE_LEVEL_NAMES = tuple(level.name for level in EvidenceLevel)


def parse_evidence_level(value: object) -> EvidenceLevel:
    if isinstance(value, EvidenceLevel):
        return value
    if not isinstance(value, str) or value not in EVIDENCE_LEVEL_NAMES:
        raise RuntimeError("BCEVIDENCE_R0_LEVEL_INVALID")
    return EvidenceLevel[value]


def _canonical_artifacts(values: Iterable[object]) -> tuple[str, ...]:
    artifacts: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError("BCEVIDENCE_R0_SOURCE_ARTIFACT_INVALID")
        artifacts.append(value)
    if not artifacts:
        raise RuntimeError("BCEVIDENCE_R0_SOURCE_ARTIFACTS_MISSING")
    if len(set(artifacts)) != len(artifacts):
        raise RuntimeError("BCEVIDENCE_R0_SOURCE_ARTIFACT_DUPLICATE")
    return tuple(sorted(artifacts))


@dataclass(frozen=True)
class EvidenceReceiptR0:
    schema_version: str
    maximum_justified_level: EvidenceLevel
    source_artifacts: tuple[str, ...]

    def validate(self) -> None:
        if self.schema_version != EVIDENCE_SCHEMA_R0:
            raise RuntimeError("BCEVIDENCE_R0_SCHEMA_MISMATCH")
        parse_evidence_level(self.maximum_justified_level)
        canonical = _canonical_artifacts(self.source_artifacts)
        if self.source_artifacts != canonical:
            raise RuntimeError("BCEVIDENCE_R0_SOURCE_ARTIFACTS_NONCANONICAL")

    def allows(self, claimed_level: object) -> bool:
        self.validate()
        claimed = parse_evidence_level(claimed_level)
        return claimed <= self.maximum_justified_level

    def serialize_claim(self, claimed_level: object) -> dict[str, object]:
        self.validate()
        claimed = parse_evidence_level(claimed_level)
        if claimed > self.maximum_justified_level:
            raise RuntimeError("BCEVIDENCE_R0_CLAIM_EXCEEDS_JUSTIFIED_LEVEL")
        return {
            "schema_version": self.schema_version,
            "claimed_level": claimed.name,
            "maximum_justified_level": self.maximum_justified_level.name,
            "source_artifacts": list(self.source_artifacts),
        }

    def canonical_json(self, claimed_level: object) -> str:
        return json.dumps(
            self.serialize_claim(claimed_level),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def claim_sha256(self, claimed_level: object) -> str:
        return hashlib.sha256(self.canonical_json(claimed_level).encode("utf-8")).hexdigest()


def make_evidence_receipt_r0(
    *,
    maximum_justified_level: object,
    source_artifacts: Iterable[object],
) -> EvidenceReceiptR0:
    receipt = EvidenceReceiptR0(
        schema_version=EVIDENCE_SCHEMA_R0,
        maximum_justified_level=parse_evidence_level(maximum_justified_level),
        source_artifacts=_canonical_artifacts(source_artifacts),
    )
    receipt.validate()
    return receipt


def validate_serialized_evidence_claim_r0(payload: Mapping[str, object]) -> EvidenceReceiptR0:
    if set(payload) != {
        "schema_version",
        "claimed_level",
        "maximum_justified_level",
        "source_artifacts",
    }:
        raise RuntimeError("BCEVIDENCE_R0_PAYLOAD_FIELDS_INVALID")
    if payload.get("schema_version") != EVIDENCE_SCHEMA_R0:
        raise RuntimeError("BCEVIDENCE_R0_SCHEMA_MISMATCH")
    sources = payload.get("source_artifacts")
    if not isinstance(sources, list):
        raise RuntimeError("BCEVIDENCE_R0_SOURCE_ARTIFACTS_INVALID")
    receipt = make_evidence_receipt_r0(
        maximum_justified_level=payload.get("maximum_justified_level"),
        source_artifacts=sources,
    )
    claimed = parse_evidence_level(payload.get("claimed_level"))
    if claimed > receipt.maximum_justified_level:
        raise RuntimeError("BCEVIDENCE_R0_CLAIM_EXCEEDS_JUSTIFIED_LEVEL")
    if payload != receipt.serialize_claim(claimed):
        raise RuntimeError("BCEVIDENCE_R0_PAYLOAD_NONCANONICAL")
    return receipt
