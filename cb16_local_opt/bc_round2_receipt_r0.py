from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Iterable, Mapping

from cb16_local_opt.evidence_level_r0 import (
    EvidenceLevel,
    EvidenceReceiptR0,
    make_evidence_receipt_r0,
    parse_evidence_level,
)

RECEIPT_SCHEMA_R0 = "CB16_R11_BC_ROUND2_RECEIPT_V1_R0"
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA64_RE = re.compile(r"^[0-9a-f]{64}$")
_TASK_RE = re.compile(r"^BC-[0-9]{3}$")


class ResultClassification(str, Enum):
    PASS = "PASS"
    SCIENTIFIC_FAIL = "SCIENTIFIC_FAIL"
    EXECUTION_BLOCKED = "EXECUTION_BLOCKED"
    HARDWARE_LIMIT = "HARDWARE_LIMIT"
    UNRESOLVED_OWNER_DECISION = "UNRESOLVED_OWNER_DECISION"


def _canon(values: Iterable[object], code: str) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(code)
        out.append(value)
    if len(set(out)) != len(out):
        raise RuntimeError(f"{code}_DUPLICATE")
    return tuple(sorted(out))


def _result(value: object) -> ResultClassification:
    if isinstance(value, ResultClassification):
        return value
    try:
        return ResultClassification(str(value))
    except ValueError as exc:
        raise RuntimeError("BCRECEIPT_R0_RESULT_CLASSIFICATION_INVALID") from exc


@dataclass(frozen=True)
class DependencyReceiptR0:
    task_id: str
    result_classification: ResultClassification
    receipt_sha256: str

    def validate(self) -> None:
        if not _TASK_RE.fullmatch(self.task_id):
            raise RuntimeError("BCRECEIPT_R0_DEPENDENCY_TASK_ID_INVALID")
        if not _SHA64_RE.fullmatch(self.receipt_sha256):
            raise RuntimeError("BCRECEIPT_R0_DEPENDENCY_RECEIPT_SHA_INVALID")
        _result(self.result_classification)


@dataclass(frozen=True)
class BCRound2ReceiptR0:
    schema_version: str
    task_id: str
    code_sha: str
    science_hash: str
    dependencies: tuple[DependencyReceiptR0, ...]
    required_checks: tuple[str, ...]
    completed_checks: tuple[str, ...]
    evidence: EvidenceReceiptR0
    evidence_claim_level: EvidenceLevel
    final_holdout_untouched: bool
    fresh_data_download: bool
    result_classification: ResultClassification

    def validate(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA_R0:
            raise RuntimeError("BCRECEIPT_R0_SCHEMA_MISMATCH")
        if not _TASK_RE.fullmatch(self.task_id):
            raise RuntimeError("BCRECEIPT_R0_TASK_ID_INVALID")
        if not _SHA40_RE.fullmatch(self.code_sha):
            raise RuntimeError("BCRECEIPT_R0_CODE_SHA_INVALID")
        if not _SHA64_RE.fullmatch(self.science_hash):
            raise RuntimeError("BCRECEIPT_R0_SCIENCE_HASH_INVALID")

        dep_ids = tuple(dep.task_id for dep in self.dependencies)
        if dep_ids != tuple(sorted(dep_ids)) or len(set(dep_ids)) != len(dep_ids):
            raise RuntimeError("BCRECEIPT_R0_DEPENDENCIES_NONCANONICAL")
        for dep in self.dependencies:
            dep.validate()

        if not self.required_checks:
            raise RuntimeError("BCRECEIPT_R0_REQUIRED_CHECKS_MISSING")
        if self.required_checks != tuple(sorted(self.required_checks)):
            raise RuntimeError("BCRECEIPT_R0_REQUIRED_CHECKS_NONCANONICAL")
        if len(set(self.required_checks)) != len(self.required_checks):
            raise RuntimeError("BCRECEIPT_R0_REQUIRED_CHECKS_DUPLICATE")
        if self.completed_checks != tuple(sorted(self.completed_checks)):
            raise RuntimeError("BCRECEIPT_R0_COMPLETED_CHECKS_NONCANONICAL")
        if len(set(self.completed_checks)) != len(self.completed_checks):
            raise RuntimeError("BCRECEIPT_R0_COMPLETED_CHECKS_DUPLICATE")
        if not set(self.completed_checks).issubset(self.required_checks):
            raise RuntimeError("BCRECEIPT_R0_UNKNOWN_COMPLETED_CHECK")

        self.evidence.validate()
        self.evidence.serialize_claim(parse_evidence_level(self.evidence_claim_level))
        _result(self.result_classification)

        if self.result_classification is ResultClassification.PASS:
            if set(self.completed_checks) != set(self.required_checks):
                raise RuntimeError("BCRECEIPT_R0_PASS_WITH_SKIPPED_REQUIRED_CHECK")
            if any(d.result_classification is not ResultClassification.PASS for d in self.dependencies):
                raise RuntimeError("BCRECEIPT_R0_PASS_WITH_NONPASS_DEPENDENCY")
            if not self.final_holdout_untouched:
                raise RuntimeError("BCRECEIPT_R0_PASS_WITH_FINAL_HOLDOUT_TOUCHED")
            if self.fresh_data_download:
                raise RuntimeError("BCRECEIPT_R0_PASS_WITH_FRESH_DATA_DOWNLOAD")

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "code_sha": self.code_sha,
            "science_hash": self.science_hash,
            "dependencies": [
                {"task_id": d.task_id, "result_classification": d.result_classification.value, "receipt_sha256": d.receipt_sha256}
                for d in self.dependencies
            ],
            "required_checks": list(self.required_checks),
            "completed_checks": list(self.completed_checks),
            "evidence": self.evidence.serialize_claim(self.evidence_claim_level),
            "final_holdout_untouched": self.final_holdout_untouched,
            "fresh_data_download": self.fresh_data_download,
            "result_classification": self.result_classification.value,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def make_dependency_receipt_r0(*, task_id: str, result_classification: object, receipt_sha256: str) -> DependencyReceiptR0:
    dep = DependencyReceiptR0(task_id, _result(result_classification), receipt_sha256)
    dep.validate()
    return dep


def compile_bc_round2_receipt_r0(
    *,
    task_id: str,
    code_sha: str,
    science_hash: str,
    required_dependency_ids: Iterable[str],
    dependency_receipts: Mapping[str, DependencyReceiptR0],
    required_checks: Iterable[object],
    completed_checks: Iterable[object],
    maximum_justified_evidence_level: object,
    evidence_claim_level: object,
    source_artifacts: Iterable[object],
    final_holdout_untouched: bool,
    fresh_data_download: bool,
    result_classification: object,
) -> BCRound2ReceiptR0:
    required_ids = _canon(required_dependency_ids, "BCRECEIPT_R0_REQUIRED_DEPENDENCY_INVALID")
    if any(not _TASK_RE.fullmatch(x) for x in required_ids):
        raise RuntimeError("BCRECEIPT_R0_REQUIRED_DEPENDENCY_INVALID")
    if set(dependency_receipts) - set(required_ids):
        raise RuntimeError("BCRECEIPT_R0_UNKNOWN_DEPENDENCY")
    if set(required_ids) - set(dependency_receipts):
        raise RuntimeError("BCRECEIPT_R0_MISSING_DEPENDENCY")
    dependencies = tuple(dependency_receipts[x] for x in required_ids)
    if any(dep.task_id != expected for expected, dep in zip(required_ids, dependencies)):
        raise RuntimeError("BCRECEIPT_R0_DEPENDENCY_ID_MISMATCH")

    required = _canon(tuple(required_checks), "BCRECEIPT_R0_REQUIRED_CHECK_INVALID")
    completed_values = tuple(completed_checks)
    completed = _canon(completed_values, "BCRECEIPT_R0_COMPLETED_CHECK_INVALID") if completed_values else ()
    evidence = make_evidence_receipt_r0(
        maximum_justified_level=maximum_justified_evidence_level,
        source_artifacts=source_artifacts,
    )
    receipt = BCRound2ReceiptR0(
        RECEIPT_SCHEMA_R0,
        task_id,
        code_sha,
        science_hash,
        dependencies,
        required,
        completed,
        evidence,
        parse_evidence_level(evidence_claim_level),
        final_holdout_untouched,
        fresh_data_download,
        _result(result_classification),
    )
    receipt.validate()
    return receipt
