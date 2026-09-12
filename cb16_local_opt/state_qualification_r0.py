from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping, Sequence

from .state_sufficiency_r0 import REPRESENTATION_GAP, StateSufficiencyReportR0


STATE_QUALIFICATION_SCHEMA_R0 = "CB16_R11_BC_STATE_QUALIFICATION_V1_R0"
STATE_QUALIFICATION_PASS = "PASS"
STATE_QUALIFICATION_COMPONENT_FAILURE = "COMPONENT_FAILURE"
STATE_QUALIFICATION_REPRESENTATION_GAP = "REPRESENTATION_GAP"
STATE_QUALIFICATION_VERDICTS_R0 = (
    STATE_QUALIFICATION_PASS,
    STATE_QUALIFICATION_COMPONENT_FAILURE,
    STATE_QUALIFICATION_REPRESENTATION_GAP,
)
REQUIRED_PHASE_C_TASKS_R0 = tuple(f"BC-{index:03d}" for index in range(21, 33))
COMPONENT_PASS = "PASS"
COMPONENT_FAIL = "FAIL"
COMPONENT_STATUSES_R0 = (COMPONENT_PASS, COMPONENT_FAIL)


def _canonical_json(payload: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ACQUAL_R0_NONCANONICAL_PAYLOAD") from exc


def _sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, code: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(code)
    return value


def _state_sufficiency_sha256(report: StateSufficiencyReportR0) -> str:
    report.validate()
    return _sha256(
        {
            "schema_version": report.schema_version,
            "case_count": report.case_count,
            "pair_count": report.pair_count,
            "assessments": [
                {
                    "left_case_id": item.left_case_id,
                    "right_case_id": item.right_case_id,
                    "same_policy_observation": item.same_policy_observation,
                    "same_optimal_action": item.same_optimal_action,
                    "classification": item.classification,
                }
                for item in report.assessments
            ],
        }
    )


@dataclass(frozen=True)
class ComponentQualificationEvidenceR0:
    task_id: str
    status: str
    evidence_sha256: str
    evidence_kind: str

    def validate(self) -> None:
        if self.task_id not in REQUIRED_PHASE_C_TASKS_R0:
            raise RuntimeError("ACQUAL_R0_TASK_ID_INVALID")
        if self.status not in COMPONENT_STATUSES_R0:
            raise RuntimeError("ACQUAL_R0_COMPONENT_STATUS_INVALID")
        _require_sha256(self.evidence_sha256, code="ACQUAL_R0_EVIDENCE_HASH_INVALID")
        if not isinstance(self.evidence_kind, str) or not self.evidence_kind.strip():
            raise RuntimeError("ACQUAL_R0_EVIDENCE_KIND_INVALID")


@dataclass(frozen=True)
class StateQualificationResultR0:
    schema_version: str
    verdict: str
    eligible_for_phase_d: bool
    required_tasks: tuple[str, ...]
    component_evidence: tuple[ComponentQualificationEvidenceR0, ...]
    state_sufficiency_sha256: str
    representation_gap_count: int
    unresolved_representation_gap_pairs: tuple[tuple[str, str], ...]
    blocking_reasons: tuple[str, ...]

    def validate(self) -> None:
        if self.schema_version != STATE_QUALIFICATION_SCHEMA_R0:
            raise RuntimeError("ACQUAL_R0_SCHEMA_MISMATCH")
        if self.verdict not in STATE_QUALIFICATION_VERDICTS_R0:
            raise RuntimeError("ACQUAL_R0_VERDICT_INVALID")
        if self.required_tasks != REQUIRED_PHASE_C_TASKS_R0:
            raise RuntimeError("ACQUAL_R0_REQUIRED_TASK_SET_MISMATCH")
        for evidence in self.component_evidence:
            evidence.validate()
        tasks = tuple(item.task_id for item in self.component_evidence)
        if tasks != REQUIRED_PHASE_C_TASKS_R0:
            raise RuntimeError("ACQUAL_R0_COMPONENT_ORDER_OR_COVERAGE_MISMATCH")
        _require_sha256(
            self.state_sufficiency_sha256,
            code="ACQUAL_R0_STATE_SUFFICIENCY_HASH_INVALID",
        )
        if isinstance(self.representation_gap_count, bool) or self.representation_gap_count < 0:
            raise RuntimeError("ACQUAL_R0_REPRESENTATION_GAP_COUNT_INVALID")
        if self.representation_gap_count != len(self.unresolved_representation_gap_pairs):
            raise RuntimeError("ACQUAL_R0_REPRESENTATION_GAP_PAIR_COUNT_MISMATCH")
        component_failed = any(item.status != COMPONENT_PASS for item in self.component_evidence)
        gap_present = self.representation_gap_count > 0
        expected_verdict = (
            STATE_QUALIFICATION_REPRESENTATION_GAP
            if gap_present
            else STATE_QUALIFICATION_COMPONENT_FAILURE
            if component_failed
            else STATE_QUALIFICATION_PASS
        )
        if self.verdict != expected_verdict:
            raise RuntimeError("ACQUAL_R0_VERDICT_INCONSISTENT")
        if self.eligible_for_phase_d != (expected_verdict == STATE_QUALIFICATION_PASS):
            raise RuntimeError("ACQUAL_R0_PHASE_D_ELIGIBILITY_INCONSISTENT")
        if expected_verdict == STATE_QUALIFICATION_PASS and self.blocking_reasons:
            raise RuntimeError("ACQUAL_R0_PASS_HAS_BLOCKING_REASON")
        if expected_verdict != STATE_QUALIFICATION_PASS and not self.blocking_reasons:
            raise RuntimeError("ACQUAL_R0_FAILURE_MISSING_BLOCKING_REASON")

    def machine_readable(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "verdict": self.verdict,
            "eligible_for_phase_d": self.eligible_for_phase_d,
            "required_tasks": list(self.required_tasks),
            "component_evidence": [
                {
                    "task_id": item.task_id,
                    "status": item.status,
                    "evidence_sha256": item.evidence_sha256,
                    "evidence_kind": item.evidence_kind,
                }
                for item in self.component_evidence
            ],
            "state_sufficiency_sha256": self.state_sufficiency_sha256,
            "representation_gap_count": self.representation_gap_count,
            "unresolved_representation_gap_pairs": [list(pair) for pair in self.unresolved_representation_gap_pairs],
            "blocking_reasons": list(self.blocking_reasons),
        }
        payload["qualification_sha256"] = _sha256(payload)
        return payload


def compile_state_qualification_r0(
    *,
    component_evidence: Sequence[ComponentQualificationEvidenceR0],
    state_sufficiency_report: StateSufficiencyReportR0,
) -> StateQualificationResultR0:
    """Compile Phase-C observation/state qualification without rescue semantics.

    Model size, parameter count, optimizer settings and training duration are
    intentionally absent from this API.  A known unresolved representation gap
    is a state-definition failure and therefore blocks Phase D regardless of
    model capacity or optimization effort.
    """

    evidence = tuple(component_evidence)
    for item in evidence:
        item.validate()
    tasks = tuple(item.task_id for item in evidence)
    if len(set(tasks)) != len(tasks):
        raise RuntimeError("ACQUAL_R0_DUPLICATE_TASK_EVIDENCE")
    if set(tasks) != set(REQUIRED_PHASE_C_TASKS_R0):
        missing = sorted(set(REQUIRED_PHASE_C_TASKS_R0) - set(tasks))
        extra = sorted(set(tasks) - set(REQUIRED_PHASE_C_TASKS_R0))
        raise RuntimeError(
            "ACQUAL_R0_TASK_EVIDENCE_COVERAGE_INVALID:"
            + ",".join(missing)
            + ":"
            + ",".join(extra)
        )
    by_task = {item.task_id: item for item in evidence}
    ordered = tuple(by_task[task_id] for task_id in REQUIRED_PHASE_C_TASKS_R0)

    state_sufficiency_report.validate()
    gap_pairs = tuple(
        (item.left_case_id, item.right_case_id)
        for item in state_sufficiency_report.assessments
        if item.classification == REPRESENTATION_GAP
    )
    component_failures = tuple(
        item.task_id for item in ordered if item.status != COMPONENT_PASS
    )

    blocking: list[str] = []
    if gap_pairs:
        blocking.append("UNRESOLVED_REPRESENTATION_GAP")
    if component_failures:
        blocking.append("COMPONENT_FAILURE:" + ",".join(component_failures))

    verdict = (
        STATE_QUALIFICATION_REPRESENTATION_GAP
        if gap_pairs
        else STATE_QUALIFICATION_COMPONENT_FAILURE
        if component_failures
        else STATE_QUALIFICATION_PASS
    )
    result = StateQualificationResultR0(
        schema_version=STATE_QUALIFICATION_SCHEMA_R0,
        verdict=verdict,
        eligible_for_phase_d=verdict == STATE_QUALIFICATION_PASS,
        required_tasks=REQUIRED_PHASE_C_TASKS_R0,
        component_evidence=ordered,
        state_sufficiency_sha256=_state_sufficiency_sha256(state_sufficiency_report),
        representation_gap_count=len(gap_pairs),
        unresolved_representation_gap_pairs=gap_pairs,
        blocking_reasons=tuple(blocking),
    )
    result.validate()
    return result
