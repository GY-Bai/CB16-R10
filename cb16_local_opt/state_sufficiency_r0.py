from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from itertools import combinations
from typing import Mapping, Sequence


STATE_SUFFICIENCY_SCHEMA_R0 = "CB16_R11_BC_STATE_SUFFICIENCY_V1_R0"
OBSERVATION_DISTINGUISHABLE = "OBSERVATION_DISTINGUISHABLE"
ALIAS_ACTION_CONSISTENT = "ALIAS_ACTION_CONSISTENT"
REPRESENTATION_GAP = "REPRESENTATION_GAP"
STATE_SUFFICIENCY_CLASSIFICATIONS_R0 = (
    OBSERVATION_DISTINGUISHABLE,
    ALIAS_ACTION_CONSISTENT,
    REPRESENTATION_GAP,
)


def _require_text(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _require_sha256(value: object, *, code: str) -> str:
    text = _require_text(value, code=code)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise RuntimeError(code)
    return text


def canonical_policy_observation_json_r0(payload: Mapping[str, object]) -> str:
    if not isinstance(payload, Mapping) or not payload:
        raise RuntimeError("ACSUFF_R0_OBSERVATION_PAYLOAD_INVALID")
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ACSUFF_R0_OBSERVATION_PAYLOAD_NONCANONICAL") from exc


def policy_observation_sha256_r0(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_policy_observation_json_r0(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KnownAnswerStateCaseR0:
    """One preregistered causal state and its analytically known optimal action."""

    case_id: str
    authoritative_state_sha256: str
    policy_observation_sha256: str
    optimal_action_key: str

    def validate(self) -> None:
        _require_text(self.case_id, code="ACSUFF_R0_CASE_ID_INVALID")
        _require_sha256(
            self.authoritative_state_sha256,
            code="ACSUFF_R0_AUTHORITATIVE_STATE_HASH_INVALID",
        )
        _require_sha256(
            self.policy_observation_sha256,
            code="ACSUFF_R0_POLICY_OBSERVATION_HASH_INVALID",
        )
        _require_text(self.optimal_action_key, code="ACSUFF_R0_OPTIMAL_ACTION_KEY_INVALID")


@dataclass(frozen=True)
class StateAliasAssessmentR0:
    schema_version: str
    left_case_id: str
    right_case_id: str
    same_policy_observation: bool
    same_optimal_action: bool
    classification: str

    def validate(self) -> None:
        if self.schema_version != STATE_SUFFICIENCY_SCHEMA_R0:
            raise RuntimeError("ACSUFF_R0_SCHEMA_MISMATCH")
        _require_text(self.left_case_id, code="ACSUFF_R0_LEFT_CASE_INVALID")
        _require_text(self.right_case_id, code="ACSUFF_R0_RIGHT_CASE_INVALID")
        if self.left_case_id == self.right_case_id:
            raise RuntimeError("ACSUFF_R0_PAIR_CASE_DUPLICATE")
        if self.classification not in STATE_SUFFICIENCY_CLASSIFICATIONS_R0:
            raise RuntimeError("ACSUFF_R0_CLASSIFICATION_INVALID")
        expected = (
            REPRESENTATION_GAP
            if self.same_policy_observation and not self.same_optimal_action
            else ALIAS_ACTION_CONSISTENT
            if self.same_policy_observation
            else OBSERVATION_DISTINGUISHABLE
        )
        if self.classification != expected:
            raise RuntimeError("ACSUFF_R0_CLASSIFICATION_INCONSISTENT")


@dataclass(frozen=True)
class StateSufficiencyReportR0:
    schema_version: str
    case_count: int
    pair_count: int
    assessments: tuple[StateAliasAssessmentR0, ...]

    def validate(self) -> None:
        if self.schema_version != STATE_SUFFICIENCY_SCHEMA_R0:
            raise RuntimeError("ACSUFF_R0_REPORT_SCHEMA_MISMATCH")
        if self.case_count < 2:
            raise RuntimeError("ACSUFF_R0_CASE_COUNT_INSUFFICIENT")
        expected_pairs = self.case_count * (self.case_count - 1) // 2
        if self.pair_count != expected_pairs or len(self.assessments) != expected_pairs:
            raise RuntimeError("ACSUFF_R0_PAIR_COUNT_MISMATCH")
        for assessment in self.assessments:
            assessment.validate()

    @property
    def representation_gap_count(self) -> int:
        return sum(a.classification == REPRESENTATION_GAP for a in self.assessments)

    @property
    def has_representation_gap(self) -> bool:
        return self.representation_gap_count > 0


def make_known_answer_state_case_r0(
    *,
    case_id: str,
    authoritative_state_sha256: str,
    policy_observation_payload: Mapping[str, object],
    optimal_action_key: str,
) -> KnownAnswerStateCaseR0:
    case = KnownAnswerStateCaseR0(
        case_id=_require_text(case_id, code="ACSUFF_R0_CASE_ID_INVALID"),
        authoritative_state_sha256=_require_sha256(
            authoritative_state_sha256,
            code="ACSUFF_R0_AUTHORITATIVE_STATE_HASH_INVALID",
        ),
        policy_observation_sha256=policy_observation_sha256_r0(policy_observation_payload),
        optimal_action_key=_require_text(
            optimal_action_key,
            code="ACSUFF_R0_OPTIMAL_ACTION_KEY_INVALID",
        ),
    )
    case.validate()
    return case


def assess_state_pair_r0(
    left: KnownAnswerStateCaseR0,
    right: KnownAnswerStateCaseR0,
) -> StateAliasAssessmentR0:
    left.validate()
    right.validate()
    if left.case_id == right.case_id:
        raise RuntimeError("ACSUFF_R0_PAIR_CASE_DUPLICATE")
    if (
        left.authoritative_state_sha256 == right.authoritative_state_sha256
        and left.optimal_action_key != right.optimal_action_key
    ):
        raise RuntimeError("ACSUFF_R0_KNOWN_ANSWER_CONTRADICTION")

    same_observation = left.policy_observation_sha256 == right.policy_observation_sha256
    same_action = left.optimal_action_key == right.optimal_action_key
    classification = (
        REPRESENTATION_GAP
        if same_observation and not same_action
        else ALIAS_ACTION_CONSISTENT
        if same_observation
        else OBSERVATION_DISTINGUISHABLE
    )
    assessment = StateAliasAssessmentR0(
        schema_version=STATE_SUFFICIENCY_SCHEMA_R0,
        left_case_id=left.case_id,
        right_case_id=right.case_id,
        same_policy_observation=same_observation,
        same_optimal_action=same_action,
        classification=classification,
    )
    assessment.validate()
    return assessment


def detect_state_aliases_r0(
    cases: Sequence[KnownAnswerStateCaseR0],
) -> StateSufficiencyReportR0:
    materialized = tuple(cases)
    if len(materialized) < 2:
        raise RuntimeError("ACSUFF_R0_CASE_COUNT_INSUFFICIENT")
    case_ids: set[str] = set()
    by_state: dict[str, str] = {}
    for case in materialized:
        case.validate()
        if case.case_id in case_ids:
            raise RuntimeError("ACSUFF_R0_DUPLICATE_CASE_ID")
        case_ids.add(case.case_id)
        previous = by_state.get(case.authoritative_state_sha256)
        if previous is not None and previous != case.optimal_action_key:
            raise RuntimeError("ACSUFF_R0_KNOWN_ANSWER_CONTRADICTION")
        by_state[case.authoritative_state_sha256] = case.optimal_action_key

    assessments = tuple(
        assess_state_pair_r0(left, right)
        for left, right in combinations(materialized, 2)
    )
    report = StateSufficiencyReportR0(
        schema_version=STATE_SUFFICIENCY_SCHEMA_R0,
        case_count=len(materialized),
        pair_count=len(assessments),
        assessments=assessments,
    )
    report.validate()
    return report
