from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

from cb16_local_opt.state_qualification_r0 import (
    COMPONENT_FAIL,
    COMPONENT_PASS,
    REQUIRED_PHASE_C_TASKS_R0,
    STATE_QUALIFICATION_COMPONENT_FAILURE,
    STATE_QUALIFICATION_PASS,
    STATE_QUALIFICATION_REPRESENTATION_GAP,
    ComponentQualificationEvidenceR0,
    compile_state_qualification_r0,
)
from cb16_local_opt.state_sufficiency_r0 import (
    detect_state_aliases_r0,
    make_known_answer_state_case_r0,
)


def _components(*, failed_task: str | None = None):
    return tuple(
        ComponentQualificationEvidenceR0(
            task_id=task_id,
            status=COMPONENT_FAIL if task_id == failed_task else COMPONENT_PASS,
            evidence_sha256=f"{index:x}" * 64,
            evidence_kind="DEDICATED_GATE_RECEIPT",
        )
        for index, task_id in enumerate(REQUIRED_PHASE_C_TASKS_R0, start=1)
    )


def _case(case_id: str, state_char: str, observation_value: float, action: str):
    return make_known_answer_state_case_r0(
        case_id=case_id,
        authoritative_state_sha256=state_char * 64,
        policy_observation_payload={"state": (observation_value,)},
        optimal_action_key=action,
    )


def _sufficient_report():
    return detect_state_aliases_r0(
        (
            _case("flat", "a", 0.0, "FLAT"),
            _case("long", "b", 1.0, "LONG:1.0"),
        )
    )


def _gap_report():
    return detect_state_aliases_r0(
        (
            _case("flat", "a", 0.0, "FLAT"),
            _case("long", "b", 0.0, "LONG:1.0"),
        )
    )


def test_exact_bc021_to_bc032_pass_compiles_machine_readable_phase_d_eligibility() -> None:
    result = compile_state_qualification_r0(
        component_evidence=_components(),
        state_sufficiency_report=_sufficient_report(),
    )
    assert result.verdict == STATE_QUALIFICATION_PASS
    assert result.eligible_for_phase_d is True
    assert result.representation_gap_count == 0
    assert result.blocking_reasons == ()
    assert tuple(item.task_id for item in result.component_evidence) == REQUIRED_PHASE_C_TASKS_R0

    payload = result.machine_readable()
    assert payload["verdict"] == STATE_QUALIFICATION_PASS
    assert payload["eligible_for_phase_d"] is True
    assert len(payload["qualification_sha256"]) == 64
    assert payload == result.machine_readable()


def test_unresolved_representation_gap_blocks_pass_even_when_every_component_gate_passes() -> None:
    result = compile_state_qualification_r0(
        component_evidence=_components(),
        state_sufficiency_report=_gap_report(),
    )
    assert result.verdict == STATE_QUALIFICATION_REPRESENTATION_GAP
    assert result.eligible_for_phase_d is False
    assert result.representation_gap_count == 1
    assert result.unresolved_representation_gap_pairs == (("flat", "long"),)
    assert "UNRESOLVED_REPRESENTATION_GAP" in result.blocking_reasons


def test_model_size_optimizer_and_training_duration_cannot_rescue_representation_gap() -> None:
    parameters = set(inspect.signature(compile_state_qualification_r0).parameters)
    forbidden_rescue_inputs = {
        "model_size",
        "parameter_count",
        "hidden_size",
        "optimizer",
        "learning_rate",
        "training_steps",
        "training_duration",
    }
    assert forbidden_rescue_inputs.isdisjoint(parameters)

    gap = compile_state_qualification_r0(
        component_evidence=_components(),
        state_sufficiency_report=_gap_report(),
    )
    assert gap.verdict == STATE_QUALIFICATION_REPRESENTATION_GAP
    assert gap.eligible_for_phase_d is False


def test_component_failure_blocks_phase_d_when_representation_is_sufficient() -> None:
    result = compile_state_qualification_r0(
        component_evidence=_components(failed_task="BC-028"),
        state_sufficiency_report=_sufficient_report(),
    )
    assert result.verdict == STATE_QUALIFICATION_COMPONENT_FAILURE
    assert result.eligible_for_phase_d is False
    assert result.blocking_reasons == ("COMPONENT_FAILURE:BC-028",)


def test_representation_gap_remains_primary_blocker_if_component_also_failed() -> None:
    result = compile_state_qualification_r0(
        component_evidence=_components(failed_task="BC-032"),
        state_sufficiency_report=_gap_report(),
    )
    assert result.verdict == STATE_QUALIFICATION_REPRESENTATION_GAP
    assert result.eligible_for_phase_d is False
    assert result.blocking_reasons == (
        "UNRESOLVED_REPRESENTATION_GAP",
        "COMPONENT_FAILURE:BC-032",
    )


def test_missing_or_duplicate_phase_c_evidence_fails_closed() -> None:
    components = _components()
    with pytest.raises(RuntimeError, match="COVERAGE_INVALID"):
        compile_state_qualification_r0(
            component_evidence=components[:-1],
            state_sufficiency_report=_sufficient_report(),
        )

    duplicate = components[:-1] + (components[0],)
    with pytest.raises(RuntimeError, match="DUPLICATE_TASK_EVIDENCE"):
        compile_state_qualification_r0(
            component_evidence=duplicate,
            state_sufficiency_report=_sufficient_report(),
        )


def test_component_input_order_is_canonicalized_to_bc021_through_bc032() -> None:
    components = tuple(reversed(_components()))
    result = compile_state_qualification_r0(
        component_evidence=components,
        state_sufficiency_report=_sufficient_report(),
    )
    assert tuple(item.task_id for item in result.component_evidence) == REQUIRED_PHASE_C_TASKS_R0


def test_bad_evidence_identity_or_unknown_status_fails_closed() -> None:
    good = _components()[0]
    with pytest.raises(RuntimeError, match="EVIDENCE_HASH_INVALID"):
        replace(good, evidence_sha256="not-a-sha").validate()
    with pytest.raises(RuntimeError, match="COMPONENT_STATUS_INVALID"):
        replace(good, status="MAYBE").validate()


def test_machine_result_tamper_cannot_turn_failure_into_pass() -> None:
    failed = compile_state_qualification_r0(
        component_evidence=_components(),
        state_sufficiency_report=_gap_report(),
    )
    tampered = replace(
        failed,
        verdict=STATE_QUALIFICATION_PASS,
        eligible_for_phase_d=True,
        blocking_reasons=(),
    )
    with pytest.raises(RuntimeError, match="VERDICT_INCONSISTENT"):
        tampered.validate()


def test_state_sufficiency_hash_changes_when_alias_classification_changes() -> None:
    sufficient = compile_state_qualification_r0(
        component_evidence=_components(),
        state_sufficiency_report=_sufficient_report(),
    )
    gap = compile_state_qualification_r0(
        component_evidence=_components(),
        state_sufficiency_report=_gap_report(),
    )
    assert sufficient.state_sufficiency_sha256 != gap.state_sufficiency_sha256
    assert sufficient.machine_readable()["qualification_sha256"] != gap.machine_readable()["qualification_sha256"]
