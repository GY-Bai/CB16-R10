from __future__ import annotations

from dataclasses import replace
import tempfile

from cb16_local_opt.cc_account_recovery_r0 import seal_runtime_r0, restore_runtime_r0
from cb16_local_opt.cc_environment_advance_r0 import CCEnvironmentIntervalR0
from cb16_local_opt.cc_runtime_boundary_r0 import PROCESS_FAILURE
from cb16_local_opt.cc_runtime_account_loop_r0 import CCExecutionSummaryR0
from cb16_local_opt.cc_runtime_wire_r0 import CCExecutionLegV1
from cb16_local_opt.cc_integration_contracts_r0 import (
    runtime_transition_raw_fact,
    runtime_transition_to_experience_environment,
    immutable_experience_from_runtime,
)
from cb16_local_opt.cc_integration_fast_path_r0 import (
    transport_facts_fast,
    decode_fast_chunks,
    exact_semantic_equivalence,
)
from tests.cc_thread_a_support_r0 import acct, runtime, decision, fake_exec, static_exec


def _roundtrip_transition_through_c_and_d(transition, captured_decision, *, failure_classification: str):
    environment = runtime_transition_to_experience_environment(transition)
    immutable = immutable_experience_from_runtime(
        transition_id=f"hostile-{transition.decision_index}",
        environment=environment,
        decision=captured_decision,
        market_lineage_id="CC_INTEGRATION_HOSTILE_SYNTHETIC_MARKET_V1",
        failure_classification=failure_classification,
    )
    raw = dict(runtime_transition_raw_fact(transition))
    semantic_id = f"{transition.account_lineage_id}:{transition.decision_index}"
    with tempfile.TemporaryDirectory(prefix="cc-integration-hostile-") as td:
        report = transport_facts_fast(
            facts=[(
                transition.account_lineage_id,
                transition.decision_index,
                captured_decision.policy_generation,
                semantic_id,
                raw,
                True,
            )],
            account_ids=(transition.account_lineage_id,),
            output_root=td,
            chunk_facts=1,
        )
        decoded = decode_fast_chunks(report.receipts)
    assert exact_semantic_equivalence({semantic_id: raw}, decoded)
    assert report.semantic_verdict == "PASS"
    return immutable


def test_negative_equity_liability_and_reject_world_continuation_survive_c_and_d_join():
    negative = acct(mark=10.0, cash=-100.0, basis=100.0, margin=100.0, liab=50.0)
    assert negative.equity < 0.0
    r = runtime(account=negative, executor=static_exec("REJECT", "HOSTILE_REJECT"))
    captured = {}

    def cb(a, c):
        d = decision(a, c, direction="LONG", risk=0.5)
        captured["decision"] = d
        return d

    before_equity = r.account.equity
    transition = r.step(
        CCEnvironmentIntervalR0(9.0, funding_cashflow=-1.0),
        cb,
        expected_predecessor_token=r.predecessor_token,
    )
    assert transition.permission_status == "REJECT"
    assert r.account.position_quantity == 1.0
    assert r.account.liabilities == 50.0
    assert r.account.equity < 0.0
    assert r.account.equity != before_equity
    immutable = _roundtrip_transition_through_c_and_d(
        transition,
        captured["decision"],
        failure_classification="REJECT_WITH_NEGATIVE_EQUITY_AND_LIABILITY",
    )
    assert immutable.environment.post_equity < 0.0
    assert immutable.environment.liability_delta == 0.0
    assert immutable.log_mu == captured["decision"].log_mu


def test_reversal_second_leg_failure_is_retained_exactly_across_c_and_d():
    def reversal_fail_executor(a, d):
        after = replace(
            a,
            position_quantity=0.0,
            position_cost_basis=0.0,
            margin_collateral=0.0,
        )
        return CCExecutionSummaryR0(
            "ACCEPT",
            "HOSTILE_REVERSAL_SECOND_LEG_REJECT",
            d.nominal_direction,
            d.nominal_target_risk,
            0.0,
            (
                CCExecutionLegV1(0, -1.0, "EXECUTED", a.mark_price),
                CCExecutionLegV1(1, 0.0, "REJECTED", None),
            ),
            after,
        )

    r = runtime(executor=reversal_fail_executor)
    captured = {}

    def cb(a, c):
        d = decision(a, c, direction="SHORT", risk=0.5)
        captured["decision"] = d
        return d

    transition = r.step(
        CCEnvironmentIntervalR0(100.0),
        cb,
        expected_predecessor_token=r.predecessor_token,
    )
    assert [leg.status for leg in transition.execution_legs] == ["EXECUTED", "REJECTED"]
    immutable = _roundtrip_transition_through_c_and_d(
        transition,
        captured["decision"],
        failure_classification="REVERSAL_SECOND_LEG_REJECTED",
    )
    assert immutable.environment.execution_legs[1]["status"] == "REJECTED"
    assert immutable.environment.execution_legs[1]["executed_quantity"] == 0.0


def test_process_failure_recovery_then_persistence_and_fast_transport_is_exact():
    r = runtime()
    captured = {}
    predecessor = r.predecessor_token
    r.begin_interval(
        CCEnvironmentIntervalR0(105.0, boundary_type=PROCESS_FAILURE),
        expected_predecessor_token=predecessor,
    )

    def cb(a, c):
        d = decision(a, c)
        captured["decision"] = d
        return d

    r.capture_decision(cb)
    restored, token = restore_runtime_r0(
        seal_runtime_r0(r, policy_memory_token="integration-hostile-recovery"),
        executor=fake_exec,
    )
    assert token == "integration-hostile-recovery"
    assert restored.predecessor_token == predecessor
    restored.execute_pending()
    restored.advance_pending_environment()
    transition = restored.publish_pending()
    assert transition.boundary_type == PROCESS_FAILURE
    immutable = _roundtrip_transition_through_c_and_d(
        transition,
        captured["decision"],
        failure_classification="PROCESS_FAILURE_RECOVERED",
    )
    assert immutable.environment.boundary_type == PROCESS_FAILURE
    assert immutable.environment.policy_decision_ref == captured["decision"].ref
