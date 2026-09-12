from __future__ import annotations

import ast
from pathlib import Path
import tempfile

import pytest

from cb16_local_opt.cc_environment_advance_r0 import CCEnvironmentIntervalR0
from cb16_local_opt.cc_fast_fact_queue_r0 import BackpressureRequired, FactOutputQueue, encode_fact
from cb16_local_opt.cc_integration_contracts_r0 import (
    W_SCHEMA_IDENTITIES,
    runtime_transition_raw_fact,
    runtime_transition_to_experience_environment,
)
from cb16_local_opt.cc_integration_fast_path_r0 import transport_facts_fast, decode_fast_chunks
from cb16_local_opt.cc_integration_runtime_r0 import run_closed_loop_canary_temp
from tests.cc_thread_a_support_r0 import runtime, decision


def test_w_contract_identity_and_no_decision_raw_only_firewall():
    assert W_SCHEMA_IDENTITIES == {
        "W-01": "CCPolicyDecisionV1",
        "W-02": "CCEnvironmentTransitionV1",
        "W-03": "CCExperienceSequenceV1",
        "W-04": "CCLearningUpdateV1",
        "W-05": "CCEconomicResultV1",
    }
    r = runtime(every=2)
    r.step(CCEnvironmentIntervalR0(101.0), lambda a, c: decision(a, c), expected_predecessor_token=r.predecessor_token)
    no_decision = r.step(CCEnvironmentIntervalR0(99.0), None, expected_predecessor_token=r.predecessor_token)
    raw = runtime_transition_raw_fact(no_decision)
    assert raw["w02_replay_eligible"] is False
    assert raw["raw_only_reason"] == "NO_POLICY_DECISION_NO_LOG_MU_FABRICATION"
    with pytest.raises(ValueError, match="RAW_ONLY_NO_POLICY_DECISION"):
        runtime_transition_to_experience_environment(no_decision)


def test_joined_closed_loop_canary_reaches_child_policy_on_same_account():
    result = run_closed_loop_canary_temp()
    assert result.verdict == "PASS", result
    assert all(result.checks.values()), result.checks
    assert result.replay_transition_count >= 4
    assert result.raw_fact_count >= result.replay_transition_count * 2
    assert result.parent_checkpoint_sha256 != result.child_checkpoint_sha256
    assert result.child_action_policy_id == "cc-integrated-policy-g1"


def test_thread_d_transport_is_exact_for_thread_a_semantic_payloads():
    facts = []
    reference = {}
    for decision_index in range(4):
        for account in ("acct-a", "acct-b"):
            sid = f"{account}-{decision_index}"
            payload = {
                "schema": "CCRuntimeRawFactV1",
                "account_lineage_id": account,
                "decision_index": decision_index,
                "post_equity": 1000.0 - 3.25 * decision_index,
                "mechanical_terminal": decision_index == 3 and account == "acct-b",
                "failure_classification": "LIABILITY" if decision_index == 2 and account == "acct-a" else "NONE",
            }
            reference[sid] = payload
            facts.append((account, decision_index, 0, sid, payload, payload["mechanical_terminal"] or payload["failure_classification"] != "NONE"))
    with tempfile.TemporaryDirectory(prefix="cc-fast-transport-test-") as td:
        report = transport_facts_fast(facts=facts, account_ids=("acct-a", "acct-b"), output_root=td, chunk_facts=3)
        decoded = decode_fast_chunks(report.receipts)
    assert report.semantic_verdict == "PASS"
    assert decoded == reference
    assert report.fact_count == 8
    assert all(receipt.durable for receipt in report.receipts)


def test_writer_backpressure_fails_closed_without_dropping_existing_failure_or_terminal_fact():
    q = FactOutputQueue(max_bytes=256, max_age_s=30.0)
    first = encode_fact({"kind": "FAILURE", "payload": "x" * 40}, semantic_id="failure-1", terminal_or_failure=True)
    q.put(first)
    assert q.depth == 1
    too_large = encode_fact({"kind": "TERMINAL", "payload": "y" * 1000}, semantic_id="terminal-2", terminal_or_failure=True)
    with pytest.raises(BackpressureRequired):
        q.put(too_large, block=False)
    # Backpressure is explicit; the already accepted failure fact remains present and unchanged.
    retained = q.get()
    assert retained.semantic_id == "failure-1" and retained.terminal_or_failure is True


def test_integration_and_fast_runtime_have_no_legacy_performance_import_or_fallback():
    forbidden = {
        "cb16_local_opt.gpu_inference_broker",
        "cb16_local_opt.multiprocess_trajectory_farm",
        "cb16_local_opt.vectorized_physics",
    }
    for pattern in ("cc_integration_*.py", "cc_fast_*.py"):
        for path in Path("cb16_local_opt").glob(pattern):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.add(node.module or "")
            assert not (forbidden & imported), (path, forbidden & imported)
