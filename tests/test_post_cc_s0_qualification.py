from pathlib import Path

from cb16_local_opt.post_cc_s0_qualification_v1 import REQUIRED_GATES, compile_s0_gates


def test_post_cc_s0_hard_gates_pass_on_frozen_authority():
    root = Path(__file__).resolve().parents[1]
    result = compile_s0_gates(root)
    assert result["status"] == "PASS", result
    assert result["classification"] == "PASS"
    assert set(result["gates"]) == set(REQUIRED_GATES)
    assert all(result["gates"].values())
    assert result["failed_gates"] == []
    assert result["evidence_if_pass"] == "POST_CC_CONTRACT_MIGRATION_QUALIFIED"
    assert result["FINAL_opened"] is False
    assert result["fresh_data_used"] is False
    assert result["ECONOMIC_evidence_claimed"] is False
    assert result["TRANSFER_evidence_claimed"] is False
