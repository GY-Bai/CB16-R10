"""S1 mandatory fail-closed integrity attack matrix and objective firewalls."""

from __future__ import annotations

import pytest

from cb16_local_opt.post_cc_s1_qualification_v1 import (
    run_fabricated_log_mu_audit_v1,
    run_integrity_attack_suite_v1,
    run_objective_firewall_audit_v1,
)

REQUIRED_ATTACKS = (
    "observation_content_hash_mismatch",
    "missing_behavior_identity",
    "fabricated_log_mu",
    "nominal_replaced_by_executed",
    "cross_account_replay_splice",
    "time_order_corruption",
    "nonflat_risk_outside_support",
    "flat_with_nonzero_risk",
    "prepared_or_staged_child_authority",
    "committed_child_checkpoint_tamper",
    "same_instance_retry_after_mutation",
    "terminal_supplied_with_bootstrap",
    "truncation_missing_durable_bootstrap",
)


@pytest.fixture(scope="module")
def integrity():
    return run_integrity_attack_suite_v1(".")


def test_integrity_attack_matrix_covers_every_mandatory_attack(integrity):
    for name in REQUIRED_ATTACKS:
        assert name in integrity["attacks"], name
    assert integrity["all_rejected"] is True
    for name, record in integrity["attacks"].items():
        assert record["rejected"] is True, (name, record)


def test_objective_firewall_audit_uses_hand_arithmetic(integrity):
    audit = run_objective_firewall_audit_v1(".")
    assert audit["all_checks_pass"] is True
    assert audit["checks"]["production_ev_matches_hand_arithmetic"] is True
    assert audit["checks"]["failure_branch_equity_is_negative_and_counted"] is True
    assert audit["checks"]["no_alternative_objective_token_in_s1_sources"] is True
    assert audit["checks"]["no_production_promotion_in_s1_sources"] is True
    assert audit["details"]["loss_branch_equity"] <= 0.0


def test_fabricated_log_mu_audit_fails_closed(integrity):
    audit = run_fabricated_log_mu_audit_v1(".")
    assert audit["all_checks_pass"] is True
    assert audit["checks"]["tensor_mismatch_rejected"] is True
    assert audit["checks"]["reconstruction_flag_rejected"] is True
    assert audit["checks"]["durable_binding_rebind_rejected"] is True
