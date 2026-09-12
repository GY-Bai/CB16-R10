from __future__ import annotations

import pytest

from cb16_local_opt.account_lineage_r0 import (
    CONTINUITY,
    NEW_ACCOUNT,
    ORIGIN,
    make_account_lineage_event_r0,
    validate_restart_lineage_r0,
)


def test_origin_binds_identity_to_initial_capital_source() -> None:
    event = make_account_lineage_event_r0(
        lineage_event_id="origin-1",
        relation=ORIGIN,
        logical_account_id="acct-A",
        capital_source="OWNER_CAPITAL",
        capital_flow_id="flow-initial",
    )
    assert event.logical_account_id == "acct-A"
    assert event.parent_account_id is None


def test_same_identity_restart_requires_explicit_continuity_authorization() -> None:
    continuity = make_account_lineage_event_r0(
        lineage_event_id="resume-1",
        relation=CONTINUITY,
        logical_account_id="acct-A",
        parent_account_id="acct-A",
        reset_reason="PROCESS_RESTART",
        continuity_authorization_id="continuity-auth-1",
    )
    assert validate_restart_lineage_r0(previous_account_id="acct-A", restart_event=continuity) == "acct-A"


def test_continuity_event_cannot_inject_new_capital() -> None:
    with pytest.raises(RuntimeError, match="CONTINUITY_CANNOT_INJECT_CAPITAL"):
        make_account_lineage_event_r0(
            lineage_event_id="bad",
            relation=CONTINUITY,
            logical_account_id="acct-A",
            parent_account_id="acct-A",
            reset_reason="PROCESS_RESTART",
            continuity_authorization_id="auth",
            capital_source="OWNER_CAPITAL",
            capital_flow_id="flow",
        )


def test_failed_account_recapitalization_requires_new_identity() -> None:
    new_account = make_account_lineage_event_r0(
        lineage_event_id="restart-funded",
        relation=NEW_ACCOUNT,
        logical_account_id="acct-B",
        parent_account_id="acct-A",
        reset_reason="PRIOR_ACCOUNT_ECONOMIC_FAILURE",
        capital_source="OWNER_CAPITAL",
        capital_flow_id="new-account-flow",
    )
    assert validate_restart_lineage_r0(previous_account_id="acct-A", restart_event=new_account) == "acct-B"
    assert new_account.logical_account_id != new_account.parent_account_id


def test_newly_funded_account_cannot_reuse_failed_account_id() -> None:
    with pytest.raises(RuntimeError, match="NEW_ACCOUNT_ID_REQUIRED"):
        make_account_lineage_event_r0(
            lineage_event_id="bad-reuse",
            relation=NEW_ACCOUNT,
            logical_account_id="acct-A",
            parent_account_id="acct-A",
            reset_reason="PRIOR_ACCOUNT_ECONOMIC_FAILURE",
            capital_source="OWNER_CAPITAL",
            capital_flow_id="flow-new",
        )


def test_restart_cannot_inherit_identity_via_new_account_relation() -> None:
    with pytest.raises(RuntimeError, match="NEW_ACCOUNT_ID_REQUIRED"):
        make_account_lineage_event_r0(
            lineage_event_id="bad",
            relation=NEW_ACCOUNT,
            logical_account_id="acct-A",
            parent_account_id="acct-A",
            reset_reason="RESET",
            capital_source="OWNER_CAPITAL",
            capital_flow_id="flow",
        )


def test_different_identity_restart_must_name_previous_account_as_parent() -> None:
    wrong_parent = make_account_lineage_event_r0(
        lineage_event_id="new",
        relation=NEW_ACCOUNT,
        logical_account_id="acct-C",
        parent_account_id="acct-X",
        reset_reason="PRIOR_ACCOUNT_ECONOMIC_FAILURE",
        capital_source="OWNER_CAPITAL",
        capital_flow_id="flow",
    )
    with pytest.raises(RuntimeError, match="NEW_ID_REQUIRES_NEW_ACCOUNT_RELATION"):
        validate_restart_lineage_r0(previous_account_id="acct-A", restart_event=wrong_parent)
