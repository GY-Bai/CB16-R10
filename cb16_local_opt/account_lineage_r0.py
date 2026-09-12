from __future__ import annotations

from dataclasses import dataclass

ACCOUNT_LINEAGE_SCHEMA_R0 = "CB16_R11_BC_ACCOUNT_LINEAGE_V1_R0"
ORIGIN = "ORIGIN"
CONTINUITY = "CONTINUITY"
NEW_ACCOUNT = "NEW_ACCOUNT"
ACCOUNT_LINEAGE_RELATIONS_R0 = (ORIGIN, CONTINUITY, NEW_ACCOUNT)


@dataclass(frozen=True)
class AccountLineageEventR0:
    schema_version: str
    lineage_event_id: str
    relation: str
    logical_account_id: str
    parent_account_id: str | None
    reset_reason: str | None
    capital_source: str | None
    capital_flow_id: str | None
    continuity_authorization_id: str | None

    def validate(self) -> None:
        if self.schema_version != ACCOUNT_LINEAGE_SCHEMA_R0:
            raise RuntimeError("ACLIN_R0_SCHEMA_MISMATCH")
        if not self.lineage_event_id or not self.logical_account_id:
            raise RuntimeError("ACLIN_R0_IDENTITY_INVALID")
        if self.relation not in ACCOUNT_LINEAGE_RELATIONS_R0:
            raise RuntimeError("ACLIN_R0_RELATION_INVALID")
        if self.relation == ORIGIN:
            if self.parent_account_id is not None or self.continuity_authorization_id is not None:
                raise RuntimeError("ACLIN_R0_ORIGIN_PARENT_INVALID")
            if not self.capital_source or not self.capital_flow_id:
                raise RuntimeError("ACLIN_R0_ORIGIN_CAPITAL_SOURCE_REQUIRED")
        elif self.relation == CONTINUITY:
            if self.parent_account_id != self.logical_account_id:
                raise RuntimeError("ACLIN_R0_CONTINUITY_IDENTITY_MISMATCH")
            if not self.continuity_authorization_id:
                raise RuntimeError("ACLIN_R0_CONTINUITY_AUTH_REQUIRED")
            if self.capital_source is not None or self.capital_flow_id is not None:
                raise RuntimeError("ACLIN_R0_CONTINUITY_CANNOT_INJECT_CAPITAL")
        elif self.relation == NEW_ACCOUNT:
            if not self.parent_account_id or self.parent_account_id == self.logical_account_id:
                raise RuntimeError("ACLIN_R0_NEW_ACCOUNT_ID_REQUIRED")
            if not self.reset_reason:
                raise RuntimeError("ACLIN_R0_NEW_ACCOUNT_RESET_REASON_REQUIRED")
            if not self.capital_source or not self.capital_flow_id:
                raise RuntimeError("ACLIN_R0_NEW_ACCOUNT_CAPITAL_SOURCE_REQUIRED")
            if self.continuity_authorization_id is not None:
                raise RuntimeError("ACLIN_R0_NEW_ACCOUNT_NOT_CONTINUITY")


def make_account_lineage_event_r0(*, lineage_event_id: str, relation: str, logical_account_id: str, parent_account_id: str | None = None, reset_reason: str | None = None, capital_source: str | None = None, capital_flow_id: str | None = None, continuity_authorization_id: str | None = None) -> AccountLineageEventR0:
    event = AccountLineageEventR0(
        schema_version=ACCOUNT_LINEAGE_SCHEMA_R0,
        lineage_event_id=lineage_event_id,
        relation=relation,
        logical_account_id=logical_account_id,
        parent_account_id=parent_account_id,
        reset_reason=reset_reason,
        capital_source=capital_source,
        capital_flow_id=capital_flow_id,
        continuity_authorization_id=continuity_authorization_id,
    )
    event.validate()
    return event


def validate_restart_lineage_r0(*, previous_account_id: str, restart_event: AccountLineageEventR0) -> str:
    restart_event.validate()
    if not previous_account_id:
        raise RuntimeError("ACLIN_R0_PREVIOUS_ACCOUNT_ID_INVALID")
    if restart_event.logical_account_id == previous_account_id:
        if restart_event.relation != CONTINUITY or restart_event.parent_account_id != previous_account_id:
            raise RuntimeError("ACLIN_R0_SAME_ID_REQUIRES_AUTHORIZED_CONTINUITY")
        if not restart_event.continuity_authorization_id:
            raise RuntimeError("ACLIN_R0_SAME_ID_REQUIRES_AUTHORIZED_CONTINUITY")
        return previous_account_id
    if restart_event.relation != NEW_ACCOUNT or restart_event.parent_account_id != previous_account_id:
        raise RuntimeError("ACLIN_R0_NEW_ID_REQUIRES_NEW_ACCOUNT_RELATION")
    return restart_event.logical_account_id
