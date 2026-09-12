from __future__ import annotations

from dataclasses import dataclass

from .cc_economic_promotion_r0 import assess as legacy_assess, UNRESOLVED_OWNER_DECISION
from .cc_experience_wire_r0 import CCEconomicResultV1
from .post_cc_economic_adapter_v1 import AdaptedEconomicResultV1, adapt_w05

LEGACY_PROMOTION_CONTRACT_ID = "CC_ECONOMIC_PROMOTION_R0"
SUCCESSOR_PROMOTION_CONTRACT_ID = "CB16_R11_POST_CC_PROMOTION_V1"
MIGRATION_CONTRACT_ID = "CB16_R11_POST_CC_ECONOMIC_MIGRATION_V1"
CONTRACT_MISMATCH = "CONTRACT_MISMATCH"


@dataclass(frozen=True)
class LegacyPromotionMigrationV1:
    contract_id: str
    legacy_contract_id: str
    successor_contract_id: str
    legacy_status: str
    adapted_result: AdaptedEconomicResultV1
    promotion_rule_required: bool
    current_owner_uncertainty: bool
    migration_status: str

    def validate(self) -> "LegacyPromotionMigrationV1":
        if self.contract_id != MIGRATION_CONTRACT_ID:
            raise ValueError("unexpected migration contract")
        if self.legacy_contract_id != LEGACY_PROMOTION_CONTRACT_ID:
            raise ValueError("unexpected legacy contract")
        if self.successor_contract_id != SUCCESSOR_PROMOTION_CONTRACT_ID:
            raise ValueError("unexpected successor contract")
        self.adapted_result.validate()
        if self.current_owner_uncertainty:
            raise ValueError(
                "legacy disagreement must not be represented as current owner uncertainty"
            )
        return self


def migrate_legacy_result(result: CCEconomicResultV1) -> LegacyPromotionMigrationV1:
    legacy = legacy_assess(result)
    adapted = adapt_w05(result)
    status = (
        "LEGACY_UNRESOLVED_MAPPED_TO_PARALLEL_COMPONENTS"
        if legacy.status == UNRESOLVED_OWNER_DECISION
        else "LEGACY_STATUS_PRESERVED_AS_PROVENANCE"
    )
    return LegacyPromotionMigrationV1(
        MIGRATION_CONTRACT_ID,
        LEGACY_PROMOTION_CONTRACT_ID,
        SUCCESSOR_PROMOTION_CONTRACT_ID,
        legacy.status,
        adapted,
        True,
        False,
        status,
    ).validate()


def require_post_cc_promotion_authority(contract_id: str) -> str:
    if contract_id == LEGACY_PROMOTION_CONTRACT_ID:
        raise ValueError(
            f"{CONTRACT_MISMATCH}: legacy promotion authority is historical-only"
        )
    if contract_id != SUCCESSOR_PROMOTION_CONTRACT_ID:
        raise ValueError(f"{CONTRACT_MISMATCH}: unknown promotion authority")
    return contract_id
