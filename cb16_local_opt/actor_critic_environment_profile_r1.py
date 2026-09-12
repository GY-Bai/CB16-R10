from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ENVIRONMENT_PROFILE_VERSION_R1 = "CB16_R11_BC_POLICY_NEUTRAL_ENVIRONMENT_PROFILE_V1_R1"
TAXONOMY_SCHEMA_V1 = "CB16_R11_BC_EXECUTION_RULE_TAXONOMY_V1"

MECHANICAL_OR_RESOURCE_RULES_R1 = (
    "fee",
    "funding",
    "leverage_cap",
    "liquidation",
    "margin",
    "min_notional",
    "min_qty",
    "risk_limits",
    "slippage",
)

STRATEGY_PREFERENCE_RULES_R1 = (
    "cooldown",
    "max_hold",
    "stop_loss",
    "take_profit",
)

HISTORICAL_ONLY_RULES_R1 = ("finalize_behavior",)


@dataclass(frozen=True)
class ActorCriticEnvironmentProfileR1:
    profile_version: str
    taxonomy_schema: str
    enabled_rules: tuple[str, ...]
    disabled_rules: tuple[str, ...]
    explicit_strategy_authorizations: tuple[str, ...]

    def validate(self, taxonomy: Mapping[str, object]) -> None:
        if self.profile_version != ENVIRONMENT_PROFILE_VERSION_R1:
            raise RuntimeError("ACENV_R1_PROFILE_VERSION_MISMATCH")
        if self.taxonomy_schema != TAXONOMY_SCHEMA_V1:
            raise RuntimeError("ACENV_R1_TAXONOMY_SCHEMA_MISMATCH")
        if taxonomy.get("schema") != TAXONOMY_SCHEMA_V1:
            raise RuntimeError("ACENV_R1_TAXONOMY_INPUT_INVALID")

        rows = taxonomy.get("rules")
        if not isinstance(rows, list):
            raise RuntimeError("ACENV_R1_TAXONOMY_RULES_INVALID")
        by_id = {row.get("rule_id"): row for row in rows if isinstance(row, Mapping)}
        if len(by_id) != len(rows):
            raise RuntimeError("ACENV_R1_TAXONOMY_DUPLICATE_RULE")

        enabled = set(self.enabled_rules)
        disabled = set(self.disabled_rules)
        authorized = set(self.explicit_strategy_authorizations)
        if enabled & disabled:
            raise RuntimeError("ACENV_R1_RULE_ENABLED_AND_DISABLED")
        if enabled | disabled != set(by_id):
            raise RuntimeError("ACENV_R1_RULE_COVERAGE_INCOMPLETE")
        if authorized - set(STRATEGY_PREFERENCE_RULES_R1):
            raise RuntimeError("ACENV_R1_UNKNOWN_STRATEGY_AUTHORIZATION")

        for rule_id, row in by_id.items():
            category = row.get("category")
            if category == "STRATEGY_PREFERENCE":
                if rule_id in enabled and rule_id not in authorized:
                    raise RuntimeError("ACENV_R1_STRATEGY_PREFERENCE_SILENTLY_ENABLED")
            elif category == "HISTORICAL_ONLY":
                if rule_id in enabled:
                    raise RuntimeError("ACENV_R1_HISTORICAL_ONLY_RULE_ENABLED")
            elif category in {"MARKET_EXCHANGE_MECHANIC", "USER_RESOURCE_BOUNDARY"}:
                if rule_id not in enabled:
                    raise RuntimeError("ACENV_R1_MECHANICAL_RULE_DISABLED")
            else:
                raise RuntimeError("ACENV_R1_UNCLASSIFIED_RULE")

    def is_enabled(self, rule_id: str) -> bool:
        return rule_id in self.enabled_rules


def policy_neutral_environment_profile_r1() -> ActorCriticEnvironmentProfileR1:
    return ActorCriticEnvironmentProfileR1(
        profile_version=ENVIRONMENT_PROFILE_VERSION_R1,
        taxonomy_schema=TAXONOMY_SCHEMA_V1,
        enabled_rules=tuple(sorted(MECHANICAL_OR_RESOURCE_RULES_R1)),
        disabled_rules=tuple(sorted(STRATEGY_PREFERENCE_RULES_R1 + HISTORICAL_ONLY_RULES_R1)),
        explicit_strategy_authorizations=(),
    )
