from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "authority/rearchitecture_r11/CB16_R11_BC_EXECUTION_RULE_TAXONOMY_V1.json"

REQUIRED_RULES = {
    "fee",
    "slippage",
    "funding",
    "margin",
    "liquidation",
    "min_qty",
    "min_notional",
    "leverage_cap",
    "stop_loss",
    "take_profit",
    "max_hold",
    "cooldown",
    "risk_limits",
    "finalize_behavior",
}

ALLOWED_CATEGORIES = {
    "MARKET_EXCHANGE_MECHANIC",
    "USER_RESOURCE_BOUNDARY",
    "STRATEGY_PREFERENCE",
    "HISTORICAL_ONLY",
}


def _load() -> dict:
    return json.loads(PATH.read_text())


def test_taxonomy_covers_every_bc009_required_execution_rule() -> None:
    doc = _load()
    assert doc["schema"] == "CB16_R11_BC_EXECUTION_RULE_TAXONOMY_V1"
    assert doc["task_id"] == "BC-009"
    rules = {row["rule_id"]: row for row in doc["rules"]}
    assert set(rules) == REQUIRED_RULES


def test_every_rule_has_classified_source_and_version() -> None:
    doc = _load()
    assert set(doc["allowed_categories"]) == ALLOWED_CATEGORIES
    for row in doc["rules"]:
        assert row["category"] in ALLOWED_CATEGORIES
        assert isinstance(row["source_path"], str) and row["source_path"]
        assert isinstance(row["source_version"], str) and row["source_version"]
        assert isinstance(row["source_field"], str) and row["source_field"]
        assert isinstance(row["round2_default"], str) and row["round2_default"]


def test_round2_does_not_silently_enable_strategy_preferences() -> None:
    rules = {row["rule_id"]: row for row in _load()["rules"]}
    for rule_id in ("stop_loss", "take_profit", "max_hold", "cooldown"):
        assert rules[rule_id]["category"] == "STRATEGY_PREFERENCE"
        assert rules[rule_id]["round2_default"] == "DISABLED"


def test_historical_finalize_is_not_round2_execution_authority() -> None:
    row = {row["rule_id"]: row for row in _load()["rules"]}["finalize_behavior"]
    assert row["category"] == "HISTORICAL_ONLY"
    assert row["round2_default"] == "DISABLED"


def test_mechanical_and_resource_rules_are_explicitly_enforced_or_bound() -> None:
    rules = {row["rule_id"]: row for row in _load()["rules"]}
    for rule_id in (
        "fee",
        "slippage",
        "funding",
        "margin",
        "liquidation",
        "min_qty",
        "min_notional",
        "leverage_cap",
        "risk_limits",
    ):
        row = rules[rule_id]
        assert row["category"] in {"MARKET_EXCHANGE_MECHANIC", "USER_RESOURCE_BOUNDARY"}
        assert row["round2_default"] in {"ENFORCED", "ENFORCED_IF_BOUND_NON_NULL"}


def test_no_round2_forced_behavior_can_be_unclassified() -> None:
    doc = _load()
    assert doc["round2_policy"]["unclassified_forced_behavior_forbidden"] is True
    for row in doc["rules"]:
        if row["round2_default"] != "DISABLED":
            assert row["category"] in {"MARKET_EXCHANGE_MECHANIC", "USER_RESOURCE_BOUNDARY"}
