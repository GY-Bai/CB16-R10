from __future__ import annotations

import json
from pathlib import Path

from scripts.r11_longtraj_training_campaign_r0 import (
    EXPECTED_DECISION_TIMES_SHA,
    EXPECTED_G0_CHAMPION,
    EXPECTED_PARENT_IDS_SHA,
    EXPECTED_PARENT_RECEIPTS_SHA,
    EXPECTED_TRAIN_EVIDENCE,
    EXPECTED_VALIDATION_EVIDENCE,
    canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_TRAINING_CAMPAIGN_R0_SPEC_V1.json"


def test_campaign_prereg_is_one_bounded_shadow_generation_zero() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["schema"] == "CB16_R11_LONGTRAJ_TRAINING_CAMPAIGN_R0_SPEC_V1"
    assert spec["status"] == "FROZEN_BEFORE_FIRST_CAMPAIGN_EXECUTION"
    assert spec["classification"] == (
        "ONE_BOUNDED_GENERATION0_SHADOW_CAMPAIGN__NO_CANONICAL_PROMOTION__NO_SCIENTIFIC_MARKET_VERDICT"
    )
    assert spec["training"]["campaign_count"] == 1
    assert spec["training"]["canonical_generation"] == 0
    assert spec["training"]["epochs"] == 12
    assert spec["training"]["batch_size"] == 512
    assert spec["training"]["expected_optimizer_steps"] == 12
    assert spec["training"]["optional_stopping"] is False
    assert spec["admission_parent"]["admission_one_step_checkpoint_is_campaign_parent"] is False
    assert spec["student_initialization"]["admission_one_step_model_continuation_allowed"] is False
    assert spec["evaluation"]["same_validation_followup_may_authorize_promotion"] is False
    assert spec["hard_outputs"]["canonical_promotion_authorized"] is False
    assert spec["hard_outputs"]["canonical_generation_advance_authorized"] is False
    assert spec["hard_outputs"]["new_scientific_market_verdict"] is False
    assert spec["hard_outputs"]["scientific_verdict"] is None


def test_campaign_prereg_exact_identities_match_runner_constants() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    cohort = spec["cohort_identity"]
    assert cohort["parent_receipts_canonical_sha256"] == EXPECTED_PARENT_RECEIPTS_SHA
    assert cohort["parent_ids_canonical_sha256"] == EXPECTED_PARENT_IDS_SHA
    assert cohort["decision_times_canonical_sha256"] == EXPECTED_DECISION_TIMES_SHA
    assert cohort["admission_prepared_train_hash"] == EXPECTED_TRAIN_EVIDENCE
    assert cohort["admission_prepared_validation_hash"] == EXPECTED_VALIDATION_EVIDENCE
    assert spec["student_initialization"]["expected_policy_hash"] == EXPECTED_G0_CHAMPION
    assert spec["r4_canonical_learning_authority"]["immutable_g0_champion_policy_hash"] == EXPECTED_G0_CHAMPION


def test_campaign_canonical_json_hash_is_stable() -> None:
    assert canonical_json_sha256([{"b": 2, "a": 1}]) == (
        "44c7deead2ed8313d29655e45c0d1469419213c93d9f44d66da7c7afe46e74e3"
    )
