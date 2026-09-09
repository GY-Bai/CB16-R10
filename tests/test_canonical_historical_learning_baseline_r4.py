from __future__ import annotations

from cb16_local_opt.canonical_historical_learning_baseline_r4 import (
    build_shadow_snapshot_r4,
    shadow_champion_challenger_ranking_r4,
)


def test_snapshot_is_deterministic_and_binds_support():
    train_ids = tuple(f"tp{i}" for i in range(40))
    train_groups = tuple(f"tg{i}" for i in range(40))
    val_ids = tuple(f"vp{i}" for i in range(10))
    val_groups = tuple(f"vg{i}" for i in range(10))
    kwargs = dict(
        champion_policy_hash="c" * 64,
        train_evidence_hash="t" * 64,
        validation_evidence_hash="v" * 64,
        train_teacher_protocol_hash="a" * 64,
        validation_teacher_protocol_hash="b" * 64,
        train_parent_ids=train_ids,
        train_dependence_group_ids=train_groups,
        validation_parent_ids=val_ids,
        validation_dependence_group_ids=val_groups,
    )
    a = build_shadow_snapshot_r4(**kwargs)
    b = build_shadow_snapshot_r4(**kwargs)
    assert a == b
    assert a["snapshot_hash"]
    assert a["train_dependence_groups"] == 40
    assert a["validation_dependence_groups"] == 10
    assert a["distributional_r3_artifacts_in_gradient_graph"] is False
    assert a["canonical_generation_advanced"] is False


def test_snapshot_rejects_insufficient_independent_support():
    try:
        build_shadow_snapshot_r4(
            champion_policy_hash="c",
            train_evidence_hash="t",
            validation_evidence_hash="v",
            train_teacher_protocol_hash="a",
            validation_teacher_protocol_hash="b",
            train_parent_ids=tuple(f"p{i}" for i in range(31)),
            train_dependence_group_ids=tuple(f"g{i}" for i in range(31)),
            validation_parent_ids=tuple(f"v{i}" for i in range(8)),
            validation_dependence_group_ids=tuple(f"h{i}" for i in range(8)),
        )
    except RuntimeError as exc:
        assert "TRAIN_INDEPENDENT_SUPPORT_BELOW_MINIMUM" in str(exc)
    else:
        raise AssertionError("expected support fail-closed")


def test_shadow_ranking_prefers_lower_loss_without_authorizing_promotion():
    champion = {"loss": 1.2, "direction_loss": 1.1, "sizing_loss": 0.1}
    challenger = {"loss": 1.0, "direction_loss": 0.92, "sizing_loss": 0.08}
    r = shadow_champion_challenger_ranking_r4(champion, challenger)
    assert r["shadow_winner"] == "CHALLENGER"
    assert r["relative_validation_improvement"] > 0.0
    assert r["canonical_promotion_authorized"] is False
    assert r["canonical_generation_advance_authorized"] is False
    assert r["legacy_r10_0_1_percent_threshold_used"] is False


def test_shadow_ranking_returns_champion_on_equality():
    x = {"loss": 1.0, "direction_loss": 0.9, "sizing_loss": 0.1}
    r = shadow_champion_challenger_ranking_r4(x, x)
    assert r["shadow_winner"] == "CHAMPION"
    assert r["ordering"] == "CHAMPION_LOWER_OR_EQUAL_VALIDATION_LOSS"
    assert r["relative_validation_improvement"] == 0.0
