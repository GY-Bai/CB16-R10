from __future__ import annotations

import pytest

from cb16_local_opt.continuous_generation_binding_r0 import (
    LEAGUE_EVALUATION_LEDGER,
    TRAINING_LEDGER,
    EvidenceIntervalR0,
    GenerationSnapshotR0,
    assert_account_continuity_r0,
    assert_future_invariance_r0,
    assert_global_quarantine_r0,
    canonical_json_sha256,
    evidence_ledger_sha256,
    freeze_roster_entry_r0,
    validate_generation_transition_r0,
)


def h(label: str) -> str:
    return canonical_json_sha256({"fixture": label})


def snap(generation: int, parent_ckpt: str, parent_policy: str, *, tag: str) -> GenerationSnapshotR0:
    return GenerationSnapshotR0(
        generation=generation,
        parent_checkpoint_sha256=parent_ckpt,
        parent_policy_hash=parent_policy,
        child_checkpoint_sha256=h(f"{tag}:child_ckpt"),
        child_policy_hash=h(f"{tag}:child_policy"),
        training_recipe_sha256=h("recipe"),
        training_evidence_ledger_sha256=h("train_ledger"),
        account_snapshot_sha256=h(f"{tag}:account"),
        optimizer_snapshot_sha256=h(f"{tag}:optimizer"),
        authorized_gradient_owners_sha256=h("six_owners"),
    )


def test_generation_lineage_plus_one_and_exact_parent() -> None:
    g0 = snap(0, h("g0_parent_ckpt"), h("g0_parent_policy"), tag="g0")
    g1 = snap(1, g0.child_checkpoint_sha256, g0.child_policy_hash, tag="g1")
    validate_generation_transition_r0(g0, g1, schedule_frozen=False)


def test_generation_lineage_rejects_checkpoint_break() -> None:
    g0 = snap(0, h("g0_parent_ckpt"), h("g0_parent_policy"), tag="g0")
    g1 = snap(1, h("wrong"), g0.child_policy_hash, tag="g1")
    with pytest.raises(RuntimeError, match="CGEN_PARENT_CHECKPOINT_LINEAGE_BREAK"):
        validate_generation_transition_r0(g0, g1, schedule_frozen=False)


def test_training_forbidden_after_schedule_freeze() -> None:
    g0 = snap(0, h("g0_parent_ckpt"), h("g0_parent_policy"), tag="g0")
    g1 = snap(1, g0.child_checkpoint_sha256, g0.child_policy_hash, tag="g1")
    with pytest.raises(RuntimeError, match="CGEN_TRAINING_AFTER_LEAGUE_SCHEDULE_FREEZE_FORBIDDEN"):
        validate_generation_transition_r0(g0, g1, schedule_frozen=True)


def test_dual_ledgers_are_separate_and_deterministic() -> None:
    rows = [
        EvidenceIntervalR0("BTCUSDT", 0, 100, TRAINING_LEDGER, "T0"),
        EvidenceIntervalR0("ETHUSDT", 200, 300, TRAINING_LEDGER, "T1"),
    ]
    assert evidence_ledger_sha256(rows, expected_kind=TRAINING_LEDGER) == evidence_ledger_sha256(list(reversed(rows)), expected_kind=TRAINING_LEDGER)
    bad = [EvidenceIntervalR0("BTCUSDT", 0, 100, LEAGUE_EVALUATION_LEDGER, "E0")]
    with pytest.raises(RuntimeError, match="CGEN_LEDGER_KIND_CROSS_CONTAMINATION"):
        evidence_ledger_sha256(bad, expected_kind=TRAINING_LEDGER)


def test_global_quarantine_is_cross_symbol_market_time() -> None:
    train = [EvidenceIntervalR0("BTCUSDT", 100, 200, TRAINING_LEDGER, "T0")]
    eval_rows = [EvidenceIntervalR0("ETHUSDT", 150, 250, LEAGUE_EVALUATION_LEDGER, "E0")]
    with pytest.raises(RuntimeError, match="CGEN_GLOBAL_TRAIN_EVAL_OVERLAP"):
        assert_global_quarantine_r0(train, eval_rows)
    assert_global_quarantine_r0(train, [EvidenceIntervalR0("ETHUSDT", 200, 300, LEAGUE_EVALUATION_LEDGER, "E1")])


def test_account_continuity_exact_and_break_detected() -> None:
    post = {"account6": [0.0, 0.1, 100.0, 4.0, -0.02, 1.0], "ledger": ["x"]}
    assert_account_continuity_r0(post, dict(post))
    with pytest.raises(RuntimeError, match="CGEN_ACCOUNT_CONTINUITY_BREAK"):
        assert_account_continuity_r0(post, {**post, "account6": [1.0, 0.1, 100.0, 4.0, -0.02, 1.0]})


def test_future_poison_cannot_change_current_generation_snapshot() -> None:
    s = snap(0, h("parent_ckpt"), h("parent_policy"), tag="g0")
    assert_future_invariance_r0(current_snapshot_before=s, current_snapshot_after_future_poison=s)
    poisoned = GenerationSnapshotR0(**{**s.__dict__, "account_snapshot_sha256": h("future_poison_illegally_leaked")})
    with pytest.raises(RuntimeError, match="CGEN_FUTURE_POISON_CHANGED_CURRENT_GENERATION"):
        assert_future_invariance_r0(current_snapshot_before=s, current_snapshot_after_future_poison=poisoned)


def test_final_and_fresh_firewalls_are_hard_failures() -> None:
    s = snap(0, h("parent_ckpt"), h("parent_policy"), tag="g0")
    with pytest.raises(RuntimeError, match="CGEN_FINAL_HOLDOUT_FORBIDDEN"):
        GenerationSnapshotR0(**{**s.__dict__, "final_holdout_touched": True}).validate()
    with pytest.raises(RuntimeError, match="CGEN_FRESH_DATA_FORBIDDEN"):
        GenerationSnapshotR0(**{**s.__dict__, "fresh_market_data_downloaded": True}).validate()


def test_roster_entry_is_bound_to_exact_generation_recipe_and_ledger() -> None:
    s = snap(3, h("parent_ckpt"), h("parent_policy"), tag="g3")
    clims = {symbol: h(f"clim:{symbol}") for symbol in ("BTCUSDT", "ETHUSDT")}
    entry = freeze_roster_entry_r0(
        s,
        training_recipe_sha256=s.training_recipe_sha256,
        training_evidence_ledger_sha256=s.training_evidence_ledger_sha256,
        climatology_hashes_by_symbol=clims,
        bootstrap_seed=2026091107,
    )
    assert entry["checkpoint_sha256"] == s.child_checkpoint_sha256
    assert entry["policy_hash"] == s.child_policy_hash
    assert entry["generation_snapshot_sha256"] == s.snapshot_sha256
    assert entry["training_locked_after_schedule_freeze"] is True
    assert len(entry["roster_entry_sha256"]) == 64


def test_snapshot_hash_commits_optimizer_account_and_lineage() -> None:
    s = snap(0, h("parent_ckpt"), h("parent_policy"), tag="g0")
    changed_optimizer = GenerationSnapshotR0(**{**s.__dict__, "optimizer_snapshot_sha256": h("changed_optimizer")})
    changed_account = GenerationSnapshotR0(**{**s.__dict__, "account_snapshot_sha256": h("changed_account")})
    changed_parent = GenerationSnapshotR0(**{**s.__dict__, "parent_checkpoint_sha256": h("changed_parent")})
    assert s.snapshot_sha256 != changed_optimizer.snapshot_sha256
    assert s.snapshot_sha256 != changed_account.snapshot_sha256
    assert s.snapshot_sha256 != changed_parent.snapshot_sha256
