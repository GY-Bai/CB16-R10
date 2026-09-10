from __future__ import annotations

from cb16_local_opt.independent_purge_alignment_replication_r5 import (
    R5_SHIFTS,
    adjudicate_alignment_replication_r5,
    compile_validation_targets_only_r5,
)
from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.r102_evidence_cache import ParentContextR102
from cb16_local_opt.r11_teacher_authority_candidate import (
    R11_TRAIN_TEACHER_CONFIG,
    R11_VALIDATION_TEACHER_CONFIG,
)
from cb16_local_opt.teacher_balanced_runtime_r11 import compile_teacher_evidence_balanced_r11


GRID = (
    (-1, 0.25), (-1, 0.50), (-1, 0.75), (-1, 1.00),
    (0, 0.00),
    (1, 0.25), (1, 0.50), (1, 0.75), (1, 1.00),
)


def _make_support(train_groups: int = 40):
    parents: dict[str, ParentContextR102] = {}
    samples: list[CounterfactualBranchSampleR5] = []
    base_t = 1_700_000_000_000
    step = 256 * 3_600_000
    total = train_groups + 1
    for g in range(total):
        split = "TRAIN" if g < train_groups else "VALIDATION"
        t = base_t + g * step
        gid = f"FUT:TEST:{t}"
        pid = f"P:TEST:{t}:SYNTH"
        # Nondegenerate but nearby feature geometry. The validation target follows
        # the same smooth manifold as historical support so admission is expected.
        scalar = g / max(train_groups, 1)
        op = tuple(float(scalar + 0.001 * j) for j in range(48))
        med = tuple(float(0.5 * scalar + 0.002 * j) for j in range(48))
        acc = tuple(float((-0.1 + 0.04 * j) + 0.1 * scalar) for j in range(6))
        parent = ParentContextR102(
            parent_id=pid,
            dependence_group_id=gid,
            symbol="BTCUSDT",
            decision_time_ms=t,
            split=split,
            scenario="SYNTH",
            operator48=op,
            medium48=med,
            account6=acc,
            ordered4h30=tuple(0.0 for _ in range(30)),
            current_mark=100.0,
            snapshot_sha256=f"{g:064x}"[-64:],
            eligible_for_economic_evidence=True,
            market_lineage_hash=f"MKT:{g}",
        )
        parents[pid] = parent
        for direction, risk in GRID:
            # Utility changes across both future group and action while remaining finite.
            utility = 0.0005 * g + 0.0015 * direction - 0.0004 * risk + 0.0002 * direction * risk
            row = CounterfactualBranchSampleR5(
                parent_id=pid,
                student_context_object_id=parent.student_context_object_id,
                timestamp=t,
                context_features=parent.student_features,
                direction=direction,
                requested_risk=risk,
                realized_utility=float(utility),
                dependence_group_id=gid,
                market_lineage_hash=parent.market_lineage_hash,
            )
            row.validate()
            samples.append(row)
    validation_id = f"P:TEST:{base_t + train_groups * step}:SYNTH"
    return parents, samples, validation_id


def test_target_only_teacher_is_content_identical_to_full_r24_compiler():
    parents, samples, validation_id = _make_support()
    full_train, full_validation, _ = compile_teacher_evidence_balanced_r11(
        samples=samples,
        parents=parents,
        train_config=R11_TRAIN_TEACHER_CONFIG,
        val_config=R11_VALIDATION_TEACHER_CONFIG,
        workers=1,
        block_targets=16,
    )
    assert len(full_train) == 40
    assert len(full_validation) == 1
    assert full_validation[0].parent_id == validation_id

    target_only, receipt = compile_validation_targets_only_r5(
        samples=samples,
        parents=parents,
        target_parent_ids=[validation_id],
        block_targets=16,
    )
    assert len(target_only) == 1
    assert target_only[0].content_hash == full_validation[0].content_hash
    assert target_only[0] == full_validation[0]
    assert receipt["execution_pruning_only"] is True
    assert receipt["teacher_semantics_changed"] is False
    assert receipt["teacher_protocol_hash"] == R11_VALIDATION_TEACHER_CONFIG.content_hash
    assert receipt["legacy_train_dependence_groups"] == 40
    assert target_only[0].admission.admitted


def _metrics(loss: float, direction: float, sizing: float = 0.01):
    return {"loss": loss, "direction_loss": direction, "sizing_loss": sizing}


def test_r5_positive_rule_requires_aligned_to_beat_champion_and_all_shuffles_on_total_and_direction():
    shuffled = {k: _metrics(1.10 + 0.001 * i, 1.05 + 0.001 * i) for i, k in enumerate(R5_SHIFTS)}
    out = adjudicate_alignment_replication_r5(
        champion=_metrics(1.20, 1.15),
        aligned=_metrics(1.00, 0.95),
        shuffled=shuffled,
    )
    assert out["mechanistic_alignment_replication_supported"] is True
    assert out["aligned_lower_total_count_vs_shuffles"] == 5
    assert out["aligned_lower_direction_count_vs_shuffles"] == 5
    assert out["canonical_promotion_authorized"] is False
    assert out["market_information_verdict_reopened"] is False


def test_r5_mixed_or_direction_failure_is_negative_even_if_total_is_good():
    shuffled = {k: _metrics(1.10 + 0.001 * i, 0.90 + 0.001 * i) for i, k in enumerate(R5_SHIFTS)}
    out = adjudicate_alignment_replication_r5(
        champion=_metrics(1.20, 1.15),
        aligned=_metrics(1.00, 0.95),
        shuffled=shuffled,
    )
    assert out["aligned_lower_total_count_vs_shuffles"] == 5
    assert out["aligned_lower_direction_count_vs_shuffles"] == 0
    assert out["mechanistic_alignment_replication_supported"] is False
    assert out["conclusion"] == "STATE_ALIGNMENT_NOT_REPLICATED_ON_INDEPENDENT_PURGE_SUPPORT"
