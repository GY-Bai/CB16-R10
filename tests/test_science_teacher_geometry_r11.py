from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.probabilistic_teacher_r6 import DependenceAwareTeacherConfigR6
from cb16_local_opt.science_teacher_geometry_r11 import (
    compare_teacher_evidence_sets_r11,
    compile_teacher_evidence_dependence_balanced_shadow_r11,
)
from cb16_local_opt.teacher_vectorized_r11 import compile_teacher_evidence_vectorized_r11


GRID = ((-1, 0.25), (-1, 0.50), (-1, 0.75), (-1, 1.0), (0, 0.0),
        (1, 0.25), (1, 0.50), (1, 0.75), (1, 1.0))


def _configs():
    common = dict(
        n_folds=3,
        embargo_groups=0,
        k_dependence_groups=5,
        min_train_dependence_groups=2,
        min_effective_dependence_n=1.0,
        max_nearest_distance=100.0,
        distance_temperature=2.0,
        direction_softmax_temperature=0.01,
        quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
    )
    train = DependenceAwareTeacherConfigR6(
        teacher_version="TEST_PREQ", mode="PREQUENTIAL", lane="CENTER", **common
    )
    val = DependenceAwareTeacherConfigR6(
        teacher_version="TEST_PREQ_VAL", mode="PREQUENTIAL", lane="CENTER", **common
    )
    return train, val


def _fixture(groups: int = 8):
    parents = {}
    samples = []
    for g in range(groups):
        dep = f"G{g:02d}"
        timestamp = 1000 + g
        for lane, offset in (("A", -0.6), ("B", 0.8)):
            pid = f"P{g:02d}{lane}"
            context_id = f"CTX{g:02d}{lane}"
            feature = (float(g) + offset, float(g * g) * 0.1 + (0.2 if lane == "A" else -0.3))
            parents[pid] = SimpleNamespace(
                parent_id=pid,
                split="TRAIN",
                dependence_group_id=dep,
            )
            for j, (direction, risk) in enumerate(GRID):
                utility = (
                    0.001 * g
                    + 0.0004 * direction
                    - 0.0001 * risk
                    + 0.00003 * j
                    + (0.0002 if lane == "A" else -0.00015)
                )
                samples.append(CounterfactualBranchSampleR5(
                    parent_id=pid,
                    student_context_object_id=context_id,
                    timestamp=timestamp,
                    context_features=feature,
                    direction=direction,
                    requested_risk=risk,
                    realized_utility=utility,
                    dependence_group_id=dep,
                    market_lineage_hash=f"M{g:02d}",
                ))
    return parents, samples


def _replicate_exact_context(parents, samples, *, source="P01A", replicas=20):
    out_parents = dict(parents)
    out_samples = list(samples)
    source_parent = parents[source]
    source_rows = [x for x in samples if x.parent_id == source]
    assert len(source_rows) == 9
    for i in range(replicas):
        pid = f"ZZ_REPLICA_{i:03d}_{source}"
        out_parents[pid] = SimpleNamespace(
            parent_id=pid,
            split=source_parent.split,
            dependence_group_id=source_parent.dependence_group_id,
        )
        # Exact same Student context identity + exact same features/utilities.
        out_samples.extend(replace(x, parent_id=pid) for x in source_rows)
    return out_parents, out_samples


def test_dependence_balanced_shadow_is_invariant_to_exact_same_future_context_replicas():
    parents, samples = _fixture()
    replica_parents, replica_samples = _replicate_exact_context(parents, samples)
    train_cfg, val_cfg = _configs()

    base, _, _, _ = compile_teacher_evidence_dependence_balanced_shadow_r11(
        samples=samples, parents=parents,
        train_config=train_cfg, val_config=val_cfg, block_targets=4,
    )
    replica, _, _, _ = compile_teacher_evidence_dependence_balanced_shadow_r11(
        samples=replica_samples, parents=replica_parents,
        train_config=train_cfg, val_config=val_cfg, block_targets=4,
    )
    base_target = [x for x in base if x.parent_id == "P07A"]
    replica_target = [x for x in replica if x.parent_id == "P07A"]
    assert len(base_target) == len(replica_target) == 1
    assert base_target[0].content_hash == replica_target[0].content_hash
    comparison = compare_teacher_evidence_sets_r11(base_target, replica_target)
    assert comparison["content_hash_changed_targets"] == 0
    assert comparison["maximum_action_mean_utility_abs_delta"] == 0.0
    assert comparison["maximum_action_quantile_abs_delta"] == 0.0


def test_current_parent_weighted_geometry_can_move_under_exact_replica_volume():
    parents, samples = _fixture()
    replica_parents, replica_samples = _replicate_exact_context(parents, samples)
    train_cfg, val_cfg = _configs()

    base, _, _ = compile_teacher_evidence_vectorized_r11(
        samples=samples, parents=parents,
        train_config=train_cfg, val_config=val_cfg, block_targets=4,
    )
    replica, _, _ = compile_teacher_evidence_vectorized_r11(
        samples=replica_samples, parents=replica_parents,
        train_config=train_cfg, val_config=val_cfg, block_targets=4,
    )
    a = [x for x in base if x.parent_id == "P07A"][0]
    b = [x for x in replica if x.parent_id == "P07A"][0]
    assert a.content_hash != b.content_hash


def test_shadow_comparison_reports_semantic_delta_without_declaring_authority():
    parents, samples = _fixture()
    train_cfg, val_cfg = _configs()
    current, _, _ = compile_teacher_evidence_vectorized_r11(
        samples=samples, parents=parents,
        train_config=train_cfg, val_config=val_cfg, block_targets=4,
    )
    shadow, _, _, stats = compile_teacher_evidence_dependence_balanced_shadow_r11(
        samples=samples, parents=parents,
        train_config=train_cfg, val_config=val_cfg, block_targets=4,
    )
    comparison = compare_teacher_evidence_sets_r11(current, shadow)
    assert comparison["targets"] == len(current)
    assert stats["support_regime_dependence_groups"] > 0
    assert stats["support_regime_unique_context_rows"] > 0
