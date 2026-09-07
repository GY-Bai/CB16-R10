from __future__ import annotations

import math
import unittest

import numpy as np

from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5, weighted_quantile
from cb16_local_opt.probabilistic_teacher_r6 import (
    DependenceAwareProbabilisticTeacherR6,
    DependenceAwareTeacherConfigR6,
)
from cb16_local_opt.r102_evidence_cache import ParentContextR102
from cb16_local_opt.teacher_vectorized_r11 import (
    _weighted_quantiles_batch_r11,
    compile_teacher_evidence_vectorized_r11,
)


GRID = ((0, 0.0),) + tuple((d, r) for d in (-1, 1) for r in (0.25, 0.50, 0.75, 1.0))


def _fixture(*, train_groups: int = 40, val_groups: int = 10, parents_per_group: int = 2):
    rng = np.random.default_rng(24680)
    parents = {}
    samples = []
    n_groups = train_groups + val_groups
    for gi in range(n_groups):
        split = "TRAIN" if gi < train_groups else "VALIDATION"
        dep = f"FUT:SYM:{gi:05d}"
        timestamp = 1_700_000_000_000 + gi * 3_600_000
        group_latent = rng.normal(0.0, 1.0, size=102)
        for pi in range(parents_per_group):
            parent_id = f"P:SYM:{gi:05d}:S{pi}"
            # Deliberately non-symmetric features avoid accidental distance ties while
            # preserving multiple account contexts sharing one future dependence group.
            feat = group_latent + rng.normal(0.0, 0.12, size=102) + pi * 0.0031
            op = tuple(float(x) for x in feat[:48])
            med = tuple(float(x) for x in feat[48:96])
            acc = tuple(float(x) for x in feat[96:102])
            p = ParentContextR102(
                parent_id=parent_id,
                dependence_group_id=dep,
                symbol="SYM",
                decision_time_ms=timestamp,
                split=split,
                scenario=f"S{pi}",
                operator48=op,
                medium48=med,
                account6=acc,
                ordered4h30=tuple(float(x) for x in np.zeros(30, dtype=np.float64)),
                current_mark=100.0 + gi,
                snapshot_sha256=(f"{gi:04x}{pi:02x}" * 11)[:64].ljust(64, "0"),
                eligible_for_economic_evidence=True,
                market_lineage_hash=(f"{gi:08x}" * 8)[:64],
            )
            parents[parent_id] = p
            # Utility is deterministic but nonlinear in context/action so nearest-parent
            # selection and all nine action laws are exercised.
            base = 0.0008 * math.tanh(float(feat[0]) + 0.2 * float(feat[17]))
            for direction, risk in GRID:
                utility = (
                    base
                    + 0.00031 * direction
                    + 0.00019 * risk * direction
                    - 0.00007 * risk * risk
                    + 0.000011 * gi
                    + 0.000003 * pi
                )
                samples.append(CounterfactualBranchSampleR5(
                    parent_id=parent_id,
                    student_context_object_id=p.student_context_object_id,
                    timestamp=timestamp,
                    context_features=p.student_features,
                    direction=int(direction),
                    requested_risk=float(risk),
                    realized_utility=float(utility),
                    dependence_group_id=dep,
                    market_lineage_hash=p.market_lineage_hash,
                ))
    return parents, samples


def _configs():
    train = DependenceAwareTeacherConfigR6(
        teacher_version="TEST_R11_BLOCKED",
        mode="BLOCKED_CROSSFIT",
        n_folds=5,
        embargo_groups=1,
        k_dependence_groups=16,
        min_train_dependence_groups=8,
        min_effective_dependence_n=4.0,
        max_nearest_distance=20.0,
        distance_temperature=2.0,
        direction_softmax_temperature=0.002,
        lane="CENTER",
    )
    val = DependenceAwareTeacherConfigR6(
        teacher_version="TEST_R11_PREQUENTIAL",
        mode="PREQUENTIAL",
        n_folds=5,
        embargo_groups=0,
        k_dependence_groups=16,
        min_train_dependence_groups=8,
        min_effective_dependence_n=4.0,
        max_nearest_distance=20.0,
        distance_temperature=2.0,
        direction_softmax_temperature=0.002,
        lane="CENTER",
    )
    return train, val


def _legacy_compile(samples, parents, train_cfg, val_cfg):
    index = DependenceAwareProbabilisticTeacherR6.index(samples)
    train_groups = {p.dependence_group_id for p in parents.values() if p.split == "TRAIN"}
    out = []
    for cfg, split in ((train_cfg, "TRAIN"), (val_cfg, "VALIDATION")):
        teacher = DependenceAwareProbabilisticTeacherR6(cfg)
        ids = sorted(p.parent_id for p in parents.values() if p.split == split)
        for parent_id in ids:
            out.append(teacher.compile_one(
                target_parent=parent_id,
                index=index,
                eligible_train_dependence_groups=train_groups,
            ))
    return {e.parent_id: e for e in out}


def _assert_evidence_equivalent(tc: unittest.TestCase, old, new, *, atol: float = 2e-10):
    tc.assertEqual(old.evidence_id, new.evidence_id)
    tc.assertEqual(old.parent_id, new.parent_id)
    tc.assertEqual(old.student_context_object_id, new.student_context_object_id)
    tc.assertEqual(old.target_dependence_group_id, new.target_dependence_group_id)
    tc.assertEqual(old.timestamp, new.timestamp)
    tc.assertEqual(old.teacher_version, new.teacher_version)
    tc.assertEqual(old.teacher_protocol_hash, new.teacher_protocol_hash)
    tc.assertEqual(old.train_dependence_group_hash, new.train_dependence_group_hash)
    tc.assertEqual(old.admission.status, new.admission.status)
    tc.assertEqual(old.admission.lane, new.admission.lane)
    tc.assertEqual(old.admission.unique_train_dependence_groups, new.admission.unique_train_dependence_groups)
    tc.assertEqual(old.admission.reasons, new.admission.reasons)
    tc.assertEqual(old.admission.protocol_hash, new.admission.protocol_hash)
    tc.assertAlmostEqual(
        old.admission.minimum_action_effective_dependence_n,
        new.admission.minimum_action_effective_dependence_n,
        delta=atol,
    )
    if math.isfinite(old.admission.maximum_action_nearest_distance):
        tc.assertAlmostEqual(
            old.admission.maximum_action_nearest_distance,
            new.admission.maximum_action_nearest_distance,
            delta=atol,
        )
    else:
        tc.assertEqual(old.admission.maximum_action_nearest_distance, new.admission.maximum_action_nearest_distance)

    tc.assertEqual(len(old.action_laws), len(new.action_laws))
    for a, b in zip(old.action_laws, new.action_laws):
        tc.assertEqual((a.direction, a.requested_risk), (b.direction, b.requested_risk))
        tc.assertEqual(a.quantile_levels, b.quantile_levels)
        tc.assertEqual(a.unique_dependence_groups, b.unique_dependence_groups)
        tc.assertEqual(a.support_dependence_group_hash, b.support_dependence_group_hash)
        for av, bv in (
            (a.mean_utility, b.mean_utility),
            (a.std_utility, b.std_utility),
            (a.effective_dependence_n, b.effective_dependence_n),
            (a.nearest_distance, b.nearest_distance),
            (a.max_distance_used, b.max_distance_used),
        ):
            tc.assertAlmostEqual(av, bv, delta=atol)
        np.testing.assert_allclose(a.quantiles, b.quantiles, rtol=0.0, atol=atol)

    np.testing.assert_allclose(old.direction_target_probs, new.direction_target_probs, rtol=0.0, atol=atol)
    tc.assertAlmostEqual(old.requested_risk_target, new.requested_risk_target, delta=atol)
    tc.assertEqual(old.direction_weight, new.direction_weight)
    tc.assertEqual(old.sizing_weight, new.sizing_weight)


class TestR11VectorizedTeacher(unittest.TestCase):
    def test_vectorized_teacher_matches_legacy_oracle(self):
        parents, samples = _fixture()
        train_cfg, val_cfg = _configs()
        legacy = _legacy_compile(samples, parents, train_cfg, val_cfg)
        train, val, stats = compile_teacher_evidence_vectorized_r11(
            samples=samples,
            parents=parents,
            train_config=train_cfg,
            val_config=val_cfg,
            block_targets=11,
        )
        new = {e.parent_id: e for e in train + val}
        self.assertEqual(set(legacy), set(new))
        for parent_id in sorted(legacy):
            _assert_evidence_equivalent(self, legacy[parent_id], new[parent_id])
        self.assertEqual(stats.targets, len(legacy))
        self.assertEqual(stats.actions, 9)
        self.assertEqual(stats.feature_dim, 102)
        # Blocked cross-fit creates only a small number of shared regimes; validation
        # targets share their exact prequential support in this fixture.
        self.assertLessEqual(stats.support_regimes, 7)

    def test_weighted_quantile_batch_matches_scalar_oracle(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(5, 17, 9))
        weights = rng.random(size=(5, 17)) + 1e-3
        weights /= weights.sum(axis=1, keepdims=True)
        qs = np.asarray((0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95))
        got = _weighted_quantiles_batch_r11(values, weights, qs)
        for b in range(values.shape[0]):
            for a in range(values.shape[2]):
                want = weighted_quantile(values[b, :, a], weights[b], qs)
                np.testing.assert_allclose(got[b, a], want, rtol=0.0, atol=2e-15)

    def test_single_support_quantile_is_constant(self):
        values = np.asarray([[[1.5, -2.0, 7.25]]], dtype=np.float64)  # [B=1,K=1,A=3]
        weights = np.ones((1, 1), dtype=np.float64)
        qs = (0.05, 0.5, 0.95)
        got = _weighted_quantiles_batch_r11(values, weights, qs)
        self.assertEqual(got.shape, (1, 3, 3))
        np.testing.assert_array_equal(got[0, 0], np.asarray([1.5, 1.5, 1.5]))
        np.testing.assert_array_equal(got[0, 1], np.asarray([-2.0, -2.0, -2.0]))
        np.testing.assert_array_equal(got[0, 2], np.asarray([7.25, 7.25, 7.25]))


if __name__ == "__main__":
    unittest.main()
