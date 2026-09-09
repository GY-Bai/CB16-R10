from __future__ import annotations

from dataclasses import replace
import unittest

from cb16_local_opt.r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102
from cb16_local_opt.r11_teacher_authority_candidate import (
    R11_TRAIN_TEACHER_CONFIG,
    R11_VALIDATION_TEACHER_CONFIG,
    DependenceBalancedProbabilisticTeacherR11,
)
from cb16_local_opt.teacher_balanced_runtime_r11 import (
    R11_BALANCED_SCHEDULER,
    R11_BALANCED_TEACHER_ENGINE,
)
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from tests.test_teacher_vectorized_r11 import _assert_evidence_equivalent, _configs, _fixture


def _replicate_one_train_parent(parents, samples, *, replicas: int = 16):
    source_id = sorted(
        p.parent_id for p in parents.values() if p.split == "TRAIN"
    )[3]
    source_parent = parents[source_id]
    source_rows = [s for s in samples if s.parent_id == source_id]
    out_parents = dict(parents)
    out_samples = list(samples)
    for i in range(int(replicas)):
        clone_id = f"ZZ_EXACT_REPLICA_{i:03d}:{source_id}"
        out_parents[clone_id] = replace(source_parent, parent_id=clone_id)
        out_samples.extend(replace(row, parent_id=clone_id) for row in source_rows)
    return out_parents, out_samples


class TestR11BalancedTeacherBinding(unittest.TestCase):
    def test_production_legacy_identity_rotates_to_r11_authority(self):
        parents, samples = _fixture(train_groups=40, val_groups=10, parents_per_group=2)
        train, val, stats = compile_teacher_evidence_r11(
            samples=samples,
            parents=parents,
            train_config=TRAIN_TEACHER_CONFIG_R102,
            val_config=VAL_TEACHER_CONFIG_R102,
            workers=1,
            block_targets=13,
        )
        self.assertTrue(train)
        self.assertTrue(val)
        self.assertEqual({x.teacher_version for x in train}, {R11_TRAIN_TEACHER_CONFIG.teacher_version})
        self.assertEqual({x.teacher_protocol_hash for x in train}, {R11_TRAIN_TEACHER_CONFIG.content_hash})
        self.assertEqual({x.teacher_version for x in val}, {R11_VALIDATION_TEACHER_CONFIG.teacher_version})
        self.assertEqual({x.teacher_protocol_hash for x in val}, {R11_VALIDATION_TEACHER_CONFIG.content_hash})
        self.assertNotEqual(TRAIN_TEACHER_CONFIG_R102.content_hash, R11_TRAIN_TEACHER_CONFIG.content_hash)
        self.assertNotEqual(VAL_TEACHER_CONFIG_R102.content_hash, R11_VALIDATION_TEACHER_CONFIG.content_hash)
        self.assertEqual(stats.core.engine, R11_BALANCED_TEACHER_ENGINE)
        self.assertEqual(stats.scheduler, R11_BALANCED_SCHEDULER)

    def test_explicit_custom_qualification_configs_keep_legacy_runtime_semantics(self):
        # Existing runtime qualification fixtures must remain reproducible byte-for-byte.
        parents, samples = _fixture(train_groups=20, val_groups=6, parents_per_group=2)
        train_cfg, val_cfg = _configs()
        a_train, a_val, _ = compile_teacher_evidence_r11(
            samples=samples, parents=parents,
            train_config=train_cfg, val_config=val_cfg,
            workers=1, block_targets=7,
        )
        from cb16_local_opt.teacher_scheduler_r11 import compile_teacher_evidence_threaded_r11
        b_train, b_val, _ = compile_teacher_evidence_threaded_r11(
            samples=samples, parents=parents,
            train_config=train_cfg, val_config=val_cfg,
            workers=1, block_targets=7,
        )
        self.assertEqual(a_train, b_train)
        self.assertEqual(a_val, b_val)

    def test_production_runtime_matches_direct_r11_candidate(self):
        parents, samples = _fixture(train_groups=40, val_groups=10, parents_per_group=2)
        _, val, _ = compile_teacher_evidence_r11(
            samples=samples,
            parents=parents,
            train_config=TRAIN_TEACHER_CONFIG_R102,
            val_config=VAL_TEACHER_CONFIG_R102,
            workers=1,
            block_targets=11,
        )
        teacher = DependenceBalancedProbabilisticTeacherR11(R11_VALIDATION_TEACHER_CONFIG)
        index = teacher.index(samples)
        train_groups = {p.dependence_group_id for p in parents.values() if p.split == "TRAIN"}
        runtime_by_id = {x.parent_id: x for x in val}
        for parent_id in sorted(runtime_by_id)[:6]:
            direct = teacher.compile_one(
                target_parent=parent_id,
                index=index,
                eligible_train_dependence_groups=train_groups,
            )
            _assert_evidence_equivalent(self, direct, runtime_by_id[parent_id], atol=3e-10)

    def test_production_runtime_is_exact_replica_invariant(self):
        parents, samples = _fixture(train_groups=40, val_groups=10, parents_per_group=2)
        replica_parents, replica_samples = _replicate_one_train_parent(parents, samples, replicas=16)
        base_train, base_val, _ = compile_teacher_evidence_r11(
            samples=samples, parents=parents,
            train_config=TRAIN_TEACHER_CONFIG_R102,
            val_config=VAL_TEACHER_CONFIG_R102,
            workers=1, block_targets=13,
        )
        replica_train, replica_val, _ = compile_teacher_evidence_r11(
            samples=replica_samples, parents=replica_parents,
            train_config=TRAIN_TEACHER_CONFIG_R102,
            val_config=VAL_TEACHER_CONFIG_R102,
            workers=1, block_targets=13,
        )
        base = {x.parent_id: x for x in base_train + base_val}
        replica = {x.parent_id: x for x in replica_train + replica_val}
        self.assertTrue(set(base).issubset(replica))
        for parent_id in sorted(base):
            self.assertEqual(base[parent_id], replica[parent_id], parent_id)

    def test_authoritative_worker_topology_is_semantically_invariant(self):
        parents, samples = _fixture(train_groups=40, val_groups=10, parents_per_group=2)
        reference = None
        for workers in (1, 4):
            train, val, stats = compile_teacher_evidence_r11(
                samples=samples, parents=parents,
                train_config=TRAIN_TEACHER_CONFIG_R102,
                val_config=VAL_TEACHER_CONFIG_R102,
                workers=workers, block_targets=9,
            )
            if reference is None:
                reference = (train, val)
            else:
                self.assertEqual(reference, (train, val))
            self.assertEqual(stats.core.engine, R11_BALANCED_TEACHER_ENGINE)
            self.assertEqual(stats.scheduler, R11_BALANCED_SCHEDULER)


if __name__ == "__main__":
    unittest.main()
