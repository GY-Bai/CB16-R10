from __future__ import annotations

import threading
import unittest
from unittest import mock

from cb16_local_opt import teacher_scheduler_r11 as scheduler
from cb16_local_opt.teacher_runtime_r11 import (
    SUPPORTED_QUALIFICATION_WORKERS_R11,
    compile_teacher_evidence_r11,
)
from cb16_local_opt.teacher_vectorized_r11 import compile_teacher_evidence_vectorized_r11
from tests.test_teacher_vectorized_r11 import _configs, _fixture


class TestR11TeacherRuntime(unittest.TestCase):
    def _compile(self, *, parents, samples, workers, block_targets=7):
        train_cfg, val_cfg = _configs()
        return compile_teacher_evidence_r11(
            samples=samples,
            parents=parents,
            train_config=train_cfg,
            val_config=val_cfg,
            workers=workers,
            block_targets=block_targets,
        )

    def test_workers_1_4_8_12_are_evidence_identical(self):
        parents, samples = _fixture(train_groups=20, val_groups=6, parents_per_group=2)
        reference = None
        for workers in SUPPORTED_QUALIFICATION_WORKERS_R11:
            train, val, stats = self._compile(
                parents=parents,
                samples=samples,
                workers=workers,
            )
            content = (train, val)
            if reference is None:
                reference = content
            else:
                self.assertEqual(content, reference)
            self.assertEqual(stats.workers, workers)
            self.assertEqual(stats.nested_blas_threads_required, 1)
            self.assertTrue(stats.nested_blas_limit_enforced)
            self.assertFalse(stats.topology_in_scientific_identity)

    def test_runtime_preserves_vectorized_support_and_admission_semantics(self):
        parents, samples = _fixture(train_groups=20, val_groups=6, parents_per_group=2)
        train_cfg, val_cfg = _configs()
        expected_train, expected_val, _ = compile_teacher_evidence_vectorized_r11(
            samples=samples,
            parents=parents,
            train_config=train_cfg,
            val_config=val_cfg,
            block_targets=7,
        )
        got_train, got_val, _ = compile_teacher_evidence_r11(
            samples=samples,
            parents=parents,
            train_config=train_cfg,
            val_config=val_cfg,
            workers=8,
            block_targets=7,
        )
        self.assertEqual(got_train, expected_train)
        self.assertEqual(got_val, expected_val)
        self.assertEqual(
            [(x.parent_id, x.admission, x.direction_weight, x.sizing_weight) for x in got_train + got_val],
            [(x.parent_id, x.admission, x.direction_weight, x.sizing_weight) for x in expected_train + expected_val],
        )

    def test_shuffled_completion_reassembles_canonical_parent_order(self):
        parents, samples = _fixture(train_groups=20, val_groups=6, parents_per_group=2)
        original = scheduler._compile_block_r11
        release_first = threading.Event()
        finished: list[str] = []
        finished_lock = threading.Lock()

        def reordered_completion(**kwargs):
            ids = tuple(kwargs["target_parent_ids"])
            is_first_canonical_block = "P:SYM:00000:S0" in ids
            if is_first_canonical_block:
                release_first.wait()
            rows = original(**kwargs)
            with finished_lock:
                finished.append(ids[0])
            if not is_first_canonical_block:
                release_first.set()
            return rows

        with mock.patch.object(scheduler, "_compile_block_r11", side_effect=reordered_completion):
            train, val, _ = self._compile(
                parents=parents,
                samples=samples,
                workers=4,
                block_targets=3,
            )

        expected_train = sorted(p.parent_id for p in parents.values() if p.split == "TRAIN")
        expected_val = sorted(p.parent_id for p in parents.values() if p.split == "VALIDATION")
        self.assertEqual([x.parent_id for x in train], expected_train)
        self.assertEqual([x.parent_id for x in val], expected_val)
        self.assertTrue(finished)
        self.assertNotEqual(finished[0], "P:SYM:00000:S0")

    def test_workers_share_one_readonly_columnar_index(self):
        parents, samples = _fixture(train_groups=16, val_groups=4, parents_per_group=2)
        original = scheduler._compile_block_r11
        index_ids: set[int] = set()
        lock = threading.Lock()

        def inspect_shared_index(**kwargs):
            index = kwargs["index"]
            with lock:
                index_ids.add(id(index))
            for name in scheduler._INDEX_ARRAY_FIELDS:
                self.assertFalse(getattr(index, name).flags.writeable)
            with self.assertRaises(TypeError):
                index.parent_row_by_id["FORBIDDEN"] = 0
            return original(**kwargs)

        with mock.patch.object(scheduler, "_compile_block_r11", side_effect=inspect_shared_index):
            self._compile(parents=parents, samples=samples, workers=8, block_targets=4)
        self.assertEqual(len(index_ids), 1)

    def test_worker_exception_fails_whole_compile_closed(self):
        parents, samples = _fixture(train_groups=16, val_groups=4, parents_per_group=2)
        with mock.patch.object(scheduler, "_compile_block_r11", side_effect=ValueError("synthetic worker fault")):
            with self.assertRaisesRegex(RuntimeError, r"R11_TEACHER_WORKER_EXCEPTION:BLOCK="):
                self._compile(parents=parents, samples=samples, workers=4, block_targets=4)

    def test_incomplete_block_fails_closed(self):
        parents, samples = _fixture(train_groups=16, val_groups=4, parents_per_group=2)
        original = scheduler._compile_block_r11

        def drop_one(**kwargs):
            rows = original(**kwargs)
            if len(rows) > 1:
                return rows[:-1]
            return rows

        with mock.patch.object(scheduler, "_compile_block_r11", side_effect=drop_one):
            with self.assertRaisesRegex(RuntimeError, r"R11_INCOMPLETE_BLOCK"):
                self._compile(parents=parents, samples=samples, workers=4, block_targets=4)

    def test_output_ordering_drift_fails_closed(self):
        parents, samples = _fixture(train_groups=16, val_groups=4, parents_per_group=2)
        original = scheduler._compile_block_r11

        def reverse_one(**kwargs):
            rows = original(**kwargs)
            if len(rows) > 1:
                return list(reversed(rows))
            return rows

        with mock.patch.object(scheduler, "_compile_block_r11", side_effect=reverse_one):
            with self.assertRaisesRegex(RuntimeError, r"R11_OUTPUT_ORDERING_DRIFT"):
                self._compile(parents=parents, samples=samples, workers=4, block_targets=4)

    def test_missing_target_fails_closed(self):
        parents, samples = _fixture(train_groups=16, val_groups=4, parents_per_group=2)
        missing = sorted(parents)[0]
        reduced_samples = [sample for sample in samples if sample.parent_id != missing]
        with self.assertRaisesRegex(RuntimeError, rf"R11_MISSING_TARGET:{missing}"):
            self._compile(parents=parents, samples=reduced_samples, workers=1)

    def test_duplicate_target_fails_closed(self):
        parents, samples = _fixture(train_groups=16, val_groups=4, parents_per_group=2)
        duplicate_parent = parents[sorted(parents)[0]]
        duplicated_mapping = dict(parents)
        duplicated_mapping["ALIAS_FOR_DUPLICATE"] = duplicate_parent
        with self.assertRaisesRegex(RuntimeError, rf"R11_DUPLICATE_TARGET:{duplicate_parent.parent_id}"):
            self._compile(parents=duplicated_mapping, samples=samples, workers=1)

    def test_repeated_invocation_is_deterministic(self):
        parents, samples = _fixture(train_groups=16, val_groups=4, parents_per_group=2)
        a_train, a_val, _ = self._compile(parents=parents, samples=samples, workers=8)
        b_train, b_val, _ = self._compile(parents=parents, samples=samples, workers=8)
        self.assertEqual(a_train, b_train)
        self.assertEqual(a_val, b_val)

    def test_runtime_requires_blas_control(self):
        parents, samples = _fixture(train_groups=16, val_groups=4, parents_per_group=2)
        with mock.patch.object(scheduler, "threadpool_limits", None):
            with self.assertRaisesRegex(RuntimeError, "R11_BLAS_SINGLE_THREAD_CONTROL_UNAVAILABLE"):
                self._compile(parents=parents, samples=samples, workers=4)
        with mock.patch.object(scheduler, "threadpool_limits", None):
            with self.assertRaisesRegex(RuntimeError, "R11_BLAS_SINGLE_THREAD_CONTROL_UNAVAILABLE"):
                self._compile(parents=parents, samples=samples, workers=1)


if __name__ == "__main__":
    unittest.main()
