from __future__ import annotations

import hashlib
import inspect
import unittest
from unittest import mock

from cb16_local_opt import teacher_scheduler_r11 as scheduler
from cb16_local_opt.cpu_runtime_r11 import CANONICAL_TEACHER_WORKERS_R11
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.teacher_scheduler_r11 import TeacherWorkerPoolR11
from tests.test_teacher_vectorized_r11 import _configs, _fixture


def _evidence_batch_hash(train, val) -> str:
    payload = "TRAIN\n" + "\n".join(x.content_hash for x in train)
    payload += "\nVALIDATION\n" + "\n".join(x.content_hash for x in val)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


class TestR11Stage2TeacherPool(unittest.TestCase):
    def _compile(self, parents, samples, *, workers, pool=None):
        train_cfg, val_cfg = _configs()
        return compile_teacher_evidence_r11(samples=samples, parents=parents, train_config=train_cfg, val_config=val_cfg, workers=workers, block_targets=5, worker_pool=pool)

    def test_canonical_runtime_default_is_eight_workers(self):
        default = inspect.signature(compile_teacher_evidence_r11).parameters["workers"].default
        self.assertEqual(default, CANONICAL_TEACHER_WORKERS_R11)
        self.assertEqual(default, 8)

    def test_worker_count_does_not_change_teacher_evidence_hash(self):
        parents, samples = _fixture(train_groups=20, val_groups=6, parents_per_group=2)
        hashes = []
        for workers in (1, 4, 8, 12):
            train, val, stats = self._compile(parents, samples, workers=workers)
            hashes.append(_evidence_batch_hash(train, val))
            self.assertFalse(stats.topology_in_scientific_identity)
        self.assertEqual(len(set(hashes)), 1)

    def test_reusable_pool_reuses_threads_without_reusing_scientific_state(self):
        parents, samples = _fixture(train_groups=20, val_groups=6, parents_per_group=2)
        with TeacherWorkerPoolR11(max_workers=8) as pool:
            executor_id = pool.executor_identity
            a_train, a_val, a_stats = self._compile(parents, samples, workers=8, pool=pool)
            b_train, b_val, b_stats = self._compile(parents, samples, workers=8, pool=pool)
            self.assertEqual(pool.executor_identity, executor_id)
            self.assertEqual(pool.executions, 2)
            self.assertTrue(a_stats.persistent_pool_supplied)
            self.assertTrue(b_stats.persistent_pool_supplied)
            self.assertLessEqual(a_stats.max_in_flight, 8)
            self.assertLessEqual(b_stats.max_in_flight, 8)
        self.assertEqual(_evidence_batch_hash(a_train, a_val), _evidence_batch_hash(b_train, b_val))
        self.assertEqual(a_train, b_train)
        self.assertEqual(a_val, b_val)

    def test_worker_exception_drains_pool_then_clean_compile_has_no_stale_state(self):
        parents, samples = _fixture(train_groups=20, val_groups=6, parents_per_group=2)
        original = scheduler._compile_block_r11
        faulted = False
        def fail_once(**kwargs):
            nonlocal faulted
            if not faulted:
                faulted = True
                raise ValueError("synthetic-stage2-worker-fault")
            return original(**kwargs)
        with TeacherWorkerPoolR11(max_workers=8) as pool:
            with mock.patch.object(scheduler, "_compile_block_r11", side_effect=fail_once):
                with self.assertRaisesRegex(RuntimeError, r"R11_TEACHER_WORKER_EXCEPTION:BLOCK="):
                    self._compile(parents, samples, workers=8, pool=pool)
            clean_train, clean_val, _ = self._compile(parents, samples, workers=8, pool=pool)
        fresh_train, fresh_val, _ = self._compile(parents, samples, workers=8)
        self.assertEqual(clean_train, fresh_train)
        self.assertEqual(clean_val, fresh_val)
        self.assertEqual(_evidence_batch_hash(clean_train, clean_val), _evidence_batch_hash(fresh_train, fresh_val))


if __name__ == "__main__":
    unittest.main()
