from __future__ import annotations

import unittest

from cb16_local_opt.cpu_runtime_r11 import (
    CANONICAL_BLAS_THREADS_R11,
    CANONICAL_TEACHER_WORKERS_R11,
    CANONICAL_TRACE_WORKERS_R11,
    CpuExecutionBudgetR11,
)


class TestR11Stage2CpuExecutionBudget(unittest.TestCase):
    def _shanxi(self) -> CpuExecutionBudgetR11:
        return CpuExecutionBudgetR11(logical_cpus=16, physical_cores_estimate=8, teacher_workers=8, trace_workers=8, storage_background_threads=1, blas_threads=1)

    def test_canonical_shanxi_policy_is_8_8_blas1_without_oversubscription(self):
        budget = self._shanxi()
        self.assertEqual(CANONICAL_TEACHER_WORKERS_R11, 8)
        self.assertEqual(CANONICAL_TRACE_WORKERS_R11, 8)
        self.assertEqual(CANONICAL_BLAS_THREADS_R11, 1)
        teacher = budget.allocation_for("TEACHER")
        self.assertEqual((teacher.teacher_workers, teacher.trace_workers), (8, 0))
        self.assertEqual(teacher.storage_background_threads, 1)
        self.assertLessEqual(teacher.total_scheduled_threads, 16)
        self.assertFalse(teacher.oversubscribed)
        trace = budget.allocation_for("TRACE")
        self.assertEqual((trace.teacher_workers, trace.trace_workers), (0, 8))
        self.assertEqual(trace.storage_background_threads, 1)
        self.assertLessEqual(trace.total_scheduled_threads, 16)
        self.assertFalse(trace.oversubscribed)

    def test_requested_workers_are_clamped_to_physical_compute_capacity(self):
        budget = CpuExecutionBudgetR11(logical_cpus=8, physical_cores_estimate=4, teacher_workers=12, trace_workers=12, storage_background_threads=2)
        self.assertEqual(budget.teacher_workers, 4)
        self.assertEqual(budget.trace_workers, 4)
        for stage in ("TEACHER", "TRACE"):
            allocation = budget.allocation_for(stage)
            self.assertEqual(allocation.compute_worker_threads, 4)
            self.assertLessEqual(allocation.total_scheduled_threads, 8)
            self.assertFalse(allocation.oversubscribed)

    def test_nested_blas_threads_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "R11_CPU_BLAS_THREADS_MUST_BE_ONE"):
            CpuExecutionBudgetR11(logical_cpus=16, physical_cores_estimate=8, blas_threads=2)

    def test_stage_transition_releases_then_trace_borrows_budget(self):
        budget = self._shanxi()
        with budget.stage_lease("TEACHER") as teacher:
            self.assertEqual(budget.snapshot(), teacher)
            self.assertEqual((teacher.teacher_workers, teacher.trace_workers), (8, 0))
        self.assertEqual(budget.snapshot().total_scheduled_threads, 0)
        with budget.stage_lease("TRACE") as trace:
            self.assertEqual(budget.snapshot(), trace)
            self.assertEqual((trace.teacher_workers, trace.trace_workers), (0, 8))
        self.assertEqual(budget.snapshot().stage, "IDLE")
        self.assertEqual(budget.transitions, 4)

    def test_overlapping_compute_stage_leases_fail_closed(self):
        budget = self._shanxi()
        with budget.stage_lease("TEACHER"):
            with self.assertRaisesRegex(RuntimeError, "R11_CPU_STAGE_ALREADY_ACTIVE"):
                with budget.stage_lease("TRACE"):
                    self.fail("unreachable")
        self.assertEqual(budget.snapshot().stage, "IDLE")

    def test_runtime_receipt_excludes_topology_from_scientific_identity(self):
        receipt = self._shanxi().runtime_receipt()
        self.assertFalse(receipt["scientific_semantics_changed"])
        self.assertFalse(receipt["topology_in_scientific_identity"])
        self.assertEqual(receipt["teacher_workers"], 8)
        self.assertEqual(receipt["trace_workers"], 8)
        self.assertEqual(receipt["blas_threads"], 1)


if __name__ == "__main__":
    unittest.main()
