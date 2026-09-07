from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cb16_local_opt.r102_common import sha256_file
from cb16_local_opt.r102_evidence_incremental import (
    _frame_equal,
    _validate_parent_manifest,
)


class TestIncrementalEvidenceGuards(unittest.TestCase):
    def test_frame_identity_is_exact(self):
        def frame(v: float):
            return SimpleNamespace(
                decision_time_ms=123,
                micro_1m_60x5=np.full((2, 2), v, dtype=np.float32),
                micro_stamps_60x5=np.full((2, 2), v + 1, dtype=np.float32),
                hourly_64x5=np.full((2, 2), v + 2, dtype=np.float32),
                hourly_stamps_64x5=np.full((2, 2), v + 3, dtype=np.float32),
                ordered4h30=np.full((3,), v + 4, dtype=np.float32),
            )

        a = frame(1.0)
        b = frame(1.0)
        self.assertTrue(_frame_equal(a, b))
        b.hourly_64x5[0, 0] += 1.0
        self.assertFalse(_frame_equal(a, b))

    def test_parent_manifest_requires_nested_stride_and_exact_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = {}
            for name in ("parents", "states", "branches"):
                p = root / f"{name}.bin"
                p.write_bytes((name + "\n").encode())
                files[name] = p
            manifest = {
                "schema": "CB16_R10_2_REAL_EVIDENCE_CACHE_MANIFEST_V1",
                "final_holdout_2025_09_accessed": False,
                "stride_hours": 512,
                "prehistory_hours": 96,
                "parents_file": str(files["parents"]),
                "parents_sha256": sha256_file(files["parents"]),
                "parent_states_file": str(files["states"]),
                "parent_states_sha256": sha256_file(files["states"]),
                "branches_file": str(files["branches"]),
                "branches_sha256": sha256_file(files["branches"]),
            }
            _validate_parent_manifest(manifest, stride_hours=256, prehistory_hours=96)
            with self.assertRaisesRegex(RuntimeError, "STRIDE_NOT_NESTED"):
                _validate_parent_manifest(manifest, stride_hours=300, prehistory_hours=96)
            with self.assertRaisesRegex(RuntimeError, "PREHISTORY_MISMATCH"):
                _validate_parent_manifest(manifest, stride_hours=256, prehistory_hours=64)
            files["branches"].write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "FILE_HASH_MISMATCH"):
                _validate_parent_manifest(manifest, stride_hours=256, prehistory_hours=96)


if __name__ == "__main__":
    unittest.main()
