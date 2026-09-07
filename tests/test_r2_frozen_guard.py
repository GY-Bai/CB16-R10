from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from cb16_local_opt.r2_frozen_guard import FrozenAuthorityGuardR2


class FrozenAuthorityGuardR2Test(unittest.TestCase):
    def test_unchanged_roundtrip_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.bin").write_bytes(b"abc")
            (root / "b.bin").write_bytes(b"def")
            guard = FrozenAuthorityGuardR2.capture(root, ("a.bin", "b.bin"))
            self.assertTrue(guard.metadata_unchanged())
            receipt = guard.finalize()
            self.assertTrue(receipt["pass"])
            self.assertTrue(receipt["full_hashes_unchanged"])

    def test_same_size_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "a.bin"
            path.write_bytes(b"abc")
            guard = FrozenAuthorityGuardR2.capture(root, ("a.bin",))
            path.write_bytes(b"xyz")
            self.assertFalse(guard.metadata_unchanged())
            receipt = guard.finalize()
            self.assertFalse(receipt["pass"])
            self.assertFalse(receipt["full_hashes_unchanged"])

    def test_restoring_mtime_does_not_hide_ctime_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "a.bin"
            path.write_bytes(b"abc")
            guard = FrozenAuthorityGuardR2.capture(root, ("a.bin",))
            start = path.stat()
            path.write_bytes(b"xyz")
            os.utime(path, ns=(start.st_atime_ns, start.st_mtime_ns))
            self.assertFalse(guard.metadata_unchanged())
            self.assertFalse(guard.finalize()["pass"])


if __name__ == "__main__":
    unittest.main()
