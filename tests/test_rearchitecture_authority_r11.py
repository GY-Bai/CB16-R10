from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cb16_local_opt import rearchitecture_authority_r11 as r11


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_archive_with_sidecar(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    Path(str(path) + ".CHECKSUM").write_text(f"{_sha(payload)}  {path.name}\n", encoding="utf-8")


class TestR11StaticAuthority(unittest.TestCase):
    def test_repo_static_contracts(self):
        root = Path(__file__).resolve().parents[1]
        result = r11.verify_static_semantic_contracts(root)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(tuple(result["symbols"]), r11.TEN_SYMBOLS_R11)
        self.assertFalse(result["legacy_python_is_runtime_authority"])


class TestR11DatasetSeal(unittest.TestCase):
    def _fake_root(self, base: Path) -> Path:
        root = base / "market"
        (root / "klines_1m").mkdir(parents=True)
        (root / "fundingRate").mkdir(parents=True)
        for symbol in r11.TEN_SYMBOLS_R11:
            pre_k = root / "klines_1m" / symbol / f"{symbol}-1m-2025-08.zip"
            hold_k = root / "klines_1m" / symbol / f"{symbol}-1m-2025-09.zip"
            pre_f = root / "fundingRate" / symbol / f"{symbol}-fundingRate-2025-08.zip"
            hold_f = root / "fundingRate" / symbol / f"{symbol}-fundingRate-2025-09.zip"
            _write_archive_with_sidecar(pre_k, ("PRE-K-" + symbol).encode())
            _write_archive_with_sidecar(hold_k, ("HOLD-K-" + symbol).encode())
            _write_archive_with_sidecar(pre_f, ("PRE-F-" + symbol).encode())
            _write_archive_with_sidecar(hold_f, ("HOLD-F-" + symbol).encode())
        (root / "DOWNLOAD_MANIFEST.json").write_text(json.dumps({"fake": True}), encoding="utf-8")
        return root

    def test_holdout_payload_is_never_hashed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._fake_root(Path(td))
            original = r11.sha256_file
            hashed_paths = []

            def guarded_sha(path, chunk=8 << 20):
                p = Path(path)
                hashed_paths.append(p)
                if p.suffix == ".zip" and ("-2025-09.zip" in p.name):
                    raise AssertionError(f"holdout payload was opened for hashing: {p}")
                return original(p, chunk)

            with mock.patch.object(r11, "sha256_file", side_effect=guarded_sha):
                seal = r11.build_read_only_dataset_seal(root)

            self.assertEqual(seal["unopened_payload_bytes_read"], 0)
            self.assertEqual(seal["archive_count"], 40)
            self.assertTrue(any(p.suffix == ".zip" and "-2025-08.zip" in p.name for p in hashed_paths))
            hold = [e for e in seal["entries"] if e["month"] == "2025-09"]
            self.assertEqual(len(hold), 20)
            self.assertTrue(all(e["mode"] == "UNOPENED_CHECKSUM_BOUND" for e in hold))
            self.assertTrue(all(e["archive_sha256"] is None for e in hold))

    def test_consumed_history_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = self._fake_root(base)
            seal_path = base / "seal.json"
            first = r11.write_read_only_dataset_seal(root, seal_path)
            self.assertEqual(first["status"], "SEALED_EXISTING_BYTES_READ_ONLY")
            victim = root / "klines_1m" / "BTCUSDT" / "BTCUSDT-1m-2025-08.zip"
            victim.write_bytes(b"MUTATED")
            with self.assertRaisesRegex(RuntimeError, "R11_EXISTING_ARCHIVE_CHECKSUM_MISMATCH"):
                r11.verify_read_only_dataset_seal(root, seal_path)

    def test_seal_cannot_be_written_inside_market_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._fake_root(Path(td))
            with self.assertRaisesRegex(RuntimeError, "MUST_NOT_BE_INSIDE_MARKET_DATA_ROOT"):
                r11.write_read_only_dataset_seal(root, root / "R11_SEAL.json")

    def test_exact_ten_symbol_set_is_required(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._fake_root(Path(td))
            (root / "klines_1m" / "EXTRAUSDT").mkdir()
            with self.assertRaisesRegex(RuntimeError, "R11_KLINE_SYMBOL_SET_DRIFT"):
                r11.build_read_only_dataset_seal(root)


if __name__ == "__main__":
    unittest.main()
