from __future__ import annotations

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from cb16_local_opt.binance_archive_input_r10 import BinanceUSDMArchiveSourceR10
from cb16_local_opt.r102_common import HOUR_MS, FUNDING_CANONICAL_JITTER_TOLERANCE_MS
from cb16_local_opt.r102_market import (
    canonicalize_funding_event_hour_r1021,
    iter_funding_bounded,
    scan_funding_timestamp_compatibility_r1021,
)


class TestR1021FundingTimestampCompatibility(unittest.TestCase):
    def test_exact_and_small_jitter_normalize_to_same_hour(self):
        hour = 1_577_923_200_000
        for delta in (0, 1, 2, 18, 999, -1, -999):
            canonical, jitter = canonicalize_funding_event_hour_r1021(hour + delta)
            self.assertEqual(canonical, hour)
            self.assertEqual(jitter, delta)

    def test_large_jitter_fails_closed(self):
        hour = 1_577_923_200_000
        with self.assertRaisesRegex(RuntimeError, "FUNDING_TIMESTAMP_JITTER_EXCEEDS_TOLERANCE"):
            canonicalize_funding_event_hour_r1021(hour + FUNDING_CANONICAL_JITTER_TOLERANCE_MS + 1)
        with self.assertRaisesRegex(RuntimeError, "FUNDING_TIMESTAMP_JITTER_EXCEEDS_TOLERANCE"):
            canonicalize_funding_event_hour_r1021(hour - FUNDING_CANONICAL_JITTER_TOLERANCE_MS - 1)

    def test_official_style_plus_2ms_event_is_applied_once_no_forward_fill(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d = root / 'fundingRate/BTCUSDT'
            d.mkdir(parents=True)
            zp = d / 'BTCUSDT-fundingRate-2020-01.zip'
            rows = [
                ['calc_time', 'funding_interval_hours', 'last_funding_rate'],
                ['1577836800000', '8', '-0.00012359'],
                ['1577865600000', '8', '-0.00012383'],
                ['1577894400000', '8', '-0.00009664'],
                ['1577923200002', '8', '0.00003662'],
            ]
            import io
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator='\n')
            w.writerows(rows)
            with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('BTCUSDT-fundingRate-2020-01.csv', buf.getvalue())
            src = BinanceUSDMArchiveSourceR10(root)
            recs = list(iter_funding_bounded(src, 'BTCUSDT'))
            self.assertEqual(len(recs), 4)
            self.assertEqual(recs[-1].funding_time, 1_577_923_200_000)
            self.assertAlmostEqual(recs[-1].funding_rate, 0.00003662)
            self.assertTrue(all(r.funding_time % HOUR_MS == 0 for r in recs))
            audit = scan_funding_timestamp_compatibility_r1021(src, ['BTCUSDT'])
            self.assertEqual(audit['status'], 'PASS')
            self.assertEqual(audit['events_total'], 4)
            self.assertEqual(audit['normalized_events_total'], 1)
            self.assertEqual(audit['max_abs_jitter_ms'], 2)
            self.assertEqual(audit['symbols']['BTCUSDT']['normalized_events'], 1)

    def test_forbidden_2025_09_funding_archive_never_opened(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d = root / 'fundingRate/BTCUSDT'
            d.mkdir(parents=True)
            for month, ts in [('2025-08', 1_754_006_400_000), ('2025-09', 1_756_684_800_000)]:
                zp = d / f'BTCUSDT-fundingRate-{month}.zip'
                text = f'calc_time,funding_interval_hours,last_funding_rate\n{ts},8,0.0001\n'
                with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr(f'BTCUSDT-fundingRate-{month}.csv', text)
            src = BinanceUSDMArchiveSourceR10(root)
            recs = list(iter_funding_bounded(src, 'BTCUSDT'))
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].funding_time, 1_754_006_400_000)

    def test_canonical_collision_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d = root / 'fundingRate/BTCUSDT'
            d.mkdir(parents=True)
            zp = d / 'BTCUSDT-fundingRate-2020-01.zip'
            text = (
                'calc_time,funding_interval_hours,last_funding_rate\n'
                '1577923200001,8,0.0001\n'
                '1577923200002,8,0.0002\n'
            )
            with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('BTCUSDT-fundingRate-2020-01.csv', text)
            src = BinanceUSDMArchiveSourceR10(root)
            with self.assertRaisesRegex(RuntimeError, 'NON_INCREASING_OR_COLLIDING_FUNDING_CANONICAL'):
                list(iter_funding_bounded(src, 'BTCUSDT'))


if __name__ == '__main__':
    unittest.main()
