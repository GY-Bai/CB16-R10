from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cb16_local_opt.binance_archive_input_r10 import HOUR_MS, MINUTE_MS, KlineRecord
from cb16_local_opt import r102_market as market


class _FakeArchiveSource:
    def __init__(self, archives):
        self._archives = list(archives)

    def monthly_kline_archives(self, symbol):
        return list(self._archives)


def _write_zip(path: Path, open_times: list[int]) -> None:
    rows = []
    for i, t in enumerate(open_times):
        px = 100.0 + i
        rows.append([
            str(t), str(px), str(px + 1), str(px - 1), str(px + 0.5), "1",
            str(t + MINUTE_MS - 1), "0", "0", "0", "0", "0",
        ])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for row in rows:
        w.writerow(row)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.name.replace(".zip", ".csv"), buf.getvalue())


def _rec(t: int, origin: int) -> KlineRecord:
    # Encode hour index into price so a yielded window can be checked for bridge-free context.
    px = float((t - origin) // HOUR_MS)
    return KlineRecord(
        open_time=t, open=px, high=px + 1.0, low=px - 1.0, close=px + 0.25,
        volume=1.0, close_time=t + MINUTE_MS - 1,
    )


def _contiguous_minutes(start: int, hours: int, origin: int):
    return [_rec(start + i * MINUTE_MS, origin) for i in range(hours * 60)]


def test_iter_1m_accepts_positive_minute_aligned_gap_without_synthesis(tmp_path: Path):
    start = 1609459200000  # 2021-01-01 UTC
    zp = tmp_path / "BTCUSDT-1m-2021-01.zip"
    times = [start, start + MINUTE_MS, start + 4 * MINUTE_MS]
    _write_zip(zp, times)
    rows = list(market.iter_1m_bounded(_FakeArchiveSource([zp]), "BTCUSDT"))
    assert [x.open_time for x in rows] == times
    assert len(rows) == 3  # no synthetic rows inserted into the gap


def test_iter_1m_rejects_non_minute_alignment_and_non_increasing(tmp_path: Path):
    start = 1609459200000
    bad_align = tmp_path / "BTCUSDT-1m-2021-01.zip"
    _write_zip(bad_align, [start, start + MINUTE_MS + 1])
    with pytest.raises(RuntimeError, match="NON_MINUTE_ALIGNED_1M"):
        list(market.iter_1m_bounded(_FakeArchiveSource([bad_align]), "BTCUSDT"))

    bad_order = tmp_path / "BTCUSDT-1m-2021-02.zip"
    _write_zip(bad_order, [start, start])
    with pytest.raises(RuntimeError, match="NON_INCREASING_1M"):
        list(market.iter_1m_bounded(_FakeArchiveSource([bad_order]), "BTCUSDT"))


def test_sensory_context_is_fully_reset_across_gap():
    origin = 1609459200000
    pre_hours = 65
    gap_hours = 72
    post_hours = 65
    pre = _contiguous_minutes(origin, pre_hours, origin)
    post_start = origin + (pre_hours + gap_hours) * HOUR_MS
    post = _contiguous_minutes(post_start, post_hours, origin)
    records = pre + post

    with patch.object(market, "iter_1m_bounded", return_value=iter(records)):
        frames = list(market.iter_sensory_frames_bounded(object(), "BTCUSDT"))

    pre_frames = [f for f in frames if f.decision_time_ms <= origin + pre_hours * HOUR_MS]
    post_frames = [f for f in frames if f.decision_time_ms >= post_start]
    assert pre_frames
    assert post_frames
    first_post = post_frames[0]
    # 64 fresh complete real hours are required again after the gap.
    assert first_post.decision_time_ms == post_start + 64 * HOUR_MS
    # Price encodes source hour index; the first hourly row must be post-gap, never pre-gap.
    post_hour_index = float((post_start - origin) // HOUR_MS)
    assert float(first_post.hourly_64x5[0, 0]) == post_hour_index
    assert all(not (origin + pre_hours * HOUR_MS < f.decision_time_ms < post_start + 64 * HOUR_MS) for f in frames)


def test_hourly_discontinuity_ranges_are_exact_and_fail_closed():
    start = 1609459200000
    ts = [start, start + HOUR_MS, start + 5 * HOUR_MS, start + 6 * HOUR_MS]
    ranges = market._hourly_discontinuity_ranges(ts)
    assert ranges == [(start + 2 * HOUR_MS, start + 4 * HOUR_MS)]
    assert market._hour_is_inside_discontinuity(start + 3 * HOUR_MS, ranges)
    assert not market._hour_is_inside_discontinuity(start + HOUR_MS, ranges)
    with pytest.raises(RuntimeError, match="NON_INCREASING_HOURLY_CACHE"):
        market._hourly_discontinuity_ranges([start, start])
    with pytest.raises(RuntimeError, match="NON_HOUR_ALIGNED_CACHE_TIMESTAMP"):
        market._hourly_discontinuity_ranges([start + 1])
