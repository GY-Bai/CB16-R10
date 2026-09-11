from __future__ import annotations

"""Streaming archive helpers for R11 long-trajectory experiments.

The FINAL month is filtered by archive filename before the ZIP is opened, so a
prefinal scan never needs to read one byte from the 2025-09 payload.
"""

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .binance_archive_input_r10 import BinanceUSDMArchiveSourceR10, KlineRecord, _is_header, _row_to_kline

_MONTH = re.compile(r"^(?P<symbol>[A-Z0-9]+)-1m-(?P<year>\d{4})-(?P<month>\d{2})\.zip$")


@dataclass(frozen=True)
class PrefinalArchiveInventoryR1:
    symbol: str
    archives: tuple[str, ...]
    first_year_month: tuple[int, int] | None
    last_year_month: tuple[int, int] | None


def prefinal_archives_r1(
    source: BinanceUSDMArchiveSourceR10,
    symbol: str,
    *,
    final_year_month: tuple[int, int] = (2025, 9),
) -> list[Path]:
    out: list[tuple[int, int, Path]] = []
    for p in source.monthly_kline_archives(symbol):
        m = _MONTH.match(p.name)
        if not m:
            continue
        ym = (int(m.group("year")), int(m.group("month")))
        if ym >= final_year_month:
            continue
        out.append((ym[0], ym[1], p))
    out.sort()
    return [p for _, _, p in out]


def inventory_prefinal_archives_r1(source: BinanceUSDMArchiveSourceR10, symbol: str) -> PrefinalArchiveInventoryR1:
    rows = prefinal_archives_r1(source, symbol)
    yms = []
    for p in rows:
        m = _MONTH.match(p.name)
        assert m is not None
        yms.append((int(m.group("year")), int(m.group("month"))))
    return PrefinalArchiveInventoryR1(
        symbol=symbol,
        archives=tuple(p.name for p in rows),
        first_year_month=yms[0] if yms else None,
        last_year_month=yms[-1] if yms else None,
    )


def iter_prefinal_observed_1m_r1(
    source: BinanceUSDMArchiveSourceR10,
    symbol: str,
    *,
    verify_checksums: bool = False,
) -> Iterator[KlineRecord]:
    from .binance_archive_input_r10 import verify_binance_checksum

    archives = prefinal_archives_r1(source, symbol)
    if not archives:
        raise RuntimeError(f"NO_PREFINAL_1M_ARCHIVES:{symbol}")
    prev = None
    for zp in archives:
        if verify_checksums:
            verify_binance_checksum(zp)
        with zipfile.ZipFile(zp, "r") as zf:
            members = [n for n in zf.namelist() if not n.endswith("/")]
            csv_members = [n for n in members if n.lower().endswith(".csv")]
            if len(members) == 1:
                member = members[0]
            elif len(csv_members) == 1:
                member = csv_members[0]
            else:
                raise RuntimeError(f"ZIP_MEMBER_AMBIGUITY:{zp}:{members[:5]}")
            with zf.open(member, "r") as raw:
                txt = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                reader = csv.reader(txt)
                for row in reader:
                    if not row or _is_header(row):
                        continue
                    rec = _row_to_kline(row)
                    if prev is not None and rec.open_time <= prev:
                        raise RuntimeError(f"NON_INCREASING_1M:{symbol}:{prev}->{rec.open_time}")
                    prev = rec.open_time
                    yield rec
