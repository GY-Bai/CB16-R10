#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.r11_m_series_m2_1_correction_quality import main

if __name__=="__main__":
    raise SystemExit(main())
