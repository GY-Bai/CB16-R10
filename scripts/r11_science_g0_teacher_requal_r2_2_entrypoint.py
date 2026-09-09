#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.r11_science_g0_teacher_requal_r2_2 import main

if __name__ == "__main__":
    raise SystemExit(main())
