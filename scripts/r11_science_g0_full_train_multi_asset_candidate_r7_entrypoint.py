#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.full_train_multi_asset_candidate_r7 import canonical_sha256_r7
from scripts import r11_science_g0_full_train_multi_asset_candidate_r7 as target

# R7 keeps R4 immutable. The executor historically references r4.sha256_obj only for
# a disposable shadow snapshot identity; bind that name to R7's canonical JSON SHA256
# helper rather than editing the frozen R4 implementation.
if not hasattr(target.r4, "sha256_obj"):
    target.r4.sha256_obj = canonical_sha256_r7

if __name__ == "__main__":
    raise SystemExit(target.main())
