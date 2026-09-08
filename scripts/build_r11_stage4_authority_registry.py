#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.stage4_authority_registry_builder_r11 import write_registry


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the explicit CB16 R11 Stage-4 S4A writer registry")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--output",
        default="authority/rearchitecture_r11/CB16_R11_STAGE4_AUTHORITY_WRITER_REGISTRY_V1.json",
    )
    args = ap.parse_args()
    registry = write_registry(args.repo_root, args.output)
    counts: dict[str, int] = {}
    for entry in registry["entries"]:
        key = entry["classification"]
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"entries": len(registry["entries"]), "classification_counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
