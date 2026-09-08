#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.stage4_authority_inventory_r11 import audit_registry, load_registry


def main() -> int:
    ap = argparse.ArgumentParser(description="CB16 R11 Stage-4 S4A authority writer surface audit")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--registry",
        default="authority/rearchitecture_r11/CB16_R11_STAGE4_AUTHORITY_WRITER_REGISTRY_V1.json",
    )
    ap.add_argument("--json-out")
    ap.add_argument("--print-discovered", action="store_true")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    registry = load_registry(root / args.registry)
    report = audit_registry(root, registry, verify_git_guards=True)
    if args.print_discovered or not report["pass"]:
        for item in report["discoveries"]:
            print("STAGE4_DISCOVERED " + json.dumps(item, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "discoveries"}, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
