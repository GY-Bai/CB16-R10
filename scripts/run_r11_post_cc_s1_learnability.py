"""CB16 R11 S1 R1 learnability runner (smoke or formal qualification).

Allowed CLI differences are execution-only: mode, output root, worker count and
repo root.  Qualification mode exposes no seed/threshold/reward/model/LR/budget
override and fails closed if the execution manifest does not match the frozen
authority chain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.post_cc_s1_qualification_v1 import S1QualificationError, run_s1_program_v1

EXIT_OK = 0
EXIT_NOT_PASS = 1
EXIT_BLOCKED = 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--mode", choices=("smoke", "qualification"), required=True)
    parser.add_argument("--output-root", default="artifacts/post_cc_s1")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--scratch-root",
        default=None,
        help="execution-only fast scratch root for durable run stores (e.g. /dev/shm/cb16-s1)",
    )
    args = parser.parse_args()
    try:
        summary = run_s1_program_v1(
            repo_root=args.repo_root,
            mode=args.mode,
            output_root=args.output_root,
            workers=args.workers,
            scratch_root=args.scratch_root,
        )
    except S1QualificationError as exc:
        print(json.dumps({"status": "CONTRACT_MISMATCH", "error": str(exc)}, indent=2, sort_keys=True))
        return EXIT_BLOCKED
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if args.mode == "smoke":
        return EXIT_OK if summary.get("status") == "SMOKE_ONLY_NOT_SCIENTIFIC_QUALIFICATION" else EXIT_NOT_PASS
    return EXIT_OK if summary.get("classification") == "PASS" else EXIT_NOT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
