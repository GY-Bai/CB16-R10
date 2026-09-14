"""Emit the exact-SHA Shanxi machine-readable initial-checkpoint identity artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.post_cc_s1_checkpoint_identity_v1 import (  # noqa: E402
    write_initial_checkpoint_identity_artifact_v1,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    payload = write_initial_checkpoint_identity_artifact_v1(args.json_out)
    import json

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
