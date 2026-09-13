"""CI-C gate: reviewer authorization must bind the exact reviewed runtime and manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.post_cc_s1_qualification_v1 import (  # noqa: E402
    S1QualificationError,
    require_qualification_authorization_v1,
)
from cb16_local_opt.post_cc_s1_execution_manifest_v1 import validate_s1_execution_manifest_v1  # noqa: E402

CANDIDATE_PATH = "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_REVIEW_CANDIDATE_V1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    root = Path(args.repo_root)
    try:
        manifest = validate_s1_execution_manifest_v1(root)
        candidate = json.loads((root / CANDIDATE_PATH).read_text(encoding="utf-8"))
        payload = require_qualification_authorization_v1(
            root,
            expected_runtime_sha=str(candidate["implementation_candidate_sha"]),
            expected_runtime_tree_sha=str(candidate["implementation_candidate_tree_sha"]),
            expected_manifest_sha256=str(manifest["manifest_sha256"]),
        )
        result = {
            "schema": "CB16_R11_POST_CC_S1_QUALIFICATION_AUTHORIZATION_VERIFICATION_V1",
            "status": "AUTHORIZED",
            "reviewed_implementation_sha": payload["reviewed_implementation_sha"],
            "reviewed_implementation_tree_sha": payload["reviewed_implementation_tree_sha"],
            "execution_manifest_sha256": payload["execution_manifest_sha256"],
        }
    except S1QualificationError as exc:
        result = {"schema": "CB16_R11_POST_CC_S1_QUALIFICATION_AUTHORIZATION_VERIFICATION_V1", "status": "BLOCKED", "error": str(exc)}
    payload_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        Path(args.json_out).write_text(payload_text, encoding="utf-8")
    print(payload_text)
    return 0 if result["status"] == "AUTHORIZED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
