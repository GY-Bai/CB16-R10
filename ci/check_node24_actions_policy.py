#!/usr/bin/env python3
"""Fail-closed GitHub Actions runtime compatibility policy.

This is a Guard sub-check, not a standalone workflow gate. It keeps selected
first-party action references on the current Node 24-compatible major releases
and prevents insecure runtime opt-outs.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ACTION_RULES: tuple[tuple[str, re.Pattern[str], int], ...] = (
    ("actions/checkout", re.compile(r"actions/checkout@([^\s#]+)"), 7),
    ("actions/upload-artifact", re.compile(r"actions/upload-artifact@([^\s#]+)"), 7),
    ("actions/download-artifact", re.compile(r"actions/download-artifact@([^\s#]+)"), 8),
)
FORBIDDEN_OPT_OUT = "ACTIONS_ALLOW_USE_" + "UNSECURE_NODE_VERSION"


def _major(ref: str) -> int | None:
    m = re.fullmatch(r"v(\d+)(?:\..*)?", ref)
    return int(m.group(1)) if m else None


def check_workflows(root: Path) -> list[str]:
    errors: list[str] = []
    wf_dir = root / ".github" / "workflows"
    if not wf_dir.is_dir():
        return ["missing .github/workflows"]

    for path in sorted((*wf_dir.glob("*.yml"), *wf_dir.glob("*.yaml"))):
        text = path.read_text(encoding="utf-8")
        if FORBIDDEN_OPT_OUT in text:
            errors.append(f"{path}: forbidden insecure Node-version opt-out")

        for lineno, line in enumerate(text.splitlines(), 1):
            for action_name, pattern, required_major in ACTION_RULES:
                match = pattern.search(line)
                if not match:
                    continue
                ref = match.group(1)
                major = _major(ref)
                if major != required_major:
                    errors.append(
                        f"{path}:{lineno}: {action_name} must use @v{required_major}; "
                        f"found @{ref}"
                    )

    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()
    errors = check_workflows(args.root.resolve())
    if errors:
        print("NODE24_ACTION_POLICY=FAIL")
        for error in errors:
            print(error)
        return 1
    print("NODE24_ACTION_POLICY=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
