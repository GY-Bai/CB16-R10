#!/usr/bin/env python3
"""Fail-closed GitHub Actions runtime compatibility policy.

This is a Guard sub-check, not a standalone workflow gate. It keeps action
versions on the Node24-compatible generation and prevents insecure opt-outs.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CHECKOUT_RE = re.compile(r"actions/checkout@([^\s#]+)")
UPLOAD_RE = re.compile(r"actions/upload-artifact@([^\s#]+)")
DOWNLOAD_RE = re.compile(r"actions/download-artifact@([^\s#]+)")
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
            m = CHECKOUT_RE.search(line)
            if m:
                ref = m.group(1)
                major = _major(ref)
                if major != 6:
                    errors.append(
                        f"{path}:{lineno}: actions/checkout must use @v6; found @{ref}"
                    )

            m = UPLOAD_RE.search(line)
            if m:
                ref = m.group(1)
                major = _major(ref)
                if major is not None and major < 5:
                    errors.append(
                        f"{path}:{lineno}: actions/upload-artifact must use v5+; found @{ref}"
                    )

            m = DOWNLOAD_RE.search(line)
            if m:
                ref = m.group(1)
                major = _major(ref)
                if major is not None and major < 5:
                    errors.append(
                        f"{path}:{lineno}: actions/download-artifact must use v5+; found @{ref}"
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
