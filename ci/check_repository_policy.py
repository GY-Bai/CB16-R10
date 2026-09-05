#!/usr/bin/env python3
"""CB16 repository policy guard.

Blocks model weights, secrets, keys, and obvious large/binary artifacts from
being committed to the GitHub source authority.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FORBIDDEN_EXTENSIONS = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pem", ".key",
    ".p12", ".pfx", ".joblib", ".npz", ".h5", ".hdf5", ".bin", ".parquet",
    ".arrow", ".zip", ".tar", ".gz", ".zst", ".lgb",
}

FORBIDDEN_NAMES = {".env", ".env.local", ".env.production", ".env.development"}

# Look for actual high-entropy secret values, not documentation keywords.
SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"CF-Access-Client-Secret\s*[:=]\s*[A-Za-z0-9_\-]{16,}"),
    re.compile(r"BEGIN (OPENSSH|RSA|EC|DSA) PRIVATE KEY"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]

MAX_TEXT_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB


def check_path(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        low = rel.lower()
        if path.name in FORBIDDEN_NAMES or any(low.endswith(ext) for ext in FORBIDDEN_EXTENSIONS):
            errors.append(f"forbidden file: {rel}")
            continue
        # no git directory files
        if rel.startswith(".git/"):
            continue
        # scan text-ish files for secrets
        size = path.stat().st_size
        if size > MAX_TEXT_FILE_BYTES:
            errors.append(f"oversize file: {rel} ({size} bytes)")
            continue
        if size > 0:
            try:
                head = path.read_bytes()[:4096].decode("utf-8", errors="ignore")
            except Exception:
                head = ""
            if any(p.search(head) for p in SECRET_PATTERNS):
                errors.append(f"possible secret in file: {rel}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--allow-large", action="store_true", help="for test only")
    args = ap.parse_args()
    errors = check_path(args.root)
    if errors:
        print("POLICY_VIOLATIONS")
        for e in errors[:100]:
            print(e)
        return 1
    print("POLICY_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
