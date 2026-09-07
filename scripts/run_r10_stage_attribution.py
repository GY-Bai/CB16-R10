#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cb16_diagnostics.stage_attribution import _read_jsonl, build_stage_attribution


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate CB16 diagnostics sidecar samples by generation/stage")
    ap.add_argument("--diagnostics-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    diagnostics_root = Path(args.diagnostics_root).resolve()
    runtime_jsonl = diagnostics_root / "runtime_samples.jsonl"
    result = build_stage_attribution(_read_jsonl(runtime_jsonl))

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "sample_count": result["sample_count"],
        "stage_bucket_count": result["stage_bucket_count"],
        "generation_count_observed": result["generation_count_observed"],
        "writes_to_canonical_run_root": result["safety"]["writes_to_canonical_run_root"],
        "scientific_semantics_changed": result["safety"]["scientific_semantics_changed"],
        "final_holdout_2025_09_accessed": result["safety"]["final_holdout_2025_09_accessed"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
