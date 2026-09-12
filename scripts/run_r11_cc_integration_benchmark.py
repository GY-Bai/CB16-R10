from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

# Direct script execution puts scripts/ rather than the repository root on
# sys.path. Qualification intentionally runs the checked-out source tree without
# installing the package or using the network, so bind the repo root explicitly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from cb16_local_opt.cc_integration_benchmark_r0 import (
    run_reference_benchmark,
    run_fast_benchmark,
    assert_reference_fast_equivalence,
)
from cb16_local_opt.cc_integration_fast_path_r0 import summarize_repetitions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--accounts", type=int, default=16)
    parser.add_argument("--market-steps", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--seed", type=int, default=9917)
    args = parser.parse_args()
    if args.repetitions < 7:
        raise SystemExit("integration benchmark requires at least 7 repetitions per topology")

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    rows = []
    reference_rates = []
    fast_rates = []
    reference_walls = []
    fast_walls = []
    canonical_semantic_checksum = None
    canonical_account_checksum = None

    for rep in range(args.repetitions):
        order = ("reference", "fast") if rep % 2 == 0 else ("fast", "reference")
        pair = {}
        for topology in order:
            root = work_root / f"rep-{rep:02d}-{topology}"
            if topology == "reference":
                result = run_reference_benchmark(
                    output_root=root,
                    account_count=args.accounts,
                    market_steps=args.market_steps,
                    seed=args.seed,
                )
            else:
                result = run_fast_benchmark(
                    output_root=root,
                    account_count=args.accounts,
                    market_steps=args.market_steps,
                    seed=args.seed,
                    chunk_facts=128,
                )
            pair[topology] = result
            rows.append({"repetition": rep, "order": list(order), **result.to_dict()})
        assert_reference_fast_equivalence(pair["reference"], pair["fast"])
        if canonical_semantic_checksum is None:
            canonical_semantic_checksum = pair["reference"].semantic_checksum
            canonical_account_checksum = pair["reference"].final_account_checksum
        if pair["reference"].semantic_checksum != canonical_semantic_checksum or pair["fast"].semantic_checksum != canonical_semantic_checksum:
            raise RuntimeError("CROSS_REPETITION_SEMANTIC_DRIFT")
        if pair["reference"].final_account_checksum != canonical_account_checksum or pair["fast"].final_account_checksum != canonical_account_checksum:
            raise RuntimeError("CROSS_REPETITION_ACCOUNT_DRIFT")
        reference_rates.append(pair["reference"].compliant_transitions_per_s)
        fast_rates.append(pair["fast"].compliant_transitions_per_s)
        reference_walls.append(pair["reference"].wall_clock_s)
        fast_walls.append(pair["fast"].wall_clock_s)

    ref_summary = summarize_repetitions(reference_rates)
    fast_summary = summarize_repetitions(fast_rates)
    ref_wall_summary = summarize_repetitions(reference_walls)
    fast_wall_summary = summarize_repetitions(fast_walls)
    selected = "CC_FAST_R0_A_ORACLE_PLUS_D_SCHEDULER_BOUNDED_CHUNK_WRITER" if fast_summary["median"] > ref_summary["median"] else "REFERENCE_A_RUNTIME_PLUS_C_RAW_FACT_STORE"
    hard_cutover_pass = selected.startswith("CC_FAST_R0")

    payload = {
        "schema": "CB16_R11_CC_INTEGRATION_SHANXI_BENCHMARK_V1",
        "code_sha": os.environ.get("GITHUB_SHA", "LOCAL"),
        "science_contract": "CB16_R11_CC_SCIENCE_SEMANTIC_V1",
        "workload": {
            "identity": "CB16_R11_CC_INTEGRATED_SYNTHETIC_WORKLOAD_V1",
            "accounts": args.accounts,
            "market_steps": args.market_steps,
            "repetitions_per_topology": args.repetitions,
            "seed": args.seed,
            "ordering": "ALTERNATING_REFERENCE_FAST",
            "FINAL_opened": False,
            "fresh_data_used": False,
        },
        "semantic_equivalence": {
            "verdict": "PASS",
            "discrete": "EXACT_BY_CANONICAL_FACT_CHECKSUM",
            "numeric_atol": 1e-10,
            "numeric_rtol": 1e-10,
            "semantic_checksum": canonical_semantic_checksum,
            "final_account_checksum": canonical_account_checksum,
        },
        "reference": {
            "transitions_per_s": ref_summary,
            "wall_clock_s": ref_wall_summary,
        },
        "fast": {
            "transitions_per_s": fast_summary,
            "wall_clock_s": fast_wall_summary,
        },
        "selection_rule": "maximum median end-to-end compliant transitions/s among semantic PASS implementations",
        "selected_topology": selected,
        "hard_cutover_pass": hard_cutover_pass,
        "legacy_fallback": False,
        "hardware": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "runs": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if not hard_cutover_pass:
        raise SystemExit("semantic PASS but Thread-D integrated fast topology did not win; hard cutover gate fails")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
