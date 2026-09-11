from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import torch

from cb16_local_opt.binance_archive_input_r10 import BinanceUSDMArchiveSourceR10
from cb16_local_opt.full_minute_long_trajectory_r1 import MinuteEnvelopeR1
from cb16_local_opt.longtraj_infra_closure_r0 import find_contiguous_prefinal_run_r0, sha256_file_r0
from cb16_local_opt.minute_sensory_adapter_r1 import iter_minute_sensory_frames_r1
from cb16_local_opt.training_runtime_r11 import PreparedEvidenceR11


def make_prepared(rows: int = 512) -> PreparedEvidenceR11:
    g = torch.Generator(device="cpu")
    g.manual_seed(881_337)
    packed = torch.zeros((rows, 107), dtype=torch.float32)
    packed[:, :102] = torch.randn((rows, 102), generator=g, dtype=torch.float32) * 0.05
    packed[:, 102:105] = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32)
    packed[:, 105] = 0.4
    packed[:, 106] = 1.0
    out = PreparedEvidenceR11(
        parent_ids=tuple(f"STRESS_PARENT_{i:04d}" for i in range(rows)),
        dependence_group_ids=tuple(f"STRESS_GROUP_{i:04d}" for i in range(rows)),
        packed=packed,
        evidence_hash="STRESS_ONLY_V2",
        host_to_device_transfers=0,
        h2d_strategy="STRESS_ONLY_DETERMINISTIC_FIXTURE",
    )
    out.validate()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=os.environ.get("CB16_RAW_ROOT", "/cb16/raw"))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    source = BinanceUSDMArchiveSourceR10(args.raw_root)
    source.validate_layout()
    records = find_contiguous_prefinal_run_r0(source, args.symbol, required_rows=64 * 60 + 64)
    frames = list(itertools.islice(
        iter_minute_sensory_frames_r1(args.symbol, [MinuteEnvelopeR1(r, False) for r in records]),
        32,
    ))
    if len(frames) != 32:
        raise RuntimeError("STRESS_FIXTURE_MINUTE_FRAME_SHORTFALL")
    prepared = make_prepared()
    fixture = {
        "prepared": {
            "parent_ids": prepared.parent_ids,
            "dependence_group_ids": prepared.dependence_group_ids,
            "packed": prepared.packed,
            "evidence_hash": prepared.evidence_hash,
        },
        "sensory_frames": [f for f, _ in frames],
        "physics_records": list(records[:4096]),
        "archive_identity": {
            "symbol": args.symbol,
            "first_ms": int(records[0].open_time),
            "last_ms": int(records[-1].open_time),
            "final_holdout_touched": False,
            "scientific_weight": "NONE__HARDWARE_STRESS_ONLY",
        },
    }
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, p)
    receipt = {
        "schema": "CB16_R11_LONGTRAJ_STRESS_FIXTURE_V2",
        "status": "PASS",
        "fixture": str(p),
        "sha256": sha256_file_r0(p),
        "sensory_frames": 32,
        "physics_records": len(fixture["physics_records"]),
        "prepared_rows": prepared.rows,
        "scientific_weight": "NONE__HARDWARE_STRESS_ONLY",
        "final_holdout_touched": False,
    }
    Path(str(p) + ".json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
