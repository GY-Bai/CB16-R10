#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cb16_diagnostics.model_probe import probe_checkpoint, probe_pair
from cb16_diagnostics.postrun import FROZEN_SOURCE_AUTHORITY, build_diagnostics
from cb16_diagnostics.runtime_observer import RuntimeObserver


def main() -> int:
    ap = argparse.ArgumentParser(description="CB16 R10 diagnostic harness R0")
    sub = ap.add_subparsers(dest="command", required=True)

    live = sub.add_parser("observe-runtime", help="sidecar runtime sampler; writes outside campaign root")
    live.add_argument("--run-root", default="/data/cb16_hdd/cb16_runtime/R10_4")
    live.add_argument("--out", required=True)
    live.add_argument("--interval", type=float, default=5.0)
    live.add_argument("--duration", type=float, default=None)
    live.add_argument("--pid", type=int, default=None)
    live.add_argument("--stop-when-complete", action="store_true")

    snap = sub.add_parser("snapshot", help="read-only incomplete campaign snapshot")
    snap.add_argument("--run-root", default="/data/cb16_hdd/cb16_runtime/R10_4")
    snap.add_argument("--out", required=True)
    snap.add_argument("--runtime-diagnostics", default=None)
    snap.add_argument("--start-checkpoint", default="/home/bgy/cb16_ssd/runtime/R10_3/generations/G19/champion_after.pt")
    snap.add_argument("--source-authority-sha", default=FROZEN_SOURCE_AUTHORITY)
    snap.add_argument("--requested", type=int, default=100)
    snap.add_argument("--no-model-probe", action="store_true")

    post = sub.add_parser("postrun", help="require a complete campaign and build durable diagnostics")
    post.add_argument("--run-root", default="/data/cb16_hdd/cb16_runtime/R10_4")
    post.add_argument("--out", required=True)
    post.add_argument("--runtime-diagnostics", default=None)
    post.add_argument("--start-checkpoint", default="/home/bgy/cb16_ssd/runtime/R10_3/generations/G19/champion_after.pt")
    post.add_argument("--source-authority-sha", default=FROZEN_SOURCE_AUTHORITY)
    post.add_argument("--requested", type=int, default=100)
    post.add_argument("--no-model-probe", action="store_true")

    cp = sub.add_parser("checkpoint", help="probe one checkpoint")
    cp.add_argument("--checkpoint", required=True)
    cp.add_argument("--out", required=True)
    cp.add_argument("--spectral", action="store_true")

    pair = sub.add_parser("compare-checkpoints", help="compare two checkpoints")
    pair.add_argument("--before", required=True)
    pair.add_argument("--after", required=True)
    pair.add_argument("--out", required=True)
    pair.add_argument("--spectral", action="store_true")

    a = ap.parse_args()

    if a.command == "observe-runtime":
        summary = RuntimeObserver(Path(a.run_root), Path(a.out), interval_s=a.interval, pid=a.pid).run(
            duration_s=a.duration, stop_when_complete=a.stop_when_complete
        )
    elif a.command in {"snapshot", "postrun"}:
        summary = build_diagnostics(
            run_root=Path(a.run_root),
            out_dir=Path(a.out),
            runtime_diag_dir=Path(a.runtime_diagnostics) if a.runtime_diagnostics else None,
            start_checkpoint=Path(a.start_checkpoint) if a.start_checkpoint else None,
            source_authority_sha=a.source_authority_sha,
            requested=a.requested,
            allow_incomplete=a.command == "snapshot",
            probe_models=not a.no_model_probe,
        )
    elif a.command == "checkpoint":
        summary = probe_checkpoint(a.checkpoint, spectral=a.spectral)
        Path(a.out).write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    else:
        summary = probe_pair(a.before, a.after, spectral_after=a.spectral)
        Path(a.out).write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
