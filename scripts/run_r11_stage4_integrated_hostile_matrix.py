#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.stage4_hostile_integration_harness_r11 import (
    NoopHostileFaultHooks,
    ReferenceProductionHostileAdapter,
    integration_binding_manifest,
    run_integrated_hostile_matrix,
)


def _load_factory(spec: str, default: Callable[[], Any]) -> Callable[[], Any]:
    if spec == "reference":
        return default
    if ":" not in spec:
        raise SystemExit("factory must be 'reference' or module:function")
    module_name, attr = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attr)
    if not callable(factory):
        raise SystemExit(f"factory is not callable: {spec}")
    return factory


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--manifest-output", type=Path)
    ap.add_argument("--adapter-factory", default="reference")
    ap.add_argument("--hooks-factory", default="reference")
    args = ap.parse_args()

    adapter_factory = _load_factory(args.adapter_factory, ReferenceProductionHostileAdapter)
    hooks_factory = _load_factory(args.hooks_factory, NoopHostileFaultHooks)
    report = run_integrated_hostile_matrix(adapter_factory, hooks_factory)
    _write(args.output, report)
    if args.manifest_output is not None:
        _write(args.manifest_output, integration_binding_manifest())

    print(
        "INTG_HOSTILE_MATRIX="
        + report["status"]
        + f" passed={report['passed_cases']}/{report['case_count']}"
    )
    print("INTEGRATED_RUNTIME_QUALIFICATION_CLAIMED=NO")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
