from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch

from cb16_local_opt.frozen_sensory_stack_r10 import FrozenSensoryStackR10
from cb16_local_opt.longtraj_infra_closure_r0 import (
    MinuteFrozenPhysicsAdapterR0,
    recursive_state_sha256_r0,
    sha256_file_r0,
)
from cb16_local_opt.training_runtime_r11 import PreparedEvidenceR11, TrainingRuntimeR11
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "CB16_R11_LONGTRAJ_INFRA_STRESS_R0_RESULT_V1"


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("INFRA_STRESS_CUDA_REQUESTED_BUT_UNAVAILABLE")
    return requested


class NvidiaSMISampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, float]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = shutil.which("nvidia-smi") is not None

    @staticmethod
    def _query() -> dict[str, float]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ]
        raw = subprocess.check_output(cmd, text=True, timeout=5).strip().splitlines()
        if not raw:
            raise RuntimeError("NVIDIA_SMI_EMPTY")
        vals = [x.strip() for x in raw[0].split(",")]
        if len(vals) != 5:
            raise RuntimeError(f"NVIDIA_SMI_COLUMN_DRIFT:{raw[0]}")
        return {
            "gpu_util_pct": float(vals[0]),
            "memory_util_pct": float(vals[1]),
            "memory_used_mib": float(vals[2]),
            "power_draw_w": float(vals[3]),
            "power_limit_w": float(vals[4]),
        }

    def start(self) -> None:
        if not self.available:
            return

        def run() -> None:
            while not self._stop.is_set():
                try:
                    self.samples.append(self._query())
                except Exception as exc:
                    self.errors.append(f"{type(exc).__name__}:{exc}")
                self._stop.wait(1.0)

        self._thread = threading.Thread(target=run, name="cb16-nvidia-smi-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {
                "available": bool(self.available),
                "sample_count": 0,
                "errors": self.errors[-10:],
            }
        keys = ("gpu_util_pct", "memory_util_pct", "memory_used_mib", "power_draw_w", "power_limit_w")
        out: dict[str, Any] = {
            "available": True,
            "sample_count": len(self.samples),
            "errors": self.errors[-10:],
        }
        for key in keys:
            values = [float(x[key]) for x in self.samples]
            out[f"{key}_avg"] = sum(values) / len(values)
            out[f"{key}_max"] = max(values)
            out[f"{key}_min"] = min(values)
        return out


def replicate_prepared(base: PreparedEvidenceR11, *, rows: int, device: str) -> PreparedEvidenceR11:
    require(base.rows > 0, "INFRA_STRESS_EMPTY_BASE_EVIDENCE")
    ids = torch.arange(rows, device=base.packed.device, dtype=torch.long) % base.rows
    packed = base.packed.index_select(0, ids).detach().clone().to(device)
    parent_ids = tuple(f"STRESS:{i}:{base.parent_ids[i % base.rows]}" for i in range(rows))
    group_ids = tuple(f"STRESS_G:{i}:{base.dependence_group_ids[i % base.rows]}" for i in range(rows))
    out = PreparedEvidenceR11(
        parent_ids=parent_ids,
        dependence_group_ids=group_ids,
        packed=packed,
        evidence_hash=f"STRESS_ONLY:{base.evidence_hash}:{rows}",
        host_to_device_transfers=0,
        h2d_strategy="STRESS_FIXTURE_DEVICE_LOCAL",
    )
    out.validate()
    return out


def make_flat_action(adapter: MinuteFrozenPhysicsAdapterR0, snapshot, risk_auth, symbol: str, row, ordinal: int):
    return adapter.step_intent(
        snapshot,
        risk_auth,
        direction_v55=1,
        risk=0.0,
        symbol=symbol,
        transition_row=row,
        funding_rate=0.0,
        trace_id=f"INFRA_STRESS_FLAT:{ordinal}",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--package-root", default=os.environ.get("CB16_PACKAGE_ROOT", "/cb16/package"))
    ap.add_argument("--duration-seconds", type=int, default=300)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--output", required=True)
    ap.add_argument("--stress-evidence-rows", type=int, default=512)
    ap.add_argument("--physics-steps-per-cycle", type=int, default=64)
    args = ap.parse_args()

    require(int(args.duration_seconds) == 300, "INFRA_STRESS_DURATION_MUST_BE_EXACTLY_300_SECONDS")
    require(int(args.stress_evidence_rows) >= 512, "INFRA_STRESS_EVIDENCE_ROWS_BELOW_512")
    require(int(args.physics_steps_per_cycle) >= 1, "INFRA_STRESS_PHYSICS_STEPS_INVALID")

    device = choose_device(args.device)
    fixture_path = Path(args.fixture).resolve()
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    require(isinstance(fixture, dict), "INFRA_STRESS_FIXTURE_INVALID")

    prepared_raw = fixture["prepared"]
    base_prepared = PreparedEvidenceR11(
        parent_ids=tuple(prepared_raw["parent_ids"]),
        dependence_group_ids=tuple(prepared_raw["dependence_group_ids"]),
        packed=prepared_raw["packed"].to(dtype=torch.float32, device=device),
        evidence_hash=str(prepared_raw["evidence_hash"]),
        host_to_device_transfers=0,
        h2d_strategy="INFRA_CLOSURE_STRESS_FIXTURE",
    )
    base_prepared.validate()
    prepared = replicate_prepared(
        base_prepared,
        rows=int(args.stress_evidence_rows),
        device=device,
    )

    sensory_frames = list(fixture["sensory_frames"])
    physics_records = list(fixture["physics_records"])
    archive_identity = dict(fixture["archive_identity"])
    require(len(sensory_frames) >= 1, "INFRA_STRESS_NO_SENSORY_FRAMES")
    require(len(physics_records) >= 2, "INFRA_STRESS_NO_PHYSICS_RECORDS")
    require(archive_identity.get("final_holdout_touched") is False, "INFRA_STRESS_FINAL_HOLDOUT_FLAG_INVALID")

    sensory = FrozenSensoryStackR10(args.package_root, device=device, verify_hashes=True)
    adapter = MinuteFrozenPhysicsAdapterR0(args.package_root)
    snapshot, risk_auth = adapter.initialize("INFRA_STRESS_ACCOUNT", 1.0)

    torch.manual_seed(24_680)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(24_680)
    model = build_g0_brain_r10("TIER_1", seed=24_680, device=device)
    runtime = TrainingRuntimeR11(device=device)
    optimizer = runtime.build_optimizer(model)
    ids = torch.arange(prepared.rows, device=prepared.packed.device, dtype=torch.long)

    sensory.encode_frames(sensory_frames[: min(8, len(sensory_frames))])
    with torch.inference_mode():
        model(prepared.operator48, prepared.medium48, prepared.account6)
    runtime._train_step(
        model=model,
        optimizer=optimizer,
        prepared=prepared,
        ids=ids,
        runtime_prevalidated=False,
        collect_diagnostics=False,
        materialize_result=False,
    )
    warm_row = physics_records[1]
    warm_step = make_flat_action(adapter, snapshot, risk_auth, str(archive_identity["symbol"]), warm_row, 0)
    snapshot = warm_step["snapshot_t1"]
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    sampler = NvidiaSMISampler()
    sampler.start()

    sensory_frames_done = 0
    inference_rows_done = 0
    optimizer_steps_done = 0
    physics_steps_done = 0
    physics_path_resets = 0
    cycles = 0
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    deadline = start_wall + float(args.duration_seconds)
    physics_index = 2

    try:
        while time.perf_counter() < deadline:
            enc_chunk = sensory_frames[: min(32, len(sensory_frames))]
            sensory.encode_frames(enc_chunk)
            sensory_frames_done += len(enc_chunk)

            with torch.inference_mode():
                out = model(prepared.operator48, prepared.medium48, prepared.account6)
                if not torch.isfinite(out["direction_logits"]).all().item():
                    raise RuntimeError("INFRA_STRESS_NONFINITE_INFERENCE")
                if not torch.isfinite(out["requested_risk_raw"]).all().item():
                    raise RuntimeError("INFRA_STRESS_NONFINITE_RISK")
            inference_rows_done += prepared.rows

            runtime._train_step(
                model=model,
                optimizer=optimizer,
                prepared=prepared,
                ids=ids,
                runtime_prevalidated=True,
                collect_diagnostics=False,
                materialize_result=False,
            )
            optimizer_steps_done += 1

            for _ in range(int(args.physics_steps_per_cycle)):
                if physics_index >= len(physics_records):
                    physics_path_resets += 1
                    snapshot, risk_auth = adapter.initialize(
                        f"INFRA_STRESS_ACCOUNT:{physics_path_resets}",
                        1.0,
                    )
                    physics_index = 1
                row = physics_records[physics_index]
                step = make_flat_action(
                    adapter,
                    snapshot,
                    risk_auth,
                    str(archive_identity["symbol"]),
                    row,
                    physics_steps_done + 1,
                )
                snapshot = step["snapshot_t1"]
                physics_steps_done += 1
                physics_index += 1
                if snapshot["termination_state"]["terminated"]:
                    raise RuntimeError("INFRA_STRESS_FLAT_ACCOUNT_TERMINATED")
            cycles += 1
    finally:
        if device == "cuda":
            torch.cuda.synchronize()
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        sampler.stop()

    wall = float(end_wall - start_wall)
    cpu_seconds = float(end_cpu - start_cpu)
    require(wall >= 300.0, f"INFRA_STRESS_WALL_UNDERRUN:{wall}")
    require(cycles > 0 and optimizer_steps_done > 0 and physics_steps_done > 0, "INFRA_STRESS_ZERO_WORK")
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "INFRA_STRESS_NONFINITE_MODEL")

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_kib = int(rusage.ru_maxrss)
    cuda_stats: dict[str, Any] = {}
    if device == "cuda":
        cuda_stats = {
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "current_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "current_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
            "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        }

    smi = sampler.summary()
    avg_gpu = smi.get("gpu_util_pct_avg")
    if isinstance(avg_gpu, (int, float)):
        if avg_gpu >= 60.0:
            utilization_class = "GOOD"
        elif avg_gpu >= 30.0:
            utilization_class = "MODERATE"
        else:
            utilization_class = "LOW__PERFORMANCE_FOLLOWUP_RECOMMENDED"
    elif device == "cuda":
        utilization_class = "CUDA_ACTIVE__NVIDIA_SMI_METRICS_UNAVAILABLE"
    else:
        utilization_class = "CPU_ONLY_RUN__GPU_UTILIZATION_NOT_APPLICABLE"

    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": "INFRA_300S_HARDWARE_STRESS_INTEGRITY_PASS",
        "fixture": {
            "path": str(fixture_path),
            "sha256": sha256_file_r0(fixture_path),
            "archive_identity": archive_identity,
            "base_evidence_hash": base_prepared.evidence_hash,
            "stress_replication_rows": prepared.rows,
            "scientific_weight": "NONE__THROUGHPUT_ONLY",
        },
        "device": device,
        "duration": {
            "requested_seconds": 300,
            "measured_wall_seconds": wall,
            "process_cpu_seconds": cpu_seconds,
            "process_cpu_to_wall_ratio": cpu_seconds / wall,
        },
        "throughput": {
            "cycles": cycles,
            "sensory_frames": sensory_frames_done,
            "sensory_frames_per_second": sensory_frames_done / wall,
            "student_inference_rows": inference_rows_done,
            "student_inference_rows_per_second": inference_rows_done / wall,
            "optimizer_steps": optimizer_steps_done,
            "optimizer_steps_per_second": optimizer_steps_done / wall,
            "scalar_frozen_physics_steps": physics_steps_done,
            "scalar_frozen_physics_steps_per_second": physics_steps_done / wall,
            "physics_path_resets_after_fixture_end": physics_path_resets,
        },
        "memory": {
            "process_peak_rss_kib": max_rss_kib,
            "torch_cuda": cuda_stats,
        },
        "nvidia_smi": smi,
        "hardware_efficiency": {
            "classification": utilization_class,
            "interpretation": (
                "GPU utilization is diagnostic throughput evidence only; low utilization is not a scientific failure. "
                "Any optimization must preserve exact causal and Frozen Physics parity gates."
            ),
        },
        "integrity_gates": {
            "EXACT_300_SECOND_WALL_PASS": wall >= 300.0,
            "NONZERO_WORK_PASS": cycles > 0 and optimizer_steps_done > 0 and physics_steps_done > 0,
            "FINITE_MODEL_PASS": True,
            "NO_RUNTIME_CRASH_OR_OOM_PASS": True,
            "FINAL_FIREWALL_PASS": archive_identity.get("final_holdout_touched") is False,
        },
        "final_holdout_touched": False,
        "fresh_market_data_downloaded": False,
        "scientific_verdict": None,
        "market_information_claim": "NONE__INFRA_STRESS_ONLY",
        "model_state_sha256": recursive_state_sha256_r0(model.state_dict()),
        "optimizer_state_sha256": recursive_state_sha256_r0(optimizer.state_dict()),
    }
    require(all(result["integrity_gates"].values()), "INFRA_STRESS_INTEGRITY_GATE_FAIL")

    out_path = Path(args.output).resolve()
    atomic_json(out_path, result)
    out_path.with_suffix(out_path.suffix + ".sha256").write_text(
        sha256_file_r0(out_path) + "  " + out_path.name + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "classification": result["classification"],
        "device": device,
        "wall_seconds": wall,
        "throughput": result["throughput"],
        "hardware_efficiency": result["hardware_efficiency"],
        "nvidia_smi": smi,
        "scientific_verdict": None,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
