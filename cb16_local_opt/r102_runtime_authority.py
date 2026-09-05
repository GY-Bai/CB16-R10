from __future__ import annotations

"""R10.2-R10.4 binding to the already-qualified Shanxi R8.1 runtime profile.

Performance authority only. Scientific knobs (H72, chronology, Teacher semantics,
optimizer epochs, R10 batch size, FINAL boundary) are never rewritten here.
The compact R9 authority is hash-bound to the full R8.1 qualification return;
R10 additionally vendors the exact R8.1 performance and final-profile JSONs so
selected worker limits remain directly auditable without carrying burn-in telemetry.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime_authority_r9 import sha256_file

EXPECTED_RUNTIME_AUTHORITY_FILE_SHA256 = "f50757da882cee0f6def11ec9c6e38e1065f415032f7d1405b249beb997e92df"
EXPECTED_RUNTIME_AUTHORITY_CONTENT_HASH = "07afedc7d42f78ed28506fc6e95a4f5193d131290991f159f3a3fd30e5820383"
EXPECTED_RUNTIME_PROFILE_HASH = "98471292867fca6fa19cc383f4fd0ef7567deec01581f488198916d99a9363a0"
EXPECTED_PERFORMANCE_FILE_SHA256 = "997492b1bdb9dcd7194a27719d890975aad27d2b80ffc9e713aada1a2eaeafca"
EXPECTED_LIMITS_FILE_SHA256 = "ed193de8e27e4bc2b29c68865f47173842cc61cc5d237fe655900865505235b1"
EXPECTED_FINAL_FILE_SHA256 = "842677cd1e79781b095745c8654a80719fa481affe4cc1d56fd5977faac2c504"


@dataclass(frozen=True)
class R102RuntimeParallelism:
    runtime_authority_content_hash: str
    runtime_profile_hash: str
    h72_workers: int
    h72_threads_per_worker: int
    h72_max_in_flight: int
    teacher_workers: int
    teacher_threads_per_worker: int
    experience_shards: int
    sqlite_synchronous: str
    loader_prefetch: int
    max_stage_concurrency: int
    do_not_run_h72_and_teacher_at_full_concurrency: bool
    ram_backpressure_high: float
    ram_hard_stop: float
    vram_backpressure_high: float
    disk_free_hard_stop_gib: float
    gpu_rollout_chunk_rows: int
    gpu_qualified_train_batch_ceiling: int
    gpu_cpu_torch_threads: int
    single_cuda_owner: bool

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "scientific_semantics_changed": False}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_r102_runtime_parallelism(
    package_root: str | Path,
    *,
    live_environment_check: bool = False,
) -> R102RuntimeParallelism:
    # Runtime-performance authority belongs to the exact Git source lineage, not
    # to the separately mounted Shanxi frozen-asset package. Keep package_root in
    # the API for caller compatibility but always bind R8.1 receipts from source.
    root = Path(__file__).resolve().parents[1]
    saved_path = root / "authority" / "SHANXI_RUNTIME_AUTHORITY_R9.json"
    limits_path = root / "authority" / "R8_1_qualification" / "R8_1_RUNTIME_LIMITS_FROZEN.json"
    final_path = root / "authority" / "R8_1_qualification" / "FINAL_QUALIFICATION_R8_1.json"

    if sha256_file(saved_path) != EXPECTED_RUNTIME_AUTHORITY_FILE_SHA256:
        raise RuntimeError("R8_1_SAVED_RUNTIME_AUTHORITY_FILE_SHA256_MISMATCH")
    if sha256_file(limits_path) != EXPECTED_LIMITS_FILE_SHA256:
        raise RuntimeError("R8_1_RUNTIME_LIMITS_FILE_SHA256_MISMATCH")
    if sha256_file(final_path) != EXPECTED_FINAL_FILE_SHA256:
        raise RuntimeError("R8_1_FINAL_FILE_SHA256_MISMATCH")

    saved = _load_json(saved_path)
    limits = _load_json(limits_path)
    final = _load_json(final_path)
    if saved.get("content_hash") != EXPECTED_RUNTIME_AUTHORITY_CONTENT_HASH:
        raise RuntimeError("R8_1_SAVED_RUNTIME_AUTHORITY_CONTENT_HASH_MISMATCH")
    if saved.get("runtime_profile_hash") != EXPECTED_RUNTIME_PROFILE_HASH:
        raise RuntimeError("R8_1_SAVED_RUNTIME_PROFILE_HASH_MISMATCH")
    source_files = saved.get("source_file_sha256", {})
    if source_files.get("performance") != EXPECTED_PERFORMANCE_FILE_SHA256:
        raise RuntimeError("R8_1_AUTHORITY_PERFORMANCE_SOURCE_HASH_MISMATCH")
    if source_files.get("final") != EXPECTED_FINAL_FILE_SHA256:
        raise RuntimeError("R8_1_AUTHORITY_FINAL_SOURCE_HASH_MISMATCH")
    final_profile = final.get("runtime_profile") or {}
    if final_profile.get("content_hash") != EXPECTED_RUNTIME_PROFILE_HASH:
        raise RuntimeError("R8_1_FINAL_PROFILE_CONTENT_HASH_MISMATCH")
    if saved.get("runtime_profile") != final_profile:
        raise RuntimeError("R8_1_SAVED_AUTHORITY_PROFILE_DIFFERS_FROM_FINAL_RECEIPT")
    if limits.get("source_performance_file_sha256") != EXPECTED_PERFORMANCE_FILE_SHA256:
        raise RuntimeError("R8_1_LIMITS_SOURCE_PERFORMANCE_HASH_MISMATCH")
    if limits.get("source_final_file_sha256") != EXPECTED_FINAL_FILE_SHA256:
        raise RuntimeError("R8_1_LIMITS_SOURCE_FINAL_HASH_MISMATCH")
    if limits.get("runtime_profile_hash") != EXPECTED_RUNTIME_PROFILE_HASH:
        raise RuntimeError("R8_1_LIMITS_PROFILE_HASH_MISMATCH")
    if int(limits["h72_worker_scaling"]["selected_workers"]) != 2 or limits["h72_worker_scaling"].get("status") != "PASS":
        raise RuntimeError("R8_1_H72_PERFORMANCE_SELECTION_DRIFT")
    if saved.get("qualification_final_status") not in {"READY_WITH_LIMITS", "READY_FOR_SHORT_REAL_CAMPAIGN"}:
        raise RuntimeError("R8_1_RUNTIME_AUTHORITY_NOT_READY")

    profile = saved["runtime_profile"]
    gpu = profile["gpu"]
    cpu = profile["cpu"]
    io = profile["io"]
    pipe = profile["pipeline"]
    lim = profile["resource_limits"]
    if profile.get("scientific_semantics_changed") is not False:
        raise RuntimeError("R8_1_RUNTIME_PROFILE_SCIENTIFIC_SEMANTICS_CHANGED")
    if gpu.get("dtype") != "fp32" or gpu.get("amp_enabled") is not False:
        raise RuntimeError("R8_1_GPU_NUMERIC_POLICY_DRIFT")
    if gpu.get("tier") != "TIER_1" or gpu.get("single_cuda_owner") is not True:
        raise RuntimeError("R8_1_GPU_OWNERSHIP_POLICY_DRIFT")
    if pipe.get("do_not_run_h72_and_teacher_at_full_concurrency") is not True:
        raise RuntimeError("R8_1_STAGE_CONCURRENCY_POLICY_DRIFT")

    # Optional live checks are deliberately narrow: this adapter must never
    # retune a machine or modify packages. The V6.3 Provisioner owns package
    # preparation; campaign prestart owns CUDA availability/device receipts.
    if live_environment_check:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("R8_1_LIVE_CUDA_UNAVAILABLE")
        cap = tuple(int(x) for x in torch.cuda.get_device_capability(0))
        if cap != (6, 1):
            raise RuntimeError(f"R8_1_LIVE_COMPUTE_CAPABILITY_DRIFT:{cap}")

    return R102RuntimeParallelism(
        runtime_authority_content_hash=EXPECTED_RUNTIME_AUTHORITY_CONTENT_HASH,
        runtime_profile_hash=EXPECTED_RUNTIME_PROFILE_HASH,
        h72_workers=int(cpu["h72_workers"]),
        h72_threads_per_worker=int(cpu["h72_threads_per_worker"]),
        h72_max_in_flight=int(cpu["h72_max_in_flight"]),
        teacher_workers=int(cpu["teacher_workers"]),
        teacher_threads_per_worker=int(cpu["teacher_threads_per_worker"]),
        experience_shards=int(io["experience_shards"]),
        sqlite_synchronous=str(io["sqlite_synchronous"]),
        loader_prefetch=int(pipe["loader_prefetch"]),
        max_stage_concurrency=int(pipe["max_stage_concurrency"]),
        do_not_run_h72_and_teacher_at_full_concurrency=bool(pipe["do_not_run_h72_and_teacher_at_full_concurrency"]),
        ram_backpressure_high=float(lim["ram_backpressure_high"]),
        ram_hard_stop=float(lim["ram_hard_stop"]),
        vram_backpressure_high=float(lim["vram_backpressure_high"]),
        disk_free_hard_stop_gib=float(lim["disk_free_hard_stop_gib"]),
        gpu_rollout_chunk_rows=int(gpu["rollout_chunk_rows"]),
        gpu_qualified_train_batch_ceiling=int(gpu["train_batch"]),
        gpu_cpu_torch_threads=int(gpu["cpu_torch_threads"]),
        single_cuda_owner=bool(gpu["single_cuda_owner"]),
    )
