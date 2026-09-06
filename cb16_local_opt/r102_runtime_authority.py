from __future__ import annotations

"""R10 binding to R8.1 authority plus R8.2 performance-only execution overlay.

R8.1 remains the immutable hardware qualification baseline. R8.2 may change only
runtime scheduling/resource controls; it cannot rewrite H72, chronology, Teacher
semantics, optimizer epochs, scientific batch size, Physics, or FINAL boundaries.
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
EXPECTED_R8_2_OVERLAY_FILE_SHA256 = "3bf2592cbab55431f798578aa72513df79f38ee1b7509c4a6223b2f4b22a0115"


@dataclass(frozen=True)
class R102RuntimeParallelism:
    runtime_authority_content_hash: str
    runtime_profile_hash: str
    performance_overlay: str
    performance_overlay_file_sha256: str
    h72_workers: int
    h72_threads_per_worker: int
    h72_max_in_flight: int
    h72_minimum_workers: int
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
    cache_hit_first: bool
    runtime_scheduling_identity_is_not_scientific_cache_identity: bool

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
    root = Path(__file__).resolve().parents[1]
    saved_path = root / "authority" / "SHANXI_RUNTIME_AUTHORITY_R9.json"
    limits_path = root / "authority" / "R8_1_qualification" / "R8_1_RUNTIME_LIMITS_FROZEN.json"
    final_path = root / "authority" / "R8_1_qualification" / "FINAL_QUALIFICATION_R8_1.json"
    overlay_path = root / "configs" / "R10_RUNTIME_PERFORMANCE_R8_2.json"

    if sha256_file(saved_path) != EXPECTED_RUNTIME_AUTHORITY_FILE_SHA256:
        raise RuntimeError("R8_1_SAVED_RUNTIME_AUTHORITY_FILE_SHA256_MISMATCH")
    if sha256_file(limits_path) != EXPECTED_LIMITS_FILE_SHA256:
        raise RuntimeError("R8_1_RUNTIME_LIMITS_FILE_SHA256_MISMATCH")
    if sha256_file(final_path) != EXPECTED_FINAL_FILE_SHA256:
        raise RuntimeError("R8_1_FINAL_FILE_SHA256_MISMATCH")
    if sha256_file(overlay_path) != EXPECTED_R8_2_OVERLAY_FILE_SHA256:
        raise RuntimeError("R8_2_PERFORMANCE_OVERLAY_FILE_SHA256_MISMATCH")

    saved = _load_json(saved_path)
    limits = _load_json(limits_path)
    final = _load_json(final_path)
    overlay = _load_json(overlay_path)

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

    if overlay.get("schema") != "CB16_R10_RUNTIME_PERFORMANCE_R8_2_V1":
        raise RuntimeError("R8_2_OVERLAY_SCHEMA_MISMATCH")
    if overlay.get("status") != "ACTIVE_PERFORMANCE_ONLY_OVERRIDE":
        raise RuntimeError("R8_2_OVERLAY_NOT_ACTIVE")
    if overlay.get("base_runtime_authority_content_hash") != EXPECTED_RUNTIME_AUTHORITY_CONTENT_HASH:
        raise RuntimeError("R8_2_BASE_AUTHORITY_HASH_MISMATCH")
    if overlay.get("base_runtime_profile_hash") != EXPECTED_RUNTIME_PROFILE_HASH:
        raise RuntimeError("R8_2_BASE_PROFILE_HASH_MISMATCH")
    if overlay.get("scientific_semantics_changed") is not False:
        raise RuntimeError("R8_2_OVERLAY_SCIENTIFIC_SEMANTICS_CHANGED")
    scientific = overlay.get("scientific_values_preserved") or {}
    required_preserved = {
        "horizon_hours": 72,
        "training_batch_size": 512,
        "epochs_per_attempt": 12,
        "learning_rate": 0.0003,
        "teacher_semantics_changed": False,
        "physics_changed": False,
        "chronology_changed": False,
        "purge_changed": False,
        "final_holdout_policy_changed": False,
    }
    for k, expected in required_preserved.items():
        if scientific.get(k) != expected:
            raise RuntimeError(f"R8_2_SCIENTIFIC_PRESERVATION_DRIFT:{k}")

    h72 = overlay.get("h72") or {}
    teacher = overlay.get("teacher") or {}
    cache = overlay.get("cache_policy") or {}
    h72_workers = int(h72.get("workers_start", 0))
    h72_threads = int(h72.get("threads_per_worker", 0))
    h72_in_flight = int(h72.get("max_in_flight", 0))
    h72_minimum = int(h72.get("minimum_workers", 0))
    ram_high = float(h72.get("ram_backpressure_high", 0.0))
    ram_hard = float(h72.get("ram_hard_stop", 0.0))
    if h72_workers != 6:
        raise RuntimeError("R8_2_H72_WORKER_START_DRIFT")
    if h72_threads != 1 or h72_minimum != 1:
        raise RuntimeError("R8_2_H72_THREAD_OR_MINIMUM_DRIFT")
    if h72_in_flight < h72_workers:
        raise RuntimeError("R8_2_H72_MAX_IN_FLIGHT_LT_WORKERS")
    if not (0.0 < ram_high < ram_hard < 1.0):
        raise RuntimeError("R8_2_RAM_THRESHOLDS_INVALID")
    if h72.get("hard_pressure_action") != "TERMINATE_ONE_ACTIVE_WORKER_AND_REQUEUE_PURE_H72_JOB":
        raise RuntimeError("R8_2_HARD_RAM_ACTION_DRIFT")

    teacher_workers = int(teacher.get("workers", 0))
    teacher_threads = int(teacher.get("threads_per_worker", 0))
    teacher_in_flight = int(teacher.get("max_in_flight", 0))
    teacher_minimum = int(teacher.get("minimum_workers", 0))
    if teacher_workers != 6:
        raise RuntimeError("R8_2_TEACHER_WORKER_START_DRIFT")
    if teacher_threads != 1 or teacher_minimum != 1:
        raise RuntimeError("R8_2_TEACHER_THREAD_OR_MINIMUM_DRIFT")
    if teacher_in_flight < teacher_workers:
        raise RuntimeError("R8_2_TEACHER_MAX_IN_FLIGHT_LT_WORKERS")
    if float(teacher.get("ram_backpressure_high", 0.0)) != ram_high or float(teacher.get("ram_hard_stop", 0.0)) != ram_hard:
        raise RuntimeError("R8_2_TEACHER_RAM_POLICY_DRIFT")
    if teacher.get("hard_pressure_action") != "TERMINATE_ONE_ACTIVE_WORKER_AND_REQUEUE_PURE_TEACHER_JOB":
        raise RuntimeError("R8_2_TEACHER_HARD_RAM_ACTION_DRIFT")

    if cache.get("cache_hit_first") is not True:
        raise RuntimeError("R8_2_CACHE_HIT_FIRST_DISABLED")
    if cache.get("runtime_scheduling_identity_is_not_scientific_cache_identity") is not True:
        raise RuntimeError("R8_2_RUNTIME_IDENTITY_ILLEGALLY_INVALIDATES_CACHE")

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
        performance_overlay="R8_2_6W_RAM_ADAPTIVE",
        performance_overlay_file_sha256=EXPECTED_R8_2_OVERLAY_FILE_SHA256,
        h72_workers=h72_workers,
        h72_threads_per_worker=h72_threads,
        h72_max_in_flight=h72_in_flight,
        h72_minimum_workers=h72_minimum,
        teacher_workers=teacher_workers,
        teacher_threads_per_worker=teacher_threads,
        experience_shards=int(io["experience_shards"]),
        sqlite_synchronous=str(io["sqlite_synchronous"]),
        loader_prefetch=int(pipe["loader_prefetch"]),
        max_stage_concurrency=int(pipe["max_stage_concurrency"]),
        do_not_run_h72_and_teacher_at_full_concurrency=bool(pipe["do_not_run_h72_and_teacher_at_full_concurrency"]),
        ram_backpressure_high=ram_high,
        ram_hard_stop=ram_hard,
        vram_backpressure_high=float(lim["vram_backpressure_high"]),
        disk_free_hard_stop_gib=float(lim["disk_free_hard_stop_gib"]),
        gpu_rollout_chunk_rows=int(gpu["rollout_chunk_rows"]),
        gpu_qualified_train_batch_ceiling=int(gpu["train_batch"]),
        gpu_cpu_torch_threads=int(gpu["cpu_torch_threads"]),
        single_cuda_owner=bool(gpu["single_cuda_owner"]),
        cache_hit_first=True,
        runtime_scheduling_identity_is_not_scientific_cache_identity=True,
    )
