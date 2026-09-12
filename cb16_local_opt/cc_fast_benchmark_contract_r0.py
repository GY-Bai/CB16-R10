
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from typing import Any, Mapping, Sequence

BENCHMARK_SCHEMA = "CB16_R11_CC_FAST_BENCHMARK_REPORT_V1"
SEMANTIC_PASS = "PASS"
SEMANTIC_FAIL = "FAIL"


def _finite_nonnegative(v: float, name: str) -> float:
    x = float(v)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name}_INVALID")
    return x


@dataclass(frozen=True)
class BenchmarkReport:
    schema_version: str
    workload_identity: str
    code_identity: str
    science_identity: str
    topology: Mapping[str, Any]
    account_count: int
    decision_rate: float
    batch_size_distribution: Mapping[str, float]
    compliant_transitions_per_s: float
    policy_decisions_per_s: float
    wall_clock_s: float
    cpu_per_core_utilization: tuple[float, ...]
    pss_bytes: int
    cgroup_memory_bytes: int | None
    page_faults: int
    swap_in_bytes: int
    swap_out_bytes: int
    disk_read_bytes_per_s: float
    disk_write_bytes_per_s: float
    io_wait_fraction: float
    gpu_vram_bytes: int | None
    gpu_kernel_ms: float | None
    gpu_transfer_ms: float | None
    queue_bytes: int
    queue_depth: int
    queue_oldest_age_s: float
    correctness_checksum: str
    semantic_verdict: str
    notes: tuple[str, ...] = ()

    def validate(self) -> "BenchmarkReport":
        if self.schema_version != BENCHMARK_SCHEMA:
            raise ValueError("BENCHMARK_SCHEMA_MISMATCH")
        for name in ("workload_identity", "code_identity", "science_identity", "correctness_checksum"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name.upper()}_EMPTY")
        if self.account_count <= 0:
            raise ValueError("ACCOUNT_COUNT_INVALID")
        for name in ("decision_rate", "compliant_transitions_per_s", "policy_decisions_per_s", "wall_clock_s", "disk_read_bytes_per_s", "disk_write_bytes_per_s", "io_wait_fraction", "queue_oldest_age_s"):
            _finite_nonnegative(getattr(self, name), name)
        if self.pss_bytes < 0 or self.page_faults < 0 or self.swap_in_bytes < 0 or self.swap_out_bytes < 0:
            raise ValueError("COUNTER_NEGATIVE")
        if self.queue_bytes < 0 or self.queue_depth < 0:
            raise ValueError("QUEUE_COUNTER_NEGATIVE")
        if not self.cpu_per_core_utilization:
            raise ValueError("CPU_UTIL_EMPTY")
        if any((not math.isfinite(float(x)) or float(x) < 0 or float(x) > 100) for x in self.cpu_per_core_utilization):
            raise ValueError("CPU_UTIL_INVALID")
        if self.semantic_verdict not in {SEMANTIC_PASS, SEMANTIC_FAIL}:
            raise ValueError("SEMANTIC_VERDICT_MISSING")
        if self.semantic_verdict != SEMANTIC_PASS:
            raise ValueError("BENCHMARK_NOT_SEMANTICALLY_QUALIFIED")
        if self.wall_clock_s <= 0:
            raise ValueError("WALL_CLOCK_ZERO")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        out = asdict(self)
        out["topology"] = dict(self.topology)
        out["batch_size_distribution"] = dict(self.batch_size_distribution)
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
