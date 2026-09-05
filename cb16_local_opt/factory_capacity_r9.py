from __future__ import annotations

"""Simple capacity projection from an observed R9 short campaign.

This is an operational projection, not a scientific extrapolation.  It uses observed
wall-clock and storage growth only to estimate the order of magnitude for 50/100/250
attempt long campaigns on the same frozen runtime profile.
"""

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .campaign_telemetry_r9 import CampaignTelemetryReceiptR9


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class FactoryCapacityProjectionRowR9:
    attempts: int
    estimated_wall_seconds: float
    estimated_wall_hours: float
    estimated_hot_and_run_bytes: float
    estimated_hot_and_run_gib: float


@dataclass(frozen=True)
class FactoryCapacityProjectionR9:
    projection_version: str
    source_telemetry_hash: str
    observed_attempts: int
    observed_mean_cycle_wall_seconds: float
    observed_total_bytes: int
    observed_bytes_per_attempt: float
    rows: tuple[FactoryCapacityProjectionRowR9, ...]
    caveat: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def project_factory_capacity_r9(
    telemetry: CampaignTelemetryReceiptR9,
    attempts: Sequence[int] = (50, 100, 250),
) -> FactoryCapacityProjectionR9:
    if telemetry.attempts <= 0:
        raise ValueError("no attempts")
    # Includes cycle dirs + current Experience Lake + checkpoints.  It intentionally
    # over-counts fixed base files slightly rather than under-budgeting disk.
    observed = (
        telemetry.total_cycle_directory_bytes
        + telemetry.experience_lake_bytes
        + telemetry.checkpoint_bytes
    )
    per = observed / telemetry.attempts
    rows = tuple(
        FactoryCapacityProjectionRowR9(
            attempts=int(n),
            estimated_wall_seconds=float(n * telemetry.mean_cycle_wall_seconds),
            estimated_wall_hours=float(n * telemetry.mean_cycle_wall_seconds / 3600.0),
            estimated_hot_and_run_bytes=float(n * per),
            estimated_hot_and_run_gib=float(n * per / 1024**3),
        )
        for n in attempts
    )
    return FactoryCapacityProjectionR9(
        projection_version="CB16_FACTORY_CAPACITY_PROJECTION_R9",
        source_telemetry_hash=telemetry.content_hash,
        observed_attempts=telemetry.attempts,
        observed_mean_cycle_wall_seconds=telemetry.mean_cycle_wall_seconds,
        observed_total_bytes=observed,
        observed_bytes_per_attempt=per,
        rows=rows,
        caveat=(
            "Operational linear projection only; Teacher support, compaction, retention, "
            "dataset growth and model tier changes can alter real scaling."
        ),
    )


def save_factory_capacity_projection_r9(r, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**asdict(r), "content_hash": r.content_hash}, indent=2) + "\n")
    return p
