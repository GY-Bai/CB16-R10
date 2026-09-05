from __future__ import annotations

"""Operational adjudication for the 5–10 generation R9 real historical short campaign.

This adjudicator deliberately separates:
1. FACTORY OPERATIONAL READINESS, and
2. MARKET/LEARNING SCIENCE RESULTS.

F0/F1/F2/F3 signs, promotion count and realized economic utility do NOT rescue or
invalidate the engineering factory gate.  They are reported as scientific findings.
The factory gate asks whether the real-data learning graph ran faithfully, durably and
without crossing FINAL boundaries.
"""

import dataclasses
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .campaign_telemetry_r9 import CampaignTelemetryReceiptR9
from .dataset_snapshot_r9 import DatasetSnapshotReceiptR9
from .encoder_authority_r9 import EncoderInstallationReceiptR9
from .real_multiasset_cache_r9 import MultiAssetCacheReceiptR9
from .runtime_authority_r9 import ShanxiRuntimeAuthorityR9


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ShortCampaignReadinessPolicyR9:
    min_attempts: int = 5
    max_attempts: int = 10
    require_controls_pass_every_generation: bool = True
    require_admitted_evidence_every_generation: bool = True
    require_nonzero_gradient_every_generation: bool = True
    ssd_hard_min_free_gib: float = 10.0
    ssd_preferred_free_gib: float = 30.0
    hdd_hard_min_free_gib: float = 20.0

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class ScientificObservationR9:
    generation_attempt: int
    f2_minus_f0_delta: float | None
    f3_minus_f2_delta: float | None
    interpretation: tuple[str, ...]


@dataclass(frozen=True)
class ShortCampaignAdjudicationR9:
    adjudication_version: str
    status: str
    operational_ready: bool
    hard_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    runtime_authority_hash: str
    encoder_authority_hash: str
    dataset_snapshot_hash: str
    multiasset_cache_hash: str
    telemetry_hash: str
    attempts: int
    promotions: int
    rejections: int
    final_generation: int
    mean_generation_wall_seconds: float
    mean_evidence_admission_fraction: float | None
    total_gradient_steps: int
    total_training_examples: int
    final_tournament_closed: bool
    ssd_free_gib: float
    hdd_free_gib: float
    scientific_observations: tuple[ScientificObservationR9, ...]
    scientific_observations_status_driving_for_factory_readiness: bool
    policy_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def _free_gib(path: str | Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def adjudicate_short_campaign_r9(
    *,
    runtime: ShanxiRuntimeAuthorityR9,
    encoder: EncoderInstallationReceiptR9,
    dataset: DatasetSnapshotReceiptR9,
    cache: MultiAssetCacheReceiptR9,
    telemetry: CampaignTelemetryReceiptR9,
    ssd_root: str | Path,
    hdd_root: str | Path,
    policy: ShortCampaignReadinessPolicyR9 | None = None,
) -> ShortCampaignAdjudicationR9:
    policy = policy or ShortCampaignReadinessPolicyR9()
    failures: list[str] = []
    warnings: list[str] = []

    if runtime.status != "SHANXI_RUNTIME_AUTHORITY_BOUND":
        failures.append("RUNTIME_AUTHORITY_NOT_BOUND")
    if runtime.scientific_semantics_changed:
        failures.append("RUNTIME_AUTHORITY_CHANGED_SCIENTIFIC_SEMANTICS")
    if encoder.authority != "USER_FROZEN_ENCODER":
        failures.append("REAL_FROZEN_ENCODER_NOT_INSTALLED")
    if dataset.status != "REAL_DATASET_SNAPSHOT_QUALIFIED":
        failures.append("REAL_DATASET_SNAPSHOT_NOT_QUALIFIED")
    if cache.status != "REAL_MULTI_ASSET_CACHE_QUALIFIED":
        failures.append("MULTIASSET_CACHE_NOT_QUALIFIED")
    if cache.scientific_dataset_hash != dataset.scientific_dataset_hash:
        failures.append("CACHE_DATASET_IDENTITY_MISMATCH")
    if cache.encoder_weight_hash != encoder.state_dict_weight_hash:
        failures.append("CACHE_ENCODER_IDENTITY_MISMATCH")
    if cache.trader_multiasset_input_enabled:
        failures.append("UNAUTHORIZED_MULTI_ASSET_TRADER_INPUT_CHANGE")

    if not policy.min_attempts <= telemetry.attempts <= policy.max_attempts:
        failures.append("SHORT_CAMPAIGN_ATTEMPT_COUNT_OUT_OF_RANGE")
    if len(telemetry.generations) != telemetry.attempts:
        failures.append("TELEMETRY_GENERATION_COUNT_MISMATCH")
    if telemetry.final_tournament_opened_anywhere_in_campaign:
        failures.append("FINAL_TOURNAMENT_WAS_OPENED")

    for g in telemetry.generations:
        if policy.require_controls_pass_every_generation and g.validation_controls_status != "PASS":
            failures.append(f"VALIDATION_CONTROLS_NOT_PASS:ATTEMPT_{g.attempt_index}")
        if policy.require_admitted_evidence_every_generation:
            if not g.evidence_admitted or g.evidence_admitted <= 0:
                failures.append(f"NO_ADMITTED_EVIDENCE:ATTEMPT_{g.attempt_index}")
        if policy.require_nonzero_gradient_every_generation:
            if (
                g.gradient_steps is None
                or g.gradient_steps <= 0
                or g.min_gradient_norm is None
                or g.min_gradient_norm <= 0
            ):
                failures.append(f"NO_NONZERO_GRADIENT:ATTEMPT_{g.attempt_index}")
        if g.adjudication_verdict not in {"PROMOTE", "REJECT"}:
            failures.append(f"MISSING_PROMOTE_REJECT_VERDICT:ATTEMPT_{g.attempt_index}")
        if g.final_tournament_opened:
            failures.append(f"FINAL_BOUNDARY_CROSSED:ATTEMPT_{g.attempt_index}")
        for k, v in g.validation_qcrps.items():
            if not isinstance(v, (float, int)) or not (float(v) >= 0.0):
                failures.append(f"NONFINITE_OR_NEGATIVE_QCRPS:{g.attempt_index}:{k}")

    ssd_free = _free_gib(ssd_root)
    hdd_free = _free_gib(hdd_root)
    if ssd_free < policy.ssd_hard_min_free_gib:
        failures.append("SSD_FREE_BELOW_HARD_MINIMUM")
    elif ssd_free < policy.ssd_preferred_free_gib:
        warnings.append("SSD_FREE_BELOW_PREFERRED_LONG_FACTORY_HEADROOM")
    if hdd_free < policy.hdd_hard_min_free_gib:
        failures.append("HDD_FREE_BELOW_HARD_MINIMUM")

    observations: list[ScientificObservationR9] = []
    for g in telemetry.generations:
        interp = []
        if g.f2_minus_f0_delta is not None:
            interp.append(
                "F2_TRUE_MARKET_ACCOUNT_BETTER_THAN_F0"
                if g.f2_minus_f0_delta < 0
                else "F2_TRUE_MARKET_ACCOUNT_NOT_BETTER_THAN_F0"
            )
        if g.f3_minus_f2_delta is not None:
            interp.append(
                "F2_TRUE_MARKET_ACCOUNT_BETTER_THAN_F3_SHUFFLE"
                if g.f3_minus_f2_delta > 0
                else "F2_TRUE_MARKET_ACCOUNT_NOT_BETTER_THAN_F3_SHUFFLE"
            )
        observations.append(ScientificObservationR9(
            generation_attempt=g.attempt_index,
            f2_minus_f0_delta=g.f2_minus_f0_delta,
            f3_minus_f2_delta=g.f3_minus_f2_delta,
            interpretation=tuple(interp),
        ))

    failures = sorted(set(failures))
    warnings = sorted(set(warnings))
    ready = not failures
    return ShortCampaignAdjudicationR9(
        adjudication_version="CB16_REAL_HISTORICAL_SHORT_CAMPAIGN_ADJUDICATION_R9",
        status=(
            "READY_FOR_LONG_HISTORICAL_FACTORY"
            if ready else "NOT_READY_FOR_LONG_HISTORICAL_FACTORY"
        ),
        operational_ready=ready,
        hard_failures=tuple(failures),
        warnings=tuple(warnings),
        runtime_authority_hash=runtime.content_hash,
        encoder_authority_hash=encoder.content_hash,
        dataset_snapshot_hash=dataset.content_hash,
        multiasset_cache_hash=cache.content_hash,
        telemetry_hash=telemetry.content_hash,
        attempts=telemetry.attempts,
        promotions=telemetry.promotions,
        rejections=telemetry.rejections,
        final_generation=telemetry.final_generation,
        mean_generation_wall_seconds=telemetry.mean_cycle_wall_seconds,
        mean_evidence_admission_fraction=telemetry.mean_evidence_admission_fraction,
        total_gradient_steps=telemetry.total_gradient_steps,
        total_training_examples=telemetry.total_training_examples,
        final_tournament_closed=not telemetry.final_tournament_opened_anywhere_in_campaign,
        ssd_free_gib=ssd_free,
        hdd_free_gib=hdd_free,
        scientific_observations=tuple(observations),
        scientific_observations_status_driving_for_factory_readiness=False,
        policy_hash=policy.content_hash,
    )


def save_short_campaign_adjudication_r9(
    receipt: ShortCampaignAdjudicationR9,
    path: str | Path,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**asdict(receipt), "content_hash": receipt.content_hash}, indent=2) + "\n")
    return p
