from __future__ import annotations

"""Gate continuous R9 historical data-factory activation on short-campaign qualification.

The activation receipt deliberately separates always-on INGEST/CATALOG from campaign
snapshots.  New bundles may arrive continuously, but training never follows a mutable
catalog head.  A new campaign requires an explicitly frozen CatalogSnapshot.
"""

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class LongFactoryPolicyR9:
    continuous_ingest: bool = True
    continuous_catalog_registration: bool = True
    auto_train_from_mutable_catalog_head: bool = False
    require_frozen_snapshot_per_campaign: bool = True
    final_holdout_auto_open: bool = False
    final_controls_auto_open: bool = False
    default_long_campaign_attempt_cap: int = 100
    maintenance_every_generations: int = 10
    wal_checkpoint_every_generations: int = 5
    retention_scan_every_generations: int = 10

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class LongFactoryActivationReceiptR9:
    activation_version: str
    status: str
    short_campaign_adjudication_hash: str
    runtime_authority_hash: str
    encoder_authority_hash: str
    dataset_snapshot_hash: str
    policy: LongFactoryPolicyR9
    final_holdout_auto_open: bool
    mutable_catalog_training_forbidden: bool

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def activate_long_factory_r9(
    *,
    short_campaign_adjudication_path: str | Path,
    policy: LongFactoryPolicyR9 | None = None,
) -> LongFactoryActivationReceiptR9:
    obj = json.loads(Path(short_campaign_adjudication_path).read_text())
    claimed = obj.get("content_hash")
    payload = dict(obj)
    payload.pop("content_hash", None)
    if claimed and canonical_hash(payload) != claimed:
        raise RuntimeError("R9_SHORT_ADJUDICATION_HASH_MISMATCH")
    if obj.get("status") != "READY_FOR_LONG_HISTORICAL_FACTORY":
        raise RuntimeError("R9_LONG_FACTORY_BLOCKED_BY_SHORT_CAMPAIGN")
    if obj.get("hard_failures"):
        raise RuntimeError("R9_LONG_FACTORY_SHORT_CAMPAIGN_HAS_HARD_FAILURES")
    if obj.get("final_tournament_closed") is not True:
        raise RuntimeError("R9_LONG_FACTORY_FINAL_BOUNDARY_NOT_CLOSED")
    policy = policy or LongFactoryPolicyR9()
    if policy.auto_train_from_mutable_catalog_head:
        raise RuntimeError("R9_MUTABLE_CATALOG_HEAD_TRAINING_FORBIDDEN")
    if policy.final_holdout_auto_open or policy.final_controls_auto_open:
        raise RuntimeError("R9_FINAL_AUTO_OPEN_FORBIDDEN")
    return LongFactoryActivationReceiptR9(
        activation_version="CB16_LONG_HISTORICAL_FACTORY_ACTIVATION_R9",
        status="LONG_HISTORICAL_FACTORY_ACTIVATED",
        short_campaign_adjudication_hash=str(claimed or canonical_hash(payload)),
        runtime_authority_hash=str(obj["runtime_authority_hash"]),
        encoder_authority_hash=str(obj["encoder_authority_hash"]),
        dataset_snapshot_hash=str(obj["dataset_snapshot_hash"]),
        policy=policy,
        final_holdout_auto_open=False,
        mutable_catalog_training_forbidden=True,
    )


def save_long_factory_activation_r9(receipt, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**asdict(receipt), "content_hash": receipt.content_hash}, indent=2) + "\n")
    return p
