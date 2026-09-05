from __future__ import annotations

"""R9 real historical 5–10 generation short-campaign runner.

The runner binds four immutable authorities before doing any work:
1. Shanxi R8.1 qualified runtime;
2. real USER_FROZEN_ENCODER installation receipt;
3. real Dataset Snapshot receipt;
4. real multi-asset Market64 cache receipt.

The status-driving Trader still uses the configured primary asset Market64.  The
other asset caches/synchronization are materialized and audited but are not silently
added to the Trader input.

FINAL TOURNAMENT/FINAL CONTROLS are never called by this runner.
"""

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .campaign_telemetry_r9 import (
    collect_campaign_telemetry_r9,
    save_campaign_telemetry_r9,
)
from .chronology_constitution_r6 import ChronologyConstitutionConfigR6
from .encoder_authority_r9 import load_encoder_install_receipt_r9
from .generation_orchestrator import GenerationStateStore, PolicyRecord, PromotionRule
from .guarded_phase_r9 import PHASE_RESOURCE
from .historical_campaign_plugins_r6 import (
    LocalSupervisorConfigR6,
    load_campaign_config,
    canonical_hash as campaign_hash,
)
from .historical_campaign_plugins_r7 import build_phase_plugin_map_r7
from .long_run_controller import LongRunState, PhasePlugin
from .market_cache_r6 import MarketLatentCacheR6
from .market_encoder_r5 import state_dict_hash
from .multi_generation_runner_r5 import (
    MultiGenerationRunConfigR5,
    MultiGenerationRunStateR5,
    MultiGenerationRunnerR5,
)
from .probabilistic_teacher_r6 import DependenceAwareTeacherConfigR6
from .real_multiasset_cache_r9 import load_multiasset_cache_receipt_r9
from .runtime_authority_r9 import load_saved_runtime_authority_r9
from .short_campaign_adjudicator_r9 import (
    ShortCampaignReadinessPolicyR9,
    adjudicate_short_campaign_r9,
    save_short_campaign_adjudication_r9,
)
from .dataset_snapshot_r9 import load_dataset_snapshot_receipt_r9
from .trader_capacity_ladder import build_trader, parameter_report
from .vectorized_physics import VectorPhysicsConfig


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def render(obj: Any, context: Mapping[str, str]) -> Any:
    if isinstance(obj, str):
        return obj.format_map(context)
    if isinstance(obj, list):
        return [render(x, context) for x in obj]
    if isinstance(obj, dict):
        return {k: render(v, context) for k, v in obj.items()}
    return obj


@dataclass(frozen=True)
class ShortCampaignSpecR9:
    run_id: str
    attempts: int = 5
    genesis_seed: int = 20260904
    account_replicas: int = 256
    primary_symbol: str = "BTCUSDT"
    min_attempts_for_long_factory: int = 5
    max_attempts_for_long_factory: int = 10
    max_soft_backpressure_wait_seconds: float = 900.0
    ssd_preferred_free_gib: float = 30.0

    def validate(self):
        if not self.run_id:
            raise ValueError("run id")
        if not 5 <= self.attempts <= 10:
            raise ValueError("R9 short campaign attempts must be 5..10")
        if self.account_replicas <= 0:
            raise ValueError("account replicas")

    @property
    def content_hash(self) -> str:
        self.validate()
        return canonical_hash(self)


@dataclass(frozen=True)
class ShortCampaignRunReceiptR9:
    run_version: str
    status: str
    run_id: str
    run_root: str
    runtime_authority_hash: str
    encoder_authority_hash: str
    dataset_snapshot_hash: str
    multiasset_cache_hash: str
    campaign_config_path: str
    campaign_config_hash: str
    genesis_policy_hash: str
    generation_run_summary: Mapping[str, Any]
    telemetry_path: str
    telemetry_hash: str
    adjudication_path: str
    adjudication_hash: str
    final_tournament_closed: bool
    spec_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def _init_genesis(
    *,
    generation_state_path: Path,
    checkpoint_path: Path,
    tier: str,
    seed: int,
    existing_checkpoint: str | Path | None = None,
) -> PolicyRecord:
    gs = GenerationStateStore(generation_state_path)
    try:
        old = gs.current_champion()
        if old is not None:
            if old.generation != 0:
                # Resuming a partially/completely run short campaign is legal.
                return old
            return old

        if existing_checkpoint:
            try:
                state = torch.load(existing_checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(existing_checkpoint, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model = build_trader(tier)
            model.load_state_dict(state, strict=True)
        else:
            rng = torch.random.get_rng_state()
            try:
                torch.manual_seed(seed)
                model = build_trader(tier)
            finally:
                torch.random.set_rng_state(rng)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        wh = state_dict_hash(model.state_dict())
        arch = campaign_hash({
            "tier": tier,
            "parameter_report": parameter_report(model),
        })
        policy = PolicyRecord(
            generation=0,
            weight_hash=wh,
            architecture_hash=arch,
            tier=tier,
            checkpoint_path=str(checkpoint_path.resolve()),
            parent_weight_hash=None,
            training_snapshot_hash=None,
        )
        gs.initialize_champion(policy)
        return policy
    finally:
        gs.close()


def _wrapped_plugins(
    raw_plugins: Mapping[str, Mapping[str, Any]],
    *,
    ssd_root: str,
    hdd_root: str,
    runtime_profile: Mapping[str, Any],
    spec: ShortCampaignSpecR9,
) -> dict[str, PhasePlugin]:
    resource = runtime_profile["resource_limits"]
    limits = {
        "ram_high": float(resource["ram_backpressure_high"]),
        "ram_hard": float(resource["ram_hard_stop"]),
        "vram_high": float(resource["vram_backpressure_high"]),
        "vram_hard": 0.96,
        "gpu_temp_high_c": 82.0,
        "gpu_temp_hard_c": 90.0,
        "min_ssd_free_gib": float(resource["disk_free_hard_stop_gib"]),
        "min_hdd_free_gib": 20.0,
        "max_running_gpu_jobs": 1,
        "max_running_cpu_heavy_jobs": 1,
        "max_running_cpu_io_jobs": 2,
        "max_running_transfer_jobs": 1,
        "max_running_maintenance_jobs": 1,
    }
    out = {}
    for phase, p in raw_plugins.items():
        out[phase] = PhasePlugin(
            dotted_callable="cb16_local_opt.guarded_phase_r9:guarded_phase_plugin_r9",
            version="R9_GUARDED_REAL_HISTORICAL_PHASE",
            config={
                "delegate_callable": p["dotted_callable"],
                "delegate_config": p["config"],
                "ssd_root": ssd_root,
                "hdd_root": hdd_root,
                "resource_limits": limits,
                "max_soft_backpressure_wait_seconds": spec.max_soft_backpressure_wait_seconds,
                "backpressure_poll_seconds": 5.0,
            },
        )
    return out


def run_short_real_campaign_r9(
    *,
    runtime_authority_path: str | Path,
    encoder_install_receipt_path: str | Path,
    dataset_snapshot_receipt_path: str | Path,
    multiasset_cache_receipt_path: str | Path,
    campaign_template_path: str | Path,
    run_root: str | Path,
    ssd_root: str | Path,
    hdd_root: str | Path,
    spec: ShortCampaignSpecR9,
    existing_genesis_checkpoint: str | Path | None = None,
) -> ShortCampaignRunReceiptR9:
    spec.validate()
    runtime = load_saved_runtime_authority_r9(runtime_authority_path)
    encoder = load_encoder_install_receipt_r9(encoder_install_receipt_path)
    dataset = load_dataset_snapshot_receipt_r9(dataset_snapshot_receipt_path)
    cache_set = load_multiasset_cache_receipt_r9(multiasset_cache_receipt_path)

    if encoder.authority != "USER_FROZEN_ENCODER":
        raise RuntimeError("R9_SHORT_CAMPAIGN_REQUIRES_REAL_ENCODER")
    if cache_set.primary_symbol != spec.primary_symbol:
        raise RuntimeError("R9_SHORT_CAMPAIGN_PRIMARY_SYMBOL_MISMATCH")
    if cache_set.scientific_dataset_hash != dataset.scientific_dataset_hash:
        raise RuntimeError("R9_SHORT_CAMPAIGN_DATASET_CACHE_MISMATCH")
    if cache_set.encoder_weight_hash != encoder.state_dict_weight_hash:
        raise RuntimeError("R9_SHORT_CAMPAIGN_ENCODER_CACHE_MISMATCH")
    if cache_set.trader_multiasset_input_enabled:
        raise RuntimeError("R9_UNAUTHORIZED_MULTI_ASSET_TRADER_INPUT")

    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / "workspace"
    experience = root / "experience"
    checkpoints = root / "checkpoints"
    state_dir = root / "state"
    for p in (workspace, experience, checkpoints, state_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Explicit final-boundary receipt. The runner itself has no code path to open FINAL.
    final_boundary = root / "FINAL_BOUNDARY_R9.json"
    boundary_obj = {
        "schema": "CB16_FINAL_BOUNDARY_R9",
        "state": "CLOSED",
        "automatic_open_allowed": False,
        "run_id": spec.run_id,
        "dataset_snapshot_hash": dataset.content_hash,
        "note": "R9 short campaign uses iterative VALIDATION only",
    }
    if final_boundary.exists():
        if json.loads(final_boundary.read_text()) != boundary_obj:
            raise RuntimeError("R9_FINAL_BOUNDARY_RECEIPT_CONFLICT")
    else:
        final_boundary.write_text(json.dumps(boundary_obj, indent=2) + "\n")

    profile = runtime.runtime_profile
    gpu = profile["gpu"]
    cpu = profile["cpu"]
    io = profile["io"]
    if spec.account_replicas > int(gpu["rollout_chunk_rows"]):
        # This would still work in chunks, but keeping the short campaign <= one chunk
        # makes the first real campaign easier to reason about and profile.
        raise RuntimeError("R9_SHORT_ACCOUNT_REPLICAS_EXCEED_ONE_ROLLOUT_CHUNK")

    primary_cache = MarketLatentCacheR6(cache_set.primary_cache_root, verify_hashes=True)
    if primary_cache.receipt.dataset_hash != dataset.scientific_dataset_hash:
        raise RuntimeError("R9_PRIMARY_CACHE_DATASET_HASH_DRIFT")
    if primary_cache.receipt.encoder_weight_hash != encoder.state_dict_weight_hash:
        raise RuntimeError("R9_PRIMARY_CACHE_ENCODER_HASH_DRIFT")

    ctx = {
        "run_root": str(root.resolve()),
        "workspace_root": str(workspace.resolve()),
        "cache_root": str(Path(cache_set.primary_cache_root).resolve()),
        "experience_root": str(experience.resolve()),
        "checkpoint_root": str(checkpoints.resolve()),
        "generation_state": str((state_dir / "generation.sqlite").resolve()),
        "dataset_hash": dataset.scientific_dataset_hash,
        "encoder_weight_hash": encoder.state_dict_weight_hash,
        "run_id": spec.run_id,
        "primary_symbol": spec.primary_symbol,
    }
    raw = render(json.loads(Path(campaign_template_path).read_text()), ctx)
    raw["campaign_version"] = (
        "CB16_R9_REAL_SHORT_"
        + dataset.scientific_dataset_hash[:12]
        + "_"
        + encoder.state_dict_weight_hash[:12]
    )
    raw["workspace_root"] = str(workspace.resolve())
    raw["market_cache_root"] = str(Path(cache_set.primary_cache_root).resolve())
    raw["generation_state_path"] = str((state_dir / "generation.sqlite").resolve())
    raw["experience_lake_root"] = str(experience.resolve())
    raw["checkpoint_root"] = str(checkpoints.resolve())
    raw["tier"] = str(gpu["tier"])
    raw["account_replicas"] = int(spec.account_replicas)
    raw.setdefault("training", {})["batch_size"] = int(gpu["train_batch"])
    raw["training"]["cpu_torch_threads"] = int(gpu["cpu_torch_threads"])
    raw["training"]["device"] = "cuda"
    materialized = root / "CAMPAIGN_CONFIG_R9.json"
    materialized.write_text(json.dumps(raw, indent=2) + "\n")
    cfg = load_campaign_config(materialized)

    genesis_receipt_path = root / "GENESIS_RECEIPT_R9.json"
    if genesis_receipt_path.exists():
        gr = json.loads(genesis_receipt_path.read_text())
        genesis = PolicyRecord(**gr["policy"])
        if genesis.generation != 0:
            raise RuntimeError("R9_STORED_GENESIS_NOT_GENERATION_ZERO")
        gs_check = GenerationStateStore(cfg.generation_state_path)
        try:
            current = gs_check.current_champion()
            if current is None:
                raise RuntimeError("R9_GENESIS_RECEIPT_EXISTS_BUT_STATE_EMPTY")
            if current.generation == 0 and current.weight_hash != genesis.weight_hash:
                raise RuntimeError("R9_GENESIS_RECEIPT_STATE_CONFLICT")
        finally:
            gs_check.close()
    else:
        genesis = _init_genesis(
            generation_state_path=Path(cfg.generation_state_path),
            checkpoint_path=checkpoints / "GENESIS_G0.pt",
            tier=cfg.tier,
            seed=spec.genesis_seed,
            existing_checkpoint=existing_genesis_checkpoint,
        )
        if genesis.generation != 0:
            raise RuntimeError("R9_CANNOT_CREATE_GENESIS_RECEIPT_AFTER_CAMPAIGN_STARTED")
        genesis_receipt_path.write_text(
            json.dumps({
                "schema": "CB16_REAL_SHORT_GENESIS_R9",
                "policy": asdict(genesis),
                "seed": None if existing_genesis_checkpoint else spec.genesis_seed,
                "source_checkpoint": (
                    None if existing_genesis_checkpoint is None
                    else str(Path(existing_genesis_checkpoint).resolve())
                ),
            }, indent=2) + "\n"
        )

    physics = VectorPhysicsConfig(**dict(cfg.physics))
    supervisor = LocalSupervisorConfigR6(**dict(cfg.supervisor))
    chronology = ChronologyConstitutionConfigR6(**dict(cfg.chronology))
    teacher = DependenceAwareTeacherConfigR6(**dict(cfg.teacher))
    promotion = PromotionRule(**dict(cfg.promotion_rule))

    raw_plugins = build_phase_plugin_map_r7(
        str(materialized),
        h72_workers=int(cpu["h72_workers"]),
        h72_cpu_threads=int(cpu["h72_threads_per_worker"]),
        h72_max_in_flight=int(cpu["h72_max_in_flight"]),
        account_chunk_rows=int(gpu["rollout_chunk_rows"]),
    )
    plugins = _wrapped_plugins(
        raw_plugins,
        ssd_root=str(Path(ssd_root).resolve()),
        hdd_root=str(Path(hdd_root).resolve()),
        runtime_profile=profile,
        spec=spec,
    )
    mg = MultiGenerationRunConfigR5(
        run_id=spec.run_id,
        experiment_version=cfg.campaign_version,
        dataset_hash=dataset.scientific_dataset_hash,
        split_hash=chronology.content_hash,
        physics_hash=physics.config_hash,
        supervisor_hash=supervisor.content_hash,
        teacher_hash=teacher.content_hash,
        promotion_rule_hash=promotion.content_hash,
        phase_plugins=plugins,
        max_attempts=spec.attempts,
        max_promotions=spec.attempts,
        max_consecutive_rejects=spec.attempts,
    )

    rs = MultiGenerationRunStateR5(root / "MULTIGEN_RUN.sqlite")
    ps = LongRunState(root / "PHASES.sqlite")
    gs = GenerationStateStore(cfg.generation_state_path)
    try:
        summary = MultiGenerationRunnerR5(
            config=mg,
            run_state=rs,
            phase_state=ps,
            generation_state=gs,
        ).run(shared_context={
            "R9_RUNTIME_AUTHORITY_HASH": runtime.content_hash,
            "R9_ENCODER_AUTHORITY_HASH": encoder.content_hash,
            "R9_DATASET_SNAPSHOT_HASH": dataset.content_hash,
            "R9_MULTI_ASSET_CACHE_HASH": cache_set.content_hash,
        })
    finally:
        rs.close(); ps.close(); gs.close()

    # The exact receipt must still say CLOSED, and no phase is allowed to mutate it.
    if json.loads(final_boundary.read_text()) != boundary_obj:
        raise RuntimeError("R9_FINAL_BOUNDARY_MUTATED")

    telemetry = collect_campaign_telemetry_r9(
        run_root=root,
        workspace_root=cfg.workspace_root,
        experience_lake_root=cfg.experience_lake_root,
        checkpoint_root=cfg.checkpoint_root,
        generation_state_path=cfg.generation_state_path,
    )
    telemetry_path = save_campaign_telemetry_r9(
        telemetry, root / "SHORT_CAMPAIGN_TELEMETRY_R9.json"
    )
    adjudication = adjudicate_short_campaign_r9(
        runtime=runtime,
        encoder=encoder,
        dataset=dataset,
        cache=cache_set,
        telemetry=telemetry,
        ssd_root=ssd_root,
        hdd_root=hdd_root,
        policy=ShortCampaignReadinessPolicyR9(
            min_attempts=spec.min_attempts_for_long_factory,
            max_attempts=spec.max_attempts_for_long_factory,
            ssd_preferred_free_gib=spec.ssd_preferred_free_gib,
        ),
    )
    adjud_path = save_short_campaign_adjudication_r9(
        adjudication, root / "SHORT_CAMPAIGN_ADJUDICATION_R9.json"
    )
    run_receipt = ShortCampaignRunReceiptR9(
        run_version="CB16_REAL_HISTORICAL_SHORT_CAMPAIGN_R9",
        status=adjudication.status,
        run_id=spec.run_id,
        run_root=str(root.resolve()),
        runtime_authority_hash=runtime.content_hash,
        encoder_authority_hash=encoder.content_hash,
        dataset_snapshot_hash=dataset.content_hash,
        multiasset_cache_hash=cache_set.content_hash,
        campaign_config_path=str(materialized.resolve()),
        campaign_config_hash=canonical_hash(raw),
        genesis_policy_hash=genesis.content_hash,
        generation_run_summary=summary,
        telemetry_path=str(telemetry_path.resolve()),
        telemetry_hash=telemetry.content_hash,
        adjudication_path=str(adjud_path.resolve()),
        adjudication_hash=adjudication.content_hash,
        final_tournament_closed=adjudication.final_tournament_closed,
        spec_hash=spec.content_hash,
    )
    (root / "SHORT_CAMPAIGN_RESULT_R9.json").write_text(
        json.dumps({**asdict(run_receipt), "content_hash": run_receipt.content_hash}, indent=2) + "\n"
    )
    return run_receipt
