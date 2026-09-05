from __future__ import annotations

"""
Concrete historical R&D phase plugins for the R4/R5 multi-generation controller.

Per-generation status-driving loop:

ROLLOUT
    Current Champion generates account-state trajectories with explicit exploration.
    At frozen TRAIN/VALIDATION anchor timestamps, R6 compiles same-future H72/H73
    counterfactual utilities. TOURNAMENT market-value bytes are not consumed.

TEACHER_CREDIT
    TRAIN-only dependence-aware prequential Teacher evidence.
    VALIDATION controls use only frozen TRAIN dependence-group support.
    TOURNAMENT data is not read.

SEAL_SNAPSHOT
    Seal admitted TRAIN Evidence.

TRAIN_CHALLENGER
    Fixed-epoch Student training from the sealed snapshot.

TOURNAMENT
    Despite the legacy phase name, the repeated generation-selection lane is VALIDATION.
    Champion and Challenger are evaluated on non-overlapping validation time blocks.

ADJUDICATE_COMMIT
    PromotionRule on VALIDATION result.

RETENTION
    Optional verified SSD->HDD archive.

The immutable TOURNAMENT split remains unopened during this loop. A separate one-time final
holdout command is provided by `final_holdout_r6.py`.
"""

import dataclasses
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .checkpoint_recovery import (
    CheckpointManager,
    ResumeIdentity,
    TrainingProgress,
)
from .chronology_constitution_r6 import (
    ChronologyConstitutionBuilderR6,
    ChronologyConstitutionConfigR6,
    ChronologyDependenceGroupR6,
    HistoricalChronologyConstitutionR6,
    ChronologySplitR6,
    ExcludedChronologyGroupR6,
)
from .generation_orchestrator import (
    ChallengerAttempt,
    GenerationStateStore,
    PolicyRecord,
    PromotionRule,
    TournamentResult,
)
from .gpu_training_policy import TrainingPolicy, build_optimizer, train_step
from .market_cache_r6 import MarketLatentCacheR6
from .market_encoder_r5 import state_dict_hash
from .probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from .probabilistic_teacher_r6 import (
    DependenceAwareProbabilisticTeacherR6,
    DependenceAwareTeacherConfigR6,
)
from .scientific_controls_r6 import (
    DependenceAwareControlSuiteConfigR6,
    DependenceAwareHistoricalControlSuiteR6,
)
from .sharded_experience_lake import (
    ExperienceRef,
    ExperienceSnapshot,
    ShardedExperienceLake,
)
from .student_training_dataset import (
    FrozenEvidenceTrainingDataset,
    StudentContextR4,
    store_student_context,
)
from .trajectory_compiler_r6 import (
    DecisionAnchorR6,
    EconomicClockR6,
    InitialAccountSnapshotR6,
    MultiBarTrajectoryCompilerR6,
    default_action_grid_r6,
)
from .vectorized_physics import (
    AccountBatchState,
    MarketBar,
    VectorPhysicsConfig,
    VectorizedPhysics,
    mark_equity,
)
from .retention_archive import (
    ArchiveCandidate,
    RetentionArchiveManager,
    RetentionPolicy,
)


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@dataclass(frozen=True)
class ExplorationConfigR6:
    enabled: bool = True
    direction_temperature: float = 1.0
    epsilon_uniform: float = 0.05
    risk_noise_std: float = 0.05
    seed: int = 20260904

    def validate(self):
        if self.direction_temperature <= 0:
            raise ValueError("direction temperature")
        if not 0 <= self.epsilon_uniform <= 1:
            raise ValueError("epsilon")
        if self.risk_noise_std < 0:
            raise ValueError("risk noise")


@dataclass(frozen=True)
class LocalSupervisorConfigR6:
    max_executable_risk: float = 1.0
    veto_drawdown_fraction: float = 0.60
    veto_margin_utilization: float = 0.95
    supervisor_version: str = "CB16_LOCAL_SUPERVISOR_R6"

    def validate(self):
        if not 0 <= self.max_executable_risk <= 1:
            raise ValueError("max executable risk")
        if not 0 < self.veto_drawdown_fraction <= 1:
            raise ValueError("drawdown veto")
        if not 0 < self.veto_margin_utilization <= 10:
            raise ValueError("margin veto")

    @property
    def content_hash(self):
        self.validate()
        return canonical_hash(self)


@dataclass(frozen=True)
class TrainingConfigR6:
    epochs: int = 2
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    cpu_torch_threads: int = 2
    device: str = "cuda"
    shuffle_seed: int = 1701

    def validate(self):
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("training epochs/batch")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("training device")


@dataclass(frozen=True)
class HistoricalCampaignConfigR6:
    campaign_version: str
    workspace_root: str
    market_cache_root: str
    generation_state_path: str
    experience_lake_root: str
    checkpoint_root: str
    tier: str
    account_replicas: int
    anchor_stride_bars: int
    warmup_bars: int
    risk_levels: tuple[float, ...]
    physics: Mapping[str, Any]
    supervisor: Mapping[str, Any]
    exploration: Mapping[str, Any]
    chronology: Mapping[str, Any]
    teacher: Mapping[str, Any]
    controls: Mapping[str, Any]
    training: Mapping[str, Any]
    promotion_rule: Mapping[str, Any]
    validation_block_bars: int = 96
    validation_bootstrap_reps: int = 1000
    retention: Mapping[str, Any] | None = None

    def validate(self):
        if not self.campaign_version:
            raise ValueError("campaign version")
        if self.account_replicas <= 0:
            raise ValueError("account replicas")
        if self.anchor_stride_bars <= 0 or self.warmup_bars < 31:
            raise ValueError("anchor schedule")
        if not self.risk_levels:
            raise ValueError("risk levels")
        if self.validation_block_bars <= 1:
            raise ValueError("validation block bars")
        if self.validation_bootstrap_reps <= 0:
            raise ValueError("bootstrap reps")
        VectorPhysicsConfig(**dict(self.physics)).validate()
        LocalSupervisorConfigR6(**dict(self.supervisor)).validate()
        ExplorationConfigR6(**dict(self.exploration)).validate()
        ChronologyConstitutionConfigR6(**dict(self.chronology)).validate()
        DependenceAwareTeacherConfigR6(**dict(self.teacher)).validate()
        TrainingConfigR6(**dict(self.training)).validate()
        PromotionRule(**dict(self.promotion_rule))

    @property
    def content_hash(self):
        self.validate()
        return canonical_hash(self)


def load_campaign_config(path: str | Path) -> HistoricalCampaignConfigR6:
    obj = json.loads(Path(path).read_text())
    obj["risk_levels"] = tuple(obj["risk_levels"])
    return HistoricalCampaignConfigR6(**obj)


def _cfg(plugin_config) -> HistoricalCampaignConfigR6:
    path = plugin_config["campaign_config_path"]
    return load_campaign_config(path)


def _cycle_dir(cfg: HistoricalCampaignConfigR6, cycle_spec) -> Path:
    safe = cycle_spec.cycle_id.replace("/", "_").replace(":", "_")
    p = Path(cfg.workspace_root) / "cycles" / safe
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_state_dict_payload(path: str | Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def _load_policy_model(policy: PolicyRecord, *, device: str):
    from .trader_capacity_ladder import build_trader

    if not Path(policy.checkpoint_path).is_file():
        raise FileNotFoundError(policy.checkpoint_path)
    model = build_trader(policy.tier)
    state = _load_state_dict_payload(policy.checkpoint_path)
    model.load_state_dict(state, strict=True)
    actual = state_dict_hash(model.state_dict())
    if actual != policy.weight_hash:
        raise RuntimeError(
            f"POLICY_CHECKPOINT_WEIGHT_HASH_MISMATCH expected={policy.weight_hash} actual={actual}"
        )
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    return model.to(dev), dev


def _policy_outputs(
    model,
    dev,
    market_latent: np.ndarray,
    account_state6: np.ndarray,
):
    m = torch.from_numpy(
        np.array(market_latent, dtype=np.float32, copy=True, order="C")
    ).to(dev)
    a = torch.from_numpy(
        np.array(account_state6, dtype=np.float32, copy=True, order="C")
    ).to(dev)
    model.eval()
    with torch.inference_mode():
        out = model(m, a)
    return (
        out["direction_logits"].detach().cpu().numpy(),
        out["direction_probs"].detach().cpu().numpy(),
        out["requested_risk_raw"].detach().cpu().numpy(),
    )


def _sample_rollout_actions(
    *,
    logits: np.ndarray,
    probs: np.ndarray,
    risk_raw: np.ndarray,
    exploration: ExplorationConfigR6,
    rng: np.random.Generator,
):
    n = len(risk_raw)
    if exploration.enabled:
        z = logits / exploration.direction_temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(np.clip(z, -60, 60))
        p /= p.sum(axis=1, keepdims=True)
        p = (
            (1 - exploration.epsilon_uniform) * p
            + exploration.epsilon_uniform / 3.0
        )
        u = rng.random(n)
        c = np.cumsum(p, axis=1)
        cls = (u[:, None] > c).sum(axis=1)
        direction = cls.astype(np.int8) - 1
        risk = risk_raw + rng.normal(
            0.0,
            exploration.risk_noise_std,
            size=n,
        )
        risk = np.clip(risk, 0.0, 1.0)
    else:
        cls = probs.argmax(axis=1)
        direction = cls.astype(np.int8) - 1
        risk = np.clip(risk_raw, 0.0, 1.0)
    risk = np.where(direction == 0, 0.0, risk)
    return direction, risk.astype(np.float64)


def _supervise(
    *,
    requested_direction: np.ndarray,
    requested_risk: np.ndarray,
    account_state6: np.ndarray,
    config: LocalSupervisorConfigR6,
):
    d = requested_direction.astype(np.int8, copy=True)
    r = np.minimum(
        requested_risk.astype(np.float64, copy=True),
        config.max_executable_risk,
    )
    # AccountState6 columns: signed exposure, entry log, remaining, drawdown,
    # risk budget fraction, margin utilization.
    veto = (
        (account_state6[:, 3] >= config.veto_drawdown_fraction)
        | (account_state6[:, 5] >= config.veto_margin_utilization)
    )
    d[veto] = 0
    r[veto] = 0.0
    r[d == 0] = 0.0
    if np.any(r - requested_risk > 1e-12):
        raise RuntimeError("SUPERVISOR_INCREASED_RISK")
    return d, r


def _account_snapshot(state: AccountBatchState, i: int) -> InitialAccountSnapshotR6:
    return InitialAccountSnapshotR6(
        balance=float(state.balance[i]),
        position_qty=float(state.position_qty[i]),
        entry_price=float(state.entry_price[i]),
        peak_equity=float(state.peak_equity[i]),
        realized_pnl=float(state.realized_pnl[i]),
        margin_used=float(state.margin_used[i]),
        holding_bars=int(state.holding_bars[i]),
        risk_budget_remaining=float(state.risk_budget_remaining[i]),
        risk_budget_capacity=float(state.risk_budget_capacity[i]),
        terminated=bool(state.terminated[i]),
        last_mark_price=float(state.last_mark_price[i]),
    )


def _constitution_from_json(obj) -> HistoricalChronologyConstitutionR6:
    def split(name):
        return ChronologySplitR6(**obj[name])
    return HistoricalChronologyConstitutionR6(
        constitution_version=obj["constitution_version"],
        protocol_hash=obj["protocol_hash"],
        dataset_hash=obj["dataset_hash"],
        horizon_clock_id=obj["horizon_clock_id"],
        source_group_hash=obj["source_group_hash"],
        train=split("train"),
        validation=split("validation"),
        tournament=split("tournament"),
        excluded=tuple(
            ExcludedChronologyGroupR6(**x) for x in obj["excluded"]
        ),
        total_source_groups=obj["total_source_groups"],
        total_included_groups=obj["total_included_groups"],
    )


def _sample_from_json(x):
    x["context_features"] = tuple(x["context_features"])
    return CounterfactualBranchSampleR5(**x)


def _campaign_group_schedule(
    *,
    cache: MarketLatentCacheR6,
    cfg: HistoricalCampaignConfigR6,
    cycle_spec,
):
    ts = cache.arrays["timestamp"]
    valid = cache.arrays["latent_valid"]
    clock = EconomicClockR6(horizon_bars=72)
    start = max(cfg.warmup_bars, cache.receipt.first_valid_index)
    groups = []
    index_by_group = {}
    for i in range(start, cache.receipt.rows):
        if not bool(valid[i]):
            continue
        if (i - start) % cfg.anchor_stride_bars != 0:
            continue
        if i + clock.horizon_bars >= cache.receipt.rows:
            break
        gid = f"DG:{int(ts[i])}"
        parents = tuple(
            f"G{cycle_spec.generation_parent}:T{int(ts[i])}:A{a:04d}"
            for a in range(cfg.account_replicas)
        )
        maturity = int(ts[i + clock.horizon_bars])
        groups.append(ChronologyDependenceGroupR6(
            group_id=gid,
            parent_ids=parents,
            decision_timestamp=int(ts[i]),
            maturity_timestamp=maturity,
            source_hash=canonical_hash({
                "dataset_hash": cache.receipt.dataset_hash,
                "decision_index": i,
                "maturity_index": i + clock.horizon_bars,
            }),
        ))
        index_by_group[gid] = i
    constitution = ChronologyConstitutionBuilderR6(
        ChronologyConstitutionConfigR6(**dict(cfg.chronology))
    ).build(
        groups,
        dataset_hash=cache.receipt.dataset_hash,
        horizon_clock_id=clock.clock_id,
    )
    return constitution, index_by_group


def historical_rollout_phase_r6(*, phase, cycle_spec, plugin_config, context):
    cfg = _cfg(plugin_config)
    if phase != "ROLLOUT":
        raise RuntimeError("WRONG_PHASE_PLUGIN")
    cdir = _cycle_dir(cfg, cycle_spec)
    receipt_path = cdir / "ROLLOUT_RECEIPT.json"
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old["cycle_hash"] != cycle_spec.content_hash:
            raise RuntimeError("ROLLOUT_RECEIPT_CYCLE_CONFLICT")
        return old

    cache = MarketLatentCacheR6(
        cfg.market_cache_root,
        verify_hashes=bool(plugin_config.get("verify_cache_hashes", False)),
    )
    if cache.receipt.dataset_hash != cycle_spec.dataset_hash:
        raise RuntimeError("CAMPAIGN_DATASET_HASH_MISMATCH")
    market = cache.market_path()
    constitution, group_index = _campaign_group_schedule(
        cache=cache,
        cfg=cfg,
        cycle_spec=cycle_spec,
    )
    atomic_json(cdir / "CHRONOLOGY_CONSTITUTION.json", asdict(constitution))

    # Repeated generation loop is forbidden from opening final TOURNAMENT market values.
    allowed_groups = set(
        constitution.train.group_ids + constitution.validation.group_ids
    )
    final_holdout_first_ts = constitution.tournament.first_decision_timestamp

    gs = GenerationStateStore(cfg.generation_state_path)
    champion = gs.current_champion()
    if champion is None:
        gs.close()
        raise RuntimeError("NO_CHAMPION")
    if (
        champion.generation != cycle_spec.generation_parent
        or champion.weight_hash != cycle_spec.parent_policy_hash
    ):
        gs.close()
        raise RuntimeError("ROLLOUT_PARENT_NOT_CURRENT_CHAMPION")
    model, dev = _load_policy_model(
        champion,
        device=TrainingConfigR6(**dict(cfg.training)).device,
    )
    gs.close()

    physics_cfg = VectorPhysicsConfig(**dict(cfg.physics))
    physics = VectorizedPhysics(physics_cfg)
    compiler = MultiBarTrajectoryCompilerR6(
        physics_cfg,
        clock=EconomicClockR6(horizon_bars=72),
    )
    action_grid = default_action_grid_r6(cfg.risk_levels)
    supervisor = LocalSupervisorConfigR6(**dict(cfg.supervisor))
    exploration = ExplorationConfigR6(**dict(cfg.exploration))
    rng = np.random.default_rng(
        exploration.seed + 1000003 * cycle_spec.generation_parent
    )

    state = AccountBatchState.empty(
        cfg.account_replicas,
        physics_cfg,
        account_prefix=f"G{cycle_spec.generation_parent}A",
    )
    lake = ShardedExperienceLake(cfg.experience_lake_root, shards=4)

    train_samples = []
    val_samples = []
    train_trajectory_rows = []
    val_trajectory_rows = []
    contexts_stored = 0

    ts = cache.arrays["timestamp"]
    valid = cache.arrays["latent_valid"]
    latent = cache.arrays["market_latent"]
    group_to_split = {
        g: "TRAIN" for g in constitution.train.group_ids
    }
    group_to_split.update({
        g: "VALIDATION" for g in constitution.validation.group_ids
    })

    start = max(cfg.warmup_bars, cache.receipt.first_valid_index)
    # We only need actual Champion account-state rollout through the final VALIDATION
    # decision. Counterfactual compilation reads its H72 maturity path separately, and the
    # constitution guarantees that maturity remains strictly before TOURNAMENT begins.
    # This prevents even the actual rollout from touching final-holdout OHLC/latent bytes.
    last_validation_decision_index = int(
        np.searchsorted(
            ts,
            constitution.validation.last_decision_timestamp,
            side="left",
        )
    )

    for t in range(
        start,
        min(last_validation_decision_index + 1, cache.receipt.rows - 1),
    ):
        if not bool(valid[t]):
            continue

        # Decision context is close-of-t state + market latent ending at t.
        account6 = physics.account_observation6(
            state,
            float(cache.arrays["close"][t]),
        )
        market_batch = np.broadcast_to(
            np.asarray(latent[t], dtype=np.float32),
            (cfg.account_replicas, 64),
        )
        logits, probs, risk_raw = _policy_outputs(
            model,
            dev,
            market_batch,
            account6,
        )
        req_d, req_r = _sample_rollout_actions(
            logits=logits,
            probs=probs,
            risk_raw=risk_raw,
            exploration=exploration,
            rng=rng,
        )
        exe_d, exe_r = _supervise(
            requested_direction=req_d,
            requested_risk=req_r,
            account_state6=account6,
            config=supervisor,
        )

        gid = f"DG:{int(ts[t])}"
        split = group_to_split.get(gid)
        if split is not None:
            # H72 maturity is guaranteed by constitution/source schedule.
            per_group_samples = []
            for a in range(cfg.account_replicas):
                parent = (
                    f"G{cycle_spec.generation_parent}:"
                    f"T{int(ts[t])}:A{a:04d}"
                )
                context_id = f"CTX:{cycle_spec.cycle_id}:{parent}"
                ctx = StudentContextR4(
                    context_id=context_id,
                    decision_event_hash=canonical_hash({
                        "cycle": cycle_spec.cycle_id,
                        "parent": parent,
                        "policy": champion.weight_hash,
                        "timestamp": int(ts[t]),
                    }),
                    timestamp=int(ts[t]),
                    symbol=str(plugin_config.get("symbol", "BTCUSDT")),
                    account_id=f"A{a:04d}",
                    policy_generation=champion.generation,
                    policy_weight_hash=champion.weight_hash,
                    market_latent=tuple(float(x) for x in latent[t]),
                    account_state6=tuple(float(x) for x in account6[a]),
                    market_lineage_hash=cache.receipt.scientific_identity_hash,
                    account_lineage_hash=canonical_hash({
                        "cycle": cycle_spec.cycle_id,
                        "account": a,
                        "timestamp": int(ts[t]),
                        "state": [
                            float(state.balance[a]),
                            float(state.position_qty[a]),
                            float(state.entry_price[a]),
                            int(state.holding_bars[a]),
                        ],
                    }),
                )
                store_student_context(lake, ctx)
                contexts_stored += 1
                anchor = DecisionAnchorR6(
                    parent_id=parent,
                    student_context_object_id=context_id,
                    decision_index=t,
                    context_features=tuple(
                        float(x) for x in np.concatenate(
                            [np.asarray(latent[t]), account6[a]]
                        )
                    ),
                    dependence_group_id=gid,
                    market_lineage_hash=cache.receipt.scientific_identity_hash,
                    initial_account=_account_snapshot(state, a),
                )
                compiled = compiler.compile_anchor(
                    market,
                    anchor,
                    action_grid,
                )
                teacher_samples = compiler.to_teacher_samples(
                    [compiled],
                    {parent: anchor},
                )
                per_group_samples.extend(teacher_samples)
                target_rows = (
                    train_trajectory_rows
                    if split == "TRAIN"
                    else val_trajectory_rows
                )
                target_rows.extend(asdict(b) for b in compiled.branches)

            if split == "TRAIN":
                train_samples.extend(per_group_samples)
            else:
                val_samples.extend(per_group_samples)

        # Advance actual Champion account trajectories using only next bar t+1.
        bar = MarketBar(
            open=float(cache.arrays["open"][t + 1]),
            high=float(cache.arrays["high"][t + 1]),
            low=float(cache.arrays["low"][t + 1]),
            close=float(cache.arrays["close"][t + 1]),
            funding_rate=float(cache.arrays["funding_rate"][t + 1]),
        )
        physics.step(
            state,
            bar,
            executable_direction=exe_d,
            executable_risk=exe_r,
            requested_direction=req_d,
            dependence_group_count=1,
        )

    lake.close()

    train_samples_path = cdir / "TRAIN_SAMPLES.jsonl"
    val_samples_path = cdir / "VALIDATION_SAMPLES.jsonl"
    write_jsonl(
        train_samples_path,
        [asdict(x) for x in train_samples],
    )
    write_jsonl(
        val_samples_path,
        [asdict(x) for x in val_samples],
    )
    write_jsonl(cdir / "TRAIN_TRAJECTORIES.jsonl", train_trajectory_rows)
    write_jsonl(cdir / "VALIDATION_TRAJECTORIES.jsonl", val_trajectory_rows)

    receipt = {
        "schema": "CB16_R6_ROLLOUT_RECEIPT",
        "cycle_id": cycle_spec.cycle_id,
        "cycle_hash": cycle_spec.content_hash,
        "champion_generation": champion.generation,
        "champion_weight_hash": champion.weight_hash,
        "market_cache_identity": cache.receipt.scientific_identity_hash,
        "encoder_weight_hash": cache.receipt.encoder_weight_hash,
        "constitution_hash": constitution.content_hash,
        "train_dependence_groups": constitution.train.group_count,
        "validation_dependence_groups": constitution.validation.group_count,
        "tournament_dependence_groups_reserved_unopened": constitution.tournament.group_count,
        "train_parent_contexts": constitution.train.parent_count,
        "validation_parent_contexts": constitution.validation.parent_count,
        "student_contexts_stored": contexts_stored,
        "train_counterfactual_branches": len(train_samples),
        "validation_counterfactual_branches": len(val_samples),
        "tournament_market_values_opened": False,
        "supervisor_hash": supervisor.content_hash,
        "exploration_hash": canonical_hash(exploration),
        "train_samples_sha256": sha256_file(train_samples_path),
        "validation_samples_sha256": sha256_file(val_samples_path),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def historical_teacher_credit_phase_r6(*, phase, cycle_spec, plugin_config, context):
    cfg = _cfg(plugin_config)
    if phase != "TEACHER_CREDIT":
        raise RuntimeError("WRONG_PHASE_PLUGIN")
    cdir = _cycle_dir(cfg, cycle_spec)
    constitution = _constitution_from_json(
        json.loads((cdir / "CHRONOLOGY_CONSTITUTION.json").read_text())
    )
    train_samples = [
        _sample_from_json(x)
        for x in read_jsonl(cdir / "TRAIN_SAMPLES.jsonl")
    ]
    val_samples = [
        _sample_from_json(x)
        for x in read_jsonl(cdir / "VALIDATION_SAMPLES.jsonl")
    ]

    teacher_cfg = DependenceAwareTeacherConfigR6(**dict(cfg.teacher))
    if teacher_cfg.mode != "PREQUENTIAL":
        raise RuntimeError(
            "R6_REAL_CAMPAIGN_REQUIRES_PREQUENTIAL_TRAIN_TEACHER"
        )
    teacher = DependenceAwareProbabilisticTeacherR6(teacher_cfg)
    train_evidence = teacher.compile_many(
        train_samples,
        target_parent_ids=constitution.train.parent_ids,
        eligible_train_dependence_groups=set(constitution.train.group_ids),
    )
    admitted = [e for e in train_evidence if e.admission.admitted]

    lake = ShardedExperienceLake(cfg.experience_lake_root, shards=4)
    refs = [
        teacher.persist_evidence(
            lake=lake,
            evidence=e,
            policy_generation=cycle_spec.generation_parent,
            policy_weight_hash=cycle_spec.parent_policy_hash,
            parent_snapshot_hash=constitution.train.group_hash,
        )
        for e in admitted
    ]

    controls_cfg = dict(cfg.controls)
    controls_cfg["teacher"] = teacher_cfg
    controls = DependenceAwareHistoricalControlSuiteR6(
        DependenceAwareControlSuiteConfigR6(**controls_cfg)
    )
    validation_controls = controls.evaluate(
        train_samples + val_samples,
        target_parent_ids=constitution.validation.parent_ids,
        eligible_train_dependence_group_ids=constitution.train.group_ids,
    )
    lake.close()

    atomic_json(
        cdir / "VALIDATION_CONTROLS.json",
        asdict(validation_controls),
    )
    atomic_json(
        cdir / "ADMITTED_EVIDENCE_REFS.json",
        [asdict(r) for r in refs],
    )
    receipt = {
        "schema": "CB16_R6_TEACHER_CREDIT_RECEIPT",
        "cycle_id": cycle_spec.cycle_id,
        "teacher_protocol_hash": teacher_cfg.content_hash,
        "train_evidence_total": len(train_evidence),
        "train_evidence_admitted": len(admitted),
        "admitted_evidence_ref_hash": canonical_hash(
            [r.identity_hash for r in refs]
        ),
        "validation_controls_hash": validation_controls.content_hash,
        "validation_controls_status": validation_controls.status,
        "tournament_data_read": False,
    }
    atomic_json(cdir / "TEACHER_CREDIT_RECEIPT.json", receipt)
    return receipt


def historical_seal_snapshot_phase_r6(*, phase, cycle_spec, plugin_config, context):
    cfg = _cfg(plugin_config)
    if phase != "SEAL_SNAPSHOT":
        raise RuntimeError("WRONG_PHASE_PLUGIN")
    cdir = _cycle_dir(cfg, cycle_spec)
    constitution = _constitution_from_json(
        json.loads((cdir / "CHRONOLOGY_CONSTITUTION.json").read_text())
    )
    refs = [
        ExperienceRef(**x)
        for x in json.loads((cdir / "ADMITTED_EVIDENCE_REFS.json").read_text())
    ]
    lake = ShardedExperienceLake(cfg.experience_lake_root, shards=4)
    snap = lake.seal_snapshot(
        snapshot_id=f"R6TRAIN:{cycle_spec.cycle_id}",
        parent_generation=cycle_spec.generation_parent,
        parent_policy_hash=cycle_spec.parent_policy_hash,
        refs=refs,
    )
    lake.close()
    atomic_json(cdir / "TRAINING_SNAPSHOT.json", asdict(snap))
    receipt = {
        "schema": "CB16_R6_SEAL_SNAPSHOT_RECEIPT",
        "cycle_id": cycle_spec.cycle_id,
        "snapshot_id": snap.snapshot_id,
        "snapshot_hash": snap.content_hash,
        "object_count": snap.object_count,
        "train_group_hash": constitution.train.group_hash,
    }
    atomic_json(cdir / "SEAL_SNAPSHOT_RECEIPT.json", receipt)
    return receipt


def _snapshot_from_json(obj) -> ExperienceSnapshot:
    return ExperienceSnapshot(
        snapshot_id=obj["snapshot_id"],
        parent_generation=obj["parent_generation"],
        parent_policy_hash=obj["parent_policy_hash"],
        object_ids=tuple(obj["object_ids"]),
        object_identity_hashes=tuple(obj["object_identity_hashes"]),
        object_count=obj["object_count"],
    )


def historical_train_challenger_phase_r6(*, phase, cycle_spec, plugin_config, context):
    cfg = _cfg(plugin_config)
    if phase != "TRAIN_CHALLENGER":
        raise RuntimeError("WRONG_PHASE_PLUGIN")
    cdir = _cycle_dir(cfg, cycle_spec)
    snap = _snapshot_from_json(
        json.loads((cdir / "TRAINING_SNAPSHOT.json").read_text())
    )
    gs = GenerationStateStore(cfg.generation_state_path)
    champion = gs.current_champion()
    if champion is None:
        gs.close()
        raise RuntimeError("NO_CHAMPION")
    if (
        champion.generation != cycle_spec.generation_parent
        or champion.weight_hash != cycle_spec.parent_policy_hash
    ):
        gs.close()
        raise RuntimeError("TRAIN_PARENT_NOT_CURRENT_CHAMPION")

    tcfg = TrainingConfigR6(**dict(cfg.training))
    model, dev = _load_policy_model(champion, device=tcfg.device)
    policy = TrainingPolicy(
        device=str(dev),
        dtype="fp32",
        amp_enabled=False,
        cpu_torch_threads=tcfg.cpu_torch_threads,
        lr=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
    )
    optimizer = build_optimizer(model, policy)

    attempt_id = f"R6:{cycle_spec.cycle_id}"
    gs.create_attempt(ChallengerAttempt(
        attempt_id=attempt_id,
        parent_generation=cycle_spec.generation_parent,
        parent_weight_hash=cycle_spec.parent_policy_hash,
        training_snapshot_hash=snap.content_hash,
        experiment_version=cfg.campaign_version,
        architecture_hash=champion.architecture_hash,
        tier=champion.tier,
    ))

    lake = ShardedExperienceLake(cfg.experience_lake_root, shards=4)
    ds = FrozenEvidenceTrainingDataset(
        lake=lake,
        snapshot=snap,
        expected_parent_generation=cycle_spec.generation_parent,
        expected_parent_policy_hash=cycle_spec.parent_policy_hash,
    )

    grad_norms = []
    examples = 0
    steps = 0
    for epoch in range(tcfg.epochs):
        for batch in ds.iter_batches(
            batch_size=tcfg.batch_size,
            shuffle_seed=tcfg.shuffle_seed + epoch,
            pin_memory=(dev.type == "cuda"),
        ):
            stats = train_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                policy=policy,
            )
            grad_norms.append(stats.grad_norm)
            examples += stats.batch_size
            steps += 1
    lake.close()

    if not grad_norms or max(grad_norms) <= 0:
        gs.close()
        raise RuntimeError("CHALLENGER_TRAINING_NO_GRADIENT")
    challenger_hash = state_dict_hash(model.state_dict())
    if challenger_hash == champion.weight_hash:
        gs.close()
        raise RuntimeError("CHALLENGER_WEIGHTS_DID_NOT_CHANGE")

    ckpt = CheckpointManager(cfg.checkpoint_root)
    identity = ResumeIdentity(
        experiment_version=cfg.campaign_version,
        dataset_hash=cycle_spec.dataset_hash,
        split_hash=cycle_spec.split_hash,
        physics_hash=cycle_spec.physics_hash,
        supervisor_hash=cycle_spec.supervisor_hash,
        teacher_hash=cycle_spec.teacher_hash,
        training_snapshot_hash=snap.content_hash,
        parent_policy_hash=cycle_spec.parent_policy_hash,
        architecture_hash=champion.architecture_hash,
        training_config_hash=canonical_hash(tcfg),
    )
    receipt = ckpt.save(
        model=model,
        optimizer=optimizer,
        identity=identity,
        progress=TrainingProgress(
            generation=cycle_spec.generation_parent + 1,
            epoch=tcfg.epochs,
            global_step=steps,
            examples_seen=examples,
            evidence_cursor=(
                snap.object_ids[-1] if snap.object_ids else "EMPTY"
            ),
        ),
        extra_metadata={
            "cycle_id": cycle_spec.cycle_id,
            "training_snapshot_hash": snap.content_hash,
        },
    )
    challenger = PolicyRecord(
        generation=cycle_spec.generation_parent + 1,
        weight_hash=receipt.model_weight_hash,
        architecture_hash=champion.architecture_hash,
        tier=champion.tier,
        checkpoint_path=receipt.tensor_path,
        parent_weight_hash=champion.weight_hash,
        training_snapshot_hash=snap.content_hash,
    )
    gs.record_challenger(attempt_id, challenger)
    gs.close()

    out = {
        "schema": "CB16_R6_TRAIN_CHALLENGER_RECEIPT",
        "cycle_id": cycle_spec.cycle_id,
        "attempt_id": attempt_id,
        "parent_weight_hash": champion.weight_hash,
        "challenger_weight_hash": challenger.weight_hash,
        "checkpoint_id": receipt.checkpoint_id,
        "checkpoint_path": receipt.tensor_path,
        "gradient_steps": steps,
        "examples_seen": examples,
        "min_gradient_norm": float(min(grad_norms)),
        "max_gradient_norm": float(max(grad_norms)),
    }
    atomic_json(cdir / "TRAIN_CHALLENGER_RECEIPT.json", out)
    return out


def _deterministic_actions(model, dev, latent, account6, supervisor):
    logits, probs, risk_raw = _policy_outputs(
        model,
        dev,
        latent,
        account6,
    )
    cls = probs.argmax(axis=1)
    d = cls.astype(np.int8) - 1
    r = np.clip(risk_raw, 0, 1).astype(np.float64)
    r[d == 0] = 0.0
    return _supervise(
        requested_direction=d,
        requested_risk=r,
        account_state6=account6,
        config=supervisor,
    )


def _terminal_close_equity(state, terminal_close, physics_cfg):
    qty = state.position_qty.copy()
    active = (qty != 0) & (~state.terminated)
    slip = physics_cfg.slippage_bps * 1e-4
    fill = np.full(state.n, terminal_close, dtype=np.float64)
    fill[active] = terminal_close * (
        1.0 - np.sign(qty[active]) * slip
    )
    pnl = np.zeros(state.n)
    pnl[active] = qty[active] * (
        fill[active] - state.entry_price[active]
    )
    fee = np.zeros(state.n)
    fee[active] = (
        np.abs(qty[active]) * fill[active] * physics_cfg.fee_rate
    )
    return state.balance + pnl - fee


def _simulate_policy_block(
    *,
    policy: PolicyRecord,
    cache: MarketLatentCacheR6,
    start_index: int,
    end_index: int,
    physics_cfg: VectorPhysicsConfig,
    supervisor: LocalSupervisorConfigR6,
    device: str,
):
    model, dev = _load_policy_model(policy, device=device)
    physics = VectorizedPhysics(physics_cfg)
    state = AccountBatchState.empty(1, physics_cfg)
    initial_eq = float(state.balance[0])

    for t in range(start_index, end_index):
        if not bool(cache.arrays["latent_valid"][t]):
            continue
        obs = physics.account_observation6(
            state,
            float(cache.arrays["close"][t]),
        )
        latent = np.asarray(
            cache.arrays["market_latent"][t],
            dtype=np.float32,
        )[None, :]
        d, r = _deterministic_actions(
            model,
            dev,
            latent,
            obs,
            supervisor,
        )
        bar = MarketBar(
            open=float(cache.arrays["open"][t + 1]),
            high=float(cache.arrays["high"][t + 1]),
            low=float(cache.arrays["low"][t + 1]),
            close=float(cache.arrays["close"][t + 1]),
            funding_rate=float(cache.arrays["funding_rate"][t + 1]),
        )
        physics.step(
            state,
            bar,
            executable_direction=d,
            executable_risk=r,
            requested_direction=d,
            dependence_group_count=1,
        )

    final_eq = float(
        _terminal_close_equity(
            state,
            float(cache.arrays["close"][end_index]),
            physics_cfg,
        )[0]
    )
    return (
        float("-inf")
        if final_eq <= 0
        else float(math.log(final_eq / initial_eq))
    )


def _validation_blocks(
    *,
    cache: MarketLatentCacheR6,
    constitution: HistoricalChronologyConstitutionR6,
    block_bars: int,
):
    ts = cache.arrays["timestamp"]
    start = int(
        np.searchsorted(
            ts,
            constitution.validation.first_decision_timestamp,
            side="left",
        )
    )
    end = int(
        np.searchsorted(
            ts,
            constitution.validation.last_maturity_timestamp,
            side="left",
        )
    )
    blocks = []
    i = start
    while i + block_bars <= end:
        blocks.append((i, i + block_bars))
        i += block_bars
    return blocks


def _bootstrap_ci(delta, reps, seed):
    delta = np.asarray(delta, dtype=np.float64)
    if len(delta) < 2:
        raise RuntimeError("INSUFFICIENT_VALIDATION_BLOCKS")
    rng = np.random.default_rng(seed)
    boot = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, len(delta), size=len(delta))
        boot[i] = np.mean(delta[idx])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def historical_validation_tournament_phase_r6(*, phase, cycle_spec, plugin_config, context):
    cfg = _cfg(plugin_config)
    if phase != "TOURNAMENT":
        raise RuntimeError("WRONG_PHASE_PLUGIN")
    cdir = _cycle_dir(cfg, cycle_spec)
    constitution = _constitution_from_json(
        json.loads((cdir / "CHRONOLOGY_CONSTITUTION.json").read_text())
    )
    cache = MarketLatentCacheR6(cfg.market_cache_root)

    gs = GenerationStateStore(cfg.generation_state_path)
    champion = gs.current_champion()
    attempt_id = f"R6:{cycle_spec.cycle_id}"
    row = gs.db.execute(
        "SELECT challenger_json FROM attempts WHERE attempt_id=?",
        (attempt_id,),
    ).fetchone()
    if champion is None or row is None or not row[0]:
        gs.close()
        raise RuntimeError("VALIDATION_TOURNAMENT_POLICIES_MISSING")
    challenger = PolicyRecord(**json.loads(row[0]))
    if champion.weight_hash != cycle_spec.parent_policy_hash:
        gs.close()
        raise RuntimeError("VALIDATION_CHAMPION_CHANGED_BEFORE_TOURNAMENT")

    physics_cfg = VectorPhysicsConfig(**dict(cfg.physics))
    supervisor = LocalSupervisorConfigR6(**dict(cfg.supervisor))
    training_cfg = TrainingConfigR6(**dict(cfg.training))
    blocks = _validation_blocks(
        cache=cache,
        constitution=constitution,
        block_bars=cfg.validation_block_bars,
    )
    if len(blocks) < 2:
        gs.close()
        raise RuntimeError("INSUFFICIENT_VALIDATION_TIME_BLOCKS")

    uc = []
    uh = []
    for start, end in blocks:
        uc.append(_simulate_policy_block(
            policy=champion,
            cache=cache,
            start_index=start,
            end_index=end,
            physics_cfg=physics_cfg,
            supervisor=supervisor,
            device=training_cfg.device,
        ))
        uh.append(_simulate_policy_block(
            policy=challenger,
            cache=cache,
            start_index=start,
            end_index=end,
            physics_cfg=physics_cfg,
            supervisor=supervisor,
            device=training_cfg.device,
        ))
    if not all(math.isfinite(x) for x in uc + uh):
        gs.close()
        raise RuntimeError(
            "BANKRUPT_VALIDATION_BLOCK_REQUIRES_EXPLICIT_ADJUDICATION_POLICY"
        )
    delta = np.asarray(uh) - np.asarray(uc)
    lo, hi = _bootstrap_ci(
        delta,
        cfg.validation_bootstrap_reps,
        seed=int(canonical_hash({
            "cycle": cycle_spec.cycle_id,
            "validation": constitution.validation.group_hash,
        })[:16], 16),
    )
    q = np.array_split(np.arange(len(delta)), min(4, len(delta)))
    regimes = {
        f"CHRONO_Q{i}": float(np.mean(delta[idx]))
        for i, idx in enumerate(q)
        if len(idx)
    }

    result = TournamentResult(
        attempt_id=attempt_id,
        champion_weight_hash=champion.weight_hash,
        challenger_weight_hash=challenger.weight_hash,
        evaluation_dataset_hash=canonical_hash({
            "lane": "ITERATIVE_VALIDATION_NOT_FINAL_HOLDOUT",
            "validation_group_hash": constitution.validation.group_hash,
            "cache_identity": cache.receipt.scientific_identity_hash,
            "block_bars": cfg.validation_block_bars,
        }),
        mean_utility_champion=float(np.mean(uc)),
        mean_utility_challenger=float(np.mean(uh)),
        delta_utility=float(np.mean(delta)),
        bootstrap_ci_low=lo,
        bootstrap_ci_high=hi,
        independent_groups=len(blocks),
        regime_deltas=regimes,
        status="COMPLETE",
    )
    gs.record_tournament(result)
    gs.close()

    out = {
        "schema": "CB16_R6_VALIDATION_TOURNAMENT_RECEIPT",
        "cycle_id": cycle_spec.cycle_id,
        "evaluation_lane": "ITERATIVE_VALIDATION_NOT_FINAL_HOLDOUT",
        "result": asdict(result),
        "result_hash": result.content_hash,
        "validation_blocks": len(blocks),
        "final_tournament_opened": False,
    }
    atomic_json(cdir / "VALIDATION_TOURNAMENT_RECEIPT.json", out)
    return out


def historical_adjudicate_commit_phase_r6(*, phase, cycle_spec, plugin_config, context):
    cfg = _cfg(plugin_config)
    if phase != "ADJUDICATE_COMMIT":
        raise RuntimeError("WRONG_PHASE_PLUGIN")
    cdir = _cycle_dir(cfg, cycle_spec)
    gs = GenerationStateStore(cfg.generation_state_path)
    attempt_id = f"R6:{cycle_spec.cycle_id}"
    rule = PromotionRule(**dict(cfg.promotion_rule))
    verdict, reasons = gs.adjudicate(attempt_id, rule)
    gs.commit(attempt_id)
    champion = gs.current_champion()
    gs.close()
    out = {
        "schema": "CB16_R6_ADJUDICATION_RECEIPT",
        "cycle_id": cycle_spec.cycle_id,
        "verdict": verdict,
        "reasons": list(reasons),
        "promotion_rule_hash": rule.content_hash,
        "champion_generation_after": champion.generation,
        "champion_weight_hash_after": champion.weight_hash,
        "final_tournament_opened": False,
    }
    atomic_json(cdir / "ADJUDICATION_RECEIPT.json", out)
    return out


def historical_retention_phase_r6(*, phase, cycle_spec, plugin_config, context):
    cfg = _cfg(plugin_config)
    if phase != "RETENTION":
        raise RuntimeError("WRONG_PHASE_PLUGIN")
    cdir = _cycle_dir(cfg, cycle_spec)
    if not cfg.retention:
        out = {
            "schema": "CB16_R6_RETENTION_RECEIPT",
            "status": "NO_RETENTION_CONFIG",
            "moved": 0,
        }
        atomic_json(cdir / "RETENTION_RECEIPT.json", out)
        return out

    r = dict(cfg.retention)
    policy = RetentionPolicy(**r.get("policy", {}))
    manager = RetentionArchiveManager(
        ssd_root=r["ssd_root"],
        hdd_root=r["hdd_root"],
        policy=policy,
    )
    gs = GenerationStateStore(cfg.generation_state_path)
    champ = gs.current_champion()
    gs.close()
    candidates = []
    for x in r.get("candidates", []):
        candidates.append(ArchiveCandidate(**x))
    receipts = manager.relieve_pressure(
        candidates,
        current_generation=champ.generation,
    )
    out = {
        "schema": "CB16_R6_RETENTION_RECEIPT",
        "status": "PASS",
        "moved": len(receipts),
        "receipts": [asdict(x) for x in receipts],
    }
    atomic_json(cdir / "RETENTION_RECEIPT.json", out)
    return out


def build_phase_plugin_map_r6(campaign_config_path: str) -> dict[str, dict[str, Any]]:
    """Helper for generating `run_multi_generation_r5.py` JSON configuration."""
    module = "cb16_local_opt.historical_campaign_plugins_r6"
    common = {"campaign_config_path": str(campaign_config_path)}
    return {
        "ROLLOUT": {
            "dotted_callable": f"{module}:historical_rollout_phase_r6",
            "version": "R6",
            "config": common,
        },
        "TEACHER_CREDIT": {
            "dotted_callable": f"{module}:historical_teacher_credit_phase_r6",
            "version": "R6",
            "config": common,
        },
        "SEAL_SNAPSHOT": {
            "dotted_callable": f"{module}:historical_seal_snapshot_phase_r6",
            "version": "R6",
            "config": common,
        },
        "TRAIN_CHALLENGER": {
            "dotted_callable": f"{module}:historical_train_challenger_phase_r6",
            "version": "R6",
            "config": common,
        },
        "TOURNAMENT": {
            "dotted_callable": f"{module}:historical_validation_tournament_phase_r6",
            "version": "R6",
            "config": common,
        },
        "ADJUDICATE_COMMIT": {
            "dotted_callable": f"{module}:historical_adjudicate_commit_phase_r6",
            "version": "R6",
            "config": common,
        },
        "RETENTION": {
            "dotted_callable": f"{module}:historical_retention_phase_r6",
            "version": "R6",
            "config": common,
        },
    }
