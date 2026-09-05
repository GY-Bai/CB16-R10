from __future__ import annotations

"""
Scale-integrated R7 historical campaign phase plugins.

Only ROLLOUT is replaced relative to R6. The remaining scientific phases reuse the R6
implementations unchanged.

R7 ROLLOUT:
1. sequential Champion account-state rollout;
2. cached Market64 -> chunked Trader account batching;
3. pre-outcome StudentContext persistence;
4. mmap H72 anchor-store capture;
5. after capture, 3700X spawn farm compiles H72 groups in parallel;
6. farm samples are concatenated into the same TRAIN/VALIDATION files expected by the
   R6 dependence-aware Teacher.

Crash recovery:
- capture stage is sealed separately;
- an incomplete capture is recomputed deterministically;
- a sealed anchor store is reused;
- farm jobs publish immutable per-group receipts and are idempotent.
"""

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .h72_anchor_farm_r7 import (
    H72AnchorStoreWriterR7,
    H72DependenceGroupFarmR7,
    H72FarmConfigR7,
    H72FarmJobR7,
)
from .h72_group_compiler_r7 import GroupAnchorBatchR7
from .historical_campaign_plugins_r6 import (
    ExplorationConfigR6,
    LocalSupervisorConfigR6,
    TrainingConfigR6,
    _account_snapshot,
    _campaign_group_schedule,
    _cfg,
    _cycle_dir,
    _load_policy_model,
    _sample_rollout_actions,
    _supervise,
    atomic_json,
    canonical_hash,
    historical_adjudicate_commit_phase_r6,
    historical_retention_phase_r6,
    historical_seal_snapshot_phase_r6,
    historical_teacher_credit_phase_r6,
    historical_train_challenger_phase_r6,
    historical_validation_tournament_phase_r6,
    sha256_file,
)
from .market_cache_r6 import MarketLatentCacheR6
from .rollout_batching_r7 import (
    CachedLatentRolloutBatcherR7,
    RolloutBatchingConfigR7,
)
from .sharded_experience_lake import ShardedExperienceLake
from .student_training_dataset import StudentContextR4,store_student_context
from .trajectory_compiler_r6 import EconomicClockR6,default_action_grid_r6
from .vectorized_physics import AccountBatchState,MarketBar,VectorPhysicsConfig,VectorizedPhysics
from .generation_orchestrator import GenerationStateStore


def _read_store_receipt(path:Path):
    return json.loads((path/"STORE_RECEIPT.json").read_text())


def _write_concatenated_samples(
    *,
    results,
    allowed_group_ids:set[str],
    output_path:Path,
):
    tmp=output_path.with_name(output_path.name+f".{os.getpid()}.partial")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with tmp.open("wb") as dst:
        for r in results:
            if r.dependence_group_id not in allowed_group_ids:
                continue
            with Path(r.samples_path).open("rb") as src:
                for chunk in iter(lambda:src.read(1<<20),b""):
                    dst.write(chunk)
            count+=r.teacher_sample_count
        dst.flush();os.fsync(dst.fileno())
    os.replace(tmp,output_path)
    return count


def historical_rollout_phase_r7(*,phase,cycle_spec,plugin_config,context):
    if phase!="ROLLOUT":raise RuntimeError("WRONG_PHASE_PLUGIN")
    cfg=_cfg(plugin_config)
    cdir=_cycle_dir(cfg,cycle_spec)
    final_receipt_path=cdir/"ROLLOUT_RECEIPT.json"
    if final_receipt_path.exists():
        old=json.loads(final_receipt_path.read_text())
        if old["cycle_hash"]!=cycle_spec.content_hash:
            raise RuntimeError("R7_ROLLOUT_RECEIPT_CYCLE_CONFLICT")
        return old

    cache=MarketLatentCacheR6(
        cfg.market_cache_root,
        verify_hashes=bool(plugin_config.get("verify_cache_hashes",False)),
    )
    if cache.receipt.dataset_hash!=cycle_spec.dataset_hash:
        raise RuntimeError("R7_CAMPAIGN_DATASET_HASH_MISMATCH")
    constitution,group_index=_campaign_group_schedule(
        cache=cache,cfg=cfg,cycle_spec=cycle_spec
    )
    atomic_json(cdir/"CHRONOLOGY_CONSTITUTION.json",asdict(constitution))
    train_groups=set(constitution.train.group_ids)
    val_groups=set(constitution.validation.group_ids)
    included_groups=sorted(
        train_groups|val_groups,
        key=lambda g:(group_index[g],g)
    )
    store_root=cdir/"H72_ANCHOR_STORE"
    capture_receipt_path=cdir/"R7_CAPTURE_RECEIPT.json"

    # Stage 1: sequential Champion rollout + mmap anchor capture.
    if not capture_receipt_path.exists():
        if store_root.exists() and (store_root/"STORE_RECEIPT.json").exists():
            # Crash may have happened after atomic store publication but before capture receipt.
            sr=_read_store_receipt(store_root)
            if sr["total_groups"]!=len(included_groups):
                raise RuntimeError("R7_EXISTING_ANCHOR_STORE_GROUP_COUNT_CONFLICT")
            atomic_json(capture_receipt_path,{
                "schema":"CB16_R7_CAPTURE_RECEIPT_RECOVERED",
                "cycle_id":cycle_spec.cycle_id,
                "cycle_hash":cycle_spec.content_hash,
                "anchor_store_identity":sr["store_identity_hash"],
                "groups":len(included_groups),
                "recovered_after_publish":True,
            })
        else:
            build_root=cdir/"H72_ANCHOR_STORE_BUILD"
            if build_root.exists():
                shutil.rmtree(build_root)
            writer=H72AnchorStoreWriterR7(
                build_root,
                total_groups=len(included_groups),
                accounts_per_group=cfg.account_replicas,
                feature_dim=70,
            )
            store_slot={g:i for i,g in enumerate(included_groups)}

            gs=GenerationStateStore(cfg.generation_state_path)
            champion=gs.current_champion()
            if champion is None:
                gs.close();raise RuntimeError("NO_CHAMPION")
            if (
                champion.generation!=cycle_spec.generation_parent
                or champion.weight_hash!=cycle_spec.parent_policy_hash
            ):
                gs.close();raise RuntimeError("R7_ROLLOUT_PARENT_NOT_CURRENT_CHAMPION")
            tcfg=TrainingConfigR6(**dict(cfg.training))
            model,dev=_load_policy_model(champion,device=tcfg.device)
            gs.close()

            batcher=CachedLatentRolloutBatcherR7(
                model,
                RolloutBatchingConfigR7(
                    device=str(dev),
                    account_chunk_rows=int(
                        plugin_config.get("account_chunk_rows",8192)
                    ),
                    pin_host_memory=bool(
                        plugin_config.get("pin_host_memory",True)
                    ),
                    non_blocking=True,
                ),
            )
            physics_cfg=VectorPhysicsConfig(**dict(cfg.physics))
            physics=VectorizedPhysics(physics_cfg)
            state=AccountBatchState.empty(
                cfg.account_replicas,physics_cfg,
                account_prefix=f"G{cycle_spec.generation_parent}A"
            )
            supervisor=LocalSupervisorConfigR6(**dict(cfg.supervisor))
            exploration=ExplorationConfigR6(**dict(cfg.exploration))
            rng=np.random.default_rng(
                exploration.seed+1000003*cycle_spec.generation_parent
            )
            lake=ShardedExperienceLake(cfg.experience_lake_root,shards=4)

            ts=cache.arrays["timestamp"]
            valid=cache.arrays["latent_valid"]
            start=max(cfg.warmup_bars,cache.receipt.first_valid_index)
            last_val_index=int(np.searchsorted(
                ts,constitution.validation.last_decision_timestamp,side="left"
            ))
            contexts=0
            inference_rows=0
            inference_ms=0.0
            capture_t0=time.perf_counter()

            for t in range(start,min(last_val_index+1,cache.receipt.rows-1)):
                if not bool(valid[t]):continue
                account6=physics.account_observation6(
                    state,float(cache.arrays["close"][t])
                )
                market_lat=np.asarray(
                    cache.arrays["market_latent"][t],dtype=np.float32
                )
                mapping=np.zeros(cfg.account_replicas,dtype=np.int64)
                logits,probs,risk_raw,br=batcher.infer_policy_heads(
                    unique_market_latent=market_lat[None,:],
                    account_state6=account6,
                    account_to_market=mapping,
                )
                inference_rows+=br.accounts
                inference_ms+=br.elapsed_ms
                req_d,req_r=_sample_rollout_actions(
                    logits=logits,probs=probs,risk_raw=risk_raw,
                    exploration=exploration,rng=rng
                )
                exe_d,exe_r=_supervise(
                    requested_direction=req_d,requested_risk=req_r,
                    account_state6=account6,config=supervisor
                )

                gid=f"DG:{int(ts[t])}"
                if gid in store_slot:
                    parent_ids=[]
                    context_ids=[]
                    for a in range(cfg.account_replicas):
                        parent=(
                            f"G{cycle_spec.generation_parent}:"
                            f"T{int(ts[t])}:A{a:04d}"
                        )
                        context_id=f"CTX:{cycle_spec.cycle_id}:{parent}"
                        ctx=StudentContextR4(
                            context_id=context_id,
                            decision_event_hash=canonical_hash({
                                "cycle":cycle_spec.cycle_id,
                                "parent":parent,
                                "policy":champion.weight_hash,
                                "timestamp":int(ts[t]),
                            }),
                            timestamp=int(ts[t]),
                            symbol=str(plugin_config.get("symbol","BTCUSDT")),
                            account_id=f"A{a:04d}",
                            policy_generation=champion.generation,
                            policy_weight_hash=champion.weight_hash,
                            market_latent=tuple(float(x) for x in market_lat),
                            account_state6=tuple(float(x) for x in account6[a]),
                            market_lineage_hash=cache.receipt.scientific_identity_hash,
                            account_lineage_hash=canonical_hash({
                                "cycle":cycle_spec.cycle_id,
                                "account":a,"timestamp":int(ts[t]),
                                "balance":float(state.balance[a]),
                                "position_qty":float(state.position_qty[a]),
                                "entry_price":float(state.entry_price[a]),
                                "holding_bars":int(state.holding_bars[a]),
                            }),
                        )
                        store_student_context(lake,ctx)
                        contexts+=1
                        parent_ids.append(parent);context_ids.append(context_id)

                    group=GroupAnchorBatchR7(
                        dependence_group_id=gid,
                        decision_index=t,
                        parent_ids=tuple(parent_ids),
                        student_context_object_ids=tuple(context_ids),
                        context_features=np.concatenate([
                            np.broadcast_to(
                                market_lat,(cfg.account_replicas,64)
                            ),
                            account6,
                        ],axis=1).astype(np.float32,copy=False),
                        balance=state.balance.copy(),
                        position_qty=state.position_qty.copy(),
                        entry_price=state.entry_price.copy(),
                        peak_equity=state.peak_equity.copy(),
                        realized_pnl=state.realized_pnl.copy(),
                        margin_used=state.margin_used.copy(),
                        holding_bars=state.holding_bars.copy(),
                        risk_budget_remaining=state.risk_budget_remaining.copy(),
                        risk_budget_capacity=state.risk_budget_capacity.copy(),
                        terminated=state.terminated.copy(),
                        last_mark_price=state.last_mark_price.copy(),
                        market_lineage_hash=cache.receipt.scientific_identity_hash,
                    )
                    writer.write_group(store_slot[gid],group)

                physics.step(
                    state,
                    MarketBar(
                        open=float(cache.arrays["open"][t+1]),
                        high=float(cache.arrays["high"][t+1]),
                        low=float(cache.arrays["low"][t+1]),
                        close=float(cache.arrays["close"][t+1]),
                        funding_rate=float(cache.arrays["funding_rate"][t+1]),
                    ),
                    executable_direction=exe_d,
                    executable_risk=exe_r,
                    requested_direction=req_d,
                    dependence_group_count=1,
                )

            lake.close()
            sr=writer.seal()
            if store_root.exists():shutil.rmtree(store_root)
            os.replace(build_root,store_root)
            capture_receipt={
                "schema":"CB16_R7_CAPTURE_RECEIPT",
                "cycle_id":cycle_spec.cycle_id,
                "cycle_hash":cycle_spec.content_hash,
                "anchor_store_identity":sr.store_identity_hash,
                "groups":sr.total_groups,
                "accounts_per_group":sr.accounts_per_group,
                "student_contexts_stored":contexts,
                "inference_account_rows":inference_rows,
                "inference_elapsed_ms":inference_ms,
                "capture_elapsed_s":time.perf_counter()-capture_t0,
                "market_cache_identity":cache.receipt.scientific_identity_hash,
                "encoder_weight_hash":cache.receipt.encoder_weight_hash,
                "tournament_market_values_opened":False,
            }
            atomic_json(capture_receipt_path,capture_receipt)

    capture=json.loads(capture_receipt_path.read_text())

    # Stage 2: parallel H72 compilation across dependence groups.
    farm_cfg=H72FarmConfigR7(
        workers=int(plugin_config.get("h72_workers",6)),
        start_method=str(plugin_config.get("h72_start_method","spawn")),
        cpu_threads_per_worker=int(plugin_config.get("h72_cpu_threads",1)),
        max_in_flight=int(plugin_config.get("h72_max_in_flight",12)),
    )
    candidate_grid=tuple(
        (c.direction,c.requested_risk)
        for c in default_action_grid_r6(cfg.risk_levels)
    )
    jobs=[
        H72FarmJobR7(
            group_index=i,
            anchor_store_root=str(store_root),
            market_cache_root=str(cfg.market_cache_root),
            result_root=str(cdir/"H72_FARM_RESULTS"),
            physics_config=dict(cfg.physics),
            candidates=candidate_grid,
            horizon_bars=72,
        )
        for i in range(len(included_groups))
    ]
    farm_t0=time.perf_counter()
    results=H72DependenceGroupFarmR7(farm_cfg).run(jobs)
    farm_elapsed=time.perf_counter()-farm_t0

    train_path=cdir/"TRAIN_SAMPLES.jsonl"
    val_path=cdir/"VALIDATION_SAMPLES.jsonl"
    train_count=_write_concatenated_samples(
        results=results,allowed_group_ids=train_groups,output_path=train_path
    )
    val_count=_write_concatenated_samples(
        results=results,allowed_group_ids=val_groups,output_path=val_path
    )
    atomic_json(cdir/"H72_FARM_RECEIPTS.json",[
        asdict(r) for r in results
    ])
    # Compact trajectory indexes rather than duplicating all branch matrices into JSON.
    atomic_json(cdir/"TRAIN_TRAJECTORIES.json",[
        asdict(r) for r in results if r.dependence_group_id in train_groups
    ])
    atomic_json(cdir/"VALIDATION_TRAJECTORIES.json",[
        asdict(r) for r in results if r.dependence_group_id in val_groups
    ])

    receipt={
        "schema":"CB16_R7_SCALE_ROLLOUT_RECEIPT",
        "cycle_id":cycle_spec.cycle_id,
        "cycle_hash":cycle_spec.content_hash,
        "champion_generation":cycle_spec.generation_parent,
        "champion_weight_hash":cycle_spec.parent_policy_hash,
        "constitution_hash":constitution.content_hash,
        "train_dependence_groups":constitution.train.group_count,
        "validation_dependence_groups":constitution.validation.group_count,
        "tournament_dependence_groups_reserved_unopened":constitution.tournament.group_count,
        "anchor_store_identity":capture["anchor_store_identity"],
        "student_contexts_stored":capture.get("student_contexts_stored"),
        "rollout_inference_account_rows":capture.get("inference_account_rows"),
        "rollout_inference_elapsed_ms":capture.get("inference_elapsed_ms"),
        "h72_farm_workers":farm_cfg.workers,
        "h72_farm_groups":len(results),
        "h72_farm_elapsed_s":farm_elapsed,
        "train_counterfactual_branches":train_count,
        "validation_counterfactual_branches":val_count,
        "train_samples_sha256":sha256_file(train_path),
        "validation_samples_sha256":sha256_file(val_path),
        "tournament_market_values_opened":False,
    }
    atomic_json(final_receipt_path,receipt)
    return receipt


def build_phase_plugin_map_r7(
    campaign_config_path:str,
    *,
    h72_workers:int=6,
    h72_cpu_threads:int=1,
    h72_max_in_flight:int=12,
    account_chunk_rows:int=8192,
)->dict[str,dict[str,Any]]:
    module="cb16_local_opt.historical_campaign_plugins_r7"
    r6="cb16_local_opt.historical_campaign_plugins_r6"
    common={"campaign_config_path":str(campaign_config_path)}
    return {
        "ROLLOUT":{
            "dotted_callable":f"{module}:historical_rollout_phase_r7",
            "version":"R7",
            "config":{
                **common,
                "h72_workers":h72_workers,
                "h72_cpu_threads":h72_cpu_threads,
                "h72_max_in_flight":h72_max_in_flight,
                "account_chunk_rows":account_chunk_rows,
            },
        },
        "TEACHER_CREDIT":{
            "dotted_callable":f"{r6}:historical_teacher_credit_phase_r6",
            "version":"R6",
            "config":common,
        },
        "SEAL_SNAPSHOT":{
            "dotted_callable":f"{r6}:historical_seal_snapshot_phase_r6",
            "version":"R6",
            "config":common,
        },
        "TRAIN_CHALLENGER":{
            "dotted_callable":f"{r6}:historical_train_challenger_phase_r6",
            "version":"R6",
            "config":common,
        },
        "TOURNAMENT":{
            "dotted_callable":f"{r6}:historical_validation_tournament_phase_r6",
            "version":"R6",
            "config":common,
        },
        "ADJUDICATE_COMMIT":{
            "dotted_callable":f"{r6}:historical_adjudicate_commit_phase_r6",
            "version":"R6",
            "config":common,
        },
        "RETENTION":{
            "dotted_callable":f"{r6}:historical_retention_phase_r6",
            "version":"R6",
            "config":common,
        },
    }
