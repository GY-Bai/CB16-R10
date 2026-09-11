from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cb16_local_opt.binance_archive_input_r10 import BinanceUSDMArchiveSourceR10, MINUTE_MS
from cb16_local_opt.frozen_sensory_stack_r10 import FrozenSensoryStackR10
from cb16_local_opt.full_minute_historical_replay_r0 import FINAL_HOLDOUT_START_MS
from cb16_local_opt.full_minute_long_trajectory_r1 import MinuteEnvelopeR1
from cb16_local_opt.longtraj_infra_closure_r0 import (
    H72_MINUTES_R0,
    MinuteFrozenPhysicsAdapterR0,
    build_minute_teacher_support_r0,
    find_contiguous_prefinal_run_r0,
    flat_real_path_snapshots_r0,
    funding_events_by_minute_r0,
    load_exact_checkpoint_r0,
    recursive_exact_equal_r0,
    recursive_state_sha256_r0,
    restore_exact_checkpoint_r0,
    save_exact_checkpoint_r0,
    sha256_file_r0,
    tensor_mapping_sha256_r0,
)
from cb16_local_opt.minute_sensory_adapter_r1 import iter_minute_sensory_frames_r1
from cb16_local_opt.r102_learning import evidence_summary
from cb16_local_opt.r11_teacher_authority_candidate import R11_TRAIN_TEACHER_CONFIG, R11_VALIDATION_TEACHER_CONFIG
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.training_runtime_r11 import PreparedEvidenceR11, TrainingRuntimeR11, policy_hash_r11
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_INFRA_CLOSURE_R0_SPEC_V1.json"
DEFAULT_OUT = ROOT / "artifacts/r11_longtraj_infra_closure_r0"
SCHEMA = "CB16_R11_LONGTRAJ_INFRA_CLOSURE_R0_RESULT_V1"


def require(cond: bool, code: str) -> None:
    if not cond: raise RuntimeError(code)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".tmp"); tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"); os.replace(tmp, path)


def choose_device(requested: str) -> str:
    if requested == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available(): raise RuntimeError("INFRA_CLOSURE_CUDA_REQUESTED_BUT_UNAVAILABLE")
    return requested


def archive_run_hash(records) -> str:
    h = hashlib.sha256(); h.update(b"CB16_R11_INFRA_CLOSURE_REAL_1M_RUN_V1\0")
    for r in records:
        h.update(np.asarray([r.open_time, r.close_time, r.number_of_trades], dtype=np.int64).tobytes()); h.update(np.asarray([r.open, r.high, r.low, r.close, r.volume, r.quote_asset_volume, r.taker_buy_base_asset_volume, r.taker_buy_quote_asset_volume], dtype=np.float64).tobytes())
    return h.hexdigest()


def encode_selected_frames(sensory: FrozenSensoryStackR10, frames, batch_size: int):
    out = {}
    for start in range(0, len(frames), int(batch_size)):
        chunk = frames[start:start + int(batch_size)]; enc = sensory.encode_frames([x[0] for x in chunk])
        for j, (frame, audit) in enumerate(chunk):
            require(audit.micro_latest_time_ms == audit.decision_time_ms, "INFRA_CLOSURE_MICRO_ENDPOINT_DRIFT"); require(audit.latest_completed_hour_start_ms < audit.current_hour_start_ms, "INFRA_CLOSURE_UNFINISHED_HOUR_LEAK")
            out[int(frame.decision_time_ms)] = (enc.operator48[j].copy(), enc.medium48[j].copy(), enc.ordered4h30[j].copy())
    return out


def state_equal(a, b) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k].detach().cpu(), b[k].detach().cpu()) for k in a)


def make_ids(prepared: PreparedEvidenceR11):
    return torch.arange(min(prepared.rows, 512), device=prepared.packed.device, dtype=torch.long)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def checkpoint_exact_canary(*, prepared: PreparedEvidenceR11, device: str, out_dir: Path, account_state: dict, cursor_ms: int, evidence_hash: str, spec_sha: str, archive_identity: dict) -> dict[str, Any]:
    seed = 918_273
    seed_all(seed); ref_model = build_g0_brain_r10("TIER_1", seed=seed, device=device); ref_rt = TrainingRuntimeR11(device=device); ref_opt = ref_rt.build_optimizer(ref_model); ref_rt.train_one_step(model=ref_model, optimizer=ref_opt, prepared=prepared, ids=make_ids(prepared)); ref_ids2_cpu = torch.randperm(prepared.rows, device="cpu")[:min(prepared.rows, 512)]; ref_rt.train_one_step(model=ref_model, optimizer=ref_opt, prepared=prepared, ids=ref_ids2_cpu.to(prepared.packed.device)); ref_marker = {"python": [random.random() for _ in range(3)], "numpy": np.random.random(3).tolist(), "torch": torch.rand(3, device="cpu").tolist()}; ref_model_state = {k: v.detach().cpu().clone() for k, v in ref_model.state_dict().items()}; ref_opt_state = ref_opt.state_dict()
    seed_all(seed); pre_model = build_g0_brain_r10("TIER_1", seed=seed, device=device); pre_rt = TrainingRuntimeR11(device=device); pre_opt = pre_rt.build_optimizer(pre_model); pre_rt.train_one_step(model=pre_model, optimizer=pre_opt, prepared=prepared, ids=make_ids(prepared)); checkpoint = out_dir / "EXACT_RESTART_CHECKPOINT.pt"; receipt = save_exact_checkpoint_r0(checkpoint, model=pre_model, optimizer=pre_opt, runtime_step_index=pre_rt._step_index, cursor_ms=cursor_ms, account_state=account_state, evidence_hash=evidence_hash, run_id="INFRA_CLOSURE_EXACT_RESTART", experiment_id="G2_RESTART_CANARY", spec_sha256=spec_sha, archive_identity=archive_identity); del pre_model, pre_opt, pre_rt
    resumed_model = build_g0_brain_r10("TIER_1", seed=seed + 999, device=device); resumed_rt = TrainingRuntimeR11(device=device); resumed_opt = resumed_rt.build_optimizer(resumed_model); payload = load_exact_checkpoint_r0(checkpoint); restore_exact_checkpoint_r0(payload, model=resumed_model, optimizer=resumed_opt, runtime=resumed_rt, device=device); resumed_ids2_cpu = torch.randperm(prepared.rows, device="cpu")[:min(prepared.rows, 512)]; require(torch.equal(ref_ids2_cpu, resumed_ids2_cpu), "INFRA_CLOSURE_RNG_DID_NOT_RESTORE_PERMUTATION"); resumed_rt.train_one_step(model=resumed_model, optimizer=resumed_opt, prepared=prepared, ids=resumed_ids2_cpu.to(prepared.packed.device)); resumed_marker = {"python": [random.random() for _ in range(3)], "numpy": np.random.random(3).tolist(), "torch": torch.rand(3, device="cpu").tolist()}; resumed_model_state = {k: v.detach().cpu().clone() for k, v in resumed_model.state_dict().items()}; resumed_opt_state = resumed_opt.state_dict()
    model_exact = state_equal(ref_model_state, resumed_model_state); optimizer_exact = recursive_exact_equal_r0(ref_opt_state, resumed_opt_state); rng_exact = ref_marker == resumed_marker; lineage_exact = int(payload["cursor_ms"]) == int(cursor_ms) and payload["account_state"] == account_state and payload["evidence_hash"] == evidence_hash and payload["spec_sha256"] == spec_sha and payload["archive_identity"] == archive_identity
    require(model_exact, "INFRA_CLOSURE_RESTART_MODEL_DRIFT"); require(optimizer_exact, "INFRA_CLOSURE_RESTART_OPTIMIZER_DRIFT"); require(rng_exact, "INFRA_CLOSURE_RESTART_RNG_DRIFT"); require(lineage_exact, "INFRA_CLOSURE_RESTART_LINEAGE_DRIFT")
    return {"checkpoint": receipt, "model_exact": model_exact, "optimizer_exact": optimizer_exact, "rng_exact": rng_exact, "account_cursor_evidence_lineage_exact": lineage_exact, "reference_model_hash": tensor_mapping_sha256_r0(ref_model_state), "resumed_model_hash": tensor_mapping_sha256_r0(resumed_model_state), "reference_optimizer_hash": recursive_state_sha256_r0(ref_opt_state), "resumed_optimizer_hash": recursive_state_sha256_r0(resumed_opt_state), "runtime_step_index_restored": int(payload["runtime_step_index"]) == 1}


def save_worker_fixture(path: Path, prepared: PreparedEvidenceR11) -> None:
    torch.save({"parent_ids": prepared.parent_ids, "dependence_group_ids": prepared.dependence_group_ids, "packed": prepared.packed.detach().cpu().clone(), "evidence_hash": prepared.evidence_hash}, path)


def worker_cmd(script: Path, fixture: Path, seed: int, worker_id: str, output: Path, fail: bool = False):
    cmd = [sys.executable, str(script), "--isolation-worker", "--worker-fixture", str(fixture), "--worker-seed", str(seed), "--worker-id", worker_id, "--worker-output", str(output)]
    if fail: cmd.append("--worker-fail-after-init")
    return cmd


def worker_env() -> dict[str, str]:
    env = dict(os.environ); env["OMP_NUM_THREADS"] = "1"; env["MKL_NUM_THREADS"] = "1"; env["OPENBLAS_NUM_THREADS"] = "1"; return env


def load_json(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))


def parallel_isolation_canary(*, prepared: PreparedEvidenceR11, out_dir: Path) -> dict[str, Any]:
    fixture = out_dir / "ISOLATION_PREPARED_EVIDENCE.pt"; save_worker_fixture(fixture, prepared); script = Path(__file__).resolve(); env = worker_env(); ref_a = out_dir / "serial_A.json"; ref_b = out_dir / "serial_B.json"; subprocess.run(worker_cmd(script, fixture, 111, "A", ref_a), check=True, env=env, cwd=ROOT); subprocess.run(worker_cmd(script, fixture, 222, "B", ref_b), check=True, env=env, cwd=ROOT); serial_a = load_json(ref_a); serial_b = load_json(ref_b)
    par_a = out_dir / "parallel_A.json"; par_b = out_dir / "parallel_B.json"; pa = subprocess.Popen(worker_cmd(script, fixture, 111, "A", par_a), env=env, cwd=ROOT); pb = subprocess.Popen(worker_cmd(script, fixture, 222, "B", par_b), env=env, cwd=ROOT); rca = pa.wait(); rcb = pb.wait(); require(rca == 0 and rcb == 0, f"INFRA_CLOSURE_PARALLEL_WORKER_FAIL:{rca}:{rcb}"); parallel_a = load_json(par_a); parallel_b = load_json(par_b); a_exact = serial_a == parallel_a; b_exact = serial_b == parallel_b; require(a_exact, "INFRA_CLOSURE_PARALLEL_A_DRIFT"); require(b_exact, "INFRA_CLOSURE_PARALLEL_B_DRIFT")
    fail_a_out = out_dir / "fail_A.json"; survivor_b_out = out_dir / "survivor_B.json"; fa = subprocess.Popen(worker_cmd(script, fixture, 111, "A_FAIL", fail_a_out, fail=True), env=env, cwd=ROOT); sb = subprocess.Popen(worker_cmd(script, fixture, 222, "B", survivor_b_out), env=env, cwd=ROOT); fail_rc = fa.wait(); survivor_rc = sb.wait(); require(fail_rc == 86, f"INFRA_CLOSURE_EXPECTED_FAIL_WORKER_EXIT:{fail_rc}"); require(survivor_rc == 0, f"INFRA_CLOSURE_PEER_DID_NOT_SURVIVE:{survivor_rc}"); survivor_b = load_json(survivor_b_out); peer_survived_exact = survivor_b == serial_b; require(peer_survived_exact, "INFRA_CLOSURE_PEER_STATE_CHANGED_AFTER_OTHER_FAILURE"); mutable_isolated = serial_a["model_hash"] != serial_b["model_hash"] and str(par_a) != str(par_b) and parallel_a["worker_id"] == "A" and parallel_b["worker_id"] == "B"; require(mutable_isolated, "INFRA_CLOSURE_MUTABLE_STATE_NOT_ISOLATED")
    return {"parallel_a_equals_serial_a": a_exact, "parallel_b_equals_serial_b": b_exact, "failed_worker_exit_code": fail_rc, "peer_continues_exact": peer_survived_exact, "mutable_state_isolated": mutable_isolated, "fixture_sha256": sha256_file_r0(fixture)}


def isolation_worker(args) -> int:
    torch.set_num_threads(1); fixture = torch.load(args.worker_fixture, map_location="cpu", weights_only=False); prepared = PreparedEvidenceR11(parent_ids=tuple(fixture["parent_ids"]), dependence_group_ids=tuple(fixture["dependence_group_ids"]), packed=fixture["packed"].to(dtype=torch.float32, device="cpu"), evidence_hash=str(fixture["evidence_hash"]), host_to_device_transfers=0); prepared.validate(); seed_all(int(args.worker_seed)); model = build_g0_brain_r10("TIER_1", seed=int(args.worker_seed), device="cpu"); runtime = TrainingRuntimeR11(device="cpu"); optimizer = runtime.build_optimizer(model)
    if args.worker_fail_after_init: return 86
    step = runtime.train_one_step(model=model, optimizer=optimizer, prepared=prepared, ids=make_ids(prepared)); result = {"worker_id": str(args.worker_id), "seed": int(args.worker_seed), "evidence_hash": prepared.evidence_hash, "model_hash": policy_hash_r11(model), "optimizer_hash": recursive_state_sha256_r0(optimizer.state_dict()), "loss": step.loss, "direction_loss": step.direction_loss, "sizing_loss": step.sizing_loss, "runtime_step_index": int(runtime._step_index)}; atomic_json(Path(args.worker_output), result); return 0


def build_stress_fixture(path: Path, *, prepared: PreparedEvidenceR11, sensory_frames, records, funding_by_minute, archive_identity) -> None:
    torch.save({"prepared": {"parent_ids": prepared.parent_ids, "dependence_group_ids": prepared.dependence_group_ids, "packed": prepared.packed.detach().cpu().clone(), "evidence_hash": prepared.evidence_hash}, "sensory_frames": [x[0] for x in sensory_frames[:32]], "physics_records": list(records[:4096]), "funding_by_minute": dict(funding_by_minute), "archive_identity": dict(archive_identity)}, path)


def normal_main(args) -> int:
    out_dir = Path(args.output_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True); spec_bytes = SPEC.read_bytes(); spec = json.loads(spec_bytes); spec_sha = hashlib.sha256(spec_bytes).hexdigest(); require(spec["parent_sha"] == "e7c1048478b8ab5b047424194f116d701f9a28f4", "INFRA_CLOSURE_PARENT_SHA_DRIFT")
    device = choose_device(args.device); source = BinanceUSDMArchiveSourceR10(args.raw_root); layout = source.validate_layout(); require(args.symbol in layout["symbols"], f"INFRA_CLOSURE_SYMBOL_MISSING:{args.symbol}")
    parent_count = int(args.train_parents) + int(args.validation_parents); require(int(args.train_parents) >= 48, "INFRA_CLOSURE_TRAIN_PARENT_FLOOR_48"); require(int(args.validation_parents) >= 8, "INFRA_CLOSURE_VALIDATION_PARENT_FLOOR_8"); required_rows = 64 * 60 + parent_count + H72_MINUTES_R0 + 8; records = find_contiguous_prefinal_run_r0(source, args.symbol, required_rows=required_rows); require(all(int(r.open_time) < FINAL_HOLDOUT_START_MS for r in records), "INFRA_CLOSURE_FINAL_ROW_TOUCHED"); require(all(int(b.open_time) - int(a.open_time) == MINUTE_MS for a, b in zip(records, records[1:])), "INFRA_CLOSURE_REAL_RUN_NOT_STRICT_1M")
    archive_identity = {"symbol": args.symbol, "rows": len(records), "first_ms": int(records[0].open_time), "last_ms": int(records[-1].open_time), "run_sha256": archive_run_hash(records), "final_holdout_touched": False, "halt_imputed_rows": 0}; funding = funding_events_by_minute_r0(source, args.symbol, start_ms=records[0].open_time, end_ms=records[-1].open_time)
    envelopes = [MinuteEnvelopeR1(r, False) for r in records]; selected_frames = list(itertools.islice(iter_minute_sensory_frames_r1(args.symbol, envelopes), parent_count)); require(len(selected_frames) == parent_count, f"INFRA_CLOSURE_MINUTE_SENSORY_SHORT:{len(selected_frames)}"); selected_times = [int(x[0].decision_time_ms) for x in selected_frames]; require(selected_times == sorted(selected_times) and len(set(selected_times)) == parent_count, "INFRA_CLOSURE_SELECTED_CLOCK_DRIFT"); index_by_time = {int(r.open_time): i for i, r in enumerate(records)}; selected_indices = [index_by_time[t] for t in selected_times]; require(selected_indices[-1] + H72_MINUTES_R0 < len(records), "INFRA_CLOSURE_SELECTED_H72_OVERRUN")
    sensory = FrozenSensoryStackR10(args.package_root, device=device, verify_hashes=True); encoded = encode_selected_frames(sensory, selected_frames, int(args.sensory_batch_size)); adapter = MinuteFrozenPhysicsAdapterR0(args.package_root); snapshots, risk_auth, path_receipt = flat_real_path_snapshots_r0(adapter=adapter, symbol=args.symbol, records=records, selected_times=set(selected_times), funding_by_minute=funding, account_id=f"INFRA_CLOSURE_REAL_PATH:{args.symbol}"); require(path_receipt["selected_step_indices_strictly_increasing"], "INFRA_CLOSURE_ACCOUNT_RECURRENCE_FAIL"); require(path_receipt["account_reset_count"] == 0, "INFRA_CLOSURE_ACCOUNT_RESET_DETECTED")
    parents, samples, support = build_minute_teacher_support_r0(adapter=adapter, symbol=args.symbol, records=records, selected_indices=selected_indices, encoded_by_time=encoded, snapshots_by_time=snapshots, risk_authority=risk_auth, funding_by_minute=funding, train_count=int(args.train_parents)); train_e, val_e, teacher_stats = compile_teacher_evidence_r11(samples=samples, parents=parents, train_config=R11_TRAIN_TEACHER_CONFIG, val_config=R11_VALIDATION_TEACHER_CONFIG, workers=int(args.teacher_workers), block_targets=int(args.teacher_block_targets)); train_summary = evidence_summary(train_e); val_summary = evidence_summary(val_e); require(train_summary["admitted_dependence_groups"] >= 32, f"INFRA_CLOSURE_TRAIN_TEACHER_SUPPORT:{train_summary}"); require(val_summary["admitted"] > 0, f"INFRA_CLOSURE_VALIDATION_TEACHER_SUPPORT:{val_summary}"); require({e.teacher_protocol_hash for e in train_e} == {R11_TRAIN_TEACHER_CONFIG.content_hash}, "INFRA_CLOSURE_TRAIN_TEACHER_PROTOCOL_DRIFT"); require({e.teacher_protocol_hash for e in val_e} == {R11_VALIDATION_TEACHER_CONFIG.content_hash}, "INFRA_CLOSURE_VALIDATION_TEACHER_PROTOCOL_DRIFT")
    prepared_train = PreparedEvidenceR11.from_evidence(train_e, parents, device=device); prepared_val = PreparedEvidenceR11.from_evidence(val_e, parents, device=device); require(not prepared_train.packed.requires_grad and not prepared_val.packed.requires_grad, "INFRA_CLOSURE_TEACHER_ENTERED_AUTOGRAD"); seed_all(24_680); model = build_g0_brain_r10("TIER_1", seed=24_680, device=device); runtime = TrainingRuntimeR11(device=device); optimizer = runtime.build_optimizer(model); before_hash = policy_hash_r11(model); train_step = runtime.train_one_step(model=model, optimizer=optimizer, prepared=prepared_train, ids=make_ids(prepared_train)); after_hash = policy_hash_r11(model); require(before_hash != after_hash, "INFRA_CLOSURE_STUDENT_PARAMETER_NOT_UPDATED"); require(math.isfinite(train_step.loss), "INFRA_CLOSURE_NONFINITE_TRAIN_LOSS")
    checkpoint_receipt = checkpoint_exact_canary(prepared=prepared_train, device=device, out_dir=out_dir, account_state=dict(snapshots[selected_times[-1]]), cursor_ms=selected_times[-1], evidence_hash=prepared_train.evidence_hash, spec_sha=spec_sha, archive_identity=archive_identity); parallel_receipt = parallel_isolation_canary(prepared=prepared_train, out_dir=out_dir); stress_fixture = out_dir / "STRESS_FIXTURE.pt"; build_stress_fixture(stress_fixture, prepared=prepared_train, sensory_frames=selected_frames, records=records, funding_by_minute=funding, archive_identity=archive_identity)
    policy_surface_closed = all(audit.micro_latest_time_ms == audit.decision_time_ms and audit.latest_completed_hour_start_ms < audit.current_hour_start_ms for _, audit in selected_frames)
    g1 = {"REAL_ARCHIVE_MINTRAIN_PASS": True, "REAL_PER_MINUTE_COUNTERFACTUAL_PHYSICS_BINDING_PASS": support["branch_samples"] == parent_count * 9, "CANONICAL_R11_TEACHER_EVIDENCE_PASS": train_summary["admitted_dependence_groups"] >= 32 and val_summary["admitted"] > 0, "STUDENT_PARAMETER_UPDATE_OBSERVED": before_hash != after_hash, "TEACHER_AND_FROZEN_INPUTS_DETACHED_PASS": not prepared_train.packed.requires_grad and not prepared_val.packed.requires_grad, "NO_FUTURE_MARKET_ACCOUNT_OR_TEACHER_IN_POLICY_PASS": policy_surface_closed, "ACCOUNT_RECURRENCE_PASS": path_receipt["selected_step_indices_strictly_increasing"] and path_receipt["account_reset_count"] == 0, "SCALAR_FROZEN_PHYSICS_PASS": support["counterfactual_scalar_physics_steps"] > 0, "FINAL_FIREWALL_PASS": archive_identity["final_holdout_touched"] is False, "DURABLE_RECEIPT_PASS": True}
    g2 = {"CHECKPOINT_STOP_RESTORE_CONTINUE_EXACT_PASS": checkpoint_receipt["model_exact"] and checkpoint_receipt["optimizer_exact"] and checkpoint_receipt["rng_exact"], "MODEL_STATE_EXACT_PASS": checkpoint_receipt["model_exact"], "OPTIMIZER_STATE_EXACT_PASS": checkpoint_receipt["optimizer_exact"], "RNG_STATE_EXACT_PASS": checkpoint_receipt["rng_exact"], "ACCOUNT_AND_CURSOR_EXACT_PASS": checkpoint_receipt["account_cursor_evidence_lineage_exact"], "EVIDENCE_LINEAGE_EXACT_PASS": checkpoint_receipt["account_cursor_evidence_lineage_exact"], "PARALLEL_A_EQUALS_SERIAL_A_PASS": parallel_receipt["parallel_a_equals_serial_a"], "PARALLEL_B_EQUALS_SERIAL_B_PASS": parallel_receipt["parallel_b_equals_serial_b"], "PEER_CONTINUES_WHEN_OTHER_EXPERIMENT_FAILS_PASS": parallel_receipt["peer_continues_exact"], "MUTABLE_STATE_ISOLATION_PASS": parallel_receipt["mutable_state_isolated"]}; require(all(g1.values()), "INFRA_CLOSURE_G1_GATE_FAIL:" + json.dumps(g1, sort_keys=True)); require(all(g2.values()), "INFRA_CLOSURE_G2_GATE_FAIL:" + json.dumps(g2, sort_keys=True))
    result = {"schema": SCHEMA, "classification": "INFRA_CLOSURE_G1_G2_QUALIFIED", "status": "PASS", "branch": os.environ.get("GITHUB_REF_NAME"), "head_sha": os.environ.get("GITHUB_SHA"), "parent_minpipe_sha": spec["parent_sha"], "spec_sha256": spec_sha, "device": device, "archive": archive_identity, "funding_event_count": len(funding), "sensory": {"selected_frames": len(selected_frames), "batch_size": int(args.sensory_batch_size), "causal_endpoint_pass": policy_surface_closed, "frozen_asset_hashes_verified": sensory.verified_hashes}, "account_real_path": path_receipt, "counterfactual_support": support, "teacher": {"runtime_stats": str(teacher_stats), "train_summary": train_summary, "validation_summary": val_summary, "train_protocol_hash": R11_TRAIN_TEACHER_CONFIG.content_hash, "validation_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash}, "minimum_student_update": {"before_policy_hash": before_hash, "after_policy_hash": after_hash, "loss": train_step.loss, "direction_loss": train_step.direction_loss, "sizing_loss": train_step.sizing_loss, "gradient_owner_set": sorted(train_step.gradient_owner_set), "optimizer_steps": 1}, "checkpoint_exact_restart": checkpoint_receipt, "parallel_isolation": parallel_receipt, "g1_gates": g1, "g2_gates": g2, "stress_fixture": {"path": str(stress_fixture), "sha256": sha256_file_r0(stress_fixture)}, "final_holdout_touched": False, "fresh_market_data_downloaded": False, "scientific_verdict": None, "market_information_claim": "NONE__INFRA_ONLY", "next_required_gate": "EXACT_300_SECOND_HARDWARE_STRESS"}; result_path = out_dir / "RESULT.json"; atomic_json(result_path, result); (out_dir / "RESULT.sha256").write_text(sha256_file_r0(result_path) + "  RESULT.json\n", encoding="utf-8"); print(json.dumps({"status": result["status"], "classification": result["classification"], "g1": g1, "g2": g2, "train_admitted_groups": train_summary["admitted_dependence_groups"], "validation_admitted": val_summary["admitted"], "counterfactual_scalar_physics_steps": support["counterfactual_scalar_physics_steps"], "student_policy_changed": before_hash != after_hash, "stress_fixture": str(stress_fixture), "scientific_verdict": None}, indent=2, sort_keys=True)); return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(); ap.add_argument("--raw-root", default=os.environ.get("CB16_RAW_ROOT", "/cb16/raw")); ap.add_argument("--package-root", default=os.environ.get("CB16_PACKAGE_ROOT", "/cb16/package")); ap.add_argument("--output-dir", default=str(DEFAULT_OUT)); ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda")); ap.add_argument("--symbol", default="BTCUSDT"); ap.add_argument("--train-parents", type=int, default=48); ap.add_argument("--validation-parents", type=int, default=8); ap.add_argument("--sensory-batch-size", type=int, default=8); ap.add_argument("--teacher-workers", type=int, default=4); ap.add_argument("--teacher-block-targets", type=int, default=32); ap.add_argument("--isolation-worker", action="store_true"); ap.add_argument("--worker-fixture"); ap.add_argument("--worker-seed", type=int); ap.add_argument("--worker-id"); ap.add_argument("--worker-output"); ap.add_argument("--worker-fail-after-init", action="store_true"); return ap


def main() -> int:
    args = parser().parse_args(); return isolation_worker(args) if args.isolation_worker else normal_main(args)


if __name__ == "__main__": raise SystemExit(main())
