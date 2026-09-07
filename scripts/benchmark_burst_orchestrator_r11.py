from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from cb16_local_opt.burst_orchestrator_r11 import BurstLimits, BurstOrchestratorR11, RamSample
from cb16_local_opt.orchestrator_r11 import R11Orchestrator
from cb16_local_opt.runtime_events_r11 import (
    AuthorityStamp,
    CommitReceipt,
    EvidenceRef,
    FrozenInputReceipt,
    JournalReceipt,
    PolicyResult,
    SnapshotSeal,
    StoreReceipt,
    TournamentDecision,
    TrainingResult,
    WorkerPool,
    WorkCompletion,
    WorkItem,
    WorkKind,
    semantic_hash,
)


AUTH = AuthorityStamp(3, "champion-c3", "hash-c3", "teacher-v7", "physics-frozen-v1")


class FakeEvidenceStore:
    def __init__(self):
        self.items = {}
        self.snapshots = {}

    def put_once(self, evidence):
        prior = self.items.get(evidence.evidence_id)
        if prior is None:
            self.items[evidence.evidence_id] = evidence
            return StoreReceipt(evidence.evidence_id, evidence.payload_hash, True, True)
        if prior != evidence:
            raise RuntimeError("evidence conflict")
        return StoreReceipt(evidence.evidence_id, evidence.payload_hash, True, False)

    def get(self, evidence_id):
        return self.items.get(evidence_id)

    def seal_snapshot(self, seal):
        self.snapshots[seal.snapshot_id] = seal
        return seal

    def get_snapshot(self, snapshot_id):
        return self.snapshots.get(snapshot_id)


class FakeJournal:
    def __init__(self):
        self.events = []
        self.ids = {}

    def append_once(self, event):
        if event.event_id in self.ids:
            return JournalReceipt(event.event_id, self.ids[event.event_id], True)
        seq = len(self.events) + 1
        self.events.append(event)
        self.ids[event.event_id] = seq
        return JournalReceipt(event.event_id, seq, True)

    def read_generation(self, generation):
        return [event for event in self.events if event.generation == generation]


class FakeCheckpointStore:
    def __init__(self):
        self.commits = {}
        self.checkpoints = {}

    def atomic_commit(self, proposal):
        if proposal.decision is TournamentDecision.PROMOTE:
            next_id, next_hash = proposal.challenger_id, proposal.challenger_hash
        else:
            next_id, next_hash = proposal.parent_champion_id, AUTH.champion_hash
        receipt = CommitReceipt(
            proposal.generation,
            proposal.decision,
            proposal.parent_champion_id,
            next_id,
            next_hash,
            f"commit-{proposal.generation}",
        )
        self.commits[proposal.generation] = receipt
        return receipt

    def read_commit(self, generation):
        return self.commits.get(generation)

    def seal_checkpoint(self, seal):
        self.checkpoints[seal.generation] = seal
        return seal

    def read_checkpoint(self, generation):
        return self.checkpoints.get(generation)


def old_evidence():
    return EvidenceRef(
        "e-old", "hash-e-old", 2, "champion-c2",
        AUTH.teacher_authority_id, AUTH.physics_authority_id, 2,
    )


def trace_evidence():
    return EvidenceRef(
        "bench-trace", "bench-trace-payload", 3, AUTH.champion_id,
        AUTH.teacher_authority_id, AUTH.physics_authority_id, 3,
    )


def new_core():
    core = R11Orchestrator(
        authority=AUTH,
        evidence_store=FakeEvidenceStore(),
        journal=FakeJournal(),
        checkpoint_store=FakeCheckpointStore(),
    )
    core.verify_frozen_inputs(FrozenInputReceipt("synthetic-inputs", "synthetic-seal"))
    core.accept_teacher_evidence(old_evidence())
    seal = SnapshotSeal("snapshot-g3", "snapshot-hash", 3, AUTH.champion_id, ("e-old",))
    core.seal_training_snapshot(seal)
    core.accept_policy_result(PolicyResult("policy-direct", 3, AUTH.champion_id, "policy-hash"))
    core.start_trace("trace-start")
    return core


def calibrate_cpu(target_seconds: float) -> int:
    payload = bytes(range(256)) * 16384
    loops = 0
    start = time.perf_counter()
    deadline = start + 0.8
    digest = b""
    while time.perf_counter() < deadline:
        digest = hashlib.sha256(payload).digest()
        loops += 1
    elapsed = max(1e-6, time.perf_counter() - start)
    estimated = max(1, int(loops / elapsed * target_seconds))
    if not digest:
        raise RuntimeError("cpu calibration failed")
    return estimated


def cpu_work(loops: int) -> str:
    payload = bytes(range(256)) * 16384
    digest = b""
    for _ in range(loops):
        digest = hashlib.sha256(payload).digest()
    return digest.hex()


def make_gpu_work(target_seconds: float):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_TASK_D_MICROBENCH")
    if tuple(torch.cuda.get_device_capability(0)) != (6, 1):
        raise RuntimeError(f"EXPECTED_SM61_GOT:{torch.cuda.get_device_capability(0)}")
    torch.manual_seed(24680)
    a = torch.linspace(-1.0, 1.0, 512 * 512, dtype=torch.float32, device="cuda").reshape(512, 512)
    b = torch.linspace(1.0, -1.0, 512 * 512, dtype=torch.float32, device="cuda").reshape(512, 512)
    out = torch.empty((512, 512), dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()

    loops = 0
    start = time.perf_counter()
    deadline = start + 0.8
    while time.perf_counter() < deadline:
        torch.mm(a, b, out=out)
        loops += 1
    torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - start)
    fixed_loops = max(1, int(loops / elapsed * target_seconds))

    def run() -> str:
        for _ in range(fixed_loops):
            torch.mm(a, b, out=out)
        torch.cuda.synchronize()
        raw = out.detach().cpu().numpy().tobytes(order="C")
        return hashlib.sha256(raw).hexdigest()

    return fixed_loops, run


def run_case(mode: str, *, cpu_loops: int, gpu_run):
    checksums = {}
    core = new_core()

    def trace_runner(work):
        checksums["cpu"] = cpu_work(cpu_loops)
        evidence = trace_evidence()
        return WorkCompletion(work.work_id, work.kind, work.generation, evidence.payload_hash, evidence)

    def training_runner(work):
        checksums["gpu"] = gpu_run()
        result = TrainingResult(
            work.work_id,
            3,
            "challenger-x",
            "hash-x",
            AUTH.champion_id,
            "snapshot-g3",
            "train-payload",
        )
        return WorkCompletion(work.work_id, work.kind, work.generation, result.payload_hash, result)

    burst = BurstOrchestratorR11(
        core,
        limits=BurstLimits(cpu_workers=1, gpu_workers=1, storage_workers=1, max_pending_work=4),
        runners={WorkKind.TRACE_EXECUTE: trace_runner, WorkKind.TRAIN_CHALLENGER: training_runner},
        ram_probe=lambda: RamSample(1, 100),
    )
    burst.start()
    burst.begin_burst()
    trace = WorkItem("trace-w", WorkKind.TRACE_EXECUTE, 3, WorkerPool.CPU_TRACE, AUTH.champion_id)
    train = WorkItem(
        "train-w", WorkKind.TRAIN_CHALLENGER, 3, WorkerPool.GPU_BRAIN,
        AUTH.champion_id, snapshot_id="snapshot-g3"
    )

    started = time.perf_counter()
    if mode == "serial":
        burst.schedule(trace, evidence_bytes=4096)
        burst.wait_for((trace.work_id,), timeout=180)
        burst.start_challenger_training(train)
        burst.wait_for((train.work_id,), timeout=180)
    elif mode == "overlap":
        burst.schedule(trace, evidence_bytes=4096)
        burst.start_challenger_training(train)
        burst.wait_for((trace.work_id, train.work_id), timeout=180)
    else:
        raise ValueError(mode)
    elapsed = time.perf_counter() - started

    identity = semantic_hash(
        {
            "state": core.record.state.value,
            "snapshot": core.record.snapshot.snapshot_id,
            "trace_evidence": sorted(core.record.trace_evidence_ids),
            "challenger_id": core.record.challenger_id,
            "challenger_hash": core.record.challenger_hash,
            "cpu_checksum": checksums["cpu"],
            "gpu_checksum": checksums["gpu"],
        }
    )
    return {"seconds": elapsed, "identity": identity, "checksums": checksums}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-seconds", type=float, default=12.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 5.0 <= args.target_seconds <= 60.0:
        raise SystemExit("target-seconds must be between 5 and 60")

    cpu_loops = calibrate_cpu(args.target_seconds)
    gpu_loops, gpu_run = make_gpu_work(args.target_seconds)
    serial = run_case("serial", cpu_loops=cpu_loops, gpu_run=gpu_run)
    overlap = run_case("overlap", cpu_loops=cpu_loops, gpu_run=gpu_run)
    identity_equal = serial["identity"] == overlap["identity"]
    speedup = serial["seconds"] / max(1e-9, overlap["seconds"])
    report = {
        "schema": "CB16_R11_STAGE2_TASK_D_MICROBENCH_V1",
        "scientific_verdict": "NONE_MICROBENCH_ONLY",
        "final_holdout_read": False,
        "raw_1m_read": False,
        "cpu_loops": cpu_loops,
        "gpu_loops": gpu_loops,
        "serial": serial,
        "overlap": overlap,
        "semantic_identity_equal": identity_equal,
        "speedup": speedup,
        "performance_gate": speedup >= 1.15,
        "status": "PASS" if identity_equal and speedup >= 1.15 else "FAIL",
    }
    text = json.dumps(report, sort_keys=True, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    if not identity_equal:
        raise SystemExit("TASK_D_SEMANTIC_IDENTITY_MISMATCH")
    if speedup < 1.15:
        raise SystemExit(f"TASK_D_OVERLAP_SPEEDUP_TOO_SMALL:{speedup:.4f}")


if __name__ == "__main__":
    main()
