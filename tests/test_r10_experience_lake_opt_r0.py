from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cb16_local_opt.r10_experience_lake_opt_r0 import (
    MODE_BATCHED,
    MODE_OBJECTWISE,
    put_many_r10_opt_r0,
    semantic_ref_tuple,
)
from cb16_local_opt.sharded_experience_lake import ExperienceObject, ShardedExperienceLake


def _objects(n: int, generation: int = 7):
    out = []
    for i in range(n):
        payload = {
            "schema": "CB16_R10_LAKE_OPT_TEST_PAYLOAD_V1",
            "i": i,
            "values": [((i * 7919 + j * 104729) % 1_000_003) / 1_000_003 for j in range(64)],
        }
        out.append(ExperienceObject(
            object_id=f"R102:G{generation}:E{i:05d}",
            object_type="EVIDENCE_PACKAGE",
            generation=generation,
            policy_weight_hash="p" * 64,
            snapshot_hash=f"snap-{i % 11}",
            lineage_hash=f"lineage-{i}",
            payload=payload,
        ))
    return out


def _write_baseline(root: Path, objects):
    lake = ShardedExperienceLake(root, shards=4, synchronous="FULL")
    refs = []
    try:
        for obj in objects:
            ref, _ = lake.put(obj)
            refs.append(ref)
        snap = lake.seal_snapshot(
            snapshot_id="R102_G7_TRAINING_SNAPSHOT",
            parent_generation=7,
            parent_policy_hash="p" * 64,
            refs=refs,
        )
        audit = lake.audit(verify_payloads=True)
        return [semantic_ref_tuple(x) for x in refs], snap.content_hash, audit
    finally:
        lake.close()


@pytest.mark.parametrize("mode", [MODE_OBJECTWISE, MODE_BATCHED])
def test_candidate_final_state_matches_baseline(mode):
    objects = _objects(80)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        baseline_refs, baseline_snap, baseline_audit = _write_baseline(root / "baseline", objects)
        lake = ShardedExperienceLake(root / "candidate", shards=4, synchronous="FULL")
        try:
            refs, receipt = put_many_r10_opt_r0(lake, objects, mode=mode, batch_size=13)
            snap = lake.seal_snapshot(
                snapshot_id="R102_G7_TRAINING_SNAPSHOT",
                parent_generation=7,
                parent_policy_hash="p" * 64,
                refs=refs,
            )
            assert [semantic_ref_tuple(x) for x in refs] == baseline_refs
            assert snap.content_hash == baseline_snap
            assert receipt.created_count == len(objects)
            assert lake.audit(verify_payloads=True)["pass"] is True
            assert baseline_audit["pass"] is True

            refs2, replay = put_many_r10_opt_r0(lake, objects, mode=mode, batch_size=13)
            assert [semantic_ref_tuple(x) for x in refs2] == baseline_refs
            assert replay.created_count == 0
            assert replay.reused_count == len(objects)
        finally:
            lake.close()


def test_same_id_different_content_conflict_is_preserved():
    objects = _objects(20)
    with tempfile.TemporaryDirectory() as td:
        lake = ShardedExperienceLake(td, shards=4, synchronous="FULL")
        try:
            put_many_r10_opt_r0(lake, objects, mode=MODE_BATCHED, batch_size=8)
            old = objects[3]
            bad = ExperienceObject(
                object_id=old.object_id,
                object_type=old.object_type,
                generation=old.generation,
                policy_weight_hash=old.policy_weight_hash,
                snapshot_hash=old.snapshot_hash,
                lineage_hash=old.lineage_hash,
                payload={"changed": True},
            )
            with pytest.raises(RuntimeError, match="EXPERIENCE_ID_CONTENT_CONFLICT"):
                put_many_r10_opt_r0(lake, [bad], mode=MODE_BATCHED, batch_size=8)
        finally:
            lake.close()


@pytest.mark.parametrize("point", ["AFTER_PAYLOADS_BEFORE_METADATA", "AFTER_METADATA_COMMIT"])
def test_batched_recovery_converges_after_injected_failure(point):
    objects = _objects(96)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        baseline_refs, baseline_snap, _ = _write_baseline(root / "baseline", objects)
        lake_root = root / "candidate"
        lake = ShardedExperienceLake(lake_root, shards=4, synchronous="FULL")
        fired = {"v": False}

        def hook(where, shard, batch):
            if not fired["v"] and where == point and shard == 0 and batch == 0:
                fired["v"] = True
                raise RuntimeError("INJECTED_R10_LAKE_OPT_FAILURE")

        try:
            with pytest.raises(RuntimeError, match="INJECTED_R10_LAKE_OPT_FAILURE"):
                put_many_r10_opt_r0(
                    lake,
                    objects,
                    mode=MODE_BATCHED,
                    batch_size=11,
                    fail_hook=hook,
                )
        finally:
            lake.close()
        assert fired["v"] is True

        recovered = ShardedExperienceLake(lake_root, shards=4, synchronous="FULL")
        try:
            refs, _ = put_many_r10_opt_r0(recovered, objects, mode=MODE_BATCHED, batch_size=11)
            snap = recovered.seal_snapshot(
                snapshot_id="R102_G7_TRAINING_SNAPSHOT",
                parent_generation=7,
                parent_policy_hash="p" * 64,
                refs=refs,
            )
            assert [semantic_ref_tuple(x) for x in refs] == baseline_refs
            assert snap.content_hash == baseline_snap
            assert recovered.audit(verify_payloads=True)["pass"] is True
        finally:
            recovered.close()
