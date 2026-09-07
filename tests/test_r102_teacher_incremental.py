from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.probabilistic_teacher_r6 import (
    DependenceAwareProbabilisticTeacherR6,
    DependenceAwareTeacherConfigR6,
)
from cb16_local_opt.r102_evidence_cache import ParentContextR102
from cb16_local_opt.r102_teacher_incremental import (
    ExactIncrementalTeacherR6,
    compile_teacher_evidence_incremental,
)


ACTIONS = ((-1, 0.5), (0, 0.0), (1, 0.5))


def _config(*, mode: str) -> DependenceAwareTeacherConfigR6:
    return DependenceAwareTeacherConfigR6(
        teacher_version=f"TEST_{mode}",
        mode=mode,
        n_folds=3,
        embargo_groups=0,
        k_dependence_groups=4,
        min_train_dependence_groups=2,
        min_effective_dependence_n=1.0,
        max_nearest_distance=1000.0,
        distance_temperature=2.0,
        direction_softmax_temperature=0.1,
        lane="CENTER",
        direction_weight=1.0,
        sizing_weight=1.0,
    )


def _fixture(groups: int = 12):
    samples = []
    parents = {}
    for i in range(groups):
        pid = f"P{i:02d}"
        dep = f"D{i:02d}"
        split = "TRAIN" if i < groups - 3 else "VALIDATION"
        features = (float(i) / 10.0, float(i % 3), 1.0 + float(i) / 100.0)
        parents[pid] = ParentContextR102(
            parent_id=pid,
            dependence_group_id=dep,
            symbol="BTCUSDT",
            decision_time_ms=i * 3600_000,
            split=split,
            scenario="CLEAN_FLAT_FULL",
            operator48=(features[0],),
            medium48=(features[1],),
            account6=(features[2],),
            ordered4h30=(0.0,),
            current_mark=100.0,
            snapshot_sha256=f"{i:064x}",
            eligible_for_economic_evidence=True,
            market_lineage_hash=f"{i + 1:064x}",
        )
        for direction, risk in ACTIONS:
            utility = 0.01 * i + 0.002 * direction - 0.001 * risk
            samples.append(CounterfactualBranchSampleR5(
                parent_id=pid,
                student_context_object_id=f"CTX:{pid}",
                timestamp=i * 3600_000,
                context_features=features,
                direction=direction,
                requested_risk=risk,
                realized_utility=utility,
                dependence_group_id=dep,
                market_lineage_hash=f"{i + 1:064x}",
            ))
    return samples, parents


class TestIncrementalTeacher(unittest.TestCase):
    def test_exact_compile_one_matches_frozen_r6(self):
        samples, parents = _fixture()
        config = _config(mode="BLOCKED_CROSSFIT")
        base = DependenceAwareProbabilisticTeacherR6(config)
        opt = ExactIncrementalTeacherR6(config)
        index = base.index(samples)
        train_groups = {p.dependence_group_id for p in parents.values() if p.split == "TRAIN"}
        target = sorted(p.parent_id for p in parents.values() if p.split == "TRAIN")[4]

        base_calls = 0
        opt_calls = 0
        base_norm = base._normalization
        opt_norm = opt._normalization

        def counted_base(**kwargs):
            nonlocal base_calls
            base_calls += 1
            return base_norm(**kwargs)

        def counted_opt(**kwargs):
            nonlocal opt_calls
            opt_calls += 1
            return opt_norm(**kwargs)

        base._normalization = counted_base  # type: ignore[method-assign]
        opt._normalization = counted_opt  # type: ignore[method-assign]
        a = base.compile_one(
            target_parent=target,
            index=index,
            eligible_train_dependence_groups=train_groups,
        )
        b = opt.compile_one(
            target_parent=target,
            index=index,
            eligible_train_dependence_groups=train_groups,
        )
        self.assertEqual(a.content_hash, b.content_hash)
        self.assertEqual(a, b)
        self.assertEqual(base_calls, len(ACTIONS))
        self.assertEqual(opt_calls, 1)

    def test_compiled_authority_cold_then_reuse(self):
        samples, parents = _fixture()
        train_config = _config(mode="BLOCKED_CROSSFIT")
        val_config = _config(mode="PREQUENTIAL")
        source = {"parents_sha256": "1" * 64, "branches_sha256": "2" * 64}
        with tempfile.TemporaryDirectory() as td:
            first_train, first_val, first = compile_teacher_evidence_incremental(
                samples=samples,
                parents=parents,
                source_identity=source,
                cache_root=td,
                train_config=train_config,
                val_config=val_config,
                workers=1,
                threads_per_worker=1,
                max_in_flight=1,
            )
            second_train, second_val, second = compile_teacher_evidence_incremental(
                samples=samples,
                parents=parents,
                source_identity=source,
                cache_root=td,
                train_config=train_config,
                val_config=val_config,
                workers=8,
                threads_per_worker=1,
                max_in_flight=8,
            )
            self.assertEqual(first["mode"], "COMPILED_AND_PUBLISHED")
            self.assertEqual(second["mode"], "REUSED_VERIFIED_AUTHORITY")
            self.assertEqual(first["authority_hash"], second["authority_hash"])
            self.assertEqual(
                [x.content_hash for x in first_train],
                [x.content_hash for x in second_train],
            )
            self.assertEqual(
                [x.content_hash for x in first_val],
                [x.content_hash for x in second_val],
            )
            self.assertEqual(second["compile_seconds"], 0.0)

    def test_payload_corruption_fails_closed(self):
        samples, parents = _fixture()
        train_config = _config(mode="BLOCKED_CROSSFIT")
        val_config = _config(mode="PREQUENTIAL")
        source = {"parents_sha256": "3" * 64, "branches_sha256": "4" * 64}
        with tempfile.TemporaryDirectory() as td:
            _, _, receipt = compile_teacher_evidence_incremental(
                samples=samples,
                parents=parents,
                source_identity=source,
                cache_root=td,
                train_config=train_config,
                val_config=val_config,
                workers=1,
                threads_per_worker=1,
                max_in_flight=1,
            )
            path = Path(receipt["payload_path"])
            with path.open("ab") as f:
                f.write(b"CORRUPT")
            with self.assertRaisesRegex(RuntimeError, "PAYLOAD_HASH_MISMATCH"):
                compile_teacher_evidence_incremental(
                    samples=samples,
                    parents=parents,
                    source_identity=source,
                    cache_root=td,
                    train_config=train_config,
                    val_config=val_config,
                    workers=1,
                    threads_per_worker=1,
                    max_in_flight=1,
                )


if __name__ == "__main__":
    unittest.main()
