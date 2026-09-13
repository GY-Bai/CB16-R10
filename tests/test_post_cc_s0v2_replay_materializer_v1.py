from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import gc
import tempfile

import pytest

from cb16_local_opt.cc_runtime_boundary_r0 import COMPUTE_CHUNK
from cb16_local_opt.post_cc_joint_replay_v1 import DurableSequenceRecordV1
from cb16_local_opt.post_cc_observation_store_v1 import ObservationStoreCorruption
from cb16_local_opt.post_cc_replay_materializer_v1 import (
    ReplayCorruption,
    ReplayMaterializerV1,
    ReplayStoreV1,
)
from tests.cc_s0v2_support import (
    collect_durable_sequence_v1,
    materialize_v1,
)


def test_restart_sentinel_reconstructs_identical_replay_from_durable_state():
    with tempfile.TemporaryDirectory(prefix="cc-s0v2-restart-") as td:
        collected = collect_durable_sequence_v1(td)
        sequence_id = collected.sequence_id
        phase_a = materialize_v1(td, sequence_id, restart_verified=False)
        expected_id = phase_a.manifest.stable_materialization_id
        expected_sample_hashes = phase_a.manifest.ordered_sample_hashes
        ReplayStoreV1(td).put_manifest(phase_a.manifest)
        del collected, phase_a
        gc.collect()
        phase_b = materialize_v1(
            td,
            sequence_id,
            restart_verified=True,
            expected_materialization_id=expected_id,
        )
        assert phase_b.manifest.ordered_sample_hashes == expected_sample_hashes
        assert phase_b.manifest.stable_materialization_id == expected_id
        persisted = ReplayStoreV1(td).get_manifest(expected_id)
        assert persisted.ordered_sample_hashes == expected_sample_hashes


def test_restart_manifest_mismatch_fails_closed():
    with tempfile.TemporaryDirectory(prefix="cc-s0v2-restart-mismatch-") as td:
        collected = collect_durable_sequence_v1(td)
        with pytest.raises(ReplayCorruption, match="RESTART_MANIFEST_MISMATCH"):
            materialize_v1(
                td,
                collected.sequence_id,
                restart_verified=True,
                expected_materialization_id="0" * 64,
            )


def test_tampered_transition_object_fails_closed():
    with tempfile.TemporaryDirectory(prefix="cc-s0v2-transition-corrupt-") as td:
        collected = collect_durable_sequence_v1(td)
        store = ReplayStoreV1(td)
        digest = store._transitions.content_sha256_for(collected.transition_ids[0])
        path = Path(td) / "replay_transitions" / "objects" / digest[:2] / f"{digest}.bin"
        assert path.is_file()
        path.write_bytes(b'{"transition_id":"tampered"}')
        with pytest.raises(ReplayCorruption):
            ReplayStoreV1(td).get_transition(collected.transition_ids[0])


def test_tampered_observation_object_fails_closed_during_materialization():
    with tempfile.TemporaryDirectory(prefix="cc-s0v2-observation-corrupt-") as td:
        collected = collect_durable_sequence_v1(td)
        files = list((Path(td) / "observations" / "objects").rglob("*.bin"))
        assert files
        files[0].write_bytes(b'{"corrupted":true}')
        with pytest.raises(ObservationStoreCorruption, match="CONTENT_OBJECT_(SIZE|HASH)_MISMATCH"):
            materialize_v1(td, collected.sequence_id, restart_verified=True)


def test_truncation_sequence_without_durable_bootstrap_reference_fails_closed():
    with tempfile.TemporaryDirectory(prefix="cc-s0v2-truncation-") as td:
        collected = collect_durable_sequence_v1(td)
        store = ReplayStoreV1(td)
        first = store.get_transition(collected.transition_ids[0])
        modified = replace(
            first,
            transition_id="truncation-transition-without-bootstrap",
            boundary_type=COMPUTE_CHUNK,
            bootstrap_state_ref_or_null=None,
        )
        modified.validate()
        store.put_transition(modified)
        sequence = DurableSequenceRecordV1(
            sequence_id="truncation-sequence-without-bootstrap",
            account_lineage_id=modified.account_lineage_id,
            science_semantic_version=modified.science_semantic_version,
            market_lineage_id="SYN",
            source_classification="CC_STOCHASTIC_TRAJECTORY",
            transition_ids=(modified.transition_id,),
            first_decision_index=modified.decision_index,
            last_decision_index=modified.decision_index,
            behavior_policy_identities=(modified.policy_id,),
            normalizer_identities=(modified.normalizer_id,),
            chunk_boundary_type=COMPUTE_CHUNK,
            bootstrap_state_ref_or_null=None,
            raw_fact_content_sha256=modified.content_sha256,
        ).validate()
        store.put_sequence(sequence)
        with pytest.raises(ReplayCorruption, match="TRUNCATION_SEQUENCE_BOOTSTRAP_REF_MISSING"):
            materialize_v1(td, sequence.sequence_id, restart_verified=True)


def test_account_truth_discontinuity_fails_closed():
    with tempfile.TemporaryDirectory(prefix="cc-s0v2-account-break-") as td:
        collected = collect_durable_sequence_v1(td)
        store = ReplayStoreV1(td)
        sequence_id = "broken-account-sequence"
        first = store.get_transition(collected.transition_ids[0])
        adjacent = store.get_transition(collected.transition_ids[1])
        first_copy = replace(first, sequence_id=sequence_id, transition_id="broken-first-transition")
        broken_final = replace(
            adjacent,
            sequence_id=sequence_id,
            transition_id="broken-final-transition",
            pre_account_truth_hash="9" * 64,
            boundary_type="OBJECTIVE_HORIZON_REACHED",
            bootstrap_state_ref_or_null=None,
        )
        first_copy.validate()
        broken_final.validate()
        store.put_transition(first_copy)
        store.put_transition(broken_final)
        sequence = DurableSequenceRecordV1(
            sequence_id=sequence_id,
            account_lineage_id=first.account_lineage_id,
            science_semantic_version=first.science_semantic_version,
            market_lineage_id="SYN",
            source_classification="CC_STOCHASTIC_TRAJECTORY",
            transition_ids=(first_copy.transition_id, broken_final.transition_id),
            first_decision_index=first_copy.decision_index,
            last_decision_index=broken_final.decision_index,
            behavior_policy_identities=tuple(dict.fromkeys((first_copy.policy_id, broken_final.policy_id))),
            normalizer_identities=(first_copy.normalizer_id,),
            chunk_boundary_type=broken_final.boundary_type,
            bootstrap_state_ref_or_null=None,
            raw_fact_content_sha256="1" * 64,
        ).validate()
        store.put_sequence(sequence)
        with pytest.raises(ReplayCorruption, match="TRANSITION_ACCOUNT_TRUTH_DISCONTINUITY"):
            materialize_v1(td, sequence.sequence_id, restart_verified=True)


def test_materialized_behavior_log_mu_is_the_persisted_decision_value():
    with tempfile.TemporaryDirectory(prefix="cc-s0v2-logmu-") as td:
        collected = collect_durable_sequence_v1(td)
        materialized = materialize_v1(td, collected.sequence_id, restart_verified=True)
        store = ReplayStoreV1(td)
        for sample in materialized.samples:
            record = store.get_transition(sample.transition_id)
            assert sample.behavior_log_mu == record.behavior_log_mu
            assert sample.log_mu_source == "DECISION_TIME_PERSISTED"
            assert sample.consequence_context is not None
            # execution consequence context must not be able to replace nominal log_mu
            with pytest.raises(ValueError, match="LOG_MU_NOT_DECISION_TIME_PERSISTED"):
                replace(sample, log_mu_source="RECONSTRUCTED_FROM_EXECUTION").validate()
