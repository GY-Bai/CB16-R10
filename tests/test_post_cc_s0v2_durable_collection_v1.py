from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cb16_local_opt.cc_policy_distribution_r0 import sample_nominal
from cb16_local_opt.cc_runtime_decision_schedule_r0 import CCDecisionScheduleR0
from cb16_local_opt.cc_runtime_wire_r0 import CCPolicyDecisionV1
from cb16_local_opt.post_cc_durable_collection_v1 import (
    DurableCollectionError,
    DurableObservationCollectorV1,
    canonical_brain_vectors_from_account_v1,
)
from cb16_local_opt.post_cc_observation_store_v1 import ObservationStoreCorruption, ObservationStoreV1
from cb16_local_opt.post_cc_replay_materializer_v1 import ReplayStoreV1
from tests.cc_s0v2_support import (
    DEMO_LINEAGE_ID,
    DEMO_MARKET_SOURCE_IDENTITY,
    DEMO_MARKET_SOURCE_VERSION,
    DEMO_POLICY_GENERATION,
    DEMO_POLICY_ID,
    DEMO_POLICY_SHA256,
    SCIENCE_SEMANTIC_VERSION,
    make_account,
    make_brain,
    make_interval,
    make_policy_rng,
    make_runtime,
)


def _collector(root: Path) -> DurableObservationCollectorV1:
    return DurableObservationCollectorV1(
        str(root),
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        market_source_identity=DEMO_MARKET_SOURCE_IDENTITY,
        market_source_version=DEMO_MARKET_SOURCE_VERSION,
    )


def _capture_callback(collector: DurableObservationCollectorV1, brain, rng, runtime, box):
    def callback(account_state, clocks):
        market, account_values, execution_values = canonical_brain_vectors_from_account_v1(account_state)
        logits, risk_loc, risk_log_scale = brain(
            torch.tensor(market), torch.tensor(account_values), torch.tensor(execution_values)
        )
        nominal = sample_nominal(logits, risk_loc, risk_log_scale, rng)
        decision, fact = collector.capture_policy_decision(
            account_state,
            account_lineage_id=runtime.account_lineage_id,
            decision_index=clocks.policy_decision_index,
            environment_time=clocks.environment_time,
            policy_generation=DEMO_POLICY_GENERATION,
            policy_id=DEMO_POLICY_ID,
            policy_sha256=DEMO_POLICY_SHA256,
            nominal=nominal,
        )
        box["decision"] = decision
        box["fact"] = fact
        return decision

    return callback


def test_observation_is_persisted_before_transition_link_and_hash_matches(tmp_path: Path):
    runtime = make_runtime(account=make_account())
    collector = _collector(tmp_path)
    box: dict = {}
    callback = _capture_callback(collector, make_brain(), make_policy_rng(), runtime, box)
    transition = runtime.step(make_interval(101.0), callback, expected_predecessor_token=runtime.predecessor_token)
    assert box["decision"].observation_hash == box["fact"].observation_hash
    assert ObservationStoreV1(tmp_path).count() == 1
    record = collector.persist_transition(
        sequence_id="seq-1",
        transition=transition,
        decision=box["decision"],
        reward=0.001,
        discount=0.99,
    )
    assert record.nominal_direction == box["decision"].nominal_direction
    assert record.behavior_log_mu == box["decision"].log_mu
    assert record.observation_hash == box["fact"].observation_hash
    assert ReplayStoreV1(tmp_path).count_transitions() == 1


def test_transition_requires_persisted_observation_in_order(tmp_path: Path):
    runtime = make_runtime(account=make_account())
    collector = _collector(tmp_path)
    box: dict = {}

    def callback(account_state, clocks):
        market, account_values, execution_values = canonical_brain_vectors_from_account_v1(account_state)
        brain = make_brain()
        rng = make_policy_rng()
        logits, risk_loc, risk_log_scale = brain(
            torch.tensor(market), torch.tensor(account_values), torch.tensor(execution_values)
        )
        nominal = sample_nominal(logits, risk_loc, risk_log_scale, rng)
        decision = CCPolicyDecisionV1(
            science_semantic_version=SCIENCE_SEMANTIC_VERSION,
            account_lineage_id=runtime.account_lineage_id,
            decision_index=clocks.policy_decision_index,
            environment_time=clocks.environment_time,
            policy_generation=DEMO_POLICY_GENERATION,
            policy_id=DEMO_POLICY_ID,
            policy_sha256=DEMO_POLICY_SHA256,
            observation_schema="OBS",
            observation_hash="f" * 64,
            normalizer_id="NORM",
            nominal_direction=nominal.direction,
            nominal_target_risk=nominal.target_risk,
            log_mu=nominal.log_prob,
            risk_measure_kind=nominal.risk_measure_kind,
            rng_stream_id=nominal.rng_stream_id,
            rng_position_or_counter=nominal.rng_counter,
        )
        decision.validate()
        box["decision"] = decision
        return decision

    transition = runtime.step(make_interval(101.0), callback, expected_predecessor_token=runtime.predecessor_token)
    with pytest.raises(DurableCollectionError, match="COLLECTION_OBSERVATION_NOT_PERSISTED"):
        collector.persist_transition(
            sequence_id="seq-1", transition=transition, decision=box["decision"], reward=0.0, discount=0.99
        )


def test_observation_hash_mismatch_after_capture_fails_closed(tmp_path: Path):
    runtime = make_runtime(account=make_account())
    collector = _collector(tmp_path)
    box: dict = {}
    callback = _capture_callback(collector, make_brain(), make_policy_rng(), runtime, box)
    good_decision = callback(runtime.account, runtime.clocks)
    corrupted = replace(good_decision, observation_hash="f" * 64)
    corrupted.validate()
    transition = runtime.step(make_interval(101.0), lambda *_: corrupted, expected_predecessor_token=runtime.predecessor_token)
    with pytest.raises(DurableCollectionError, match="DURABLE_OBSERVATION_HASH_MISMATCH"):
        collector.persist_transition(
            sequence_id="seq-1", transition=transition, decision=corrupted, reward=0.0, discount=0.99
        )


def test_mechanical_no_decision_advance_is_raw_only(tmp_path: Path):
    runtime = make_runtime(account=make_account())
    runtime.schedule = CCDecisionScheduleR0(2)
    collector = _collector(tmp_path)
    box: dict = {}
    callback = _capture_callback(collector, make_brain(), make_policy_rng(), runtime, box)
    decision_transition = runtime.step(make_interval(101.0), callback, expected_predecessor_token=runtime.predecessor_token)
    collector.persist_transition(
        sequence_id="seq-1", transition=decision_transition, decision=box["decision"], reward=0.0, discount=0.99
    )
    observations_before = ObservationStoreV1(tmp_path).count()
    no_decision = runtime.step(make_interval(99.0), None, expected_predecessor_token=runtime.predecessor_token)
    assert no_decision.policy_decision_ref is None
    digest = collector.persist_no_decision_advance(no_decision)
    assert len(digest) == 64
    assert ObservationStoreV1(tmp_path).count() == observations_before
    assert ReplayStoreV1(tmp_path).count_raw_advances() == 1
    with pytest.raises(DurableCollectionError, match="MECHANICAL_ADVANCE_MUST_NOT_BE_REPLAY_TRANSITION"):
        collector.persist_transition(
            sequence_id="seq-1", transition=no_decision, decision=box["decision"], reward=0.0, discount=0.99
        )


def test_observation_corruption_after_capture_fails_closed(tmp_path: Path):
    runtime = make_runtime(account=make_account())
    collector = _collector(tmp_path)
    box: dict = {}
    callback = _capture_callback(collector, make_brain(), make_policy_rng(), runtime, box)
    transition = runtime.step(make_interval(101.0), callback, expected_predecessor_token=runtime.predecessor_token)
    object_files = list((tmp_path / "observations" / "objects").rglob("*.bin"))
    object_files[0].write_bytes(b'{"corrupted":true}')
    with pytest.raises(ObservationStoreCorruption, match="CONTENT_OBJECT_(SIZE|HASH)_MISMATCH"):
        collector.persist_transition(
            sequence_id="seq-1", transition=transition, decision=box["decision"], reward=0.0, discount=0.99
        )
