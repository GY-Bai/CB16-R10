from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cb16_local_opt.cc_runtime_boundary_r0 import COMPUTE_CHUNK, ECONOMIC_TERMINAL
from cb16_local_opt.post_cc_joint_batch_v1 import (
    JointActionBatchV1,
    build_joint_batch_v1,
)
from tests.cc_s0v2_support import make_batch_v1, make_canonical_observation_v1, make_joint_sample_v1


def test_valid_batch_exposes_canonical_nominal_action_and_provenance(tmp_path: Path):
    first = make_joint_sample_v1(
        root=tmp_path, decision_index=0, environment_time=0, nominal_direction="LONG", boundary_type="CONTINUE"
    )
    second = make_joint_sample_v1(
        root=tmp_path,
        transition_id="cc-s0v2-unit-transition-1",
        decision_index=1,
        environment_time=1,
        nominal_direction="FLAT",
        nominal_target_risk=0.0,
        boundary_type=ECONOMIC_TERMINAL,
    )
    batch = make_batch_v1((first, second))
    assert batch.direction_indices.tolist() == [2, 1]
    assert batch.target_risks.tolist() == pytest.approx([0.4, 0.0])
    assert batch.risk_point_mass_mask.tolist() == [False, True]
    assert batch.behavior_log_mu.tolist() == pytest.approx([-1.25, -1.25])
    assert batch.provenance.durable_replay_only is True
    assert batch.provenance.collector_private_records_used is False


@pytest.mark.parametrize(
    "changes,error",
    (
        ({"nominal_direction": "FLAT", "nominal_target_risk": 0.4, "risk_measure_kind": "point_mass"}, "FLAT must be exact risk 0"),
        ({"nominal_direction": "FLAT", "nominal_target_risk": 0.0, "risk_measure_kind": "continuous_density"}, "FLAT must be exact risk 0"),
        ({"nominal_target_risk": 1.0}, "non-FLAT risk must be interior"),
        ({"nominal_target_risk": 0.0}, "non-FLAT risk must be interior"),
        ({"behavior_log_mu": float("nan")}, "must be a finite number"),
        ({"behavior_policy_id": ""}, "behavior_policy_id must be non-empty text"),
        ({"log_mu_source": "RECONSTRUCTED"}, "LOG_MU_NOT_DECISION_TIME_PERSISTED"),
        ({"sampling_probability_or_weight": 1.5}, "sampling_probability_or_weight must be in"),
        ({"source_fact_hashes": ()}, "SOURCE_FACT_HASHES_REQUIRED"),
    ),
)
def test_sample_level_invalid_combinations_fail_closed(tmp_path: Path, changes, error):
    sample = make_joint_sample_v1(root=tmp_path, nominal_direction="LONG")
    with pytest.raises(ValueError, match=error):
        replace(sample, **changes).validate()


def test_observation_hash_mismatch_on_batch_fails_closed(tmp_path: Path):
    sample = make_joint_sample_v1(root=tmp_path)
    batch = make_batch_v1((sample,))
    corrupted = replace(batch, observation_hashes=("0" * 64,))
    with pytest.raises(ValueError, match="OBSERVATION_HASH_MISMATCH"):
        corrupted.validate()


def test_time_order_violation_within_account_lineage_fails_closed(tmp_path: Path):
    first = make_joint_sample_v1(root=tmp_path, decision_index=0, environment_time=0)
    second = make_joint_sample_v1(
        root=tmp_path,
        transition_id="cc-s0v2-unit-transition-2",
        decision_index=1,
        environment_time=1,
        boundary_type=ECONOMIC_TERMINAL,
    )
    with pytest.raises(ValueError, match="TIME_ORDER_VIOLATION_WITHIN_ACCOUNT_LINEAGE"):
        make_batch_v1((second, first))


def test_direction_index_mapping_inversion_fails_closed(tmp_path: Path):
    sample = make_joint_sample_v1(root=tmp_path, nominal_direction="LONG")
    batch = make_batch_v1((sample,))
    corrupted = replace(batch, direction_indices=__import__("torch").tensor([0], dtype=__import__("torch").long))
    with pytest.raises(ValueError, match="DIRECTION_INDEX_MAPPING_MISMATCH"):
        corrupted.validate()


def test_truncation_requires_durable_bootstrap_observation(tmp_path: Path):
    with pytest.raises(ValueError, match="BOOTSTRAP_OBSERVATION_REQUIRED"):
        make_batch_v1((make_joint_sample_v1(root=tmp_path, boundary_type=COMPUTE_CHUNK),))
    sample = make_joint_sample_v1(
        root=tmp_path,
        boundary_type=COMPUTE_CHUNK,
        bootstrap_state_ref_or_null="b" * 64,
    )
    with pytest.raises(ValueError, match="BOOTSTRAP_OBSERVATION_REQUIRED"):
        make_batch_v1((sample,))
    from tests.cc_s0v2_support import make_canonical_observation_v1

    bootstrap_fact = make_canonical_observation_v1(decision_index=1, environment_time=1)
    batch = make_batch_v1((sample,), bootstrap_observations_by_sequence={sample.sequence_id: bootstrap_fact})
    assert batch.bootstrap_observations is not None
    assert batch.bootstrap_sequence_indices == (0,)
    unexpected = make_joint_sample_v1(
        root=tmp_path,
        transition_id="cc-s0v2-unit-transition-3",
        decision_index=1,
        environment_time=1,
        boundary_type=ECONOMIC_TERMINAL,
    )
    with pytest.raises(ValueError, match="UNEXPECTED_BOOTSTRAP_OBSERVATION"):
        make_batch_v1(
            (unexpected,),
            bootstrap_observations_by_sequence={unexpected.sequence_id: bootstrap_fact},
        )
