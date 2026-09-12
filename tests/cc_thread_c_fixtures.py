from cb16_local_opt.cc_experience_wire_r0 import CCEnvironmentTransitionV1
from cb16_local_opt.cc_experience_transition_r0 import ImmutableExperienceTransitionV1

def env(i=0, *, lineage='acct', pre=None, post=None, terminal=False, boundary='NONE', equity=100.0, legs=()):
    return CCEnvironmentTransitionV1(
        account_lineage_id=lineage, decision_index=i,
        environment_time_before=f'2026-01-01T00:{i:02d}:00Z', environment_time_after=f'2026-01-01T00:{i+1:02d}:00Z',
        pre_account_truth_hash=pre or f'h{i}', policy_decision_ref=f'd{i}', permission_status='ALLOW', permission_reason='OK',
        permitted_target_direction='LONG', permitted_target_risk=.5, target_quantity=1.0, execution_legs=tuple(legs),
        fees=.1, funding=0.0, realized_pnl=0.0, unrealized_pnl_delta=0.0, liability_delta=0.0,
        post_account_truth_hash=post or f'h{i+1}', post_equity=equity, boundary_type=boundary, mechanical_terminal=terminal,
        external_capital_flow_ref_or_null=None).validate()

def tx(i=0, *, lineage='acct', generation='G3', pre=None, post=None, failure='NONE', terminal=False, boundary='NONE', equity=100.0):
    return ImmutableExperienceTransitionV1(
        transition_id=f't-{lineage}-{i}', environment=env(i,lineage=lineage,pre=pre,post=post,terminal=terminal,boundary=boundary,equity=equity),
        science_semantic_version='R1_CC_ROUND2', policy_generation=generation, policy_id=f'p-{generation}', policy_sha256='a'*64,
        observation_schema='obs-v1', observation_hash='b'*64, normalizer_id='norm-v1', nominal_direction='LONG', nominal_target_risk=.5,
        log_mu=-.7, risk_measure_kind='continuous_density', rng_stream_id='rng-1', rng_position_or_counter=str(i),
        market_lineage_id='market-frozen', source_classification='CC_STOCHASTIC_TRAJECTORY', failure_classification=failure).validate()
