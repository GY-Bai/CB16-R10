import math
import pytest
from cb16_local_opt.cc_fast_wire_r0 import *
def _decision(**kw):
    base=dict(science_semantic_version=SCIENCE_SEMANTIC_VERSION,account_lineage_id="acct-1",decision_index=0,environment_time_ns=10,policy_generation=1,policy_id="p",policy_sha256="1"*64,observation_schema="obs-v1",observation_hash="2"*64,normalizer_id="n",nominal_direction="LONG",nominal_target_risk=0.4,log_mu=-1.2,risk_measure_kind="continuous_density",rng_stream_id="acct-1:g1",rng_counter=0); base.update(kw); return FastPolicyDecision(**base)
def test_policy_wire_compacts_and_validates():
    d=_decision().validate(); rec=d.to_record(); assert rec.dtype==POLICY_DECISION_DTYPE; assert int(rec["direction"])==LONG; assert math.isclose(float(rec["log_mu"]),-1.2)
@pytest.mark.parametrize("kw",[{"nominal_direction":"FLAT","nominal_target_risk":0.1,"risk_measure_kind":"point_mass"},{"nominal_direction":"LONG","nominal_target_risk":0.0},{"log_mu":float("nan")},{"policy_sha256":"bad"}])
def test_policy_wire_fail_closed(kw):
    with pytest.raises(ValueError): _decision(**kw).validate()
def test_transition_order_and_signed_economics():
    t=FastEnvironmentTransition(account_lineage_id="acct-1",decision_index=2,environment_time_before_ns=10,environment_time_after_ns=11,pre_account_truth_hash="3"*64,policy_decision_ref="4"*64,permission_status="ALLOW",permission_reason="ok",permitted_target_direction="SHORT",permitted_target_risk=0.4,target_quantity=-2.0,execution_legs=(ExecutionLeg(0,-1.0,100.0,0.1,"REVERSAL_CLOSE"),ExecutionLeg(1,-1.0,100.0,0.1,"REVERSAL_OPEN")),fees=0.2,funding=-0.01,realized_pnl=-5.0,unrealized_pnl_delta=-3.0,liability_delta=7.0,post_account_truth_hash="5"*64,post_equity=-12.0,boundary_type="ECONOMIC_TERMINAL",mechanical_terminal=True).validate(); assert len(t.content_sha256)==64; assert t.post_equity<0
def test_transition_rejects_reordered_legs():
    with pytest.raises(ValueError):
        FastEnvironmentTransition(account_lineage_id="a",decision_index=0,environment_time_before_ns=0,environment_time_after_ns=1,pre_account_truth_hash="1"*64,policy_decision_ref="2"*64,permission_status="ALLOW",permission_reason="ok",permitted_target_direction="LONG",permitted_target_risk=.2,target_quantity=1,execution_legs=(ExecutionLeg(1,1,100,0,"OPEN"),),fees=0,funding=0,realized_pnl=0,unrealized_pnl_delta=0,liability_delta=0,post_account_truth_hash="3"*64,post_equity=100,boundary_type="NONE",mechanical_terminal=False).validate()
