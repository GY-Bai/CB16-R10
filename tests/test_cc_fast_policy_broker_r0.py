import hashlib
import pytest
from cb16_local_opt.cc_fast_policy_broker_r0 import *
P1=PolicySpec(1,"p1","1"*64,(-0.2,-1.0,0.4)); P2=PolicySpec(2,"p2","2"*64,(0.2,-1.0,-0.4))
def req(acct,idx,gen=1,sha="1"*64): return PolicyRequest(account_lineage_id=acct,decision_index=idx,environment_time_ns=idx,policy_generation=gen,policy_sha256=sha,observation=(1.0,2.0),observation_hash=hashlib.sha256(f"{acct}:{idx}".encode()).hexdigest(),normalizer_id="n",rng_stream_id=f"{acct}:g{gen}",rng_counter=idx)
def test_response_has_joint_log_mu_and_rng_provenance():
    b=BatchedPolicyBroker([P1]); r=b.infer([req("a",0)])[0]; assert r.policy_sha256=="1"*64; assert r.rng_stream_id=="a:g1"; assert r.rng_counter==0; assert r.nominal_direction in {"SHORT","FLAT","LONG"}; assert r.log_mu==pytest.approx(r.log_mu); r.to_wire(req("a",0)).validate()
def test_partitioned_batching_by_generation():
    b=BatchedPolicyBroker([P1,P2]); out=b.infer([req("a",0),req("b",0,2,"2"*64),req("c",0)]); assert [x.policy_generation for x in out]==[1,2,1]; assert sorted(b.batch_sizes)==[1,2]
def test_stale_generation_fails_closed():
    b=BatchedPolicyBroker([P1])
    with pytest.raises(RuntimeError,match="STALE"): b.infer([req("a",0,2,"2"*64)])
def test_owner_pid_enforced(monkeypatch):
    b=BatchedPolicyBroker([P1],execution_mode="CUDA_SINGLE_OWNER"); monkeypatch.setattr("os.getpid",lambda:b.owner_pid+1)
    with pytest.raises(RuntimeError,match="OWNER_PROCESS"): b.infer([req("a",0)])
