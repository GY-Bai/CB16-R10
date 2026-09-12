import hashlib
from cb16_local_opt.cc_fast_policy_broker_r0 import *
P=PolicySpec(7,"p","7"*64,(0.1,-0.2,0.3))
def req(acct,idx): return PolicyRequest(account_lineage_id=acct,decision_index=idx,environment_time_ns=idx,policy_generation=7,policy_sha256="7"*64,observation=(1.0,),observation_hash=hashlib.sha256(f"{acct}:{idx}".encode()).hexdigest(),normalizer_id="n",rng_stream_id=f"stream:{acct}",rng_counter=idx)
def sig(x): return (x.nominal_direction,x.nominal_target_risk,x.log_mu,x.rng_stream_id,x.rng_counter)
def test_scheduler_arrival_order_cannot_change_sample():
    logical=[req("a",0),req("b",0),req("a",1),req("c",0)]; a=BatchedPolicyBroker([P]).infer(logical); perm=[2,0,3,1]; b_perm=BatchedPolicyBroker([P]).infer([logical[i] for i in perm]); by_key={(x.account_lineage_id,x.decision_index):sig(x) for x in b_perm}; assert [sig(x) for x in a]==[by_key[(x.account_lineage_id,x.decision_index)] for x in a]
