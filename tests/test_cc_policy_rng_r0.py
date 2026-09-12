from cb16_local_opt.cc_policy_rng_r0 import PolicyRNG
def test_rng_restore_exact_next_sequence():
 r=PolicyRNG('p','a',3); _=[r.random() for _ in range(5)]; s=r.state_dict(); expect=[r.random() for _ in range(5)]; rr=PolicyRNG.from_state_dict(s); assert [rr.random() for _ in range(5)]==expect
def test_account_streams_order_independent():
 a=PolicyRNG('p','A',7); b=PolicyRNG('p','B',7); x1=(a.random(),b.random(),a.random(),b.random())
 aa=PolicyRNG('p','A',7); bb=PolicyRNG('p','B',7); x2=(aa.random(),bb.random(),aa.random(),bb.random()); assert x1==x2 and a.stream_id!=b.stream_id
