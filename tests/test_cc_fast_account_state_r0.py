from cb16_local_opt.cc_fast_account_state_r0 import *
def test_soa_is_contiguous_and_signed():
    s=AccountStateSoA.allocate(["a","b"],initial_equity=100); s.liability[0]=-7.0; s.equity[0]=-3.0; s.validate(); assert s.count==2; assert s.nbytes>0; assert s.liability[0]==-7; assert s.equity[0]==-3; assert all(getattr(s,n).flags.c_contiguous for n in ("quantity","cash","equity","liability"))
