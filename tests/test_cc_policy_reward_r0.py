from cb16_local_opt.cc_policy_reward_r0 import *
def test_signed_equity_and_telescoping():
 eq=[100.,90.,95.,-5.,-20.]; rs=path_rewards(eq,100.); assert abs(sum(rs)-((eq[-1]-eq[0])/100.))<1e-12 and rs[-1]<0
