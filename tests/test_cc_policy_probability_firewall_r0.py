from cb16_local_opt.cc_policy_probability_firewall_r0 import *
from test_cc_policy_wire_r0 import decision
def test_execution_cannot_replace_nominal_probability():
 d=decision(nominal_direction='LONG',nominal_target_risk=.7,log_mu=-2.)
 for executed in [('LONG',.2),('FLAT',0.),('SHORT',.9),(None,None)]: assert behavior_probability_input(d,*executed)==('LONG',.7,-2.)
 assert abs(importance_ratio(-1.,d)-2.718281828459045)<1e-12
