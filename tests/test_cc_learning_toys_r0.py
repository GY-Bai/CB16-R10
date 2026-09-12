from cb16_local_opt.cc_learning_toys_r0 import *
def test_account_dependent_same_market(): assert account_dependent_optimal_action(1.,0.,0.)=='LONG' and account_dependent_optimal_action(1.,3.,0.)=='SHORT'
def test_delayed_and_horizon_reversal():
 c=delayed_consequence_case(); assert c.short_horizon()>0 and c.long_horizon()<0; h=horizon_reversal(); assert h['action_x_h1']>h['action_y_h1'] and h['action_x_h3']<h['action_y_h3']
 learned=delayed_long_horizon_learning(); assert learned['short_horizon']['preferred']=='A' and learned['long_horizon']['preferred']=='B'
def test_high_bankruptcy_arithmetic_expectation_wins():
 x=high_bankruptcy_learning(); assert x['risky_bankruptcy_frequency']>x['safe_bankruptcy_frequency'] and x['risky_expectation']>x['safe_expectation'] and x['learned']['preferred']=='A'
def test_offpolicy_known_answer_and_controls():
 x=offpolicy_known_answer(.2,.8,1.,0.,0.,0.); assert x['rho']==4 and x['clipped_rho']==1 and x['vs']==1
 c=offpolicy_comparison(); assert abs(c['ratio']-.125)<1e-12 and c['correct_vtrace']!=c['uncorrected_replay'] and abs(c['same_policy_reduction']-c['uncorrected_replay'])<1e-12
