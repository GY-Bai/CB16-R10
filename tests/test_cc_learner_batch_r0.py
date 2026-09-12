from cb16_local_opt.cc_learner_batch_r0 import build_batch
def test_deterministic_order_and_no_survivor_filter():
 rec=[{'sequence_id':'b','step':0,'observation':[2.],'critic_observation':[2.],'action':1,'reward':-5.,'discount':0.,'log_mu':-1.,'mask':1.},{'sequence_id':'a','step':0,'observation':[1.],'critic_observation':[1.],'action':0,'reward':1.,'discount':0.,'log_mu':-1.,'mask':1.}]
 b=build_batch(rec); assert b.sequence_ids==('a','b') and b.rewards.tolist()==[1.,-5.]
