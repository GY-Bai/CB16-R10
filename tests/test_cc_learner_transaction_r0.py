import copy, pytest
from cb16_local_opt.cc_learner_transaction_r0 import ExactlyOnceLedger
def test_exactly_once_commit_and_rebind_fail():
 l=ExactlyOnceLedger(); l.prepare('u','p',('s',),0); assert l.commit('u','c',1) and not l.commit('u','c',1)
 assert l.prepare('u','p',('s',),0)['status']=='COMMITTED'
 x=ExactlyOnceLedger(); x.prepare('u','p',('s',),0)
 with pytest.raises(RuntimeError): x.prepare('u','DIFF',('s',),0)
def test_crash_boundaries_exactly_once():
 # before gradient: no transaction exists
 l=ExactlyOnceLedger(); assert not l.prepared and not l.committed
 # after gradient calculation / before durable commit: PREPARED survives recovery and commits once
 l.prepare('u','parent',('s1','s2'),5); recovered=copy.deepcopy(l); assert 'u' in recovered.prepared and recovered.commit('u','child',6)
 # after durable commit / before acknowledgement: retry is a no-op, not a second optimizer commit
 crash_after_commit=copy.deepcopy(recovered); assert not crash_after_commit.commit('u','child',6) and crash_after_commit.committed['u']['step_after']==6
