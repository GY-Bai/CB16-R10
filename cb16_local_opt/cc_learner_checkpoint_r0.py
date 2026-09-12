from __future__ import annotations
import base64, copy, pickle, torch
from .cc_learner_transaction_r0 import ExactlyOnceLedger


def save_checkpoint(learner, policy_rng_state:dict, learner_rng_state:object, identities:dict)->dict:
    return {"actor":copy.deepcopy(learner.actor.state_dict()),"critic":copy.deepcopy(learner.critic.state_dict()),
            "actor_opt":copy.deepcopy(learner.actor_opt.state_dict()),"critic_opt":copy.deepcopy(learner.critic_opt.state_dict()),
            "step":learner.step,"ledger":copy.deepcopy(learner.ledger),"policy_rng":copy.deepcopy(policy_rng_state),
            "learner_rng_b64":base64.b64encode(pickle.dumps(learner_rng_state,protocol=4)).decode(),"identities":copy.deepcopy(identities)}

def restore_checkpoint(learner,bundle:dict)->tuple[dict,object,dict]:
    learner.actor.load_state_dict(bundle["actor"]); learner.critic.load_state_dict(bundle["critic"])
    learner.actor_opt.load_state_dict(bundle["actor_opt"]); learner.critic_opt.load_state_dict(bundle["critic_opt"])
    learner.step=int(bundle["step"]); learner.ledger=copy.deepcopy(bundle["ledger"])
    return copy.deepcopy(bundle["policy_rng"]),pickle.loads(base64.b64decode(bundle["learner_rng_b64"])),copy.deepcopy(bundle["identities"])
