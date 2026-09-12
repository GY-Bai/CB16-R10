import torch
from cb16_local_opt.cc_policy_loss_r0 import actor_policy_gradient_loss
def test_policy_gradient_sign():
 x=torch.tensor(0.,requires_grad=True); lp=torch.log(torch.sigmoid(x)); loss=actor_policy_gradient_loss(lp.reshape(1),torch.tensor([1.])); loss.backward(); assert x.grad<0
 x2=torch.tensor(0.,requires_grad=True); lp2=torch.log(torch.sigmoid(x2)); loss2=actor_policy_gradient_loss(lp2.reshape(1),torch.tensor([-1.])); loss2.backward(); assert x2.grad>0
