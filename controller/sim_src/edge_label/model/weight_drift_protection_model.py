import networkx as nx
import torch
from torch_geometric.utils import from_networkx

from sim_src.edge_label.model.base_model import base_model
from sim_src.util import to_tensor, to_numpy


class weight_drift_protection(base_model):
    def _train_actor(self,batch):
        self._print("_train_actor")
        loss = torch.zeros(1)
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            e_index:torch.Tensor = from_networkx(G).edge_index
            with torch.no_grad():
                state = to_tensor(sample['state'],requires_grad=False)
                state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))

                actor_input = state

                x = to_tensor(sample['state'],requires_grad=False)

                label = self.actor.forward(actor_input)

            label_no_grad = label.clone()
            label.requires_grad_()

            s_a = torch.hstack((state,label))
            self._printa("_train_actor states\n",to_numpy(state).T)
            self._printa("_train_actor sapair\n",to_numpy(s_a).T)

            q = self.critic_target.forward(x,e_index,label)
            q = -torch.mean(q)
            q.backward()
            lable_g = label.grad.detach()
            scale = label_no_grad
            scale[lable_g>0.] = 1. - scale[lable_g>0.]
            label_differentiable = self.actor.forward(actor_input)
            self._printa("_train_actor label.grad\n",to_numpy(label.grad).T)
            self._printa("_train_actor lable_g\n",to_numpy(lable_g).T)
            self._printa("_train_actor scale\n",to_numpy(scale).T)

            loss += (-torch.mean((lable_g*scale)*label_differentiable))

        loss/=len(batch)
        self._print("_train_actor loss",loss)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return to_numpy(loss)