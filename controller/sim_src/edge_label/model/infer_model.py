import numpy as np
import torch

from sim_src.edge_label.model.base_model import *


class infer_model(base_model):
    def __init__(self, id, edge_dim=4, node_dim=4):
        base_model.__init__(self,id,edge_dim,node_dim)

    def gen_action(self, state_np):
        n_node = state_np.shape[0]
        return np.zeros((n_node,n_node))

    def setup_infer(self):
        self.infer = INFNN(edge_feature=self.node_dim*2,edge_type=2)
        self.infer_target = INFNN(edge_feature=self.node_dim*2,edge_type=2)
        self.infer_optim = optim.Adam(self.infer.parameters(), lr=self.INF_LR)
        hard_update_inplace(self.infer_target, self.infer)

    def _train_infer(self,batch):
        self._print("_train_infer")
        loss = to_device(torch.zeros(1))
        accu = 0.
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            e_index = to_device(from_networkx(G).edge_index)

            state = to_tensor(sample['state'],requires_grad=False)
            state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))

            target = to_tensor(sample['target'],requires_grad=False)
            target = target[e_index.transpose(0,1)[:,0],e_index.transpose(0,1)[:,1]]
            target = nn.functional.one_hot(target.long(), num_classes=2).float()

            result = self.infer.forward(state)

            self._printa("_train_infer diff\n",torch.hstack((target[:,1:2],result[:,1:2])).transpose(0,1))
            self._printa("_train_infer accu",torch.mean(torch.eq(target[:,1:2],torch.bernoulli(result[:,1:2])).float()))

            accu += to_numpy(torch.mean(torch.eq(target[:,1:2],torch.bernoulli(result[:,1:2])).float()))
            loss += nn.functional.cross_entropy(torch.log(result),target)

        loss/=len(batch)
        accu/=len(batch)
        self._print("_train_infer loss",loss)
        self.infer_optim.zero_grad()
        loss.backward()
        self.infer_optim.step()
        self._add_np_log("loss", np.reshape(np.array([to_numpy(loss)[0],accu]), (1, -1)))

        return to_numpy(loss)

    def _train_actor(self,batch):
       pass
    def _train_critic(self,batch):
       pass

    def update_nn(self):
       soft_update_inplace(self.infer_target, self.infer, self.TAU)
