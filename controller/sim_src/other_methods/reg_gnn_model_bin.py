import numpy as np
import torch

from sim_src.edge_label.model.base_model import *
from sim_src.edge_label.nn import REGNN_BIN
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import p_true


class reg_gnn_model_bin(base_model):
    def __init__(self, id, edge_dim=1, node_dim=4, n_action=2):
        self.n_acton = n_action
        base_model.__init__(self,id,edge_dim,node_dim)

    def setup_actor(self):
        self.actor = REGNN_BIN(out_dim=self.n_acton)
        self.actor_target = REGNN_BIN(out_dim=self.n_acton)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.01)
        hard_update_inplace(self.actor_target, self.actor)

    def gen_action(self, state_np):
        n_node = state_np.shape[0]
        with torch.no_grad():
            G = nx.complete_graph(n_node)
            e_index = to_device(from_networkx(G).edge_index)
            state = to_tensor(state_np)

            min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

            x = min_path_loss

            state_B = state[e_index[1,:]]
            state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])

            H_mat = to_dense_adj(edge_index=e_index,batch=None,edge_attr=state_B_to_A).view(n_node,n_node)

            action = self.actor_target.forward(H_mat, x)
            action = to_numpy(action)
            action = np.clip(action, 1e-5, 1-1e-5)
            action = np.random.binomial(1,action)

        if self.EXPLORATION:
            action = self.add_noise(action)

        return action

    def add_noise(self,action):
        return action

    def _train_actor(self,batch):
        self._print("_train_actor")
        loss = to_device(torch.zeros(1))
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            e_index = to_device(from_networkx(G).edge_index)
            with torch.no_grad():
                state = to_tensor(sample['state'],requires_grad=False)
                min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

                x = min_path_loss

                state_B = state[e_index[1,:]]
                state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])

                H_mat = to_dense_adj(edge_index=e_index,batch=None,edge_attr=state_B_to_A).view(sample['n_node'],sample['n_node'])

                rwd = to_tensor(sample['reward'])
                # rwd = self._fair_q(rwd)
                rwd = torch.min(rwd)
                action = to_tensor(sample['action'],requires_grad=False)

            a = self.actor.forward(H_mat,x)
            e_pi = action*torch.log(a) + (1.-action)* torch.log(1.-a)
            e_pi = torch.sum(e_pi)

            loss += (-(e_pi*rwd))

        loss/=len(batch)
        self._print("_train_actor loss",loss)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return to_numpy(loss)

    def _train_critic(self,batch):
        pass

class sim_env_reg_gnn_bin(sim_env):
    def format_act_to_sta_twt_idx(self, action):
        action = np.copy(action)
        twt_id = np.zeros((self.pl_model.n_sta,1))
        for i in range(self.twt_log2_n_slot):
            twt_id += np.reshape((2**(self.twt_log2_n_slot-i-1))*action[:,i],(-1,1))
        return twt_id.T