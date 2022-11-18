import numpy as np

from sim_src.edge_label.model.base_model import *
from sim_src.edge_label.nn import MPGNN
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import p_true


class dac_gnn_model(base_model):
    def __init__(self, id, edge_dim=1, node_dim=4, n_action=2):
        self.n_acton = n_action
        base_model.__init__(self,id,edge_dim,node_dim)

    def setup_actor(self):
        self.actor = MPGNN(in_node_channels=self.node_dim,hidden_node_channels=self.node_dim,out_node_channels=self.n_acton)
        self.actor_target = MPGNN(in_node_channels=self.node_dim,hidden_node_channels=self.node_dim,out_node_channels=self.n_acton)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.ACT_LR)
        hard_update_inplace(self.actor_target, self.actor)

    def setup_critic(self):
        self.critic = MPGNN(in_node_channels=self.node_dim+self.n_acton,hidden_node_channels=self.node_dim+self.n_acton,out_node_channels=1)
        self.critic_target = MPGNN(in_node_channels=self.node_dim+self.n_acton,hidden_node_channels=self.node_dim+self.n_acton,out_node_channels=1)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.CRI_LR)
        hard_update_inplace(self.critic_target, self.critic)

    def gen_action(self, state_np):
        n_node = state_np.shape[0]
        with torch.no_grad():
            G = nx.complete_graph(n_node)
            e_index = to_device(from_networkx(G).edge_index)
            state = to_tensor(state_np)

            action = self.actor_target.forward(state,e_index).sigmoid()

            action = to_numpy(action)

        if self.EXPLORATION:
            action = self.add_noise(action)

        return action

    def add_noise(self,action):
        if p_true(self.EXPLORATION_PROB):
            action += np.random.randn(action.shape[0],action.shape[1])*0.2
            action[action>1.] = 1.
            action[action<0.] = 0.
        return action

    def _train_actor(self,batch):
        self._print("_train_actor")
        loss = to_device(torch.zeros(1))
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            e_index = to_device(from_networkx(G).edge_index)
            with torch.no_grad():
                state = to_tensor(sample['state'],requires_grad=False)

            action = self.actor_target.forward(state,e_index).sigmoid()

            s_a = torch.hstack((state,action))
            self._printa("_train_actor states\n",to_numpy(state).T)
            self._printa("_train_actor sapair\n",to_numpy(s_a).T)

            q = self.critic_target.forward(s_a,e_index)
            # self._printa(q)
            q = self._fair_q(q)
            # self._printa(q)

            loss += (-torch.mean(q))

        loss/=len(batch)
        self._print("_train_actor loss",loss)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return to_numpy(loss)

    def _train_critic(self,batch):
        self._print("_train_critic")
        loss = to_device(torch.zeros(1))
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            e_index = to_device(from_networkx(G).edge_index)
            with torch.no_grad():
                action = to_tensor(sample['action'],requires_grad=False)
                state = to_tensor(sample['state'],requires_grad=False)

                s_a = torch.hstack((state,action))
                self._printa("_train_critic sapair\n",to_numpy(s_a).T)

            q = self.critic.forward(s_a,e_index)
            loss += nn.functional.mse_loss(q,to_tensor(sample['reward']))
            self._printa("_train_critic critic",to_numpy(q).T)
            self._printa("_train_critic reward",sample['reward'].T)

        loss/=len(batch)
        self._print("_train_critic loss",loss)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return to_numpy(loss)

class sim_env_dac_gnn(sim_env):
    def format_act_to_sta_twt_idx(self, action):
        action = np.copy(action)
        twt_id = np.zeros((self.pl_model.n_sta,1))
        action[action>0.5] = 1
        action[action<=0.5] = 0
        for i in range(self.twt_log2_n_slot):
            twt_id += np.reshape((2**(self.twt_log2_n_slot-i-1))*action[:,i],(-1,1))
        return twt_id.T