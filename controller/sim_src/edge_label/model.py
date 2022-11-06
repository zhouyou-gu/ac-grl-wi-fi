import os.path

import networkx as nx
import numpy as np
import torch
from torch import optim, nn
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_dense_adj

from sim_src.edge_label.nn import INFNN, ELNN, WGNN
from sim_src.util import USE_CUDA, hard_update_inplace, counted, StatusObject, soft_update_inplace, to_numpy, to_tensor


class learning_model(StatusObject):
    def step(self, batch):
        pass

    def gen_action(self, state):
        pass

class gnn_edge_label(learning_model):
    INF_LR = 0.001
    ACT_LR = 0.001
    CRI_LR = 0.001
    TAU = 0.01
    EQUALIZATION_FACTOR = 0.1
    FAIRNESS_ALPHA = 10
    def __init__(self, id, edge_dim=4, node_dim=4):
        self.id = id
        self.edge_dim = edge_dim # loss sta-y to ap-x, loss sta-x to ap-y, Pr sta-x sta-y contending
        self.node_dim = node_dim # loss sta-x to ap-1..., loss sta-y to ap-1...

        self.infer = None
        self.infer_optim = None

        self.actor = None
        self.actor_optim = None

        self.critic = None
        self.critic_optim = None

        self.infer_target = None
        self.actor_target = None
        self.critic_target = None

        self.init_model()


        if USE_CUDA:
            self.move_nn_to_gpu()

    def move_nn_to_gpu(self):
        self.infer.to(torch.cuda.current_device())
        self.infer_target.to(torch.cuda.current_device())
        self.actor.to(torch.cuda.current_device())
        self.actor_target.to(torch.cuda.current_device())
        self.critic.to(torch.cuda.current_device())
        self.critic_target.to(torch.cuda.current_device())

    def setup_infer(self):
        self.infer = INFNN(edge_feature=self.node_dim*2,edge_type=2)
        self.infer_target = INFNN(edge_feature=self.node_dim*2,edge_type=2)
        self.infer_optim = optim.Adam(self.infer.parameters(), lr=self.INF_LR)
        hard_update_inplace(self.infer_target, self.infer)

    def setup_actor(self):
        self.actor = ELNN(in_edge_channels=self.edge_dim+2, out_edge_channels=1)
        self.actor_target = ELNN(in_edge_channels=self.edge_dim+2, out_edge_channels=1)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.ACT_LR)
        hard_update_inplace(self.actor_target, self.actor)

    def setup_critic(self):
        self.critic = WGNN(in_node_channels=1,hidden_node_channels=5,out_node_channels=1,in_edge_channels=self.edge_dim+1,hidden_edge_channels=5)
        self.critic_target = WGNN(in_node_channels=1,hidden_node_channels=5,out_node_channels=1,in_edge_channels=self.edge_dim+1,hidden_edge_channels=5)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.CRI_LR)
        hard_update_inplace(self.critic_target, self.critic)

    def init_model(self):
        self.setup_infer()
        self.setup_actor()
        self.setup_critic()

    def load_infer(self, path, path_target = None):
        if not path_target:
            path_target = path
        self.infer = self._load(path)
        self.actor_target = self._load(path_target)

    def load_actor(self, path, path_target = None):
        if not path_target:
            path_target = path
        self.actor = self._load(path)
        self.actor_target = self._load(path_target)

    def load_critic(self, path, path_target = None):
        if not path_target:
            path_target = path
        self.critic = self._load(path)
        self.critic_target = self._load(path_target)

    def _load(self, path):
        if USE_CUDA:
            return torch.load(path, map_location=torch.cuda.current_device())
        else:
            return torch.load(path, map_location=torch.device('cpu'))

    def update_nn(self):
        soft_update_inplace(self.infer_target, self.infer, self.TAU)
        soft_update_inplace(self.actor_target, self.actor, self.TAU)
        soft_update_inplace(self.critic_target, self.critic, self.TAU)

    def save(self, path: str, postfix: str):
        try:
            os.mkdir(path)
        except:
            pass
        torch.save(self.infer, os.path.join(path,"infer." + postfix + ".pt"))
        torch.save(self.infer_target, os.path.join(path,"infer_target." + postfix + ".pt"))

        torch.save(self.actor, os.path.join(path,"actor." + postfix + ".pt"))
        torch.save(self.actor_target, os.path.join(path,"actor_target." + postfix + ".pt"))

        torch.save(self.critic, os.path.join(path,"critic." + postfix + ".pt"))
        torch.save(self.critic_target, os.path.join(path,"critic_target." + postfix + ".pt"))

    def gen_action(self, state_np):
        n_node = state_np.shape[0]
        with torch.no_grad():
            G = nx.complete_graph(n_node)
            e_index:torch.Tensor = from_networkx(G).edge_index
            state = to_tensor(state_np)
            state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))

            result = self.infer_target.forward(state)
            p_contention = result[:,1:2]

            state = to_tensor(state_np,requires_grad=False)
            min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

            state_A = state[e_index[0,:]]
            state_A_to_B = torch.gather(state_A,1,min_path_loss_idx[e_index[1,:]])
            state_B = state[e_index[1,:]]
            state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])
            interference = torch.hstack((state_A_to_B,state_B_to_A))
            interference_and_min_pl_and_p_contention = torch.hstack((interference,min_path_loss[e_index[0,:]],min_path_loss[e_index[1,:]],p_contention))

            asso = torch.eq(min_path_loss_idx[e_index[0,:]], min_path_loss_idx[e_index[1,:]] ).float()
            actor_input = torch.hstack((asso,interference_and_min_pl_and_p_contention))

            label = self.actor_target.forward(actor_input)

            action_mat = to_dense_adj(edge_index=e_index,batch=None,edge_attr=label).view(n_node,n_node)
            action = to_numpy(action_mat)

        return action

    @counted
    def step(self, batch):
        self._print("learn")
        if not batch:
            self._print("batch is none")
            return

        l_i = self._train_infer(batch)
        l_c = self._train_critic(batch)
        l_a = self._train_actor(batch)

        print(l_i,l_a,l_c)
        self.update_nn()

    def _train_infer(self,batch):
        self._print("_train_infer")
        loss = torch.zeros(1)
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            e_index:torch.Tensor = from_networkx(G).edge_index

            state = to_tensor(sample['state'],requires_grad=False)
            state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))

            target = to_tensor(sample['target'],requires_grad=False)
            target = target[e_index.transpose(0,1)[:,0],e_index.transpose(0,1)[:,1]]

            target = nn.functional.one_hot(target.long(), num_classes=2).float()
            result = self.infer.forward(state)

            self._print("_train_infer diff",torch.hstack((target,result,state)).transpose(0,1)[:,0])

            loss += nn.functional.cross_entropy(result,target)

        loss/=len(batch)
        self._print("_train_infer loss",loss)
        self.infer_optim.zero_grad()
        loss.backward()
        self.infer_optim.step()

        return to_numpy(loss)

    def _train_actor(self,batch):
        self._print("_train_actor")
        loss = torch.zeros(1)
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            e_index:torch.Tensor = from_networkx(G).edge_index
            with torch.no_grad():
                state = to_tensor(sample['state'],requires_grad=False)
                state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))
                result = self.infer.forward(state)
                p_contention = result[:,1:2]

                state = to_tensor(sample['state'],requires_grad=False)
                min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

                x = min_path_loss

                state_A = state[e_index[0,:]]
                state_A_to_B = torch.gather(state_A,1,min_path_loss_idx[e_index[1,:]])
                state_B = state[e_index[1,:]]
                state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])
                interference = torch.hstack((state_A_to_B,state_B_to_A))
                interference_and_min_pl_and_p_contention = torch.hstack((interference,min_path_loss[e_index[0,:]],min_path_loss[e_index[1,:]],p_contention))
                asso = torch.eq(min_path_loss_idx[e_index[0,:]], min_path_loss_idx[e_index[1,:]] ).float()
                actor_input = torch.hstack((asso,interference_and_min_pl_and_p_contention))

            label = self.actor.forward(actor_input)

            s_a = torch.hstack((asso,interference,p_contention,label))
            self._printa("_train_actor states\n",to_numpy(state).T)
            self._printa("_train_actor minidA\n",to_numpy(min_path_loss_idx[e_index[0,:]]).T)
            self._printa("_train_actor minidB\n",to_numpy(min_path_loss_idx[e_index[1,:]]).T)
            self._printa("_train_actor sapair\n",to_numpy(s_a).T)
            self._printa("_train_actor minlos\n",to_numpy(torch.hstack((min_path_loss[e_index[0,:]],min_path_loss[e_index[1,:]]))).T)

            q = self.critic_target.forward(x,e_index,s_a)
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
    def _fair_q(self,q):
        assert self.FAIRNESS_ALPHA >=0
        self._printa("_fair_q qraw",q.transpose(0,1))
        if self.FAIRNESS_ALPHA == 1:
            q = torch.log(q)
        else:
            q = torch.pow(q,1-self.FAIRNESS_ALPHA)/(1-self.FAIRNESS_ALPHA)
        self._printa("_fair_q fair",q.transpose(0,1))
        return q
    def _train_critic(self,batch):
        self._print("_train_critic")
        loss = torch.zeros(1)
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            e_index:torch.Tensor = from_networkx(G).edge_index
            with torch.no_grad():
                target = to_tensor(sample['target'],requires_grad=False)
                target = target[e_index[0],e_index[1]].view(-1,1)

                action = to_tensor(sample['action'],requires_grad=False)
                action = action[e_index[0],e_index[1]].view(-1,1)

                state = to_tensor(sample['state'],requires_grad=False)
                min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

                x = min_path_loss

                state_A = state[e_index[0,:]]
                state_A_to_B = torch.gather(state_A,1,min_path_loss_idx[e_index[1,:]])
                state_B = state[e_index[1,:]]
                state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])
                interference = torch.hstack((state_A_to_B,state_B_to_A))

                asso = torch.eq(min_path_loss_idx[e_index[0,:]], min_path_loss_idx[e_index[1,:]] ).float()
                s_a = torch.hstack((asso,interference,target,action))
                self._printa("_train_critic sapair\n",to_numpy(s_a).T)

            q = self.critic.forward(x,e_index,s_a)
            loss += nn.functional.mse_loss(q,to_tensor(sample['reward']))
            self._printa("_train_critic critic",to_numpy(q).T)
            self._printa("_train_critic reward",sample['reward'].T)
        loss/=len(batch)
        self._print("_train_critic loss",loss)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return to_numpy(loss)
