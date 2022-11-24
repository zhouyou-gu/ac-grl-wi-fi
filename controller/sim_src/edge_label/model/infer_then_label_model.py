import networkx as nx
import torch
from torch import optim, nn
from torch_geometric.utils import from_networkx, to_dense_adj

from sim_src.edge_label.model.base_model import base_model
from sim_src.edge_label.nn import INFNN, WGNN, ELNN
from sim_src.util import hard_update_inplace, to_tensor, to_numpy, to_device


class infer_then_label(base_model):
    def __init__(self, id, edge_dim=4, node_dim=4):
        base_model.__init__(self,id,edge_dim,node_dim)

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

    def gen_action(self, state_np):
        n_node = state_np.shape[0]
        with torch.no_grad():
            G = nx.complete_graph(n_node)
            e_index = to_device(from_networkx(G).edge_index)
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

        action = self.add_noise(action)
        return action

    def _train_infer(self,batch):
        self._print("_train_infer")
        loss = to_device(torch.zeros(1))
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            e_index = to_device(from_networkx(G).edge_index)

            state = to_tensor(sample['state'],requires_grad=False)
            state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))

            target = to_tensor(sample['target'],requires_grad=False)
            target = target[e_index.transpose(0,1)[:,0],e_index.transpose(0,1)[:,1]]

            target = nn.functional.one_hot(target.long(), num_classes=2).float()
            result = self.infer.forward(state)

            self._printa("_train_infer diff",torch.hstack((target,result,state)).transpose(0,1)[:,0])
            loss += nn.functional.cross_entropy(torch.log(result),target)

        loss/=len(batch)
        self._print("_train_infer loss",loss)
        self.infer_optim.zero_grad()
        loss.backward()
        self.infer_optim.step()

        return to_numpy(loss)

    def _train_actor(self,batch):
        self._print("_train_actor")
        loss = to_device(torch.zeros(1))
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            e_index = to_device(from_networkx(G).edge_index)
            with torch.no_grad():
                state = to_tensor(sample['state'],requires_grad=False)
                state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))
                result = self.infer_target.forward(state)
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

            label.requires_grad_()

            s_a = torch.hstack((asso,interference,p_contention,label))
            self._printa("_train_actor states\n",to_numpy(state).T)
            self._printa("_train_actor minidA\n",to_numpy(min_path_loss_idx[e_index[0,:]]).T)
            self._printa("_train_actor minidB\n",to_numpy(min_path_loss_idx[e_index[1,:]]).T)
            self._printa("_train_actor sapair\n",to_numpy(s_a).T)
            self._printa("_train_actor minlos\n",to_numpy(torch.hstack((min_path_loss[e_index[0,:]],min_path_loss[e_index[1,:]]))).T)

            qq = self.critic_target.forward(x,e_index,s_a)
            qq.retain_grad()
            q = self._fair_q(qq)
            q = -torch.mean(q)
            q.backward()
            lable_g = label.grad.detach()
            label_differentiable = self.actor.forward(actor_input)
            loss += (-torch.mean((lable_g)*label_differentiable))
            self._printa("_train_actor qq.grad\n",to_numpy(qq.grad).T)
            self._printa("_train_actor label.grad\n",to_numpy(label.grad).T)
            self._printa("_train_actor label\n",to_numpy(label).T)

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