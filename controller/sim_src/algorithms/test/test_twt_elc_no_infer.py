import os
import random

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx

from sim_src.edge_label.model.base_model import base_model
from sim_src.ns3_ctrl.ns3_ctrl import build_ns3
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import to_tensor, to_numpy, get_current_time_str, ParameterConfig

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

class norm_model(base_model):
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

            label_no_grad = label.clone()
            label.requires_grad_()

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
            q = -torch.mean(q)
            q.backward()
            # w_g = label.grad - torch.mean(label.grad)
            lable_g = label.grad.detach()
            # lable_g = lable_g - torch.mean(lable_g)
            scale = label_no_grad
            scale[lable_g>0.] = 1. - scale[lable_g>0.]
            label_differentiable = self.actor.forward(actor_input)
            loss += (-torch.mean((lable_g*scale)*label_differentiable))
            self._printa("_train_actor label.grad\n",to_numpy(label.grad).T)
            self._printa("_train_actor lable_g\n",to_numpy(lable_g).T)
            self._printa("_train_actor scale\n",to_numpy(scale).T)
            # normalized_label = label-torch.min(label)
            # normalized_label /= torch.max(normalized_label)
            # loss += (self.EQUALIZATION_FACTOR*nn.functional.mse_loss(label,normalized_label.detach(),reduction="sum")/sample['n_node'])

        loss/=len(batch)
        self._print("_train_actor loss",loss)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return to_numpy(loss)


def run_sim(ALPHA=0.):
    OUT_FOLDER = os.path.splitext(os.path.basename(__file__))[0] + "-" + get_current_time_str()
    OUT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), OUT_FOLDER)


    build_ns3("/home/soyo/wifi-ai/ns-3-dev")
    # exit(0)
    e = sim_env(id=random.randint(40,60))
    e.PROG_PATH = "/home/soyo/wifi-ai/ns-3-dev"
    e.PROG_NAME = "wifi-ai/env"
    e.DEBUG = True

    n_step = 1000
    batch_size = 1
    model = norm_model(0)
    model.DEBUG_STEP = 10
    model.DEBUG = True

    cfg = ParameterConfig()
    cfg['ALPHA'] = ALPHA
    model.FAIRNESS_ALPHA = cfg['ALPHA']
    cfg.save(OUT_FOLDER,"NaN")

    e.set_actor(model)
    for i in range(n_step):
        batch = []
        for j in range(batch_size):
            e.init_env()
            sample = e.step(no_run=False)
            batch.append(sample)
        model.step(batch)
        if (i+1) % 100 == 0:
            model.save(OUT_FOLDER,str(i))
            e.save_np(OUT_FOLDER,str(i))

for A in [10.]:
    run_sim(A)