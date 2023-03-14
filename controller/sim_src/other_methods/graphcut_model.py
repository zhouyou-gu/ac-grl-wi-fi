import networkx
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx, to_dense_adj

from sim_src.edge_label.gw_cut import cut_into_2_k
from sim_src.edge_label.model.infer_model import infer_model
from sim_src.sim_env.path_loss import path_loss
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import StatusObject, to_numpy, to_tensor, \
    to_device, DbToRatio, RatioToDb


class graphcut_interference_model(StatusObject):
    W = 15
    M = 6
    PACKET_SIZE = 100
    NOISE_FLOOR_1MHZ_DBM = -93.9763
    MIN_RSSI = -95
    N_GROUP = 4
    def __init__(self, id):
        self.id = id
    def gen_action(self,state_np):
        n_node = state_np.shape[0]
        with torch.no_grad():
            G = nx.complete_graph(n_node)
            e_index = to_device(from_networkx(G).edge_index)
            # state = to_tensor(state_np)
            # state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))

            state = to_tensor(state_np,requires_grad=False)
            min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

            # state_A = state[e_index[0,:]]
            # state_A_to_B = torch.gather(state_A,1,min_path_loss_idx[e_index[1,:]])
            state_B = state[e_index[1,:]]
            state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])
            # print(state_B_to_A)
            interference_noise = DbToRatio((state_B_to_A+1.)*(self.MIN_RSSI)) + DbToRatio(self.NOISE_FLOOR_1MHZ_DBM)
            # print(interference_noise)
            # print(min_path_loss[e_index[0,:]])
            # print(DbToRatio((min_path_loss[e_index[0,:]]+1.)*(self.MIN_RSSI)))

            sinr = DbToRatio((min_path_loss[e_index[0,:]]+1.)*(self.MIN_RSSI))/interference_noise
            # print(DbToRatio((min_path_loss[e_index[0,:]]+1.)*(self.MIN_RSSI)))
            # sinr = 10*torch.log10(sinr)
            sinr = 1./sinr
            action_mat = to_dense_adj(edge_index=e_index,batch=None,edge_attr=sinr).view(n_node,n_node)
            action = to_numpy(action_mat)

        return action

# class sim_env_min_cut(sim_env):
#     def format_act_to_sta_twt_idx(self,action):
#         G = networkx.from_numpy_matrix(action)
#         print(G.edges)
#         return action

class graphcut_hidden_model(infer_model):
    MIN_RSSI = -95
    def gen_action(self,state_np):
        n_node = state_np.shape[0]
        with torch.no_grad():
            G = nx.complete_graph(n_node)
            e_index = to_device(from_networkx(G).edge_index)
            state = to_tensor(state_np)
            state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))

            result = self.infer_target.forward(state)
            p_contention = result

            state = to_tensor(state_np,requires_grad=False)
            min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

            # state_A = state[e_index[0,:]]
            # state_A_to_B = torch.gather(state_A,1,min_path_loss_idx[e_index[1,:]])
            state_B = state[e_index[1,:]]
            state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])


            interference = (state_B_to_A >= 0.1)

            # print(interference)
            # label = interference.float()
            # print(label)
            label = (1.-p_contention)

            action_mat = to_dense_adj(edge_index=e_index,batch=None,edge_attr=label).view(n_node,n_node)
            action = to_numpy(action_mat)

        return action


class graphcut_contention_model(infer_model):
    MIN_RSSI = -95
    def gen_action(self,state_np):
        n_node = state_np.shape[0]
        with torch.no_grad():
            G = nx.complete_graph(n_node)
            e_index = to_device(from_networkx(G).edge_index)
            state = to_tensor(state_np)
            state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))

            result = self.infer_target.forward(state)
            p_contention = result

            state = to_tensor(state_np,requires_grad=False)
            min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

            # state_A = state[e_index[0,:]]
            # state_A_to_B = torch.gather(state_A,1,min_path_loss_idx[e_index[1,:]])
            state_B = state[e_index[1,:]]
            state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])


            interference = (state_B_to_A >= 0.1)

            # print(interference)
            # label = interference.float()
            # print(label)
            label = (p_contention)

            action_mat = to_dense_adj(edge_index=e_index,batch=None,edge_attr=label).view(n_node,n_node)
            action = to_numpy(action_mat)

        return action

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    # G = nx.complete_graph(10)
    pl_model = path_loss(n_sta=20)
    bm = graphcut_hidden_model(0)

    states = pl_model.get_loss_sta_ap()
    states = pl_model.convert_loss_sta_ap_threshold(states) / (-bm.MIN_RSSI) - 1
    action = bm.gen_action(states)
    print(action)
    id = cut_into_2_k(action,20,2)
    print(id)
    # sim_env(0).format_act_to_sta_twt_idx(action)
    # bm = bianchi_model(0,global_allocation=False)
    # print(bm.gen_action(states))
    # G = networkx.from_numpy_array(action)
    # print(G)
    # ret, set= networkx.minimum_cut(G)
    # print(set)
    # print(id)