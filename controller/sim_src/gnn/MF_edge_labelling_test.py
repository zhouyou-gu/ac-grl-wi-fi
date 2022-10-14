import math

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, Batch

from sim_src.env.env import path_loss_model, p_true
from sim_src.cut.gw_cut import gw_cut
from sim_src.gnn.model import ELNN, WGNN, WGNN_GAT
from sim_src.memory.replay_memory import memory

import torch
import torch.optim as optim

from sim_src.util import *

def sym_mat_to_vec(H, dim):
    ret = np.zeros((int(dim*(dim-1)/2),1))
    idx = 0
    for i in range(dim):
        for j in range(i+1,dim):
            ret[idx,0] = H[i,j]
            idx += 1
    return ret

def vec_to_sym_mat(vec,dim,b_size):
    ret_list = []
    idx = 0
    for b in range(b_size):
        ret = np.zeros((dim,dim))
        for i in range(dim):
            for j in range(i+1,dim):
                ret[i,j] = vec[idx,0]
                ret[j,i] = ret[i,j]
                idx += 1
        ret_list.append(ret)
    return ret_list

def get_vec_to_mat_idx(f,t,dim):
    assert f != t
    idx = 0
    for i in range(dim):
        for j in range(i+1,dim):
            if (i == f and j == t) or (i == t and j == f):
                return idx
            idx += 1





np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

n_ap = 4
n_sta = 5
n_out_dim = 1
el_model = ELNN(1)
rc_model = WGNN(1,5,1,edge_dim=2)


el_optimizer = optim.Adam(el_model.parameters(), lr=0.0001)
rc_optimizer = optim.Adam(rc_model.parameters(), lr=0.0001)

mm = path_loss_model(n_ap=4, range=1000)
mm.set_ap_locs(500, 500)
mm.set_ap_locs(500, -500)
mm.set_ap_locs(-500, -500)
mm.set_ap_locs(-500, 500)

Acutter = gw_cut(n_sta)
Bcutter = gw_cut(n_sta)

n_solution = 100
net_per_batch = 20
edge_per_net = 10

plot_res_1 = []
plot_res_2 = []
plot_res_3 = []
plot_res_4 = []
figure_c = 0

rmemory = memory(size=100000,asynchronization=True)

for loop in range(1000):
    node_a_b_batch = torch.empty((0, 1))
    sample_edge_weight = torch.empty((0, 1))
    reward_value = torch.empty((0, 1))
    performance = 0.
    for n in range(net_per_batch):
        loss_sta_to_ap, H, sta_loc = mm.gen_n_sta_info(n_sta)
        D = np.zeros((n_sta, n_sta))
        for i in range(n_sta):
            for j in range(i + 1, n_sta):
                D[i, j] = np.linalg.norm(sta_loc[i] - sta_loc[j], ord=2)/1200
                D[j, i] = D[i, j]

        W_true = np.zeros((n_sta, n_sta))
        for i in range(n_sta):
            for j in range(i + 1, n_sta):
                W_true[i, j] = 0. if D[i,j] > 1. else 1.
                W_true[j, i] = W_true[i, j]

        Acutter.set_edge_weights(W_true)
        Acutter.solve()
        s, c = Acutter.get_m_solutions(n_solution)

        R_true = 0.
        R_elnn_per_sta = np.zeros((net_per_batch,n_sta))
        for ss in s:
            a, b = gw_cut.split(ss)
            R_true += gw_cut.cut_cost(a, b, W_true)
        R_true /= n_solution

        sample = (D, W_true, R_true)
        rmemory.step(sample)

    data = rmemory.sample(net_per_batch)
    Dis = np.vstack([sym_mat_to_vec(dd[0], n_sta) for dd in data])
    Dis = to_tensor(Dis)
    lw_ts = el_model.forward(Dis)
    lw_np = to_numpy(lw_ts)
    rand_W = np.random.uniform(low=0.,high=1.,size=lw_np.shape)
    # print(rand_W,lw_np.shape)
    if p_true(0.5):
        W_true = np.vstack([sym_mat_to_vec(dd[1],n_sta) for dd in data])
        W_elnn = vec_to_sym_mat(W_true,n_sta,net_per_batch)
    else:
        # lw_np = lw_np-np.min(lw_np)
        # lw_np = lw_np/np.max(lw_np)
        W_elnn = vec_to_sym_mat(lw_np,n_sta,net_per_batch)

    # print(W_elnn)

    R_elnn = np.zeros((net_per_batch,1))
    R_elnn_per_sta = np.zeros((net_per_batch,n_sta))
    for n in range(net_per_batch):
        W_true = data[n][1]
        Acutter.set_edge_weights(W_elnn[n])
        Acutter.solve()
        s, c = Acutter.get_m_solutions(n_solution)
        for ss in s:
            a, b = gw_cut.split(ss)
            R_elnn[n] += gw_cut.cut_cost(a, b, W_true)
            for aa in a:
                for bb in b:
                    R_elnn_per_sta[n,aa] += (1./2. * W_true[aa,bb])
                    R_elnn_per_sta[n,bb] += (1./2. * W_true[aa,bb])

        R_elnn[n] /= n_solution
        plot_res_1.append(R_elnn[n]-data[n][2])
        # print(R_elnn[n]-data[n][2])
        R_elnn_per_sta[n,:] /= n_solution

    # print(R_elnn,R_elnn_per_sta,np.sum(R_elnn_per_sta,axis=1,keepdims=True), [dd[2] for dd in data])

    G = nx.complete_graph(n_sta)
    e_index = from_networkx(G).edge_index
    # print(e_index)


    rc_data_list = []
    for n in range(net_per_batch):
        lw_vec = to_tensor(sym_mat_to_vec(W_elnn[n],n_sta))
        lw_mat_ts = to_tensor(W_elnn[n])[e_index[0,:],e_index[1,:]]
        lw_mat_ts = lw_mat_ts-torch.min(lw_mat_ts)
        lw_mat_ts = lw_mat_ts/torch.max(lw_mat_ts)
        D_ts = to_tensor(data[n][0][e_index[0,:],e_index[1,:]])
        # print(lw_mat_ts,D_ts)
        e_attr = torch.vstack((lw_mat_ts,D_ts)).transpose(0,1)
        print("e_attr",e_attr)

        # exit(0)
        d = Data(x=torch.ones((n_sta,1)),edge_attr=e_attr,edge_index=e_index)
        rc_data_list.append(d)

    batch = Batch.from_data_list(rc_data_list)
    # print(batch.edge_attr)
    y = rc_model.forward(batch.x,batch.edge_index,batch.edge_attr)
    target_y = to_tensor(R_elnn_per_sta.flatten()[:,np.newaxis])
    # print(target_y)
    loss = torch.nn.functional.mse_loss(y,target_y)
    rc_optimizer.zero_grad()
    loss.backward()
    rc_optimizer.step()
    print(to_numpy(loss),"loss")
    plot_res_2.append(to_numpy(loss))


    el_ts_idx = np.empty((0,1))
    for n in range(net_per_batch):
        idx = np.array([get_vec_to_mat_idx(to_numpy(x)[0],to_numpy(x)[1],dim=n_sta) for x in e_index.transpose(0,1)])
        el_ts_idx = np.vstack((el_ts_idx, idx[:,np.newaxis]))
        # print(el_ts_idx)
    el_ts_idx = to_tensor(el_ts_idx,dtype=LONG_TYPE)
    # print(el_ts_idx.dtype)
    # print(lw_ts[el_ts_idx,0])
    lw_ts_tmp = el_model.forward(Dis)
    # lw_ts = lw_ts-torch.min(lw_ts_tmp)
    # lw_ts = lw_ts/torch.max(lw_ts)
    e_attr = torch.hstack((lw_ts_tmp[el_ts_idx[:,0]],batch.edge_attr[:,1].view(-1, 1)))
    y = rc_model.forward(batch.x,batch.edge_index,e_attr)
    l = -torch.mean(y)

    el_optimizer.zero_grad()
    l.backward()
    for param in el_model.parameters():
        print("++++++++",param.grad)
        pass
    el_optimizer.step()
    print(np.hstack((Dis, to_numpy(lw_ts), to_numpy(lw_ts_tmp))))
    print(loop)

    # for e in data.edge_index.transpose():
    #     print(e)

    #
    # W_elnn
    # R_elnn_per_sta


plt.figure(figure_c)
figure_c += 1
plt.plot(np.array(plot_res_1), color='r', linewidth=1)
plt.figure(figure_c)
figure_c += 1
plt.plot(np.array(plot_res_2), color='b', linewidth=1)
# plt.figure(figure_c)
# figure_c += 1
# plt.plot(np.array(ratio) - np.array(ratio_by_mean), color='g', linewidth=1)
plt.show()
# plt.plot(res)
