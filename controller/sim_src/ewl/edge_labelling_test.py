import math

import numpy as np
import matplotlib.pyplot as plt

from sim_src.env.env import path_loss_model
from sim_src.cut.gw_cut import gw_cut
from sim_src.ewl.model import BernulliNetwork
from sim_src.memory.replay_memory import memory

import torch
import torch.optim as optim

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

n_ap = 4
n_sta = 5
n_out_dim = 1
model = BernulliNetwork(1)
model_target = BernulliNetwork(1)
tau = 0.01
for target_param, param in zip(model_target.parameters(), model.parameters()):
    target_param.data.copy_(param.data)

optimizer = optim.Adam(model.parameters(), lr=0.001)

mm = path_loss_model(n_ap=4, range=1000)
mm.set_ap_locs(500, 500)
mm.set_ap_locs(500, -500)
mm.set_ap_locs(-500, -500)
mm.set_ap_locs(-500, 500)

Acutter = gw_cut(n_sta)
Bcutter = gw_cut(n_sta)

n_solution = 100
net_per_batch = 10
edge_per_net = 10

ratio = []
ratio_by_mean = []
figure_c = 0

rmemory = memory(size=100000,asynchronization=True)

for loop in range(1000):
    # for target_param, param in zip(model.parameters(), model_target.parameters()):
    #     target_param.data.copy_(param.data)

    node_a_b_batch = torch.empty((0, 1))
    sample_edge_weight = torch.empty((0, 1))
    reward_value = torch.empty((0, 1))
    performance = 0.
    for n in range(net_per_batch):
        loss_sta_to_ap, H, sta_loc = mm.gen_n_sta_info(n_sta)
        D = np.zeros((n_sta, n_sta))
        D_array = []
        for i in range(n_sta):
            for j in range(i + 1, n_sta):
                D[i, j] = 0. if np.linalg.norm(sta_loc[i] - sta_loc[j], ord=2) > 600 else 1.
                # D[i, j] = np.linalg.norm(sta_loc[i] - sta_loc[j], ord=2)/1200
                D[j, i] = D[i, j]
                D_array.append(D[i, j])

        D_array = np.array(D_array)
        D_array = torch.Tensor(D_array).detach_()

        node_a_b = torch.empty((0, 1))
        for i in range(n_sta):
            for j in range(i + 1, n_sta):
                a_b = np.linalg.norm(sta_loc[i] - sta_loc[j], ord=2)/1200
                a_b = torch.Tensor(np.expand_dims(a_b, axis=0))
                node_a_b = torch.vstack((node_a_b, a_b))

        node_a_b_batch = torch.vstack((node_a_b_batch,node_a_b))

        H_from_nn = np.zeros((n_sta, n_sta))
        with torch.no_grad():
            out_tensor, m, sd = model_target.sample(node_a_b)
            out = out_tensor.data.numpy()
            out = np.argmax(out,axis=1,keepdims=True)
            idx_tmp = 0
            for i in range(n_sta):
                for j in range(i+1,n_sta):
                    H_from_nn[i,j] = out[i]
                    H_from_nn[j,i] = H_from_nn[i,j]
                    idx_tmp += 1
        if n == 0:
            print(out[0],D_array[0],node_a_b[0])
        # print(sample_edge_weight,out)
        sample_edge_weight = torch.vstack((sample_edge_weight,torch.Tensor(out)))

        # H_from_nn /= np.mean(np.abs(H_from_nn))
        Acutter.set_edge_weights(H_from_nn)
        Acutter.solve()
        s, c = Acutter.get_m_solutions(n_solution)

        ## get the cut performance from bcutter test only. in the actual sim should be network performance
        cut_performance = 0.
        cut_performance_per_edge = np.zeros((int(n_sta*(n_sta-1)/2),1))
        for ss in s:
            a, b = gw_cut.split(ss)
            cut_matrix = np.zeros((n_sta, n_sta))
            # print(a,b)
            for aa in a:
                for bb in b:
                    if H_from_nn[aa,bb]>0:
                        cut_matrix[aa,bb] = 1
                        cut_matrix[bb,aa] = cut_matrix[aa,bb]
                    else:
                        cut_matrix[aa,bb] = -1
                        cut_matrix[bb,aa] = cut_matrix[aa,bb]
            idx_tmp = 0
            for i in range(n_sta):
                for j in range(i+1,n_sta):
                    if cut_matrix[i,j] == 0 and H_from_nn[i,j]>0:
                        cut_performance_per_edge[idx_tmp,0] += (-1)
                    elif cut_matrix[i,j] == 0 and H_from_nn[i,j]==0:
                        cut_performance_per_edge[idx_tmp,0] += (1)
                    else:
                        cut_performance_per_edge[idx_tmp,0] += cut_matrix[i,j]
                    idx_tmp += 1
            cut_performance += gw_cut.cut_cost(a, b, D)
        cut_performance_per_edge/=n_solution
        # print("---",cut_performance_per_edge)
        cut_performance /= n_solution
        # print(cut_performance)

        Bcutter.set_edge_weights(D)
        Bcutter.solve()
        s, c = Bcutter.get_m_solutions(n_solution)
        perfect_performance = np.mean(c)
        # print(perfect_performance,cut_performance)
        p_tmp = math.exp((cut_performance - perfect_performance)*10)
        if n == 0:
            print(H_from_nn)
            print(D)
            print(cut_performance,perfect_performance)
            print(torch.hstack((model_target.forward(node_a_b).probs,node_a_b)))
        reward_value = torch.vstack((reward_value, torch.ones((int(n_sta * (n_sta - 1) / 2), 1)) * p_tmp))
        # reward_value = torch.vstack((reward_value, torch.Tensor(cut_performance_per_edge)))
        performance += cut_performance-perfect_performance
        rmemory.step((node_a_b[0],out[0], p_tmp))

    data = rmemory.sample(100)
    if data is None:
        continue

    # sample_edge_weight = torch.nn.functional.one_hot(sample_edge_weight.to(torch.int64),num_classes=2)
    in_f = torch.Tensor(np.vstack([d[0] for d in data]))
    out_f = torch.nn.functional.one_hot(torch.Tensor(np.vstack([d[1] for d in data])).to(torch.int64),num_classes=2)
    r_f = torch.Tensor(np.vstack([d[2] for d in data]))

    # print(in_f,out_f,r_f)
    # loss = model.loss(node_a_b_batch, sample_edge_weight.detach_(), reward_value, entropy=True)
    loss = model.loss(in_f, out_f, r_f, entropy=False)
    loss = torch.mean(loss)
    print(loop, loss, torch.mean(reward_value),performance/net_per_batch, "++++++++++++++++++++++++++++++")
    ratio.append(torch.mean(reward_value).data.numpy())
    ratio_by_mean.append(torch.mean(reward_value).data.numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for target_param, param in zip(model_target.parameters(), model.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

# if (loop+1) % 10 == 0 and loop > 10:
plt.figure(figure_c)
figure_c += 1
plt.plot(np.array(ratio), color='r', linewidth=1)
plt.figure(figure_c)
figure_c += 1
plt.plot(np.array(ratio_by_mean), color='b', linewidth=1)
plt.figure(figure_c)
figure_c += 1
plt.plot(np.array(ratio) - np.array(ratio_by_mean), color='g', linewidth=1)
plt.show()
# plt.plot(res)
