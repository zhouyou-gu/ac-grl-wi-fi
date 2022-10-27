import math

import numpy as np
import matplotlib.pyplot as plt

from sim_src.env.env import path_loss_model
from sim_src.cut.gw_cut import gw_cut
from sim_src.ewl.model import GaussianDensityNetwork, FNN, BernulliNetwork

import torch
import torch.optim as optim

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

n_ap = 4
n_sta = 2
n_out_dim = 1
model = BernulliNetwork(4)
model_target = BernulliNetwork(4)
tau = 0.01
for target_param, param in zip(model_target.parameters(), model.parameters()):
    target_param.data.copy_(param.data)

optimizer = optim.Adam(model.parameters(), lr=0.0001)


mm = path_loss_model(n_ap=4, range=1000)
mm.set_ap_locs(500,500)
mm.set_ap_locs(500,-500)
mm.set_ap_locs(-500,-500)
mm.set_ap_locs(-500,500)

Acutter = gw_cut(n_sta)
Bcutter = gw_cut(n_sta)

n_solution = 100
net_per_batch = 100
edge_per_net = 10


ratio = []
ratio_by_mean = []
figure_c = 0
for loop in range(5000):
    # for target_param, param in zip(model.parameters(), model_target.parameters()):
    #     target_param.data.copy_(param.data)

    # loss_batch = 0.
    node_a_b = torch.empty((0,4))
    true_value = torch.empty((0,2))
    for n in range(net_per_batch):
        loss_sta_to_ap, H, sta_loc = mm.gen_n_sta_info(n_sta)
        D = np.zeros((n_sta,n_sta))
        D_array = []
        for i in range(n_sta):
            for j in range(i+1,n_sta):
                D[i,j] = 0 if np.linalg.norm(sta_loc[i]-sta_loc[j],ord=2) < 1200 else 1
                # D[i,j] = np.linalg.norm(sta_loc[i]-sta_loc[j],ord=2)
                D[j,i] = D[i,j]
                D_array.append(D[i,j])

        D_array = np.array(D_array)
        D_array = torch.Tensor([D_array[0]])
        # print(D)
        a_b = np.hstack((sta_loc[0],sta_loc[1]))
        a_b = torch.Tensor(np.expand_dims(a_b,axis=0))
        node_a_b = torch.vstack((node_a_b,a_b))

        true_value_tmp = torch.nn.functional.one_hot(D_array.to(torch.int64),num_classes=2)
        true_value = torch.vstack((true_value,true_value_tmp))
        H_from_nn = np.zeros((n_sta,n_sta))
        with torch.no_grad():
            out_tensor, m, sd = model.sample(node_a_b)
            out = out_tensor.data.numpy()
            out = np.argmax(out,axis=1,keepdims=True)
            idx_tmp = 0
            for i in range(n_sta):
                for j in range(i+1,n_sta):
                    H_from_nn[i,j] = out[i]
                    H_from_nn[j,i] = H_from_nn[i,j]
                    idx_tmp += 1
        if n == 0:
            print(out[0],D_array[0])

    # print(node_a_b,true_value)
    loss = model.loss(node_a_b,true_value,1,entropy=False)
    loss = torch.mean(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # for target_param, param in zip(model_target.parameters(), model.parameters()):
    #     target_param.data.copy_(
    #         target_param.data * (1.0 - tau) + param.data * tau
    #     )
    print(loop,loss,"+++++++++++++++++++++++++++++")
    ratio.append(loss.data.numpy())
    ratio_by_mean.append(loss.data.numpy())

# if (loop+1) % 10 == 0 and loop > 10:
plt.figure(figure_c)
figure_c +=1
plt.plot(np.array(ratio), color='r', linewidth=1)
plt.figure(figure_c)
figure_c +=1
plt.plot(np.array(ratio_by_mean), color='b', linewidth=1)
plt.figure(figure_c)
figure_c +=1
plt.plot(np.array(ratio)-np.array(ratio_by_mean), color='g', linewidth=1)
plt.show()
# plt.plot(res)