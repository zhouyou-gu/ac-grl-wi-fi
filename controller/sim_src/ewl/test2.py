import numpy as np

from sim_src.env.env import path_loss_model
from sim_src.cut.gw_cut import gw_cut
from sim_src.ewl.model import FNN

import torch
import torch.optim as optim

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

n_ap = 4
n_sta = 5
n_out_dim = 1
model = FNN(n_ap,out_dim=n_out_dim)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


mm = path_loss_model(n_ap=n_ap, range=1000)
mm.set_ap_locs(500,500)
mm.set_ap_locs(500,-500)
mm.set_ap_locs(-500,-500)
mm.set_ap_locs(-500,500)

Acutter = gw_cut(n_sta)
Bcutter = gw_cut(n_sta)

n_solution = 100
batch_size = 1

res = []
for loop in range(1000):
    loss = 0.
    temp_res = 0.
    for it in range(20):
        d, H, _ = mm.gen_n_sta_info(n_sta)
        for i in range(n_sta):
            for j in range(i+1,n_sta):
                # H[i,j] = H[i,j] + np.max(d[i]) + np.max(d[j])
                H[i,j] = H[i,j]
                H[j,i] = H[i,j]
        np.fill_diagonal(H,0)
        # print(H)
        Acutter.set_edge_weights(H)
        Acutter.solve()
        s, c = Acutter.get_m_solutions(n_solution)
        goal = np.sum(c)/n_solution
        d = torch.Tensor(d)
        d.detach_()
        for it_it in range(1):
            with torch.no_grad():
                out = model.forward(d)
            G = np.zeros((n_sta,n_sta))
            for i in range(n_sta):
                for j in range(i+1,n_sta):
                    G[i,j] = np.linalg.norm((out[i]-out[j]),ord=2)
                    G[j,i] = G[i,j]

            G = G/np.abs(np.sum(G)/(n_sta**2))
            Bcutter.set_edge_weights(G)
            Bcutter.solve()
            s, c = Bcutter.get_m_solutions(n_solution)
            local_old = 0
            for ss in s:
                a, b = Bcutter.split(ss)
                local_old += Bcutter.cut_cost(a,b,H)

            batch = []
            batch_idx = np.random.choice([u for u in range(n_sta)],size=batch_size,replace=False)
            # print(batch_idx)
            for i in batch_idx:
                batch.append(d[i])
            batch = torch.vstack(batch)
            # print(batch)
            out_grad = model.forward(batch)
            # print(out_grad)
            d_interference_idx = []
            d_interference = []
            for i in batch_idx:
                # j = np.random.randint(0,n_out_dim)
                a = np.random.randn(1,n_out_dim)*np.linalg.norm(out[i],ord=1)/n_out_dim/10
                out[i] += a
                d_interference.append(torch.Tensor(a))

            G = np.zeros((n_sta,n_sta))
            for i in range(n_sta):
                for j in range(i+1,n_sta):
                    G[i,j] = np.linalg.norm((out[i]-out[j]),ord=2)
                    G[j,i] = G[i,j]

            G = G/np.abs(np.sum(G)/(n_sta**2))
            Bcutter.set_edge_weights(G)
            Bcutter.solve()
            s, c = Bcutter.get_m_solutions(n_solution)
            local_new = 0
            for ss in s:
                a, b = Bcutter.split(ss)
                local_new += Bcutter.cut_cost(a,b,H)

            d_local = local_new/n_solution - local_old/n_solution
            mask = torch.zeros((batch_size,n_out_dim)).detach_()
            for b in range(batch_size):
                d_proportional = d_local/(local_old/n_solution)
                loss -= torch.sum(out_grad[b] * (d_proportional / d_interference[b]))
            # print("++++++++++++++++",it,it_it, d_local, local_old/local_new, (goal*n_solution/local_old-1)*100)
            temp_res += (goal*n_solution/local_old-1)*100
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    res.append(temp_res)
    print("------------",loop,temp_res)

import matplotlib.pyplot as plt

plt.plot(res)
plt.show()
