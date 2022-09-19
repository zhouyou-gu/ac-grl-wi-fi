import os

import numpy as np
from sim_src.env.env import path_loss_model
import matplotlib.pyplot as plt

from sim_src.util import cat_str_dot_txt

OUT_FOLDER = os.path.splitext(os.path.basename(__file__))[0]
OUT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), OUT_FOLDER)
try:
    os.mkdir(OUT_FOLDER)
except:
    pass

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

sig = 0.
n_ap = 4
mm = path_loss_model(n_ap=n_ap, range=1000, shadowing_sigma=sig)
mm.set_ap_locs(500, 500)
mm.set_ap_locs(500, -500)
mm.set_ap_locs(-500, -500)
mm.set_ap_locs(-500, 500)

import numpy as np
import torch
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sim_src.infer.model import MixtureDensityNetwork, MixtureDensityNetworkCombined1OutDim

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model = MixtureDensityNetworkCombined1OutDim(n_ap * 2, n_components=3)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

y_mean = 0.
y_sigma = 0.
for i in range(1000):
    data = mm.gen_sta_pair(20)
    y = data[1]
    y_mean = y_mean * 0.99 + np.mean(y) * 0.01
    y_sigma = y_sigma * 0.99 + np.mean(np.sqrt(np.square(y - y_mean))) * 0.01
    if i % 100 == 0:
        print(y_mean, y_sigma)

i_max = 100000
for i in range(i_max):
    if i % 100 == 0:
        data = mm.gen_sta_pair(20)
        x = data[0]
        y = data[1]
        y_scaled = (y - y_mean) / y_sigma
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        samples, pi, mean, scale = model.sample(x)
        samples_scaled = samples * y_sigma + y_mean
        # print((x.data,y.data,samples.data,pi.data,mean.data,scale.data))
        out = np.hstack((y.data, y_scaled, samples.data, pi.data, mean.data, scale.data))
        # out = np.hstack((x.data,y.data,samples.data,pi.data,mean.data,scale.data))
        print(out)
        print(np.cov(np.hstack((samples.data, y.data)).transpose()))
    data = mm.gen_sta_pair(100)
    x = data[0]
    y = data[1]
    y = (y - y_mean) / y_sigma
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    optimizer.zero_grad()
    loss = model.loss(x, y).mean()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}")
        print(y_mean)
    if i % 1000 == 0:
        data = mm.gen_sta_pair(300)
        x = data[0]
        y = data[1]
        x = torch.Tensor(x)
        samples, pi, mean, scale = model.sample(x)
        samples_scaled = samples * y_sigma + y_mean
        y_pred = samples_scaled.numpy()

        yy = np.hstack((y, y_pred))

        out = cat_str_dot_txt(["out", os.path.splitext(os.path.basename(__file__))[0], str(sig), "y", str(i)])
        out = os.path.join(OUT_FOLDER, out)
        with open(out, "w") as f:
            np.savetxt(f, yy, delimiter=",")

        plt.figure()
        plt.scatter(y, y_pred)
        plt.savefig(out+".png")


data = mm.gen_sta_pair(300)
x = data[0]
y = data[1]
x = torch.Tensor(x)
samples, pi, mean, scale = model.sample(x)
samples_scaled = samples * y_sigma + y_mean
y_pred = samples_scaled.numpy()

yy = np.hstack((y, y_pred))

out = cat_str_dot_txt(["out", os.path.splitext(os.path.basename(__file__))[0], str(sig), "y", "final"])
out = os.path.join(OUT_FOLDER, out)
with open(out, "w") as f:
    np.savetxt(f, yy, delimiter=",")
plt.figure()
plt.scatter(y, y_pred)
plt.savefig(out+".png")
