import random

import numpy as np

from sim_src.edge_label.gw_cut import cut_into_2_k
from sim_src.edge_label.model import gnn_edge_label
from sim_src.ns3_ctrl.ns3_ctrl import build_ns3
from sim_src.ns3_ctrl.wifi_net_ctrl import sim_wifi_net, wifi_net_config
from sim_src.sim_env.path_loss import path_loss
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import to_tensor, to_numpy


np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

build_ns3("/home/soyo/wifi-ai/ns-3-dev")
# exit(0)
e = sim_env(id=random.randint(40,60),ns3_sim_time_s=2)
e.PROG_PATH = "/home/soyo/wifi-ai/ns-3-dev"
e.PROG_NAME = "wifi-ai/env"
e.DEBUG = True
n_step = 100000

model = gnn_edge_label(0)
model.DEBUG_STEP = 100
model.DEBUG = True

e.set_actor(model)
for i in range(n_step):
    e.init_env()
    sample = e.step(no_run=False)
    model.step([sample])
