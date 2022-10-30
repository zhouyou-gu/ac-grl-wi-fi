import random

import numpy as np

from sim_src.edge_label.gw_cut import cut_into_2_k
from sim_src.edge_label.model import gnn_edge_label
from sim_src.ns3_ctrl.ns3_ctrl import build_ns3
from sim_src.ns3_ctrl.wifi_net_ctrl import sim_wifi_net, wifi_net_config
from sim_src.sim_env.path_loss import path_loss
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import to_tensor, to_numpy

build_ns3("/home/soyo/wifi-ai/ns-3-dev")
# exit(0)
e = sim_env(id=random.randint(40,60))
e.PROG_PATH = "/home/soyo/wifi-ai/ns-3-dev"
e.PROG_NAME = "wifi-ai/env"
class test_actor:
    def __init__(self, n_sta):
        self.n_sta = n_sta

    def gen_action(self,state):
        ret = np.random.uniform(0,1,(self.n_sta,self.n_sta))
        np.fill_diagonal(ret,0)
        return to_tensor(ret)

a = test_actor(e.get_n_sta())
e.set_actor(a)
n_step = 100000

model = gnn_edge_label(0)
model.DEBUG_STEP = 100
model.DEBUG = True
for i in range(n_step):
    e.init_env()
    sample = e.step(no_run=True)
    model.step([sample])
