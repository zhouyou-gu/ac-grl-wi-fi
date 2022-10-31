import os
import random

import numpy as np
import torch

from sim_src.edge_label.gw_cut import cut_into_2_k
from sim_src.edge_label.model import gnn_edge_label
from sim_src.ns3_ctrl.ns3_ctrl import build_ns3
from sim_src.ns3_ctrl.wifi_net_ctrl import sim_wifi_net, wifi_net_config
from sim_src.sim_env.path_loss import path_loss
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import to_tensor, to_numpy, get_current_time_str, ParameterConfig

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

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
    model = gnn_edge_label(0)
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

for A in [0., 1., 2., 5., 10.]:
    run_sim(A)