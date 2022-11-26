import os
import random
from os.path import expanduser

import numpy as np
import torch
from sim_src.edge_label.model.base_model import base_model
from sim_src.ns3_ctrl.ns3_ctrl import build_ns3
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import get_current_time_str, ParameterConfig


np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

def run_sim(ALPHA=0.):
    OUT_FOLDER = os.path.splitext(os.path.basename(__file__))[0] + "-" + get_current_time_str()
    OUT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), OUT_FOLDER)

    home = expanduser("~")
    ns3_path = os.path.join(home,"wifi-ai/ns-3-dev")
    build_ns3(ns3_path)
    # exit(0)
    e = sim_env(id=random.randint(40,60))
    e.PROG_PATH = ns3_path
    e.PROG_NAME = "wifi-ai/env"
    e.DEBUG = True

    n_step = 1000
    batch_size = 1
    model = base_model(0)
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
            sample = e.step(run_ns3=True)
            batch.append(sample)
        model.step(batch)
        if (i+1) % 100 == 0:
            model.save(OUT_FOLDER,str(i))
            e.save_np(OUT_FOLDER,str(i))

for A in [0., 1., 2., 5., 10.]:
    print("hello")
    run_sim(A)