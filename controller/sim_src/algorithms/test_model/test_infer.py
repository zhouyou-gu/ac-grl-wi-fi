import os
import time
from os.path import expanduser
import random

import numpy as np
import torch

from sim_src.algorithms.test_model.run_test import run_test
from sim_src.edge_label.model.infer_model import infer_model
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)
OUT_FOLDER = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

class sim_env_tmp(sim_env):
    def format_act_to_sta_twt_idx(self, action):
        return np.zeros(self.pl_model.n_sta)


ns3_path = os.path.join(expanduser("~"),"wifi-ai/ns-3-dev")
e = sim_env_tmp(id=random.randint(40,200),noise=0.)
e.PROG_PATH = ns3_path
e.PROG_NAME = "wifi-ai/env"
e.DEBUG = True

n_step = 1000
model = infer_model(0)
model.DEBUG_STEP = 10
model.DEBUG = True

e.set_actor(model)


batch_size = 20
for i in range(n_step):
    batch = []
    for ii in range(batch_size):
        e.init_env()
        sample = e.step(run_ns3=False)
        batch.append(sample)
    model.step(batch)
    if (i+1) % 100 == 0:
        model.save(OUT_FOLDER,str(i))
    if (i+1) in [10, 100, 500, 1000]:
        e.save_np(OUT_FOLDER,str(i))
