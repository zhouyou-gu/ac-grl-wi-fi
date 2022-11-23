import os
import random
from os.path import expanduser
import time

import numpy as np
import torch

from sim_src.edge_label.model.complete_model import complete_model
from sim_src.edge_label.model.online_model import online_model
from sim_src.sim_env.sim_env import sim_env, sim_env_online
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT, ParameterConfig

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

OUT_FOLDER = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

ns3_path = os.path.join(expanduser("~"),"wifi-ai/ns-3-dev")
e = sim_env_online(id=random.randint(40,200))
e.PROG_PATH = ns3_path
e.PROG_NAME = "wifi-ai/env"
e.DEBUG = True

n_step = 1000
model = online_model(0)
model.DEBUG_STEP = 10
model.DEBUG = True

nn_path = os.path.join(expanduser("~"),"wifi-ai/controller/sim_src/algorithms/selected_nn/test_complete_model-2022-November-23-09-04-43-ail")

model.load_actor(os.path.join(nn_path,"actor_target.299.pt"))
model.load_infer(os.path.join(nn_path,"infer_target.299.pt"))
model.load_critic(os.path.join(nn_path,"critic_target.299.pt"))
# print(model.actor_target)
cfg = ParameterConfig()
# cfg['ALPHA'] = ALPHA
# model.FAIRNESS_ALPHA = cfg['ALPHA']
cfg.save(OUT_FOLDER,"NaN")

e.set_actor(model)
for i in range(n_step):
    batch = []
    e.init_env()
    sample = e.step(no_run=False)
    batch.append(sample)
    model.step(batch)
    if (i+1) % 100 == 0:
        model.save(OUT_FOLDER,str(i))
    if (i+1) in [10, 100, 500, 1000]:
        e.save_np(OUT_FOLDER,str(i))