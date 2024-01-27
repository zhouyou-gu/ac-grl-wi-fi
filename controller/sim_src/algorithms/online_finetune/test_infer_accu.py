import os
import random
from os.path import expanduser
import time

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx

from sim_src.edge_label.model.complete_model import complete_model
from sim_src.edge_label.model.infer_then_label_model import infer_then_label_model
from sim_src.edge_label.model.online_infer_model import online_infer_model
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT, ParameterConfig, to_device, to_tensor

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

OUT_FOLDER = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

from working_dir_path import *
ns3_path = get_ns3_path()

e = sim_env(id=random.randint(40,200))
e.PROG_PATH = ns3_path
e.PROG_NAME = "wifi-ai/env"
e.DEBUG = True

n_step = 1000
model = online_infer_model(0)
model.DEBUG_STEP = 10
model.DEBUG = True
model.FAIRNESS_ALPHA = 100

INFER_PATH = "sim_src/algorithms/selected_nn/infer/infer.1999.pt"
ACTOR_PATH = "sim_src/algorithms/selected_nn/training/actor_target.599.pt"
CRITIC_PATH = "sim_src/algorithms/selected_nn/training/critic_target.599.pt"

path = os.path.join(get_controller_path(),INFER_PATH)
model.load_infer(path)
path = os.path.join(get_controller_path(),ACTOR_PATH)
model.load_actor(path)
path = os.path.join(get_controller_path(),CRITIC_PATH)
model.load_critic(path)


e.set_actor(model)
model.setup_p_infer(e.cfg.n_sta,e.formate_np_state(e.pl_model.convert_loss_sta_ap_threshold(e.cfg.loss_sta_ap)))
e.init_env()
for i in range(n_step):
    batch = []
    sample = e.step(run_ns3=True)
    batch.append(sample)
    model.step(batch)
    if (i+1) % 100 == 0:
        model.save(OUT_FOLDER,str(i))
    if (i+1) in [10, 100, 500, 1000]:
        e.save_np(OUT_FOLDER,str(i))