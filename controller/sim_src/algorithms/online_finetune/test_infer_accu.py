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

ns3_path = os.path.join(expanduser("~"),"wifi-ai/ns-3-dev")
e = sim_env(id=random.randint(40,200))
e.PROG_PATH = ns3_path
e.PROG_NAME = "wifi-ai/env"
e.DEBUG = True

n_step = 1000
model = online_infer_model(0)
model.DEBUG_STEP = 10
model.DEBUG = True
model.FAIRNESS_ALPHA = 100

nn_path = os.path.join(expanduser("~"),"wifi-ai/controller/sim_src/algorithms/selected_nn/test_infer_then_label_model-2022-November-24-23-07-13-ail")
infer_path = os.path.join(expanduser("~"),"wifi-ai/controller/sim_src/algorithms/selected_nn/infer/infer.999.pt")

model.load_actor(os.path.join(nn_path,"actor_target.499.pt"))
model.load_infer(infer_path)
model.load_critic(os.path.join(nn_path,"critic_target.499.pt"))



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