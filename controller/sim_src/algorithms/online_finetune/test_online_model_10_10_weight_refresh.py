import os
import random
from os.path import expanduser
import numpy as np
import torch
from sim_src.edge_label.model.online_actor_with_w_refresh_model import online_actor_with_w_refresh_model
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import ParameterConfig, StatusObject, GET_LOG_PATH_FOR_SIM_SCRIPT

INFER_PATH = "wifi-ai/controller/sim_src/algorithms/selected_nn/infer/infer.1999.pt"
ACTOR_PATH = "wifi-ai/controller/sim_src/algorithms/selected_nn/training/actor_target.599.pt"
CRITIC_PATH = "wifi-ai/controller/sim_src/algorithms/selected_nn/training/critic_target.599.pt"

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

StatusObject.DISABLE_ALL_DEBUG = True
StatusObject.MOVING_AVERAGE_TIME_WINDOW = 20
OUT_FOLDER = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

from sim_src.algorithms.online_finetune.shared_config import user_mobility
e = sim_env(id=random.randint(40,200),ns3_sim_time_s=2.,n_user=(10,10),mobility_in_meter_per_sec=user_mobility)
OUT_FOLDER = OUT_FOLDER + '.'+ str(int(user_mobility))

ns3_path = os.path.join(expanduser("~"),"wifi-ai/ns-3-dev")
e.PROG_PATH = ns3_path
e.PROG_NAME = "wifi-ai/env"
e.DEBUG = True

n_step = 200
batch_size = 1
model = online_actor_with_w_refresh_model(0)


path = os.path.join(expanduser("~"),INFER_PATH)
model.load_infer(path)
path = os.path.join(expanduser("~"),ACTOR_PATH)
model.load_actor(path)
path = os.path.join(expanduser("~"),CRITIC_PATH)
model.load_critic(path)

model.EXPLORATION = False
model.DEBUG_STEP = 10
model.DEBUG = True

cfg = ParameterConfig()
cfg['ALPHA'] = 10
model.FAIRNESS_ALPHA = cfg['ALPHA']
cfg.save(OUT_FOLDER,str(int(user_mobility)))

e.set_actor(model)
e.init_env(np.random.randint(0,10000))
model.setup_weight(e.formate_np_state(e.pl_model.convert_loss_sta_ap_threshold(e.cfg.loss_sta_ap)))
for i in range(n_step):
    batch = []
    sample = e.step(run_ns3=True)
    batch.append(sample)
    model.step(batch)
    # if (i+1) % 100 == 0:
        # model.save(OUT_FOLDER,str(i))
    if (i+1) == n_step:
        e.save_np(OUT_FOLDER,str(i))