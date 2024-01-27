import os
import random
from os.path import expanduser

from sim_src.edge_label.model.itl_bidirection_interference import itl_bidirection_interference
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import ParameterConfig, StatusObject, GET_LOG_PATH_FOR_SIM_SCRIPT

INFER_PATH = "sim_src/algorithms/selected_nn/infer/infer.1999.pt"
ACTOR_PATH = "sim_src/algorithms/selected_nn/training/actor_target.599.pt"
CRITIC_PATH = "sim_src/algorithms/selected_nn/training/critic_target.599.pt"

StatusObject.DISABLE_ALL_DEBUG = False
OUT_FOLDER = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

# the number of users and the number of groups can be varied by setting different twt_log2_n_slot and n_user in sim_env
# (default is 4 groups and 20 users, see sim_env)
# for example, to set 8 groups and 40 users, do
# e = sim_env(id=random.randint(40,200),twt_log2_n_slot=3,n_user=(40,40))

e = sim_env(id=random.randint(40,200))

from working_dir_path import *
ns3_path = get_ns3_path()

e.PROG_PATH = ns3_path
e.PROG_NAME = "wifi-ai/env"
e.DEBUG = True

n_step = 1000
batch_size = 1
model = itl_bidirection_interference(0)


path = os.path.join(get_controller_path(),INFER_PATH)
model.load_infer(path)
path = os.path.join(get_controller_path(),ACTOR_PATH)
model.load_actor(path)
path = os.path.join(get_controller_path(),CRITIC_PATH)
model.load_critic(path)

model.EXPLORATION = False
model.DEBUG_STEP = 10
model.DEBUG = True

cfg = ParameterConfig()
cfg['ALPHA'] = 100
model.FAIRNESS_ALPHA = cfg['ALPHA']
cfg.save(OUT_FOLDER,"NaN")

e.set_actor(model)
for i in range(n_step):
    batch = []
    for j in range(batch_size):
        e.init_env()
        sample = e.step(run_ns3=True)
        batch.append(sample)
    # model.step(batch)
    # if (i+1) % 100 == 0:
    #     model.save(OUT_FOLDER,str(i))
    if (i+1) in [10, 100, 500, 1000]:
        e.save_np(OUT_FOLDER,str(i))


