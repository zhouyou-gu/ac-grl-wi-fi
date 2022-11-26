import os
import random
from os.path import expanduser
import time

import numpy as np
import torch

from sim_src.algorithms.test_model.run_test import run_test
from sim_src.other_methods.bianchi_model import bianchi_model, sim_env_bianchi_model
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT, ParameterConfig

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

mc = bianchi_model
ec = sim_env_bianchi_model

OUT_FOLDER = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)

ns3_path = os.path.join(expanduser("~"),"wifi-ai/ns-3-dev")
e = ec(id=random.randint(40,200))
e.PROG_PATH = ns3_path
e.PROG_NAME = "wifi-ai/env"
e.DEBUG = True

n_step = 1000
model = bianchi_model(0,global_allocation=False)
model.DEBUG_STEP = 10
model.DEBUG = True

cfg = ParameterConfig()
# cfg['ALPHA'] = ALPHA
# model.FAIRNESS_ALPHA = cfg['ALPHA']
cfg.save(OUT_FOLDER,"NaN")

e.set_actor(model)
for i in range(n_step):
    e.init_env()
    sample = e.step(run_ns3=True)