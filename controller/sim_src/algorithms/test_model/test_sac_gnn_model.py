import time

import numpy as np
import torch

from sim_src.algorithms.test_model.run_test import run_test
from sim_src.other_methods.sac_gnn_model import sac_gnn_model, sim_env_sac_gnn
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

model_class = sac_gnn_model
run_test(model_class,GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),ec=sim_env_sac_gnn)
time.sleep(5)
