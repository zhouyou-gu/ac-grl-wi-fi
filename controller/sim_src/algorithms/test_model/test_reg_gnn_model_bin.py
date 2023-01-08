import time

import numpy as np
import torch

from sim_src.algorithms.test_model.run_test import run_test
from sim_src.other_methods.reg_gnn_model_bin import reg_gnn_model_bin, sim_env_reg_gnn_bin
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

model_class = reg_gnn_model_bin
run_test(model_class,GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),ec=sim_env_reg_gnn_bin)
time.sleep(5)
