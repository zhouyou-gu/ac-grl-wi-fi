import time

import numpy as np
import torch

from sim_src.algorithms.test_model.run_test import run_test
from sim_src.other_methods.dac_gnn_model import dac_gnn_model, sim_env_dac_gnn
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

model_class = dac_gnn_model
run_test(model_class,GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),ec=sim_env_dac_gnn,ALPHA=100)
time.sleep(5)
