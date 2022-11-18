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
n_test = 1
for ALPHA in [10., 1., 0.]:
    for i in range(n_test):
        print(model_class.__name__,ALPHA)
        run_test(ALPHA,model_class,GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),ec=sim_env_dac_gnn)
        time.sleep(5)