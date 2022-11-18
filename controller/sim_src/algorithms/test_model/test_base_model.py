import time

import numpy as np
import torch

from sim_src.algorithms.test_model.run_test import run_test
from sim_src.edge_label.model.base_model import base_model
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

model_class = base_model
n_test = 1
for ALPHA in [10., 1., 0.]:
    for i in range(n_test):
        print(model_class.__name__,ALPHA)
        run_test(ALPHA,model_class,GET_LOG_PATH_FOR_SIM_SCRIPT(__file__))
        time.sleep(5)
