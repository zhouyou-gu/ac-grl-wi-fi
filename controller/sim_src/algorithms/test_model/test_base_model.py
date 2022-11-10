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

for ALPHA in [0., 1., 2., 5., 10.]:
    for i in range(10):
        run_test(ALPHA,model_class,GET_LOG_PATH_FOR_SIM_SCRIPT(__file__))
        time.sleep(5)
