import time

import numpy as np
import torch

from sim_src.algorithms.test_model.run_test import run_test
from sim_src.edge_label.model.infer_then_label_model import infer_then_label_model
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

np.set_printoptions(threshold=5)
np.set_printoptions(linewidth=1000)

torch.set_printoptions(threshold=5)
torch.set_printoptions(linewidth=1000)

model_class = infer_then_label_model
run_test(model_class,GET_LOG_PATH_FOR_SIM_SCRIPT(__file__),load_infer=True)
time.sleep(5)
