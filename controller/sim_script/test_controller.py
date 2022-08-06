import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
from sim_src.controller.controller import controller
from sim_src.learning.model import model
from sim_src.memory.replay_memory import memory

import os
print(os.getcwd())
# os.system('echo $HOME/wifi-ai/ns-3-dev')
os.system('cd '+"$HOME/wifi-ai/ns-3-dev")
print(os.getcwd())
os.chdir(os.path.expandvars("$HOME/wifi-ai/ns-3-dev"))
c = controller(0,5,1000)
c.config()
c.run_rn()
c.lm.input_rn_obs(c.obs)
c.start()
c.join()
#

