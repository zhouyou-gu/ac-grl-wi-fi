import numpy as np
from ns3gym import ns3env

from sim_src.sim_ctrl.rn_ctrl import ns3_rn_env
from sim_src.sim_ctrl.dt_ctrl import ns3_dt_env

rn = ns3_rn_env(0)
rn.path_to_ns3 = "/home/soyo/wifi-ai/ns-3-dev"
rn.program_name = "rn-4ap-ksta-out-gain-pm"
rn.set_port(3000)
rn.start()
rn.join()
print(rn.obs['aoi'])
obs = rn.get_rn_observation()
a = np.ones((10,10))
b = a*1.
inference = np.random.binomial(1,b)
np.fill_diagonal(inference,0.)

dt = ns3_dt_env(0)
dt.input_rn_obs(obs)
dt.input_dt_inference(inference)
dt.path_to_ns3 = "/home/soyo/wifi-ai/ns-3-dev"
dt.program_name = "dt-4ap-ksta-out-pm-in-gains"
dt.set_port(6000)
dt.start()

# env = ns3env.Ns3Env(port=dt.port, startSim=False)


