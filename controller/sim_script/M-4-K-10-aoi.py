import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
from ns3gym import ns3env

from sim_src.memory.replay_memory import memory
from sim_src.sim_ctrl.rn_ctrl import ns3_rn_env
from sim_src.sim_ctrl.dt_ctrl import ns3_dt_env

n_sta = 20
n_packet = 10000
interval_us = 50000
rn = ns3_rn_env(0,n_sta)
rn.path_to_ns3 = "/home/soyo/wifi-ai/ns-3-dev"
rn.program_name = "rn-4ap-ksta-out-gain-pm"
rn.set_port(3000)
rn.args['numPackets'] = n_packet
rn.args['interval_in_us'] = interval_us
rn.start()
rn.join()
# exit(0)
# print(rn.obs['loss_sta_sta'])

ls = rn.obs['loss_sta_sta']

tt = np.copy(ls)
np.fill_diagonal(tt,0.)
inference = np.triu(tt,k=1)
dt = ns3_dt_env(0,n_sta)
rm = memory()
dt.set_replay_memory(rm)
dt.input_rn_obs(rn.obs)
dt.input_dt_inference(inference)
dt.path_to_ns3 = "/home/soyo/wifi-ai/ns-3-dev"
dt.program_name = "dt-4ap-ksta-out-pm-in-gains"
dt.set_port(6000)
dt.args['numPackets'] = n_packet
dt.args['interval_in_us'] = interval_us
dt.start()
dt.join()
a = dt.obs['aoi']

tt = np.copy(ls)
tt[tt>=-72.] = 0.
tt[(tt < -72.) * (tt >= -92.)] = -82.
tt[tt<-92.] = -200.
np.fill_diagonal(tt,0.)
inference = np.triu(tt,k=1)
dt = ns3_dt_env(1,n_sta)
rm = memory()
dt.set_replay_memory(rm)
dt.input_rn_obs(rn.obs)
dt.input_dt_inference(inference)
dt.path_to_ns3 = "/home/soyo/wifi-ai/ns-3-dev"
dt.program_name = "dt-4ap-ksta-out-pm-in-gains"
dt.set_port(6000)
dt.args['numPackets'] = n_packet
dt.args['interval_in_us'] = interval_us
dt.start()
dt.join()
b = dt.obs['aoi']

tt = np.copy(ls)
tt[tt>=-92.] = 0.
tt[tt<-92.] = -200.
np.fill_diagonal(tt,0.)
inference = np.triu(tt,k=1)
dt = ns3_dt_env(2,n_sta)
rm = memory()
dt.set_replay_memory(rm)
dt.input_rn_obs(rn.obs)
dt.input_dt_inference(inference)
dt.path_to_ns3 = "/home/soyo/wifi-ai/ns-3-dev"
dt.program_name = "dt-4ap-ksta-out-pm-in-gains"
dt.set_port(6000)
dt.args['numPackets'] = n_packet
dt.args['interval_in_us'] = interval_us
dt.start()
dt.join()
c = dt.obs['aoi']

res = np.vstack((rn.obs['aoi'],a,b,c))
print(res.transpose())

# env = ns3env.Ns3Env(port=dt.port, startSim=False)


