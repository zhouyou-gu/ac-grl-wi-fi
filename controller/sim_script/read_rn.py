import argparse
from ns3gym import ns3env

startSim = False
port = 5000

env = ns3env.Ns3Env(port=port, startSim=startSim)

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

try:
    obs = env.reset()
    print("---obs: ", obs['loss_sta_ap'].__class__)
    print("---obs: ", len(obs['loss_sta_ap']))
    for i in range(len(obs['loss_sta_ap'])):
        print(obs['loss_sta_ap'][i])
    # print("---obs: ", len(obs['loss_sta_ap']))
    # print("---obs: ", obs['loss_sta_ap'].__class__)
    # print("---obs: ", obs['loss_sta_ap'].__class__)
    # print("---obs: ", obs['loss_sta_ap'].__class__)
except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    env.close()
    print("Done")