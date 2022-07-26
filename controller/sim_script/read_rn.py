import argparse
from ns3gym import ns3env

startSim = False
port = 5555

env = ns3env.Ns3Env(port=port, startSim=startSim)

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

try:
    obs = env.reset()
    print("---obs: ", obs)
except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    env.close()
    print("Done")