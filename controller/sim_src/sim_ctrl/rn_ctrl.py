import os
import subprocess
from threading import Thread

import numpy as np
from ns3gym import ns3env

from sim_src.sim_ctrl.ns3_ctrl import run_ns3, ns3_env


class rn_env:
    def get_rn_observation(self):
        raise NotImplementedError


class ns3_rn_env(rn_env, ns3_env, Thread):
    N_AP = 4
    def __init__(self, id, n_sta = 10,need_rn_action = False):
        Thread.__init__(self)
        self.id = id
        self.seed = id + 1000
        self.n_sta = n_sta
        self.need_rn_action = need_rn_action

        self.obs = None

        self.path_to_ns3 = None
        self.program_name = None
        self.port = None
        self.args = {}
        self.args['n_sta'] =self.n_sta

        self.ns3_proc = None
        self.rn_config = None

        self.proc = None

    def set_port(self, port):
        self.port = port

    def run(self):
        self.proc = self._run_ns3_proc()
        env = ns3env.Ns3Env(port=self.port, startSim=False)
        try:
            obs = env.reset()
            self._rn_obs(obs)
            while self.need_rn_action:
                obs, reward, done, info = env.step(self._get_rn_action())
                self._rn_obs(obs)
                if done:
                    break
        except Exception as e:
            print("Error", str(e))
        finally:
            env.close()
            print("ns3 gym rn agent",self.id,"is done",)
        self.proc.wait()

    def _rn_obs(self, obs):
        print(obs.keys())
        self.obs={}
        for k in obs.keys():
            self.obs[k] = np.array(obs[k][:])
            # print(k,self.obs[k].shape)
        self.obs['loss_ap_ap'] = np.resize(self.obs['loss_ap_ap'],(self.N_AP,self.N_AP))
        self.obs['loss_sta_ap'] = np.resize(self.obs['loss_sta_ap'],(self.N_AP,self.n_sta))
        # print(self.obs['loss_sta_ap'].shape)
        # print(self.obs['loss_sta_ap'])
        # print(self.obs['loss_ap_ap'].shape)
        # print(self.obs['aoi'])

    def get_rn_observation(self):
        return self.obs

    def _get_rn_action(self):
        pass

    def _run_ns3_proc(self) -> subprocess.Popen:
        return run_ns3(path_to_ns3=self.path_to_ns3, program_name=self.program_name, port=self.port, sim_seed=self.seed, sim_args=self.args)







if __name__ == '__main__':
    a = np.array([1,2,3,4,5,6,7,8,])
    b = np.resize(a, (2,4))
    print(b)
    print(np.resize(b, (1,8)))


    exit(0)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    rn = ns3_rn_env(0)
    rn.path_to_ns3 = "/home/soyo/wifi-ai/ns-3-dev"
    rn.program_name = "rn-4ap-ksta-out-gain-pm"
    rn.set_port(5000)
    rn.run()
    # os.system("sleep 100")
    # pro.wait()
