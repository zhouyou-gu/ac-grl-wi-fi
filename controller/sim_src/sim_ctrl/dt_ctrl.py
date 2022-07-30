import subprocess
from threading import Thread

import numpy as np
from ns3gym import ns3env

from sim_src.sim_ctrl.ns3_ctrl import run_ns3, ns3_env


class dt_env:
    def _return_dt_observation(self, obs):
        raise NotImplementedError

    def set_replay_memory(self, rm):
        raise NotImplementedError

    def input_rn_obs(self, obs):
        raise NotImplementedError

    def input_dt_inference(self, inference):
        raise NotImplementedError


class ns3_dt_env(dt_env, ns3_env, Thread):
    N_AP = 4
    def __init__(self, id, n_sta = 10):
        Thread.__init__(self)
        self.id = id
        self.seed = id + 1000
        self.n_sta = n_sta

        self.replay_memory = None

        self.path_to_ns3 = None
        self.program_name = None
        self.port = None
        self.args = {}
        self.args['n_sta'] = self.n_sta

        self.ns3_proc = None

        self.G_ap_ap = None
        self.G_sta_ap = None
        self.G_sta_sta = None
        self.H = None
        self.pm = None

        self.obs = None

        self.proc = None

    def _return_dt_observation(self, obs):
        print(obs.keys())
        self.obs= {}
        for k in obs.keys():
            self.obs[k] = np.array(obs[k][:])
        print(self.obs)

    def input_rn_obs(self, obs):
        self.G_ap_ap = np.reshape(obs['loss_ap_ap'],(1,self.N_AP*self.N_AP))
        self.G_sta_ap = np.reshape(obs['loss_sta_ap'],(1,self.n_sta*self.N_AP))

    def input_dt_inference(self, inference):
        self.H = inference
        self.G_sta_sta = np.zeros((self.n_sta,self.n_sta))
        for i in range(self.n_sta):
            for j in range(i+1,self.n_sta):
                if self.H[i,j] == 1:
                    self.G_sta_sta[i,j] = 0.
                    self.G_sta_sta[j,i] = 0.
                else:
                    self.G_sta_sta[i,j] = -200.
                    self.G_sta_sta[j,i] = -200.

        self.G_sta_sta = np.reshape(self.G_sta_sta, (1,self.n_sta*self.n_sta))

    def set_replay_memory(self, rm):
        self.replay_memory = rm

    def set_port(self, port):
        self.port = port

    def run(self):
        # self.proc = self._run_ns3_proc()
        env = ns3env.Ns3Env(port=self.port, startSim=False)
        # try:
        obs = env.reset()
        print("here",obs)
        obs, reward, done, info = env.step(self._generate_dt_configure())
        self._return_dt_observation(obs)
        # except Exception as e:
        #     print("Error", str(e))
        # finally:
        env.close()
        print("ns3 gym dt agent",self.id,"is done")
        # self.proc.wait()

    def _generate_dt_configure(self):
        cfg = {}
        cfg['loss_ap_ap'] = self.G_ap_ap.flatten().tolist()
        cfg['loss_sta_ap'] = self.G_sta_ap.flatten().tolist()
        cfg['loss_sta_sta'] = self.G_sta_sta.flatten().tolist()
        print(cfg)
        return cfg

    def _run_ns3_proc(self) -> subprocess.Popen:
        return run_ns3(path_to_ns3=self.path_to_ns3, program_name=self.program_name, port=self.port, sim_seed=self.seed, sim_args=self.args, debug=True)







if __name__ == '__main__':
    dt = ns3_dt_env(0)
    dt.path_to_ns3 = "/home/soyo/wifi-ai/ns-3-dev"
    dt.program_name = "dt-4ap-ksta-out-pm-in-gains"
    dt.set_port(5000)
    dt.start()