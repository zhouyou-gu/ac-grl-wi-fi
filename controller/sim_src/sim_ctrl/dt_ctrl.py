import os
import subprocess
from threading import Thread
from ns3gym import ns3env
from ns3_ctrl import run_ns3, ns3_env


class dt_env:
    def _return_dt_observation(self):
        raise NotImplementedError

    def set_replay_memory(self, rm):
        raise NotImplementedError

    def input_rn_observation(self, obs):
        raise NotImplementedError

    def input_dt_inference(self, inference):
        raise NotImplementedError


class ns3_dt_env(dt_env, ns3_env, Thread):
    def __init__(self, id):
        Thread.__init__(self)
        self.id = id
        self.seed = id + 1000

        self.replay_memory = None

        self.path_to_ns3 = None
        self.program_name = None
        self.port = None
        self.args = {}

        self.ns3_proc = None

        self.G_ap_ap = None
        self.G_sta_ap = None
        self.H = None
        self.pm = None


    def _return_dt_observation(self):
        raise NotImplementedError

    def input_rn_obs(self, obs):
        self.G_ap_ap = obs.G_ap_ap
        self.G_sta_ap = obs.G_sta_ap

    def input_dt_inference(self, inference):
        raise NotImplementedError

    def set_replay_memory(self, rm):
        self.replay_memory = rm

    def set_port(self, port):
        self.port = port

    def run(self):
        env = ns3env.Ns3Env(port=self.port, startSim=False)
        self.proc = self._run_ns3_proc()
        try:
            env.reset()
            obs, reward, done, info = env.step(self._generate_dt_configure())
            self.replay_memory.add_obs(obs)
        except Exception:
            print("Error")
        finally:
            env.close()
            print("ns3 gym dt agent",self.id,"is done")
        self.proc.wait()

    def _generate_dt_configure(self):
        raise NotImplementedError

    def _run_ns3_proc(self) -> subprocess.Popen:
        raise NotImplementedError







if __name__ == '__main__':
    pro = subprocess.Popen("sleep 100", shell=True, stdout=None, stderr=None)
    # os.system("sleep 100")
    # pro.wait()
