import os
import subprocess
from threading import Thread
from ns3gym import ns3env
from ns3_ctrl import run_ns3, ns3_env


class rn_env:
    def get_rn_observation(self):
        raise NotImplementedError


class ns3_rn_env(rn_env, ns3_env, Thread):
    def __init__(self, id, need_rn_action = False):
        Thread.__init__(self)
        self.id = id
        self.seed = id + 1000
        self.need_rn_action = need_rn_action

        self.obs = None

        self.path_to_ns3 = None
        self.program_name = None
        self.port = None
        self.args = {}

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
        except Exception:
            print("Error")
        finally:
            env.close()
            print("ns3 gym rn agent",self.id,"is done",)
        self.proc.wait()

    def _rn_obs(self, obs):
        self.obs = obs

    def get_rn_observation(self):
        return self.obs

    def _get_rn_action(self):
        pass

    def _run_ns3_proc(self) -> subprocess.Popen:
        print("hello")
        return run_ns3(path_to_ns3=self.path_to_ns3, program_name=self.program_name, port=self.port, sim_seed=self.seed, sim_args=self.args)







if __name__ == '__main__':
    rn = ns3_rn_env(0)
    rn.path_to_ns3 = "/home/soyo/wifi-ai/ns-3-dev"
    rn.program_name = "rn-4ap-ksta-out-gain-pm"
    rn.set_port(5000)
    rn.run()
    # os.system("sleep 100")
    # pro.wait()
