import time
from threading import Thread
import numpy as np
from typing import List

from sim_src.learning.model import model
from sim_src.memory.replay_memory import memory
from sim_src.sim_ctrl.dt_ctrl import ns3_dt_env
from sim_src.sim_ctrl.rn_ctrl import ns3_rn_env


class controller(Thread):
    PORT_BASE = 5000
    D_TIME = 0.01

    def __init__(self, id, n_concurrent_dt, n_total_dt):
        Thread.__init__(self)
        self.id = id

        self.n_sta = 0
        self.rm:memory = None
        self.lm:model = None

        self.rn = None
        self.obs = None

        self.n_concurrent_dt = n_concurrent_dt
        self.n_total_dt = n_total_dt
        self.dt_list:List[ns3_dt_env] = []
        self.ports_availability = [True for x in range(self.n_concurrent_dt)]

        self.config()

        self.step_counter = 0
        self.dt_counter = 0

    def config(self):
        self.n_sta = 10
        self.rm = memory(0)
        self.lm = model(0,n_sta=10)

    def input_rn_obs(self, obs):
        self.obs = obs

    def run(self):
        self.dt_counter = 0
        while self.dt_counter < self.n_total_dt:
            ## check running dts
            n_dt_now = self._check_dt_list()
            if n_dt_now < self.n_concurrent_dt:
                print("current n dt", n_dt_now)
                dt = self._get_dt(self.dt_counter)
                self._add_dt(dt)
                self.dt_counter += 1
                self.step()
            time.sleep(self.D_TIME)
        self._wait_dt_end()

    def run_rn(self):
        self.rn = ns3_rn_env(0,n_sta=self.n_sta)
        self.rn.path_to_ns3 = "$HOME/wifi-ai/ns-3-dev"
        self.rn.program_name = "rn-4ap-ksta-out-gain-pm"
        self.rn.set_port(3000)
        self.rn.start()
        self.rn.join()
        self.obs = self.rn.get_rn_observation()

    def _wait_rn_end(self):
        pass

    def _wait_dt_end(self):
        for t in self.dt_list:
            t.join()

    def _check_dt_list(self):
        ret = 0
        new_list:List[Thread] = []
        self.ports_availability = [True for x in range(self.n_concurrent_dt)]
        for t in self.dt_list:
            if t.is_alive():
                ret += 1
                new_list.append(t)
                self.ports_availability[t.port-self.PORT_BASE] = False
        self.dt_list.clear()
        self.dt_list = new_list
        return ret

    def _add_dt(self, dt):
        self.dt_list.append(dt)

    def _get_dt(self, id):
        dt = ns3_dt_env(id, n_sta=self.n_sta)
        dt.set_replay_memory(self.rm)
        dt.set_port(self._get_port())
        dt.input_rn_obs(self.obs)
        dt.input_dt_inference(self.lm.get_dt_inference())
        dt.path_to_ns3 = "$HOME/wifi-ai/ns-3-dev"
        dt.program_name = "dt-4ap-ksta-out-pm-in-gains"
        dt.start()
        return dt

    def _get_port(self):
        for x in range(self.n_concurrent_dt):
            if self.ports_availability[x]:
                self.ports_availability[x] = False
                return x + self.PORT_BASE
        raise Exception("port error")




    def step(self):
        self.step_counter += 1
        sample = self.rm.async_sample()
        if sample:
            self.lm.step(sample)
