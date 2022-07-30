import time
from threading import Thread
import numpy as np
from typing import List

from sim_src.learning.model import model
from sim_src.memory.replay_memory import memory
from sim_src.sim_ctrl.dt_ctrl import ns3_dt_env

class controller(Thread):
    PORT_BASE = 5000
    D_TIME = 0.01

    def __init__(self, id, n_concurrent_dt, n_total_dt):
        Thread.__init__(self)
        self.id = id

        self.rm:memory = None
        self.lm = None

        self.n_concurrent_dt = n_concurrent_dt
        self.n_total_dt = n_total_dt
        self.dt_list:List[Thread] = []

        self.config()

        self.step_counter = 0

    def config(self):
        self.rm = memory()
        self.lm = model()

    def input_rn_obs(self, obs):
        self.obs = obs

    def run(self):
        n_dt_tot = 0
        while n_dt_tot < self.n_total_dt:
            ## check running dts
            n_dt_now = self._check_dt_list()
            if n_dt_now < self.n_concurrent_dt:
                dt = self._get_dt(n_dt_now)
                self._add_dt(dt)
                self.n_total_dt += 1
            self.step()
            time.sleep(self.D_TIME)

        self._wait_dt_end()

    def _wait_dt_end(self):
        for t in self.dt_list:
            t.join()

    def _check_dt_list(self):
        ret = 0
        new_list:List[Thread] = []
        for t in self.dt_list:
            if t.is_alive():
                ret += 1
                new_list.append(t)
        self.dt_list.clear()
        self.dt_list = new_list
        return ret

    def _add_dt(self, dt):
        self.dt_list.append(dt)

    def _get_dt(self, id):
        dt = ns3_dt_env(id)
        dt.set_replay_memory(self.rm)
        dt.set_port(id%self.n_concurrent_dt+self.PORT_BASE)
        dt.input_rn_obs(self.obs)
        dt.input_dt_inference(self.lm.get_dt_inference())
        dt.start()
        return dt

    def step(self):
        self.step_counter += 1
        self.lm.step(self.rm.sample())
