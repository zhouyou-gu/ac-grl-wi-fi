import subprocess
from threading import Thread

import numpy as np
from ns3gym import ns3env

from sim_src.ns3_ctrl.ns3_ctrl import run_ns3, ns3_env

class wifi_net_config:
    def __init__(self):
        self.PROG_PATH = ""
        self.PROG_NAME = ""
        self.PROG_PORT = 5000
        self.PROG_SEED = 0
        self.PROG_TIME = 10

        self.id = 0
        self.n_ap = 0
        self.n_sta = 0

        self.app_packet_interval = 40000
        self.app_packet_size = 20

        self.loss_ap_ap = None
        self.loss_sta_ap = None
        self.loss_sta_sta = None

        self.twtstarttime = None
        self.twtoffset = None
        self.twtduration = None
        self.twtperiodicity = None


class wifi_net_instance:
    def set_config(self, config):
        pass

    def get_return(self):
        pass

class sim_wifi_net(wifi_net_instance, ns3_env, Thread):
    DEBUG = False
    def __init__(self, id):
        Thread.__init__(self)
        self.id = id

        self.cfg:wifi_net_config = wifi_net_config()
        self.proc = None

        self.ret = None

    def set_config(self, cfg:wifi_net_config):
        self.cfg = cfg

    def get_return(self):
        return self.ret

    def run(self):
        print("ns3 gym dt agent",self.id,"starts")
        self.proc = self._run_ns3_proc()
        this_env = ns3env.Ns3Env(port=self.cfg.PROG_PORT, startSim=False)
        try:
            obs = this_env.reset()
            obs, reward, done, info = this_env.step(self._gen_ns3gym_act())
            self._ret_ns3gym_obs(obs)
        except Exception as e:
            print("sim_wifi_net run Error", str(e))
        finally:
            this_env.close()
        print("ns3 gym dt agent",self.id,"is done")
        self.proc.wait()

    def _gen_ns3gym_act(self):
        act = {}
        act['loss_ap_ap'] = self.cfg.loss_ap_ap.flatten().tolist()
        act['loss_sta_ap'] = self.cfg.loss_sta_ap.flatten().tolist()
        act['loss_sta_sta'] = self.cfg.loss_sta_sta.flatten().tolist()
        act['twtstarttime'] = self.cfg.twtstarttime.flatten().tolist()
        act['twtoffset'] = self.cfg.twtoffset.flatten().tolist()
        act['twtduration'] = self.cfg.twtduration.flatten().tolist()
        act['twtperiodicity'] = self.cfg.twtperiodicity.flatten().tolist()
        # print(act)
        return act

    def _ret_ns3gym_obs(self, obs):
        assert self.ret is None, "this ns3 instance already has return value"
        self.ret= {}
        for k in obs.keys():
            self.ret[k] = np.array(obs[k][:])

    def _get_ns3_args(self):
        args = {}
        args['n_ap'] = self.cfg.n_ap
        args['n_sta'] = self.cfg.n_sta
        args['interval_in_us'] = self.cfg.app_packet_interval
        args['packetSize'] = self.cfg.app_packet_size
        return args

    def _run_ns3_proc(self) -> subprocess.Popen:
        path = self.cfg.PROG_PATH
        name = self.cfg.PROG_NAME
        port = self.cfg.PROG_PORT
        seed = self.cfg.PROG_SEED
        time = self.cfg.PROG_TIME
        args = self._get_ns3_args()
        return run_ns3(path_to_ns3=path, program_name=name, port=port, sim_seed=seed, sim_time=time, sim_args=args, debug=self.DEBUG)