import random

import numpy as np

from sim_src.edge_label.gw_cut import cut_into_2_k
from sim_src.ns3_ctrl.ns3_ctrl import build_ns3
from sim_src.ns3_ctrl.wifi_net_ctrl import sim_wifi_net, wifi_net_config
from sim_src.sim_env.path_loss import path_loss
from sim_src.util import to_tensor, to_numpy, StatusObject, counted

UDP_IP_WIFI_HEADER_SIZE = 64


class sim_env_to_controller_interface(StatusObject):
    def init_env(self):
        pass

    def step(self):
        pass

    def set_memory(self, memory):
        pass

    def set_actor(self, model):
        pass

class sim_env(sim_env_to_controller_interface):
    PROG_PATH = ''
    PROG_NAME = ''

    TWT_START_TIME = 10000000
    TWT_ASLOT_TIME = 10000

    def __init__(self, id=0, ns3_sim_time_s=1., app_packet_interval=20000, mac_packet_size=100, twt_log2_n_slot = 1):
        self.id = id
        self.ns3_sim_time_s = ns3_sim_time_s
        self.app_packet_interval = app_packet_interval
        assert mac_packet_size > UDP_IP_WIFI_HEADER_SIZE
        self.app_packet_size = mac_packet_size - UDP_IP_WIFI_HEADER_SIZE
        self.mac_packet_size = mac_packet_size
        self.twt_log2_n_slot = twt_log2_n_slot

        self.memory = None
        self.pl_model:path_loss = None
        self.ns3_env:sim_wifi_net = None
        self.actor = None

        self.MOVING_AVERAGE_DICT["reward"] = 0.
        self.MOVING_AVERAGE_DICT["min_reward"] = 0.

    def init_env(self):
        self.sample = {}

        self.pl_model = path_loss(n_sta=self.get_n_sta())
        self.ns3_env = sim_wifi_net(self.id)

        cfg = wifi_net_config()
        cfg.PROG_PATH = self.PROG_PATH
        cfg.PROG_NAME = self.PROG_NAME
        cfg.PROG_PORT = (self.N_STEP % 1000) + 5000 + self.id * 1000
        cfg.PROG_SEED = (self.N_STEP % 1000) + 5000
        cfg.PROG_TIME = self.ns3_sim_time_s

        cfg.id = self.id
        cfg.n_ap = self.pl_model.n_ap
        cfg.n_sta = self.pl_model.n_sta

        cfg.app_packet_interval = self.app_packet_interval
        cfg.app_packet_size = self.app_packet_size

        cfg.loss_ap_ap = self.pl_model.get_loss_ap_ap()
        cfg.loss_sta_ap = self.pl_model.get_loss_sta_ap()
        cfg.loss_sta_sta = self.pl_model.get_loss_sta_sta()

        state = self.pl_model.convert_loss_sta_ap_threshold(cfg.loss_sta_ap)
        # state = cfg.loss_sta_ap
        state = self.formate_np_state(state)
        action = self.gen_action(state)
        # print(action)
        twt_cfg = self.formate_np_action(action)

        cfg.twtstarttime = twt_cfg['twtstarttime']
        cfg.twtoffset = twt_cfg['twtoffset']
        cfg.twtduration = twt_cfg['twtduration']
        cfg.twtperiodicity = twt_cfg['twtperiodicity']

        self.ns3_env.set_config(cfg)

        self.sample['state'] = state
        self.sample['target'] = self.pl_model.convert_loss_sta_sta_binary(cfg.loss_sta_sta)
        self.sample['action'] = action
        self.sample['n_node'] = self.pl_model.n_sta

    @counted
    def step(self, no_run = False):
        if not no_run:
            self.ns3_env.start()
            self.ns3_env.join()
            rwd = self.ns3_env.get_return()
            self.sample['reward'] = self.formate_dict_reward(rwd)
        else:
            self.sample['reward'] = np.zeros((self.pl_model.n_sta, 1))

        assert self.sample['state'].shape == (self.pl_model.n_sta, self.pl_model.n_ap)
        assert self.sample['target'].shape == (self.pl_model.n_sta, self.pl_model.n_sta)
        assert self.sample['action'].shape == (self.pl_model.n_sta, self.pl_model.n_sta)
        assert self.sample['reward'].shape == (self.pl_model.n_sta, 1)

        if not np.isnan(np.sum(self.sample['reward'])):
            if self.memory and not np.isnan(np.sum(self.sample['reward'])):
                self.memory.step(self.sample.copy())
        else:
            print("nan in reward drop sample")

        return self.sample.copy()
    def set_memory(self, memory):
        self.memory = memory

    def set_actor(self, model):
        self.actor = model

    def gen_action(self, state):
        ## exploration?
        action = self.actor.gen_action(state)
        # print(action)
        action -= np.min(action[~np.eye(action.shape[0],dtype=bool)])
        action /= np.max(action[~np.eye(action.shape[0],dtype=bool)])
        np.fill_diagonal(action,0)
        return action

    def formate_np_state(self, state) -> np.array:
        ## normalization etc
        state /= (-self.pl_model.min_rssi_dbm)
        state -= 1.
        return state

    def formate_np_action(self, action) -> dict:
        ## action to twt list
        sta_twt_slot_id = cut_into_2_k(action,self.pl_model.n_sta,self.twt_log2_n_slot)
        self._printa(sta_twt_slot_id)

        twt_cfg = {}
        twt_cfg['twtstarttime'] = np.ones(self.pl_model.n_sta)*self.TWT_START_TIME
        twt_cfg['twtoffset'] = sta_twt_slot_id*self.TWT_ASLOT_TIME
        twt_cfg['twtduration'] = np.ones(self.pl_model.n_sta)*self.TWT_ASLOT_TIME
        twt_cfg['twtperiodicity'] = np.ones(self.pl_model.n_sta)*self.TWT_ASLOT_TIME * (2**self.twt_log2_n_slot)

        return twt_cfg

    def formate_dict_reward(self, reward) -> np.array:
        # print(reward,"+++++++++++++++++")
        ret = reward['thr']/(1e6/self.app_packet_interval)
        self._printa(ret)
        self._printa(self._moving_average("reward",np.mean(ret)))
        self._printa(self._moving_average("min_reward",np.min(ret)))

        return ret[:,np.newaxis]

    def get_n_sta(self):
        return 20

if __name__ == '__main__':
    build_ns3("/home/soyo/wifi-ai/ns-3-dev")
    # exit(0)
    e = sim_env(id=random.randint(40,60))
    e.PROG_PATH = "/home/soyo/wifi-ai/ns-3-dev"
    e.PROG_NAME = "wifi-ai/env"
    class test_actor:
        def __init__(self, n_sta):
            self.n_sta = n_sta

        def gen_action(self,state):
            ret = np.random.uniform(0,1,(self.n_sta,self.n_sta))
            np.fill_diagonal(ret,0)
            return to_tensor(ret)

    a = test_actor(e.get_n_sta())
    e.set_actor(a)
    e.init_env()
    e.ns3_env.DEBUG = True
    e.step()

