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

    TWT_START_TIME = 500000
    TWT_ASLOT_TIME = 10000
    TWT_START_TIME_OFFSET = 10000000

    def __init__(self, id=0, ns3_sim_time_s=5., app_packet_interval=20000, mac_packet_size=100, twt_log2_n_slot = 2, noise=0., n_user=(20,20), mobility_in_meter_per_sec=0.):
        self.LOGGED_CLASS_NAME = "sim_env"
        self.id = id
        self.seed = None
        self.ns3_sim_time_s = ns3_sim_time_s
        self.app_packet_interval = app_packet_interval
        assert mac_packet_size > UDP_IP_WIFI_HEADER_SIZE
        self.app_packet_size = mac_packet_size - UDP_IP_WIFI_HEADER_SIZE
        self.mac_packet_size = mac_packet_size
        self.twt_log2_n_slot = twt_log2_n_slot

        self.noise = noise

        self.cfg = None
        self.sample = None
        self.memory = None
        self.pl_model:path_loss = None
        self.ns3_env:sim_wifi_net = None
        self.actor = None

        self.n_user_range = n_user
        self.n_user = 0

        self.mobility_in_meter = mobility_in_meter_per_sec*self.ns3_sim_time_s

        self.n_user_rand_gen = np.random.default_rng(0)

        self.init_env()

    def init_env(self,seed = None):
        self.seed = seed
        if seed:
            self.pl_model = path_loss(n_sta=self.set_n_sta(), shadowing_sigma=self.noise, seed=seed)
        else:
            self.pl_model = path_loss(n_sta=self.set_n_sta(), shadowing_sigma=self.noise, seed=self.N_STEP)

        self.cfg = wifi_net_config()


        self.cfg.id = self.id
        self.cfg.n_ap = self.pl_model.n_ap
        self.cfg.n_sta = self.pl_model.n_sta

        self.cfg.app_packet_interval = self.app_packet_interval
        self.cfg.app_packet_size = self.app_packet_size

        self.cfg.loss_ap_ap = self.pl_model.get_loss_ap_ap()
        self.cfg.loss_sta_ap = self.pl_model.get_loss_sta_ap()
        self.cfg.loss_sta_sta = self.pl_model.get_loss_sta_sta()
    @counted
    def step(self, run_ns3 = True, seed = None):

        self.pl_model.rand_user_mobility(self.mobility_in_meter)
        self.cfg.loss_ap_ap = self.pl_model.get_loss_ap_ap()
        self.cfg.loss_sta_ap = self.pl_model.get_loss_sta_ap()
        self.cfg.loss_sta_sta = self.pl_model.get_loss_sta_sta()

        self.sample = {}
        state = self.pl_model.convert_loss_sta_ap_threshold(self.cfg.loss_sta_ap)
        # state = self.cfg.loss_sta_ap
        state = self.formate_np_state(state)
        action = self.gen_action(state)
        # print(action)
        twt_cfg = self.formate_np_action(action)

        self.cfg.PROG_PATH = self.PROG_PATH
        self.cfg.PROG_NAME = self.PROG_NAME
        self.cfg.PROG_PORT = 0
        if seed:
            self.cfg.PROG_SEED = seed
        else:
            self.cfg.PROG_SEED = (self.N_STEP % 1000) + 5000
        self.cfg.PROG_TIME = self.ns3_sim_time_s

        self.cfg.twtstarttime = twt_cfg['twtstarttime']
        self.cfg.twtoffset = twt_cfg['twtoffset']
        self.cfg.twtduration = twt_cfg['twtduration']
        self.cfg.twtperiodicity = twt_cfg['twtperiodicity']

        self.ns3_env = sim_wifi_net(self.id)
        self.ns3_env.set_config(self.cfg)

        self.sample['state'] = state
        self.sample['target'] = self.pl_model.convert_loss_sta_sta_binary(self.cfg.loss_sta_sta)
        self.sample['action'] = action
        self.sample['n_node'] = self.pl_model.n_sta
        if run_ns3:
            self.ns3_env.start()
            self.ns3_env.join()
            rwd = self.ns3_env.get_return()
            self.sample['reward'] = self.formate_dict_reward(rwd)
        else:
            self.sample['reward'] = np.zeros((self.pl_model.n_sta, 1))

        self._add_np_log("sta_loc", np.reshape(np.array(self.pl_model.sta_locs), (1, -1)))
        self._add_np_log("state", np.reshape(self.sample['state'], (1, -1)))
        self._add_np_log("action", np.reshape(self.sample['action'], (1, -1)))
        self._add_np_log("reward", np.reshape(self.sample['reward'], (1, -1)))
        self._add_np_log("target", np.reshape(self.sample['target'], (1, -1)))

        if not np.isnan(np.sum(self.sample['reward'])):
            if self.memory and not np.isnan(np.sum(self.sample['reward'])):
                assert self.sample['state'].shape == (self.pl_model.n_sta, self.pl_model.n_ap)
                assert self.sample['target'].shape == (self.pl_model.n_sta, self.pl_model.n_sta)
                assert self.sample['action'].shape[0] == self.pl_model.n_sta
                assert self.sample['reward'].shape == (self.pl_model.n_sta, 1)
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
        return action

    def formate_np_state(self, state) -> np.array:
        ## normalization etc
        state /= (-self.pl_model.min_rssi_dbm)
        state -= 1.
        return state

    def formate_np_action(self, action) -> dict:
        ## action to twt list
        sta_twt_slot_id = self.format_act_to_sta_twt_idx(action)
        self._printa(sta_twt_slot_id)
        self._add_np_log("sta_twt_slot_id", np.reshape(sta_twt_slot_id, (1, -1)))

        twt_cfg = {}
        twt_cfg['twtstarttime'] = np.ones(self.pl_model.n_sta)* (self.TWT_START_TIME * float(self.pl_model.n_sta) + self.TWT_START_TIME_OFFSET)
        twt_cfg['twtoffset'] = sta_twt_slot_id*self.TWT_ASLOT_TIME
        twt_cfg['twtduration'] = np.ones(self.pl_model.n_sta)*self.TWT_ASLOT_TIME
        twt_cfg['twtperiodicity'] = np.ones(self.pl_model.n_sta)*self.TWT_ASLOT_TIME * (2**self.twt_log2_n_slot)

        return twt_cfg

    def format_act_to_sta_twt_idx(self, action):
        # self._printa("gen_action, araw\n", action,np.min(action[~np.eye(action.shape[0],dtype=bool)]),np.max(action[~np.eye(action.shape[0],dtype=bool)]))
        # action -= np.min(action[~np.eye(action.shape[0],dtype=bool)])
        # self._printa("gen_action, -min\n", action,np.min(action[~np.eye(action.shape[0],dtype=bool)]),np.max(action[~np.eye(action.shape[0],dtype=bool)]))
        # action /= np.max(action[~np.eye(action.shape[0],dtype=bool)])
        # self._printa("gen_action. /max\n", action,np.min(action[~np.eye(action.shape[0],dtype=bool)]),np.max(action[~np.eye(action.shape[0],dtype=bool)]))
        # np.fill_diagonal(action,0)
        str = ""
        for i in range(9):
            lvl = float(i+1)/10
            str += (">%.1f=") % lvl
            str += ("%3d~") % ((action>lvl).sum())
            str += ("%.2f, ") % ((action>lvl).sum()/(action.shape[0]*(action.shape[0]-1)))
        self._printa("gen_action, ", str)

        return cut_into_2_k(action,self.pl_model.n_sta,self.twt_log2_n_slot)

    def formate_dict_reward(self, reward) -> np.array:
        # print(reward,"+++++++++++++++++")
        ret = reward['thr']/(1e6/self.app_packet_interval)
        self._printa(ret)
        self._printa("_moving_average reward",ret)
        self._printa("_moving_average reward",self._moving_average("reward",np.mean(ret)),np.mean(ret))
        self._printa("_moving_average min_reward",self._moving_average("min_reward",np.min(ret)),np.min(ret))
        self._printa("_moving_average n_none_rwd",self._moving_average("n_none_rwd",ret.size - np.count_nonzero(ret)),ret.size - np.count_nonzero(ret))

        return ret[:,np.newaxis]

    def set_n_sta(self):
        self.n_user = self.n_user_rand_gen.integers(low=self.n_user_range[0],high=self.n_user_range[1],endpoint=True)
        return self.n_user


if __name__ == '__main__':
    # build_ns3("/home/soyo/wifi-ai/ns-3-dev")
    # exit(0)
    e = sim_env(id=random.randint(40,60))
    e.PROG_PATH = "/home/soyo/wifi-ai/ns-3-dev"
    e.PROG_NAME = "wifi-ai/env"
    class test_actor:
        def __init__(self, n_sta):
            print(n_sta)
            self.n_sta = n_sta

        def gen_action(self,state):
            ret = np.random.uniform(0,1,(self.n_sta,self.n_sta))
            np.fill_diagonal(ret,0)
            return ret

    a = test_actor(e.set_n_sta())
    e.set_actor(a)
    e.init_env()
    # e.ns3_env.DEBUG = True
    e.step()

