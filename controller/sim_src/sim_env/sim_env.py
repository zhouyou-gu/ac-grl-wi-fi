import numpy as np

from sim_src.cut.gw_cut import gw_cut, cut_into_2_k
from sim_src.ns3_ctrl.wifi_net_ctrl import sim_wifi_net, wifi_net_config
from sim_src.sim_env.path_loss import path_loss
from sim_src.util import to_tensor, to_numpy

UDP_IP_WIFI_HEADER_SIZE = 64


class sim_env_to_controller_interface:
    def init_env(self):
        pass

    def step(self):
        pass

    def set_memory(self, memory):
        pass

    def set_actor(self, model):
        pass


class sim_env(sim_env_to_controller_interface):
    def __init__(self, id=0, ns3_sim_time_s=10, app_packet_interval=40000, mac_packet_size=100, twt_log2_n_slot = 1):
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

        self.sample = {}

    def init_env(self):
        self.sample = {}

        self.pl_model = path_loss(n_sta=self.get_n_sta())
        self.ns3_env = sim_wifi_net(self.id)

        cfg = wifi_net_config()
        cfg.PROG_PORT = (self.id % 1000) + 5000
        cfg.PROG_SEED = (self.id % 1000) + 5000
        cfg.PROG_TIME = self.ns3_sim_time

        cfg.id = self.id
        cfg.n_ap = self.pl_model.n_ap
        cfg.n_sta = self.pl_model.n_sta

        cfg.app_packet_interval = self.app_packet_interval
        cfg.app_packet_size = self.app_packet_size

        cfg.loss_ap_ap = self.pl_model.get_loss_ap_ap()
        cfg.loss_sta_ap = self.pl_model.get_loss_sta_ap()
        cfg.loss_sta_sta = self.pl_model.get_loss_sta_sta()

        state = path_loss.convert_loss_sta_ap(cfg.loss_sta_ap)
        state = self.formate_np_state(state)
        action = self.gen_action(state)
        twt_cfg = self.formate_np_action(action)

        cfg.twtstarttime = twt_cfg['twtstarttime']
        cfg.twtoffset = twt_cfg['twtoffset']
        cfg.twtduration = twt_cfg['twtduration']
        cfg.twtperiodicity = twt_cfg['twtperiodicity']

        self.ns3_env.set_config(cfg)

        self.sample['state'] = state
        self.sample['target'] = cfg.loss_sta_sta
        self.sample['action'] = action

    def step(self):
        self.ns3_env.start()
        self.ns3_env.join()
        rwd = self.ns3_env.get_return()
        self.sample['reward'] = self.formate_dict_reward(rwd)

        assert self.sample['state'].shape == (self.pl_model.n_sta, self.pl_model.n_ap)
        assert self.sample['target'].shape == (self.pl_model.n_sta, self.pl_model.n_sta)
        assert self.sample['action'].shape == (self.pl_model.n_sta, 1)
        assert self.sample['reward'].shape == (self.pl_model.n_sta, 1)

    def set_memory(self, memory):
        self.memory = memory

    def set_actor(self, model):
        self.actor = model

    def gen_action(self, state):
        ## exploration?
        to_tensor(state)
        action = self.actor.get_action(state)
        to_numpy(action)
        return action

    def formate_np_state(self, state) -> np.arry:
        ## normalization etc
        state /= (self.pl_model.min_rssi_dbm)
        state -= 1.
        return state

    def formate_np_action(self, action) -> dict:
        ## action to twt list
        sta_twt_slot_id = cut_into_2_k(action,self.pl_model.n_sta,self.twt_log2_n_slot)

        twt_cfg = {}
        twt_cfg['twtstarttime'] = np.ones(self.pl_model.n_sta)*self.ns3_env.TWT_START_TIME
        twt_cfg['twtoffset'] = sta_twt_slot_id*self.ns3_env.TWT_START_TIME
        twt_cfg['twtduration'] = np.ones(self.pl_model.n_sta)*self.ns3_env.TWT_ASLOT_TIME
        twt_cfg['twtperiodicity'] = np.ones(self.pl_model.n_sta)*self.ns3_env.TWT_ASLOT_TIME * (2**self.twt_log2_n_slot)

        return twt_cfg

    def formate_dict_reward(self, reward) -> np.array:
        ret = reward['thr']/(self.app_packet_interval/1e6)/self.ns3_sim_time_s
        return ret

    def get_n_sta(self):
        return 20.
