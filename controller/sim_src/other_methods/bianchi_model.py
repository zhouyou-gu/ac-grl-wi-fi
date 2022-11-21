import math

import networkx as nx
import numpy as np

from sim_src.sim_env.path_loss import path_loss
from sim_src.sim_env.sim_env import sim_env
from sim_src.util import DbToRatio, StatusObject


class bianchi_model(StatusObject):
    W = 15
    M = 6
    PACKET_SIZE = 100
    NOISE_FLOOR_1MHZ_DBM = -93.9763
    N_GROUP = 4
    def __init__(self, id,global_allocation=False):
        self.id = id
        self.global_allocation = global_allocation

    @staticmethod
    def tau_to_pcol(tau,n):
        p = 1 - (1-tau)**(n-1)
        return p

    @staticmethod
    def pcol_to_tau(p):
        nu = 2.
        do_a = np.sum([(2.*p)**m for m in range(bianchi_model.M)])
        do = (1 + bianchi_model.W) + p*bianchi_model.W*do_a
        return nu/(do)
    @staticmethod
    def get_tau(n):
        N_STEP = 100
        D_STEP = 1./N_STEP
        dif = np.inf
        min_tau = 0.
        for i in range(N_STEP+1):
            tau = D_STEP*i
            pp = bianchi_model.tau_to_pcol(tau,n)
            tau_tmp = bianchi_model.pcol_to_tau(pp)
            d = abs(tau_tmp - tau)
            if d <= dif:
                min_tau = tau
                dif = d
        return min_tau
    @staticmethod
    def tau_to_ptra(tau,n):
        if n == 0:
            return 1.
        return 1 - (1-tau)**n

    @staticmethod
    def tau_to_psuc(tau,n):
        if n == 0:
            return 0.

        nu = n * tau * (1-tau)**(n-1)
        do = 1 - (1-tau)**n
        return nu/do

    @staticmethod
    def tau_to_rate(tau,n,t_packet):
        nu = bianchi_model.tau_to_ptra(tau,n) * bianchi_model.tau_to_psuc(tau,n)
        do = (1-bianchi_model.tau_to_ptra(tau,n)) + bianchi_model.tau_to_ptra(tau,n)*t_packet
        return nu/do

    def gen_action(self,states):
        if self.global_allocation:
            return self.gen_action_global(states)
        else:
            return self.gen_action_per_ap(states)

    def gen_action_global(self, states):
        self.states = states
        self.n_ap = states.shape[1]
        self.n_sta = states.shape[0]

        sta_slot_id = np.zeros(self.n_sta)
        grouping_decision = [[] for i in range(self.N_GROUP)]
        sta_a = []
        u_group = np.ones(self.N_GROUP)*np.inf
        for s in range(self.n_sta):
                sta_a.append(s)
        while sta_a:
            ui = np.argmax(u_group)
            rate_list = np.zeros(len(sta_a))
            for si in range(len(sta_a)):
                s = sta_a[si]
                t_packet = self.get_t_packet(grouping_decision[ui],s)
                nn = len(sta_a)+1
                tau = bianchi_model.get_tau(nn)
                rate = bianchi_model.tau_to_rate(tau,nn,t_packet)
                rate_list[si] = rate
            si = np.argmax(rate_list)
            grouping_decision[ui].append(sta_a[si])
            sta_a.pop(si)
            u_group[ui] = rate_list[si]

        for g in range(self.N_GROUP):
            for gg in grouping_decision[g]:
                sta_slot_id[gg] = g

        return sta_slot_id

    def gen_action_per_ap(self, states):
        self.states = states
        self.n_ap = states.shape[1]
        self.n_sta = states.shape[0]

        sta_slot_id = np.zeros(self.n_sta)
        max_a = np.argmax(self.states,axis=1)
        for a in range(self.n_ap):
            grouping_decision = [[] for i in range(self.N_GROUP)]
            sta_a = []
            u_group = np.ones(self.N_GROUP)*np.inf
            for s in range(self.n_sta):
                if max_a[s] == a:
                    sta_a.append(s)
            while sta_a:
                ui = np.argmax(u_group)
                rate_list = np.zeros(len(sta_a))
                for si in range(len(sta_a)):
                    s = sta_a[si]
                    t_packet = self.get_t_packet(grouping_decision[ui],s)
                    nn = len(sta_a)+1
                    tau = bianchi_model.get_tau(nn)
                    rate = bianchi_model.tau_to_rate(tau,nn,t_packet)
                    rate_list[si] = rate
                si = np.argmax(rate_list)
                grouping_decision[ui].append(sta_a[si])
                sta_a.pop(si)
                u_group[ui] = rate_list[si]

            for g in range(self.N_GROUP):
                for gg in grouping_decision[g]:
                    sta_slot_id[gg] = g

        return sta_slot_id

    def get_t_packet(self,sta_list,ss):
        t_packet = 0.
        n_s = 0.
        max_g = np.max(self.states,axis=1)
        for s in sta_list:
            t_packet += self.PACKET_SIZE*8/math.log2(1+DbToRatio(-max_g[s]-self.NOISE_FLOOR_1MHZ_DBM))/1e6
            n_s += 1

        t_packet += self.PACKET_SIZE*8/math.log2(1+DbToRatio(-max_g[ss]-self.NOISE_FLOOR_1MHZ_DBM))/1e6
        n_s += 1

        t_packet /= n_s
        return t_packet

class sim_env_bianchi_model(sim_env):
    def format_act_to_sta_twt_idx(self,action):
        return action



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    # G = nx.complete_graph(10)
    pl_model = path_loss(n_sta=20)
    states = pl_model.get_loss_sta_ap()
    bm = bianchi_model(0,global_allocation=True)
    print(bm.gen_action(states))
    bm = bianchi_model(0,global_allocation=False)
    print(bm.gen_action(states))