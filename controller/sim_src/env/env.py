import math

import numpy as np

def p_true(probability_of_true):
    return np.random.choice([True, False], p=[probability_of_true, 1 - probability_of_true])

class path_loss_model():
    C = 299792458.0
    PI = 3.14159265358979323846
    def __init__(self, n_ap = 4, range = 1000., fre_Hz = 1e9, txp_dbm = 10, min_rssi_dbm = -82, shadowing_sigma = 0):
        self.n_ap = n_ap
        self.range = range
        self.ap_locs = []
        self.fre_Hz = fre_Hz
        self.lam = self.C / self.fre_Hz
        self.txp_dbm = txp_dbm
        self.min_rssi_dbm = min_rssi_dbm
        self.shadowing_sigma = shadowing_sigma

    def set_ap_locs(self,x,y):
        assert len(self.ap_locs) < self.n_ap
        self.ap_locs.append((x,y))

    def gen_sta_pair(self,batch_size):
        ret_d = []
        ret_t = []
        for b in range(batch_size):
            d = []
            rad_loc_a = self._get_random_loc()
            for a in range(self.n_ap):
                loss = self._get_loss_between_locs(self._get_ap_loc(a),rad_loc_a)
                if self.txp_dbm - loss <= self.min_rssi_dbm:
                    loss = 200.
                d.append(loss)
            rad_loc_b = self._get_random_loc()
            for a in range(self.n_ap):
                loss = self._get_loss_between_locs(self._get_ap_loc(a),rad_loc_b)
                if self.txp_dbm - loss <= self.min_rssi_dbm:
                   loss = 200.
                d.append(loss)
            d = np.array(d)
            t = self._get_loss_between_locs(rad_loc_a,rad_loc_b)
            ret_d.append(d)
            ret_t.append(t)

        ret_d = np.vstack(ret_d)
        ret_t = np.vstack(ret_t)

        return (ret_d, ret_t)

    def gen_n_sta_info(self, n_sta):
        ret_d = []
        ret_l = []
        ret_t = np.zeros((n_sta,n_sta))
        for b in range(n_sta):
            d = []
            rad_loc_a = self._get_random_loc()
            ret_l.append(rad_loc_a)
            for a in range(self.n_ap):
                loss = self._get_loss_between_locs(self._get_ap_loc(a),rad_loc_a)
                if self.txp_dbm - loss <= self.min_rssi_dbm:
                    loss = 200.
                d.append(loss)
            d = np.array(d)
            ret_d.append(d)
        for a in range(n_sta):
            for b in range(a+1,n_sta):
                t = self._get_loss_between_locs(ret_d[a],ret_d[b])
                ret_t[a,b] = t
                ret_t[b,a] = t


        ret_d = np.vstack(ret_d)
        ret_l = np.vstack(ret_l)
        return ret_d, ret_t, ret_l

    def _get_random_loc(self):
        x = np.random.uniform(-self.range, self.range)
        y = np.random.uniform(-self.range, self.range)
        return (x,y)

    def _get_ap_loc(self, i):
        return self.ap_locs[i]

    def _get_loss_between_locs(self, a, b):
        dis = np.linalg.norm(np.array(a)-np.array(b),ord=2)
        return self._get_loss_distance(dis)

    def _get_loss_distance(self, dis):
        numerator = self.lam ** 2
        denominator = 16 * (self.PI ** 2) * ((dis + 1e-5) ** 2)
        loss = - 10. * math.log10(numerator / denominator) + np.random.randn() * self.shadowing_sigma
        loss = np.max((loss,0))
        return loss
