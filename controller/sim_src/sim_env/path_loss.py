import math

import numpy as np

class path_loss():
    C = 299792458.0
    PI = 3.14159265358979323846
    HIDDEN_LOSS = 200.
    NOISE_FLOOR_1MHZ_DBM = -93.9763
    def __init__(self, n_ap = 4, n_sta = 10,range = 1000., fre_Hz = 1e9, txp_dbm = 0., min_rssi_dbm = -95, shadowing_sigma = 5., seed=0):
        self.rand_gen = np.random.default_rng(seed)

        self.memory = None
        self.model = None

        self.n_ap = n_ap
        self.n_sta = n_sta
        self.range = range
        self.ap_locs = []
        self.sta_locs = []

        self.fre_Hz = fre_Hz
        self.lam = self.C / self.fre_Hz
        self.txp_dbm = txp_dbm
        self.min_rssi_dbm = min_rssi_dbm
        assert shadowing_sigma >= 0.
        self.shadowing_sigma = shadowing_sigma

        self._config_ap_locs()
        self._config_sta_locs()

    def get_loss_ap_ap(self):
        ret = np.zeros((self.n_ap,self.n_ap))
        for i in range(self.n_ap):
            for j in range(self.n_ap):
                if i == j:
                    continue
                ret[i,j] = self._get_loss_between_locs(self.ap_locs[i],self.ap_locs[j])
        return ret

    def get_loss_sta_ap(self):
        ret = np.ones((self.n_sta,self.n_ap))*np.inf
        for i in range(self.n_sta):
            while np.min(ret[i,:]) > 90:
                for j in range(self.n_ap):
                    ret[i,j] = self._get_loss_between_locs(self.sta_locs[i],self.ap_locs[j],noise=True)
        return ret

    def get_loss_sta_sta(self):
        ret = np.zeros((self.n_sta,self.n_sta))
        for i in range(self.n_sta):
            for j in range(i,self.n_sta):
                if i == j:
                    continue
                ret[i,j] = self._get_loss_between_locs(self.sta_locs[i],self.sta_locs[j],noise=True)
                ret[j,i] = ret[i,j]
        return ret

    def _config_ap_locs(self):
        grid_ap = self.range/2.
        self.ap_locs.append((grid_ap, grid_ap))
        self.ap_locs.append((-grid_ap, grid_ap))
        self.ap_locs.append((grid_ap, -grid_ap))
        self.ap_locs.append((-grid_ap, -grid_ap))
        assert len(self.ap_locs) == self.n_ap

    def _config_sta_locs(self):
        for i in range(self.n_sta):
            self.sta_locs.append(self._get_random_loc())

        assert len(self.sta_locs) == self.n_sta

    def _get_random_loc(self):
        x = self.rand_gen.uniform(-self.range, self.range)
        y = self.rand_gen.uniform(-self.range, self.range)
        return (x,y)

    def _get_loss_between_locs(self, a, b, noise=False):
        dis = np.linalg.norm(np.array(a)-np.array(b),ord=2)
        return self._get_loss_distance(dis, noise)

    def _get_loss_distance(self, dis, noise=False):
        numerator = self.lam ** 2
        denominator = 16 * (self.PI ** 2) * ((dis + 1e-5) ** 2)
        loss = - 10. * math.log10(numerator / denominator)
        if noise:
            loss += self.rand_gen.standard_normal() * self.shadowing_sigma
        loss = np.max((loss,0))
        return loss

    def convert_loss_sta_ap_threshold(self, loss):
        ret = np.copy(loss)
        ret[ret>(self.txp_dbm-self.min_rssi_dbm)] = self.HIDDEN_LOSS
        return ret

    def convert_loss_sta_sta_binary(self, loss):
        ret = np.copy(loss)
        ret[np.logical_or(ret>(self.txp_dbm-self.min_rssi_dbm), self.txp_dbm-ret-self.NOISE_FLOOR_1MHZ_DBM<0.)] = 0.
        ret[ret>0.] = 1.
        return ret

if __name__ == '__main__':
    pl = path_loss()
    # x = pl.get_loss_sta_ap()
    # # print(x)
    # print(pl.convert_loss_sta_ap_threshold(x))
    # print(x)
    # print(pl.convert_loss_sta_ap_threshold(pl.get_loss_ap_ap()))
    # print(pl.convert_loss_sta_ap_threshold(pl.get_loss_sta_sta()))

    # y = pl.get_loss_sta_sta()
    # print(y)
    # print(pl.convert_loss_sta_sta_binary(y))
    l_max = pl._get_loss_between_locs((1500,1500),(750,750))
    print(l_max)
    print(pl.get_loss_sta_ap())