import numpy as np


class model():
    ALPHA = 0.99
    DELTA = 1e-5
    def __init__(self, id = 0, n_sta = 10):
        self.id = id
        self.n_sta = n_sta

        self.pm = None
        self.H = np.triu(np.ones((n_sta,n_sta))*1.,k=1)

        self.step_counter = 0

    def input_rn_obs(self, obs):
        self.pm = obs['aoi']

    def step(self, samples):
        print("learning step", self.step_counter)
        w = []
        for s in samples:
            w.append(1/model.diff_pm(self.pm,s[1]))

        w = np.array(w)
        w = w/np.sum(w)

        sum_H = np.zeros((self.n_sta,self.n_sta), dtype=float)
        i = 0
        for s in samples:
            sum_H += w[i] * s[0]
            i += 1

        self.H = self.H * self.ALPHA + sum_H * (1-self.ALPHA)
        print(self.H)

        self.step_counter += 1

    def get_dt_inference(self):
        self.H = np.triu(self.H,k=1)
        return np.random.binomial(1,self.H)

    @classmethod
    def diff_pm(cls, a, b):
        return np.linalg.norm(a-b,1) + model.DELTA



if __name__ == '__main__':
    a = np.ones((5,5))
    b = np.triu(a,k=1)*0.1
    print(np.triu(a,k=1)*0.1)
    print(np.random.binomial(1,b))