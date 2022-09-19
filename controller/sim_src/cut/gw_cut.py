import cvxpy as cp
import numpy as np




class gw_cut():
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.H = None
        self.L = None

        self.prob = None
        self.X = None
        self.obj = None
        self.constraints = None

    def get_m_solutions(self, num_trials):
        u, s, v = np.linalg.svd(self.X.value)
        U = u * np.sqrt(s)

        cut_results = []
        cost_results = []
        for i in range(num_trials):
            r = np.random.randn(self.n_nodes)
            r = r / np.linalg.norm(r)
            cut = np.sign(r @ U.T)
            cut_results.append(cut)
            a, b = gw_cut.split(cut)
            cost_results.append(gw_cut.cut_cost(a,b,self.H))

        return cut_results, cost_results

    def set_edge_weights(self, H):
        self.H = H
        self.L = - H
        d = np.sum(self.H,axis=1)
        np.fill_diagonal(self.L,d)
        self._config_prob()

    def _config_prob(self):
        self.X = cp.Variable((self.n_nodes, self.n_nodes), PSD=True)
        self.obj = 0.25 * cp.trace(self.L @ self.X)
        self.constraints = [cp.diag(self.X) == 1]
        self.prob = cp.Problem(cp.Maximize(self.obj), constraints=self.constraints)

    def solve(self):
        self.prob.solve(solver=cp.SCS)

    @staticmethod
    def split(cut):
        n = len(cut)
        S = [i for i in range(n) if cut[i]>0]
        T = [i for i in range(n) if cut[i]<0]
        return S, T

    @staticmethod
    def cut_cost(cut_a, cut_b, H):
        ret = 0.
        for x in cut_a:
            for y in cut_b:
                ret += H[x,y]
        return ret



if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    def gen_rand_H(n):
        ret = np.random.randn(n,n)
        for i in range(n):
            ret[i,i] = 0.
            for j in range(i+1,n):
                ret[j,i] = ret[i,j]

        return ret

    n = 200
    m_solutions  = 10
    H = gen_rand_H(n)
    o = gw_cut(n)
    o.set_edge_weights(H)
    o.solve()
    s, c = o.get_m_solutions(m_solutions)
    print(s)
    print(c)
