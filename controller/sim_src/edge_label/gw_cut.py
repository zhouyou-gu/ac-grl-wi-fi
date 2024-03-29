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

        self.rand_gen = np.random.default_rng(0)

    def get_m_solutions(self, num_trials=1):
        u, s, v = np.linalg.svd(self.X.value)
        U = u * np.sqrt(s)

        cut_results = []
        cost_results = []
        for i in range(num_trials):
            r = self.rand_gen.standard_normal(self.n_nodes)
            r = r / np.linalg.norm(r)
            cut = np.sign(r @ U.T)
            cut_results.append(cut)
            a, b = gw_cut.split(cut)
            cost_results.append(gw_cut.cut_cost(a,b,self.H))

        return cut_results, cost_results

    def get_best_solution(self, num_trials=1):
        u, s, v = np.linalg.svd(self.X.value)
        U = u * np.sqrt(s)

        cut_results = []
        cost_results = []
        for i in range(num_trials):
            r = self.rand_gen.standard_normal(self.n_nodes)
            r = r / np.linalg.norm(r)
            cut = np.sign(r @ U.T)
            cut_results.append(cut)
            a, b = gw_cut.split(cut)
            cost_results.append(gw_cut.cut_cost(a,b,self.H))

        max_value = max(cost_results)
        i = cost_results.index(max_value)
        return cut_results[i], cost_results[i]

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
                ret += H[y,x]
        return ret


def cut_into_2_k(W,n_node,k):
    assert W.shape == (n_node,n_node)
    node_idx = [i for i in range(n_node)]
    return cut_into_2_k_recur(W,n_node,0,k,node_idx)


def cut_into_2_k_recur(W,n_node,lvl,total_lvl,node_idx):
    ret = np.zeros(n_node)
    if lvl == total_lvl:
        return ret
    if not node_idx:
        return ret
    cutter = gw_cut(len(node_idx))
    cutter.set_edge_weights(W[np.ix_(node_idx,node_idx)])
    cutter.solve()
    s, c = cutter.get_best_solution(10)
    a, b = gw_cut.split(s)
    a_ind = [node_idx[i] for i in a]
    ret[a_ind] += 2**(total_lvl-lvl-1)
    ret += cut_into_2_k_recur(W,n_node,lvl+1,total_lvl,a_ind)

    b_ind = [node_idx[i] for i in b]
    ret += cut_into_2_k_recur(W,n_node,lvl+1,total_lvl,b_ind)
    return ret





if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    def gen_rand_H(n):
        ret = np.random.randn(n,n)
        ret = ret * ret
        ret += 100
        for i in range(n):
            ret[i,i] = 0.
            for j in range(i+1,n):
                ret[j,i] = ret[i,j]

        return ret

    n = 20
    m_solutions  = 100
    H = gen_rand_H(n)
    o = gw_cut(n)
    o.set_edge_weights(H)
    o.solve()
    s, c = o.get_m_solutions(m_solutions)
    print(s)
    print(c)
    s, c = o.get_best_solution(m_solutions)
    print(s)
    print(c)
    print(cut_into_2_k(H,n,2))
    print(H)
