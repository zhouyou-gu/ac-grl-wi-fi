import cvxpy as cp
import numpy as np



class max_k_cut():
    def __init__(self, n_nodes, k):
        self.k_part = k
        self.n_nodes = n_nodes
        self.H = None
        self.L = None

        self.prob = None
        self.X = None
        self.obj = None
        self.constraints = None

    def get_m_solutions(self, num_trials=1):
        u, s, v = np.linalg.svd(self.X.value)
        U = u * np.sqrt(s) ## each row of U is v_i

        cut_results = []
        cost_results = []
        for i in range(num_trials):
            r = np.random.randn(self.k_part,self.n_nodes)
            r = r / np.linalg.norm(r,axis=1,keepdims=True)
            cut = np.argmax(r @ U.T,axis=0)
            cut_results.append(cut)
            cost_results.append(max_k_cut.cut_cost(cut,self.H))

        return cut_results, cost_results

    def set_edge_weights(self, H):
        self.H = H
        self.L = - H
        d = np.sum(self.H,axis=1)
        np.fill_diagonal(self.L,d)
        self._config_prob()

    def _config_prob(self):
        self.X = cp.Variable((self.n_nodes, self.n_nodes), PSD=True)
        self.obj = ((self.k_part-1)/(self.k_part)) * cp.trace(self.L @ self.X)
        self.constraints = [cp.diag(self.X) == 1]
        self.constraints += [cp.min(self.X) >= -1./(self.k_part-1)]
        self.prob = cp.Problem(cp.Maximize(self.obj), constraints=self.constraints)

    def solve(self):
        self.prob.solve(solver=cp.SCS)

    @staticmethod
    def cut_cost(cut, H):
        ret = 0.
        n_node = cut.size
        cut = cut.ravel()
        for x in range(n_node):
            for y in range(n_node):
                if cut[x] != cut[y]:
                    ret += H[x,y]
        return ret




if __name__ == '__main__':
    from sim_src.edge_label.gw_cut import gw_cut, cut_into_2_k

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    def gen_rand_H(n):
        ret = np.random.randn(n,n)
        ret = ret * ret
        for i in range(n):
            ret[i,i] = 0.
            for j in range(i+1,n):
                ret[j,i] = ret[i,j]

        return ret
    avg = 0.
    for ii in range(1):
        n = 20
        m_solutions  = 100
        H = gen_rand_H(n)
        o = max_k_cut(n,2)
        o.set_edge_weights(H)
        o.solve()
        s1, c = o.get_m_solutions(m_solutions)
        a = np.mean(c)

        H = gen_rand_H(n)
        o = gw_cut(n)
        o.set_edge_weights(H)
        o.solve()
        s2, c = o.get_m_solutions(m_solutions)
        b = np.mean(c)

        for mm in range(m_solutions):
            print(s2 == (s1[mm]*2-1))
            print(s2 == -(s1[mm]*2-1))
        avg += a/b
        print(a/b)

    print(avg)

    avg = 0.
    for ii in range(100):
        n = 20
        H = gen_rand_H(n)
        o = max_k_cut(n,4)
        o.set_edge_weights(H)
        o.solve()
        s1, c = o.get_m_solutions(1)
        a = np.mean(c)

        id = cut_into_2_k(H,n,2)
        print(s1[0],id)

        c1 = max_k_cut.cut_cost(s1[0],H)
        c2 = max_k_cut.cut_cost(id,H)
        avg += c1/c2
    print(avg)