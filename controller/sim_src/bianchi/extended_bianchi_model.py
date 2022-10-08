import networkx as nx
import numpy as np

class extended_bianchi_model():
    W = 32
    M = 3
    sigma = 1
    def __init__(self, con_adj_m, hid_adj_m, ts, tc):
        self.con_adj_m = con_adj_m
        self.hid_adj_m = hid_adj_m

        self.ts = ts
        self.tc = tc


        self.con_graph = nx.from_numpy_matrix(self.con_adj_m)
        self.hid_graph = nx.from_numpy_matrix(self.hid_adj_m)
        assert self.con_graph.number_of_nodes == self.con_graph.number_of_nodes

        self.N = self.con_graph.number_of_nodes()
        self.tau = np.ones(self.N)*0.5
        self.p_collision = np.ones(self.N)*0.5

        self.tau_next = np.ones(self.N)*0.5
        self.p_collision_next = np.ones(self.N)*0.5


    @staticmethod
    def p_collision_to_tau(p):
        nu = 2.
        do_a = np.sum([(2.*p)**m for m in range(extended_bianchi_model.M)])
        do = (1 + extended_bianchi_model.W) + p*extended_bianchi_model.W*do_a
        return nu/(do)

    @staticmethod
    def p_collision_to_tau_do(p):
        do_a = np.sum([(2.*p)**m for m in range(extended_bianchi_model.M)])
        do = (1 + extended_bianchi_model.W) + p*extended_bianchi_model.W*do_a
        return do

    def update_p_collision(self):
        for n in range(self.N):
            c_neigbour = list(self.con_graph.neighbors(n))
            c_tmp = 1.-self.tau[c_neigbour]

            h_neigbour = list(self.hid_graph.neighbors(n))
            h_tmp = np.power(1.-self.tau[h_neigbour],0.1)

            self.p_collision_next[n] = 1 - np.prod(c_tmp)*np.prod(h_tmp)

    def update_tau(self):
        for n in range(self.N):
            self.tau_next[n] = extended_bianchi_model.p_collision_to_tau(self.p_collision[n])

    def update_cur(self,i):
        nn = i % self.N
        self.tau[nn] = self.tau_next[nn]
        self.p_collision[nn] = self.p_collision_next[nn]

    def get_Jacobian(self):
        J = np.zeros((2*self.N,2*self.N))
        for i in range(self.N):
            # df/dp_i
            J[i,i] = -1.

        return J

    def iter_Jaco(self, I, p = None):
        if p is not None:
            self.p_collision = np.ones(self.N)*p
            self.tau = np.ones(self.N) * extended_bianchi_model.p_collision_to_tau(p)

        for it in range(I):
            self.update_p_collision()

            self.p_collision = np.copy(self.p_collision_next)
            self.p_collision[self.p_collision>1.] = 1.-1e-5
            self.p_collision[self.p_collision<0.] = 0.+1e-5

            for n in range(self.N):
                self.tau[n] = extended_bianchi_model.p_collision_to_tau(self.p_collision[n])

            self.update_p_collision()

            print("--------", it)
            print("p_c", self.p_collision)
            print("p_c", self.p_collision_next)
            print("tau", self.tau)

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    N = 100
    C = np.random.choice([0. , 1.], (N,N),p=[0.6, 0.4])
    np.fill_diagonal(C,0)
    C = np.tril(C,-1) + np.tril(C,-1).T
    H = np.random.choice([0. , 1.], (N,N),p=[0.6, 0.4])
    np.fill_diagonal(H,0)
    print(C)
    print(H)
    m = extended_bianchi_model(C,H)
    m.iter_Jaco(100)