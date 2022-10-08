import networkx as nx
import numpy as np


class bianchi_model():
    W = 32
    M = 3

    def __init__(self, con_adj_m):
        self.con_adj_m = con_adj_m
        self.graph = nx.from_numpy_matrix(self.con_adj_m)
        self.N = self.graph.number_of_nodes()
        self.tau = np.ones(self.N)*0.5
        self.p_collision = np.ones(self.N)*0.5

        self.tau_next = np.ones(self.N)*0.5
        self.p_collision_next = np.ones(self.N)*0.5


    @staticmethod
    def p_collision_to_tau(p):
        nu = 2.
        do_a = np.sum([(2.*p)**m for m in range(bianchi_model.M)])
        do = (1 + bianchi_model.W) + p*bianchi_model.W*do_a
        return nu/(do)

    @staticmethod
    def p_collision_to_tau_do(p):
        do_a = np.sum([(2.*p)**m for m in range(bianchi_model.M)])
        do = (1 + bianchi_model.W) + p*bianchi_model.W*do_a
        return do

    def update_p_collision(self):
        for n in range(self.N):
            neighbour = list(self.graph.neighbors(n))
            # print(neighbour.__len__())
            # print(self.tau[neighbour])
            tmp = 1.-self.tau[neighbour]
            self.p_collision_next[n] = 1 - np.prod(tmp)

    def update_tau(self):
        for n in range(self.N):
            self.tau_next[n] = bianchi_model.p_collision_to_tau(self.p_collision[n])

    def update_cur(self,i):
        nn = i % self.N
        self.tau[nn] = self.tau_next[nn]
        self.p_collision[nn] = self.p_collision_next[nn]
    #
    def iter(self, n_iter):
        for i in range(n_iter):
            print('----')
            print("tau", self.tau)
            print("p_c", self.p_collision)
            self.update_p_collision()
            self.update_tau()
            self.update_cur(i)
            print("tau", self.tau)
            print("p_c", self.p_collision)

    def get_Jacobian(self):
        J = np.zeros((2*self.N,2*self.N))
        for i in range(self.N):
            # df/dp_i
            J[i,i] = -1.

            # # dg/dp_i
            # tmp_do = bianchi_model.W * np.sum([(m+1)*(2.*self.p_collision[i])**m for m in range(bianchi_model.M)])
            # J[i,i+self.N] = - 2 * 1 / (bianchi_model.p_collision_to_tau_do(self.p_collision[i])**2) * tmp_do

        # for i in range(self.N):
        #     # df/dtau_i
        #     neighbour = list(self.graph.neighbors(i))
        #     for j in neighbour:
        #         tmp = 1.-self.tau[neighbour]
        #         tmp = np.prod(tmp)/(1-self.tau[j])
        #         J[i+self.N,j] = tmp
        #
        #     # dg/dtau_i
        #     J[i+self.N,i+self.N] = -1.

        return J

    def get_Jacobian_2(self):
        J = np.zeros((2*self.N,2*self.N))
        for i in range(self.N):
            # df/dp_i
            neighbour = list(self.graph.neighbors(i))
            tmp = 1. - self.tau[neighbour]
            tmp = 1. - np.prod(tmp)
            J[i,i] = -tmp/(self.p_collision[i]**2)

            # dg/dp_i
            tmp_do = bianchi_model.W * np.sum([(m+1)*(2.*self.p_collision[i])**m for m in range(bianchi_model.M)])
            J[i,i+self.N] = - 2 * 1 / (bianchi_model.p_collision_to_tau_do(self.p_collision[i])**2) * tmp_do / self.tau[i]

        # for i in range(self.N):
        #     # df/dtau_i
        #     neighbour = list(self.graph.neighbors(i))
        #     for j in neighbour:
        #         tmp = 1.-self.tau[neighbour]
        #         tmp = np.prod(tmp)/(1-self.tau[j])
        #         J[i+self.N,j] = tmp/self.p_collision[j]
        #
        #     # dg/dtau_i
        #     J[i+self.N,i+self.N] = -bianchi_model.p_collision_to_tau(self.p_collision[i])/ (self.tau[i]**2)

        return J



    def iter_Jaco(self, I, p = None):
        if p is not None:
            self.p_collision = np.ones(self.N)*p
            self.tau = np.ones(self.N) * bianchi_model.p_collision_to_tau(p)

        for it in range(I):
            f= np.zeros(self.N)
            g= np.zeros(self.N)
            for i in range(self.N):
                neighbour = list(self.graph.neighbors(i))
                tmp = 1.-self.tau[neighbour]
                f[i] = 1 - self.p_collision[i] - np.prod(tmp)

                g[i] = - self.tau[i] + bianchi_model.p_collision_to_tau(self.p_collision[i])

            J = self.get_Jacobian()
            # J_inv = np.linalg.inv(J)

            x_cur = np.hstack((self.p_collision,self.tau))
            F_cur = np.hstack((f,g))
            s = np.linalg.solve(J[0:self.N,0:self.N], -F_cur[0:self.N])
            x_nxt = x_cur[0:self.N] + s
            # if int(it/10) % 2 ==0:
            #     self.p_collision = x_nxt[0:self.N]
            #     self.p_collision[self.p_collision>1.] = 1.-1e-5
            #     self.p_collision[self.p_collision<0.] = 0.+1e-5
            # else:
            #     self.tau = x_nxt[self.N:2*self.N]
            #     self.tau[self.tau>1.] = 1.-1e-5
            #     self.tau[self.tau<0.] = 0.+1e-5
            # nn = it % self.N
            self.p_collision = x_nxt
            self.p_collision[self.p_collision>1.] = 1.-1e-5
            self.p_collision[self.p_collision<0.] = 0.+1e-5

            for n in range(self.N):
                self.tau[n] = bianchi_model.p_collision_to_tau(self.p_collision[n])

            # self.tau[nn] = x_nxt[nn+self.N]
            # self.tau[self.tau>1.] = 1.-1e-5
            # self.tau[self.tau<0.] = 0.+1e-5

            self.update_tau()
            self.update_p_collision()
            print('----',it)

            print("p_c", self.p_collision)
            print("p_c", self.p_collision_next)
            print("tau", self.tau)
            print("tau", self.tau_next)
            print("F_cur", F_cur)
            print("x_cur", x_cur)
            # print("J", J)
            # print("J_inv", J_inv)
            print("s", s)
            # print("F_cur", self.p_collision)


    def test(self):
        for i in range(100):
            p = i*1e-2
            self.p_collision = np.ones(self.N)*p
            self.tau = np.ones(self.N) * bianchi_model.p_collision_to_tau(p)

            self.update_p_collision()
            self.update_tau()
            print('----',p)
            print("tau", self.tau)
            print("tau", self.tau_next)
            print("p_c", self.p_collision)
            print("p_c", self.p_collision_next)

    def test_p(self, p):
        self.p_collision = np.ones(self.N)*p
        self.tau = np.ones(self.N) * bianchi_model.p_collision_to_tau(p)

        self.update_p_collision()
        self.update_tau()
        print('----',p)
        print("tau", self.tau)
        print("tau", self.tau_next)
        print("p_c", self.p_collision)
        print("p_c", self.p_collision_next)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    # G = nx.complete_graph(10)
    G = nx.dorogovtsev_goltsev_mendes_graph(3)
    adj_m = nx.adjacency_matrix(G).todense()
    print(adj_m)
    m = bianchi_model(adj_m)
    # m.test()
    # m.test_p(0.60171571)
    # m.iter(10000)
    m.iter_Jaco(100)
    # m.iter_Jaco(10000)
    nx.draw(G)
    plt.show()