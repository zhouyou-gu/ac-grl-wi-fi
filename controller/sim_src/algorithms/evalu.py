class evalu_algorithm:
    def __init__(self):
        self.n_step = 1

    def run(self):
        for i in range(self.n_step):
            self.do_sim_step()

    def do_sim_step(self):
        env = self.setup_env()
        env.run()
        ## save info in memory
        ## log performance

    def setup_env(self):
        obs = self.env.get_obs()
        self.model.get_action(obs)


class elnn_evalu_algorithm:
    pass

class elnn_combine_evalu_algorithm:
    pass

class gnnc_evalu_algorithm:
    pass

class rand_evalu_algorithm:
    pass

class bian_evalu_algorithm:
    pass

class heur_evalu_algorithm:
    pass