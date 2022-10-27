from sim_src.ns3_ctrl.wifi_net_ctrl import wifi_net_instance


class train_algorithm:
    def __init__(self):
        self.n_step = 10
        self.batch_size = 10

        self.env = None
        self.memory = None
        self.model = None

    def set_env(self, env):
        self.env = env

    def run(self):
        self.env.set_memory(self.memory)
        self.env.set_actor(self.model)
        for i in range(self.n_step):
            self.do_sim_step()
            self.do_tra_step()

    def do_sim_step(self):
        self.env.init_env()
        self.env.step()

    def do_tra_step(self):
        batch = self.memory.sample(self.batch_size)
        if batch:
            self.model.step(batch)
