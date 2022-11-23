from sim_src.edge_label.model.base_model import *


class equal_weight_model(base_model):
    def gen_action(self, state_np):
        n_node = state_np.shape[0]
        action = np.ones((n_node,n_node))
        np.fill_diagonal(action,0.)
        return action

    def _train_infer(self,batch):
        pass

    def _train_actor(self,batch):
        pass

    def _train_critic(self,batch):
        pass