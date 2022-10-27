import os.path

import torch

from sim_src.tsb import GLOBAL_LOGGER
from sim_src.util import USE_CUDA, hard_update_inplace, counted


class learning_model:
    def step(self, batch):
        pass

    def get_action(self, state):
        pass

class gnn_label(learning_model):
    def __init__(self, id):
        self.id = id
        self.n_step = 0
        GLOBAL_LOGGER.get_tb_logger().add_text_of_object("GNN_LABEL_CONFIG", self.config)

        self.infer = None
        self.infer_optim = None

        self.actor = None
        self.actor_optim = None

        self.critic = None
        self.critic_optim = None

        self.actor_target = None
        self.critic_target = None

        self.init_model()
        GLOBAL_LOGGER.get_tb_logger().add_text_of_object("infer_arch", self.infer)

        GLOBAL_LOGGER.get_tb_logger().add_text_of_object("actor_target_arch", self.actor_target)
        GLOBAL_LOGGER.get_tb_logger().add_text_of_object("critic_target_arch", self.critic_target)

        GLOBAL_LOGGER.get_tb_logger().add_text_of_object("actor_arch", self.actor)
        GLOBAL_LOGGER.get_tb_logger().add_text_of_object("critic_arch", self.critic)

        if USE_CUDA:
            self.move_nn_to_gpu()

    def move_nn_to_gpu(self):
        self.infer.to(torch.cuda.current_device())
        self.actor.to(torch.cuda.current_device())
        self.actor_target.to(torch.cuda.current_device())
        self.critic.to(torch.cuda.current_device())
        self.critic_target.to(torch.cuda.current_device())

    def setup_infer(self):
        pass

    def setup_actor(self):
        if self.config.actor_config.load_path is not None:
            self.actor = self._load(self.config.actor_config.load_path)
        else:
            self.actor = Net(self.config.actor_config.nn_arch,
                             self.config.actor_config.af_config,
                             self.config.actor_config.init_w)

        self.actor_optim = self.config.actor_config.optim(self.actor.parameters(),
                                                          lr=self.config.actor_config.lr)

        self.actor_target = Net(self.config.actor_config.nn_arch,
                                self.config.actor_config.af_config,
                                self.config.actor_config.init_w)

        hard_update_inplace(self.actor_target, self.actor)

    def setup_critic(self):
        if self.config.critic_config.load_path is not None:
            self.critic = self._load(self.config.critic_config.load_path)
        else:
            self.critic = Net(self.config.critic_config.nn_arch,
                              self.config.critic_config.af_config,
                              self.config.critic_config.init_w)

        self.critic_optim = self.config.critic_config.optim(self.critic.parameters(),
                                                            lr=self.config.critic_config.lr)

        self.critic_target = Net(self.config.critic_config.nn_arch,
                                 self.config.critic_config.af_config,
                                 self.config.critic_config.init_w)

        hard_update_inplace(self.critic_target, self.critic)

    def init_model(self):
        self.setup_infer()
        self.setup_actor()
        self.setup_critic()

    def set_actor_target(self, path):
        self.actor_target = self._load(path)

    def set_critic_target(self, path):
        self.critic_target = self._load(path)

    def _load(self, path):
        if USE_CUDA:
            return torch.load(path)
        else:
            return torch.load(path, map_location=torch.device('cpu'))

    def _reward(self, reward, states):
        return reward

    def _action_match(self, action):
        action[action > 0] = 1.
        action[action <= 0] = -1.
        return action

    @counted
    def step(self, batch):
        self._print("learn")
        states = to_tensor(batch[0])
        actions = to_tensor(batch[1])
        rewards = self._reward(to_tensor(batch[2]), states)
        next_states = to_tensor(batch[3])
        done = to_tensor(batch[4])

        a = self.actor_target.forward(next_states)
        a = self._action_match(a)
        s_a = torch.cat((next_states, a), dim=1)
        q = self.critic_target.forward(s_a)
        y = torch.mul(q, self.config.rl_config.gamma)
        self._print("gamma", self.config.rl_config.gamma)
        self._print("rewards", rewards)
        self._print("q", q)

        y = torch.add(rewards, y).detach()
        self._print("y", y)

        actions = self._action_match(actions)
        s_a = torch.cat((states, actions), dim=1)
        q = self.critic.forward(s_a)
        l_critic = F.smooth_l1_loss(q, y, reduction='none')
        self._print("loss", l_critic)

        l_critic_per_batch = torch.sum(l_critic, dim=1, keepdim=True)
        ret_per_e = to_numpy(l_critic_per_batch)
        self._print('l_critic_per_batch', ret_per_e)

        if len(batch) > 5:
            weights = to_tensor(batch[5])
            self._print("weights", weights)
            l_critic = torch.mul(l_critic_per_batch, weights)
            self._print("w_l_critic", l_critic)

        l_critic = torch.mean(l_critic)

        self.critic_optim.zero_grad()
        l_critic.backward()
        self.critic_optim.step()

        a = self.actor.forward(states)
        s_a = torch.cat((states, a), dim=1)
        l_actor = self.critic.forward(s_a)

        l_actor_per_batch = torch.sum(l_actor, dim=1, keepdim=True)
        if len(batch) > 5:
            weights = to_tensor(batch[5])
            self._print("weights", weights)
            l_actor = torch.mul(l_actor_per_batch, weights)
            self._print("w_l_actor", l_actor)

        l_actor = torch.mean(torch.neg(l_actor))

        self.actor_optim.zero_grad()
        l_actor.backward()
        self.actor_optim.step()

        GLOBAL_LOGGER.get_tb_logger().add_scalar('DDPG.loss_actor', to_numpy(l_actor), self.n_step)
        GLOBAL_LOGGER.get_tb_logger().add_scalar('DDPG.loss_critic', to_numpy(l_critic), self.n_step)

        self.update_nn()

        self.step_counter += 1

        return ret_per_e

    def update_nn(self):
        if self.config.update_config.no_update:
            return
        if self.config.update_config.is_soft:
            soft_update_inplace(self.actor_target, self.actor, self.config.update_config.tau)
            soft_update_inplace(self.critic_target, self.critic, self.config.update_config.tau)
        elif self.step_counter % self.config.update_config.c_step == 0:
            hard_update_inplace(self.actor_target, self.actor)
            hard_update_inplace(self.critic_target, self.critic)

    def get_actor(self):
        return self.actor_target

    def save(self, path: str, postfix: str):
        torch.save(self.infer, os.path.join(path,"infer_" + postfix + ".pt"))

        torch.save(self.actor, os.path.join(path,"actor_" + postfix + ".pt"))
        torch.save(self.actor_target, os.path.join(path,"actor_target_" + postfix + ".pt"))

        torch.save(self.critic, os.path.join(path,"critic_" + postfix + ".pt"))
        torch.save(self.critic_target, os.path.join(path,"critic_target_" + postfix + ".pt"))

    def get_action(self, state):
        with torch.no_grad:
            ret = self.actor_target.forward(state)
        return ret