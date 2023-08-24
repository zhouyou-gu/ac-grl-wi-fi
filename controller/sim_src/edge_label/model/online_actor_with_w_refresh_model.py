from sim_src.edge_label.model.itl_bidirection_interference import *


class online_actor_with_w_refresh_model(itl_bidirection_interference):
    N_INFER_STEP = 10
    N_WEIGHT_REFRESH_STEP = 20
    def setup_weight(self,states):
        n_node = states.shape[0]
        with torch.no_grad():
            G = nx.complete_graph(n_node)
            e_index = to_device(from_networkx(G).edge_index)
            state = to_tensor(states)
            state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))

            result = self.infer_target.forward(state)
            p_contention = result

            state = to_tensor(states,requires_grad=False)
            min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

            # state_A = state[e_index[0,:]]
            # state_A_to_B = torch.gather(state_A,1,min_path_loss_idx[e_index[1,:]])
            state_B = state[e_index[1,:]]
            state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])
            interference = state_B_to_A
            interference_and_min_pl_and_p_contention = torch.hstack((interference,min_path_loss[e_index[0,:]],min_path_loss[e_index[1,:]],p_contention))

            # asso = torch.eq(min_path_loss_idx[e_index[0,:]], min_path_loss_idx[e_index[1,:]] ).float()
            # actor_input = torch.hstack((asso,interference_and_min_pl_and_p_contention))
            actor_input = interference_and_min_pl_and_p_contention

        label = self.actor_target.forward(actor_input).detach().clone()
        self.label = torch.log(label/(1-label)).clone()
        self.label.requires_grad_()

        self.w_optim = optim.Adam([self.label], lr=0.001)

    def gen_action(self, state_np):
        if self.N_STEP % self.N_WEIGHT_REFRESH_STEP == 0:
            self.setup_weight(state_np)
        n_node = state_np.shape[0]
        with torch.no_grad():
            G = nx.complete_graph(n_node)
            e_index = to_device(from_networkx(G).edge_index)

            label = self.label.sigmoid()

            action_mat = to_dense_adj(edge_index=e_index,batch=None,edge_attr=label).view(n_node,n_node)
            action = to_numpy(action_mat)

        return action


    def _train_infer(self,batch):
        pass

    def _train_actor(self,batch):
        self._print("_train_actor")
        loss = to_device(torch.zeros(1))
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            self._printa("_train_actor n_node\t",sample['n_node'])

            e_index = to_device(from_networkx(G).edge_index)
            with torch.no_grad():
                reward = to_tensor(sample['reward'],requires_grad=False)
                min_r_idx = torch.argmin(reward)

                state = to_tensor(sample['state'],requires_grad=False)
                state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))
                result = self.infer_target.forward(state)
                p_contention = result

                state = to_tensor(sample['state'],requires_grad=False)
                min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

                x = min_path_loss

                # state_A = state[e_index[0,:]]
                # state_A_to_B = torch.gather(state_A,1,min_path_loss_idx[e_index[1,:]])
                state_B = state[e_index[1,:]]
                state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])
                interference = state_B_to_A
                interference_and_min_pl_and_p_contention = torch.hstack((interference,min_path_loss[e_index[0,:]],min_path_loss[e_index[1,:]],p_contention))
                # asso = torch.eq(min_path_loss_idx[e_index[0,:]], min_path_loss_idx[e_index[1,:]] ).float()
                # actor_input = torch.hstack((asso,interference_and_min_pl_and_p_contention))
                actor_input = interference_and_min_pl_and_p_contention

                label = self.label.sigmoid()

            label.requires_grad_()

            # s_a = torch.hstack((asso,interference,p_contention,label))
            s_a = torch.hstack((interference,p_contention,label))
            self._printa("_train_actor states\n",to_numpy(state).T)
            self._printa("_train_actor minidA\n",to_numpy(min_path_loss_idx[e_index[0,:]]).T)
            self._printa("_train_actor minidB\n",to_numpy(min_path_loss_idx[e_index[1,:]]).T)
            self._printa("_train_actor sapair\n",to_numpy(s_a).T)
            self._printa("_train_actor minlos\n",to_numpy(torch.hstack((min_path_loss[e_index[0,:]],min_path_loss[e_index[1,:]]))).T)

            qq = self.critic_target.forward(x,e_index,s_a)
            qq.retain_grad()
            self._printa("_train_actor qq.min\n",qq[min_r_idx])

            q = -qq[min_r_idx]
            q.backward()
            lable_g = label.grad.detach()
            label_differentiable = self.label.sigmoid()
            loss += (-torch.mean((lable_g)*label_differentiable))
            self._printa("_train_actor qq.grad\n",to_numpy(qq.grad).T)
            self._printa("_train_actor label.grad\n",to_numpy(label.grad).T)
            self._printa("_train_actor label\n",to_numpy(label).T)

        loss/=len(batch)
        self._print("_train_actor loss",loss)
        self.w_optim.zero_grad()
        loss.backward()
        self.w_optim.step()

        return to_numpy(loss)

    def _train_critic(self,batch):
        pass

    def _fair_q_r(self,q,r):
        assert self.FAIRNESS_ALPHA >=0
        self._printa("_fair_q qraw",q.transpose(0,1))
        if self.FAIRNESS_ALPHA == 1:
            q = q/r
            q = torch.mean(q,dim=0,keepdim=True)
            q = torch.exp(q)
            self._printa("_fair_q fair pf",q.transpose(0,1))
        elif self.FAIRNESS_ALPHA > self.MAX_FAIR_ALPHA:
            min_r_idx = torch.argmin(r)
            q = q[min_r_idx]
            self._printa("_fair_q fair min",q)
        else:
            q = q*torch.pow(r+0.001,-self.FAIRNESS_ALPHA)/(1-self.FAIRNESS_ALPHA)
            q = torch.mean(q,dim=0,keepdim=True)
            q = torch.pow(q*(1-self.FAIRNESS_ALPHA),1/(1-self.FAIRNESS_ALPHA))
            self._printa("_fair_q fair",self.FAIRNESS_ALPHA,q.transpose(0,1))
        return q