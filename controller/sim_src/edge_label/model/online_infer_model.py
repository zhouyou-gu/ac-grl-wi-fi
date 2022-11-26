from sim_src.edge_label.model.complete_model import *


class online_infer_model(infer_then_label_model):
    N_INFER_STEP = 10
    def setup_p_infer(self,n_nodes,states):
        self.n_nodes = n_nodes
        G = nx.complete_graph(n_nodes)
        e_index = to_device(from_networkx(G).edge_index)
        state = to_tensor(states)
        state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))
        # init_p = self.infer_target.forward(state).detach().clone()
        # self.p_contention = torch.log(init_p[:,1:2]/(1-init_p[:,1:2])).clone()
        self.p_contention = torch.zeros((state.shape[0],1))
        self.p_contention.requires_grad_()
        self.p_optim = optim.Adam([self.p_contention], lr=0.1)
    def gen_action(self, state_np):
        n_node = state_np.shape[0]
        with torch.no_grad():
            G = nx.complete_graph(n_node)
            e_index = to_device(from_networkx(G).edge_index)
            state = to_tensor(state_np)
            state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))

            self._printa("gen_action p_contention\n", to_numpy(self.p_contention.sigmoid()))
            inference = torch.bernoulli(self.p_contention.sigmoid())

            state = to_tensor(state_np,requires_grad=False)
            min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

            state_A = state[e_index[0,:]]
            state_A_to_B = torch.gather(state_A,1,min_path_loss_idx[e_index[1,:]])
            state_B = state[e_index[1,:]]
            state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])
            interference = torch.hstack((state_A_to_B,state_B_to_A))
            interference_and_min_pl_and_p_contention = torch.hstack((interference,min_path_loss[e_index[0,:]],min_path_loss[e_index[1,:]],self.p_contention.sigmoid()))

            asso = torch.eq(min_path_loss_idx[e_index[0,:]], min_path_loss_idx[e_index[1,:]] ).float()
            actor_input = torch.hstack((asso,interference_and_min_pl_and_p_contention))

            label = self.actor_target.forward(actor_input)

            action_mat = to_dense_adj(edge_index=e_index,batch=None,edge_attr=label).view(n_node,n_node)
            action = to_numpy(action_mat)

        return action

    def _train_infer(self,batch):
        self._print("_train_infer")
        loss = to_device(torch.zeros(1))
        for sample in batch:
            for ii in range(self.N_INFER_STEP):
                G = nx.complete_graph(sample['n_node'])
                e_index = to_device(from_networkx(G).edge_index)

                action = to_tensor(sample['action'],requires_grad=False)
                action = action[e_index[0],e_index[1]].view(-1,1)

                state = to_tensor(sample['state'],requires_grad=False)
                min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

                x = min_path_loss

                state_A = state[e_index[0,:]]
                state_A_to_B = torch.gather(state_A,1,min_path_loss_idx[e_index[1,:]])
                state_B = state[e_index[1,:]]
                state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])
                interference = torch.hstack((state_A_to_B,state_B_to_A))

                asso = torch.eq(min_path_loss_idx[e_index[0,:]], min_path_loss_idx[e_index[1,:]] ).float()

                inference = torch.bernoulli(self.p_contention.sigmoid()).detach()

                s_a = torch.hstack((asso,interference,inference,action))
                self._printa("_train_infer sapair\n",to_numpy(s_a).T)

                q = self.critic_target.forward(x,e_index,s_a)

                e_pi = (1-inference)*torch.log(1-self.p_contention.sigmoid()) + inference*torch.log(self.p_contention.sigmoid())
                q_diff = nn.functional.mse_loss(q.detach(),to_tensor(sample['reward']),reduction="none")
                q_diff = q_diff[e_index[0,:]]+q_diff[e_index[1,:]]
                # print(e_pi.shape,q_diff.shape)

                loss += torch.mean(q_diff * e_pi)
                self._printa("_train_infer critic",to_numpy(q).T)
                self._printa("_train_infer reward",sample['reward'].T)

                target = to_tensor(sample['target'],requires_grad=False)
                target = target[e_index[0],e_index[1]].view(-1,1)

                target_and_predict = torch.hstack((target,self.p_contention.sigmoid()))
                self._printa("_train_infer tarpre\n",to_numpy(target_and_predict).T)

                self._printa("_train_infer accu",torch.mean(torch.eq(target,torch.bernoulli(self.p_contention.sigmoid())).float()))

        loss/=len(batch)
        self._print("_train_infer loss",loss)
        self.p_optim.zero_grad()
        loss.backward()
        print(self.p_contention.grad)
        self.p_optim.step()

        return to_numpy(loss)

    def _train_actor(self,batch):
        pass
    def _train_critic(self,batch):
        pass
