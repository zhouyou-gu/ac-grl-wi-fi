from sim_src.edge_label.model.complete_model import *

class online_model(complete_model):
    def _train_infer(self,batch):
        self._print("_train_infer")
        loss = to_device(torch.zeros(1))
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            e_index = to_device(from_networkx(G).edge_index)
            state = to_tensor(sample['state'],requires_grad=False)
            state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))

            result = self.infer.forward(state)
            p_contention = result[:,1:2]

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
            s_a = torch.hstack((asso,interference,p_contention,action))
            self._printa("_train_infer sapair\n",to_numpy(s_a).T)

            q = self.critic_target.forward(x,e_index,s_a)
            loss += nn.functional.mse_loss(q,to_tensor(sample['reward']))
            self._printa("_train_infer critic",to_numpy(q).T)
            self._printa("_train_infer reward",sample['reward'].T)

            target = to_tensor(sample['target'],requires_grad=False)
            target = target[e_index.transpose(0,1)[:,0],e_index.transpose(0,1)[:,1]]
            target = nn.functional.one_hot(target.long(), num_classes=2).float()

            print(target.shape,p_contention.shape)
            target_and_predict = torch.hstack((target,result))
            self._printa("_train_infer tarpre\n",to_numpy(target_and_predict).T)


        loss/=len(batch)
        self._print("_train_infer loss",loss)
        self.infer_optim.zero_grad()
        loss.backward()
        self.infer_optim.step()

        return to_numpy(loss)

    def _train_actor(self,batch):
        pass
    def _train_critic(self,batch):
        pass


