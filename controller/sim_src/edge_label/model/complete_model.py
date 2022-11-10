from sim_src.edge_label.model.infer_then_label_model import *

class complete_model(infer_then_label):
    def _train_actor(self,batch):
        self._print("_train_actor")
        loss = to_device(torch.zeros(1))
        for sample in batch:
            G = nx.complete_graph(sample['n_node'])
            e_index = to_device(from_networkx(G).edge_index)
            with torch.no_grad():
                state = to_tensor(sample['state'],requires_grad=False)
                state = torch.hstack((state[e_index[0,:]],state[e_index[1,:]]))
                result = self.infer.forward(state)
                p_contention = result[:,1:2]

                state = to_tensor(sample['state'],requires_grad=False)
                min_path_loss, min_path_loss_idx = torch.min(state,dim=1,keepdim=True)

                x = min_path_loss

                state_A = state[e_index[0,:]]
                state_A_to_B = torch.gather(state_A,1,min_path_loss_idx[e_index[1,:]])
                state_B = state[e_index[1,:]]
                state_B_to_A = torch.gather(state_B,1,min_path_loss_idx[e_index[0,:]])
                interference = torch.hstack((state_A_to_B,state_B_to_A))
                interference_and_min_pl_and_p_contention = torch.hstack((interference,min_path_loss[e_index[0,:]],min_path_loss[e_index[1,:]],p_contention))
                asso = torch.eq(min_path_loss_idx[e_index[0,:]], min_path_loss_idx[e_index[1,:]] ).float()
                actor_input = torch.hstack((asso,interference_and_min_pl_and_p_contention))

                label = self.actor.forward(actor_input)

            label_no_grad = label.clone()
            label.requires_grad_()

            s_a = torch.hstack((asso,interference,p_contention,label))
            self._printa("_train_actor states\n",to_numpy(state).T)
            self._printa("_train_actor minidA\n",to_numpy(min_path_loss_idx[e_index[0,:]]).T)
            self._printa("_train_actor minidB\n",to_numpy(min_path_loss_idx[e_index[1,:]]).T)
            self._printa("_train_actor sapair\n",to_numpy(s_a).T)
            self._printa("_train_actor minlos\n",to_numpy(torch.hstack((min_path_loss[e_index[0,:]],min_path_loss[e_index[1,:]]))).T)

            q = self.critic_target.forward(x,e_index,s_a)
            q = self._fair_q(q)
            q = -torch.mean(q)
            q.backward()
            lable_g = label.grad.detach()
            scale = label_no_grad
            scale[lable_g>0.] = 1. - scale[lable_g>0.]
            label_differentiable = self.actor.forward(actor_input)
            loss += (-torch.mean((lable_g*scale)*label_differentiable))
            self._printa("_train_actor label.grad\n",to_numpy(label.grad).T)
            self._printa("_train_actor lable_g\n",to_numpy(lable_g).T)
            self._printa("_train_actor scale\n",to_numpy(scale).T)

        loss/=len(batch)
        self._print("_train_actor loss",loss)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return to_numpy(loss)

