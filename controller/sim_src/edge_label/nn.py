import torch
import torch.nn as nn
import torch_geometric as pyg

from torch_geometric.nn import GCNConv, SAGEConv

HIDDEN_DIM_MULTIPLIER = 10

def init_weights(m,gain=1.):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=gain)
        torch.nn.init.zeros_(m.bias)


class WGNN(nn.Module):
    def __init__(self, in_node_channels=5, hidden_node_channels=2, out_node_channels=1, in_edge_channels = 1, hidden_edge_channels = 2, n_layer = 3):
        nn.Module.__init__(self)
        assert n_layer > 1

        in_dims = []
        out_dims = []
        for l in range(n_layer):
            in_dims.append(hidden_node_channels)
            out_dims.append(hidden_node_channels)
        in_dims.append(hidden_node_channels)
        out_dims.append(out_node_channels)
        for i, j in enumerate(in_dims):
            in_dims[i] = int(in_dims[i])
        for i, j in enumerate(out_dims):
            out_dims[i] = int(out_dims[i])
        self.conv_list = nn.ModuleList()
        for l in range(n_layer):
            self.conv_list.append(nn.ModuleList())
            for c in range(hidden_edge_channels):
                self.conv_list[l].append(GCNConv(in_dims[l], out_dims[l],add_self_loops=True,normalize=True))

        self.in_edge_emb = nn.Sequential(
                nn.Linear(in_edge_channels, in_edge_channels * HIDDEN_DIM_MULTIPLIER),
                nn.ReLU(),
                nn.Linear(in_edge_channels * HIDDEN_DIM_MULTIPLIER, in_edge_channels * HIDDEN_DIM_MULTIPLIER),
                nn.ReLU(),
                nn.Linear(in_edge_channels * HIDDEN_DIM_MULTIPLIER, hidden_edge_channels),
                nn.ReLU(),
            )

        self.in_node_emb = nn.Sequential(
                nn.Linear(in_node_channels, in_node_channels * HIDDEN_DIM_MULTIPLIER),
                nn.ReLU(),
                nn.Linear(in_node_channels * HIDDEN_DIM_MULTIPLIER, in_node_channels * HIDDEN_DIM_MULTIPLIER),
                nn.ReLU(),
                nn.Linear(in_node_channels * HIDDEN_DIM_MULTIPLIER, hidden_node_channels),
                nn.ReLU(),
            )

        self.hidden_node_emb = nn.Sequential(
                        nn.Linear(hidden_node_channels * hidden_edge_channels, hidden_node_channels * hidden_edge_channels * HIDDEN_DIM_MULTIPLIER),
                        nn.ReLU(),
                        nn.Linear(hidden_node_channels * hidden_edge_channels * HIDDEN_DIM_MULTIPLIER, hidden_node_channels * hidden_edge_channels * HIDDEN_DIM_MULTIPLIER),
                        nn.ReLU(),
                        nn.Linear(hidden_node_channels * hidden_edge_channels * HIDDEN_DIM_MULTIPLIER, hidden_node_channels),
                        nn.ReLU(),
                    )

        self.read_out = nn.Sequential(
                    nn.Linear(hidden_node_channels + in_node_channels, (hidden_node_channels + in_node_channels) * HIDDEN_DIM_MULTIPLIER),
                    nn.ReLU(),
                    nn.Linear((hidden_node_channels + in_node_channels) * HIDDEN_DIM_MULTIPLIER, (hidden_node_channels + in_node_channels) * HIDDEN_DIM_MULTIPLIER),
                    nn.ReLU(),
                    nn.Linear((hidden_node_channels + in_node_channels) * HIDDEN_DIM_MULTIPLIER, out_node_channels),
                )
    def forward(self, in_x, edge_index, edge_weight):
        x = self.in_node_emb.forward(in_x)
        edge_weight = self.in_edge_emb(edge_weight)

        layer:nn.ModuleList
        for layer in self.conv_list:
            # print(layer[0].forward(x, edge_index, edge_weight[:,0]).relu())
            x = torch.cat([e.forward(x, edge_index, edge_weight[:,i]).relu() for i, e in enumerate(layer)], dim=1)
            x = self.hidden_node_emb.forward(x)

        y = self.read_out(torch.cat((x,in_x),dim=1)).sigmoid()
        # y = nn.functional.elu(y-1.) + 1.001
        return y

class ELNN(nn.Module):
    def __init__(self, in_edge_channels,  out_edge_channels= 1, hidden_dim=None):
        nn.Module.__init__(self)
        if hidden_dim is None:
            hidden_dim = in_edge_channels * HIDDEN_DIM_MULTIPLIER
        self.network = nn.Sequential(
            nn.Linear(in_edge_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_edge_channels),
        )
    def forward(self, x):
        out = self.network(x).sigmoid()
        return out


class INFNN(nn.Module):
    def __init__(self, edge_feature, edge_type = 1, hidden_dim=None):
        nn.Module.__init__(self)
        if hidden_dim is None:
            hidden_dim = edge_feature * HIDDEN_DIM_MULTIPLIER
        self.network = nn.Sequential(
            nn.Linear(edge_feature, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_type),
        )
    def forward(self, x):
        # out = nn.functional.softmax(self.network(x), dim=1)
        out = self.network(x).sigmoid()
        return out


class MPGNN(nn.Module):
    def __init__(self, in_node_channels=5, hidden_node_channels=5, out_node_channels=1, n_layer = 3):
        nn.Module.__init__(self)
        assert n_layer > 1

        in_dims = []
        out_dims = []
        for l in range(n_layer):
            in_dims.append(hidden_node_channels)
            out_dims.append(hidden_node_channels)
        in_dims.append(hidden_node_channels)
        out_dims.append(out_node_channels)
        for i, j in enumerate(in_dims):
            in_dims[i] = int(in_dims[i])
        for i, j in enumerate(out_dims):
            out_dims[i] = int(out_dims[i])
        self.conv_list = nn.ModuleList()
        for l in range(n_layer):
            self.conv_list.append(SAGEConv(in_dims[l], out_dims[l]))

        self.in_node_emb = nn.Sequential(
                nn.Linear(in_node_channels, in_node_channels * HIDDEN_DIM_MULTIPLIER),
                nn.ReLU(),
                nn.Linear(in_node_channels * HIDDEN_DIM_MULTIPLIER, in_node_channels * HIDDEN_DIM_MULTIPLIER),
                nn.ReLU(),
                nn.Linear(in_node_channels * HIDDEN_DIM_MULTIPLIER, hidden_node_channels),
                nn.ReLU(),
            )

        self.read_out = nn.Sequential(
                    nn.Linear(hidden_node_channels + in_node_channels, (hidden_node_channels + in_node_channels) * HIDDEN_DIM_MULTIPLIER),
                    nn.ReLU(),
                    nn.Linear((hidden_node_channels + in_node_channels) * HIDDEN_DIM_MULTIPLIER, (hidden_node_channels + in_node_channels) * HIDDEN_DIM_MULTIPLIER),
                    nn.ReLU(),
                    nn.Linear((hidden_node_channels + in_node_channels) * HIDDEN_DIM_MULTIPLIER, out_node_channels),
                )
    def forward(self, in_x, edge_index):
        x = self.in_node_emb.forward(in_x)
        layer:nn.ModuleList
        for layer in self.conv_list:
            x = layer.forward(x, edge_index).relu()

        y = self.read_out(torch.cat((x,in_x),dim=1))
        return y

class REGNN(nn.Module):
    def __init__(self, hidden_layer = 5, filter_coefficient = 5, out_dim = 4):
        nn.Module.__init__(self)
        self.alpha_hidden = torch.nn.Parameter(torch.ones(hidden_layer,filter_coefficient)/filter_coefficient/10)
        self.alpha_out = torch.nn.Parameter(torch.ones(out_dim,filter_coefficient)/filter_coefficient/10)

    def forward(self, H, x):
        H = (H+1)/2.
        x = (x+1)/2
        for i in range(self.alpha_hidden.shape[0]):
            y = []
            for j in range(self.alpha_hidden.shape[1]):
                yy = self.alpha_hidden[i,j] * torch.matmul(torch.linalg.matrix_power(H,j),x)
                y.append(yy)
            y = torch.cat(y,dim=1)
            y = torch.sum(y,dim=1,keepdim=True)
            x = nn.functional.relu(y)
            # print(x)
            x = x / torch.mean(x)
            print(x)

            # x = x / torch.max(x)

        out = []
        for i in range(self.alpha_out.shape[0]):
            y = torch.cat([self.alpha_out[i,j] * torch.matmul(torch.linalg.matrix_power(H,j),x) for j in range(self.alpha_out.shape[1])], dim=1)
            o = torch.sum(y,dim=1,keepdim=True)
            out.append(o)
        out = torch.cat(out,dim=1)
        print(out)
        out = torch.softmax(out,dim=1)
        print(out)
        return out


class REGNN_BIN(nn.Module):
    def __init__(self, hidden_layer = 5, filter_coefficient = 5, out_dim = 2):
        nn.Module.__init__(self)

        self.alpha_hidden = torch.nn.Parameter(torch.randn(out_dim,hidden_layer,filter_coefficient))
    def forward(self, H, x):
        H = (H+1)/2.
        x = (x+1)/2
        print(self.alpha_hidden)
        out = []
        for a in range(self.alpha_hidden.shape[0]):
            for i in range(self.alpha_hidden.shape[1]):
                y = []
                for j in range(self.alpha_hidden.shape[2]):
                    yy = self.alpha_hidden[a,i,j] * torch.matmul(torch.linalg.matrix_power(H,j),x)
                    y.append(yy)
                print(y)
                y = torch.cat(y,dim=1)
                y = torch.sum(y,dim=1,keepdim=True)
                x = nn.functional.leaky_relu(y)
                print("x",a,i,x)
                # x = x / torch.numel(x) / self.alpha_hidden.shape[2]
                x = x / torch.mean(x).detach()
                x = x - torch.mean(x).detach()
                print("x",a,i,x)
            out.append(x)

        out = torch.cat(out,dim=1)
        print(out)
        out = nn.functional.sigmoid(out)
        print(out)
        return out