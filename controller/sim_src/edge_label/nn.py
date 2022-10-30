import torch
import torch.nn as nn
import torch_geometric as pyg

from torch_geometric.nn import GCNConv

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
        in_dims.append(in_node_channels)
        out_dims.append(hidden_node_channels)
        for l in range(n_layer-1):
            in_dims.append(hidden_node_channels * hidden_edge_channels)
            out_dims.append(hidden_node_channels)
        in_dims.append(hidden_node_channels * hidden_edge_channels)
        out_dims.append(out_node_channels)
        for i, j in enumerate(in_dims):
            in_dims[i] = int(in_dims[i])
        for i, j in enumerate(out_dims):
            out_dims[i] = int(out_dims[i])
        self.conv_list = nn.ModuleList()
        for l in range(n_layer):
            self.conv_list.append(nn.ModuleList())
            for c in range(hidden_edge_channels):
                self.conv_list[l].append(GCNConv(in_dims[l], out_dims[l],add_self_loops=False,normalize=False))

        self.read_out = nn.Sequential(
                    nn.Linear(hidden_node_channels * hidden_edge_channels, hidden_node_channels * hidden_edge_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_node_channels * hidden_edge_channels, hidden_node_channels * hidden_edge_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_node_channels * hidden_edge_channels, out_node_channels),
                )
        self.edge_emb = nn.Sequential(
                            nn.Linear(in_edge_channels, in_edge_channels * HIDDEN_DIM_MULTIPLIER),
                            nn.ReLU(),
                            nn.Linear(in_edge_channels * HIDDEN_DIM_MULTIPLIER, in_edge_channels * HIDDEN_DIM_MULTIPLIER),
                            nn.ReLU(),
                            nn.Linear(in_edge_channels * HIDDEN_DIM_MULTIPLIER, hidden_edge_channels),
                        )

    def forward(self, x, edge_index, edge_weight):
        edge_weight = self.edge_emb(edge_weight)

        layer:nn.ModuleList
        for layer in self.conv_list:
            x = torch.cat([e.forward(x, edge_index, edge_weight[:,i]).relu() for i, e in enumerate(layer)], dim=1)

        y = self.read_out(x)
        y = nn.functional.elu(y) + 1.001
        return y

class ELNN(nn.Module):
    def __init__(self, in_edge_channels,  out_edge_channels= 1, hidden_dim=None):
        super().__init__()
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
    def __init__(self, edge_feature, edge_type = 2, hidden_dim=None):
        super().__init__()
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
        out = nn.functional.softmax(self.network(x), dim=1)
        return out