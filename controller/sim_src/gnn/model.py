import torch
import torch.nn as nn
import torch_geometric as pyg

from torch_geometric.nn import GATv2Conv, GCNConv
def init_weights(m,gain=1.):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=gain)
        torch.nn.init.zeros_(m.bias)

class WGNN(nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, edge_dim = 2):
        super().__init__()
        self.conv01 = GCNConv(in_channels, hidden_channels,add_self_loops=False,normalize=False)
        self.conv02 = GCNConv(in_channels, hidden_channels,add_self_loops=False,normalize=False)
        self.conv11 = GCNConv(hidden_channels*2, hidden_channels,add_self_loops=False,normalize=False)
        self.conv12 = GCNConv(hidden_channels*2, hidden_channels,add_self_loops=False,normalize=False)
        self.conv21 = GCNConv(hidden_channels*2, hidden_channels,add_self_loops=False,normalize=False)
        self.conv22 = GCNConv(hidden_channels*2, hidden_channels,add_self_loops=False,normalize=False)

        self.read_out = nn.Sequential(
                    nn.Linear(hidden_channels*2, hidden_channels*2),
                    nn.ELU(),
                    nn.Linear(hidden_channels*2, hidden_channels*2),
                    nn.ELU(),
                    nn.Linear(hidden_channels*2, out_channels),
                )
        self.edge_emb = nn.Sequential(
                            nn.Linear(edge_dim, hidden_channels*2),
                            nn.ELU(),
                            nn.Linear(hidden_channels*2, hidden_channels*2),
                            nn.ELU(),
                            nn.Linear(hidden_channels*2, edge_dim),
                        )

    def forward(self, x, edge_index, edge_weight):
        edge_weight = self.edge_emb(edge_weight)
        edge_weight1 = edge_weight[:,0]
        edge_weight2 = edge_weight[:,1]
        # print(edge_weight,"0000000000000000000")

        y1 = self.conv01.forward(x, edge_index, edge_weight1).relu()
        y2 = self.conv02.forward(x, edge_index, edge_weight2).relu()
        y = torch.cat((y1,y2),dim=1)

        # print(y,"1111111111111111")

        y1 = self.conv11.forward(y, edge_index, edge_weight1).relu()
        y2 = self.conv12.forward(y, edge_index, edge_weight2).relu()
        y = torch.cat((y1,y2),dim=1)

        # print(y,"2222222222222222")

        y1 = self.conv21.forward(y, edge_index, edge_weight1).relu()
        y2 = self.conv22.forward(y, edge_index, edge_weight2).relu()
        y = torch.cat((y1,y2),dim=1)

        # print(y,"3333333333333333")

        y = self.read_out(y)

        return y

class WGNN_GAT(nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, edge_dim = 1):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim,add_self_loops=False)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim,add_self_loops=False)
        self.conv3 = GATv2Conv(hidden_channels, out_channels, edge_dim=edge_dim,add_self_loops=False)

    def forward(self, x, edge_index, edge_weight):
        print("0000000",x,edge_weight)
        y = self.conv1.forward(x, edge_index, edge_weight).relu()
        print("1111111",y)
        y = self.conv2.forward(y, edge_index, edge_weight).relu()
        print("2222222",y)
        y = torch.nn.functional.leaky_relu(self.conv3.forward(y, edge_index, edge_weight))
        print("3333333",y)
        return y

class ELNN(nn.Module):
    def __init__(self, in_dim, out_dim = 1, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (in_dim+out_dim) * 10
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # self.network.apply(init_weights)

    def forward(self, x):
        out = self.network(x).sigmoid()
        return out

if __name__ == '__main__':
    import networkx as nx
    G = nx.complete_graph(10)
    d = pyg.utils.from_networkx(G)

    print(d.edge_weight)