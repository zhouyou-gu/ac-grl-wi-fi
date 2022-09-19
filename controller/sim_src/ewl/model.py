import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, OneHotCategorical

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=0.3)
        torch.nn.init.zeros_(m.bias)


class FNN(nn.Module):
    def __init__(self, in_dim, out_dim = 1, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (in_dim+out_dim) * 10
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh(),
        )
        self.network.apply(init_weights)

    def forward(self, x):
        out = self.network(x)
        return out