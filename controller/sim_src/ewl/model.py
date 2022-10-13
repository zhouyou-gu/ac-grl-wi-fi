import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, OneHotCategorical, LogNormal

def init_weights(m,gain=0.3):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=gain)
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
            nn.Sigmoid()
        )
        self.network.apply(init_weights)

    def forward(self, x):
        out = self.network(x)
        return out

class GaussianDensityNetwork(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim * 10
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim*2),
        )
        self.network.apply(init_weights)

    def forward(self, x):
        params = self.network(x)
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        # print(mean,sd)
        mean = 3*torch.tanh(mean.transpose(0, 1))
        sd = torch.mul(0.1*torch.abs(mean), torch.sigmoid(sd.transpose(0, 1)))
        # print(mean,sd)
        return LogNormal(mean, sd)

    def loss(self, x, y, sample_weighting, entropy = True):
        print("sample_weighting",sample_weighting)
        dis = self.forward(x)
        loglik = dis.log_prob(y)
        if entropy:
            en = dis.entropy()
            loss = -torch.mul(loglik,sample_weighting) - en
        else:
            loss = -torch.mul(loglik,sample_weighting)
        return loss

    def sample(self, x):
        dis = self.forward(x)
        samples = dis.sample()
        samples = torch.clamp(samples,min=0.1*dis.mean,max=10.*dis.mean)
        return samples, torch.squeeze(dis.mean), torch.squeeze(dis.scale)

class BernulliNetwork(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim * 10
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim*2),
        )
        init_weights(self,gain=0.5)

    def forward(self, x):
        params = self.network(x)
        pi = F.softmax(params, dim=1)
        return OneHotCategorical(probs=pi)

    def loss(self, x, y, sample_weighting, entropy = False):
        dis = self.forward(x)
        loglik = dis.log_prob(y)
        if entropy:
            en = dis.entropy()
            loss = -torch.mul(loglik,sample_weighting) - 0.05 * en
        else:
            loss = -torch.mul(loglik,sample_weighting)
        return loss

    def sample(self, x):
        dis = self.forward(x)
        samples = dis.sample()
        return samples, torch.squeeze(dis.mean), torch.squeeze(dis.variance)