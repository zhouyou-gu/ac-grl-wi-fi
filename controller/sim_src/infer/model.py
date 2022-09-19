import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, OneHotCategorical

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=0.1)
        torch.nn.init.zeros_(m.bias)

class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.normal_network = MixtureDiagNormalNetwork(dim_in, dim_out,
                                                       n_components)

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def loss(self, x, y):
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        # use pi.logits directly instead of torch.log(pi.probs) to
        # avoid numerical problem
        loss = -torch.logsumexp(pi.logits + loglik, dim=1)
        return loss

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples, pi.probs, torch.squeeze(normal.mean), torch.squeeze(normal.scale)


class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, n_components, hidden_dim=None):
        super().__init__()
        self.n_components = n_components
        if hidden_dim is None:
            hidden_dim = in_dim * 10
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim * n_components),
        )
        self.network.apply(init_weights)

    def forward(self, x):
        params = self.network(x)
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        # replaced torch.exp(sd) with ELU plus to improve numerical stability
        # added epsilon to avoid zero scale
        # due to non associativity of floating point add, 1 and 1e-7 need to be added seperately
        return Normal(mean.transpose(0, 1), (F.elu(sd)+1+1e-7).transpose(0, 1))

class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
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
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax()
        )
        self.network.apply(init_weights)

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(probs=params)


class MixtureDensityNetworkCombined1OutDim(nn.Module):
    def __init__(self, in_dim, n_components, hidden_dim=None):
        super().__init__()
        self.n_components = n_components
        if hidden_dim is None:
            hidden_dim = in_dim * 10
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 3 * n_components),
        )
        self.network.apply(init_weights)

    def forward(self, x):
        params = self.network(x)
        pi, mean, sd = torch.split(params, params.shape[1] // 3, dim=1)
        pi = torch.squeeze(torch.stack(pi.split(pi.shape[1] // self.n_components, 1)))
        pi = F.softmax(pi.transpose(0, 1), dim=1)
        # print(pi.transpose(0, 1))
        mean = torch.squeeze(torch.stack(mean.split(mean.shape[1] // self.n_components, 1)))
        sd = torch.squeeze(torch.stack(sd.split(sd.shape[1] // self.n_components, 1)))
        # replaced torch.exp(sd) with ELU plus to improve numerical stability
        # added epsilon to avoid zero scale
        # due to non associativity of floating point add, 1 and 1e-7 need to be added seperately
        return OneHotCategorical(probs=pi), Normal(mean.transpose(0, 1), (F.elu(sd)+1+1e-7).transpose(0, 1))

    def loss(self, x, y):
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.expand_as(normal.loc))
        # loglik = torch.sum(loglik, dim=1)
        # use pi.logits directly instead of torch.log(pi.probs) to
        # avoid numerical problem
        loss = -torch.logsumexp(pi.logits + loglik, dim=1)
        return loss

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample() * normal.sample(), dim=1,keepdim=True)
        return samples, pi.probs, torch.squeeze(normal.mean), torch.squeeze(normal.scale)