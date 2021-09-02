import torch
from torch import nn


class Gaussian(nn.Module):

    def __init__(self, hidden_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)

    def forward(self, h):
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = self.mu_layer(h)
        return mu_t, sigma_t

def gaussian_sample(mu, sigma):

    gaussian = torch.distributions.normal.Normal(mu, sigma)
    ypred = gaussian.sample(mu.size())
    return ypred

class NegativeBinomial(nn.Module):

    def __init__(self, raw_params):
        '''
        Negative Binomial likelihood
        Args:
            hidden_size(int) : columns size of hidden h_{i, t}
            output_size(int) : embedding size
        '''

        super(NegativeBinomial, self).__init__()
        params = dict(raw_params)
        self.mu = nn.Linear(params["lstm_hidden_dim"] * params["lstm_layers"], 1)
        self.sigma = nn.Linear(params["lstm_hidden_dim"] * params["lstm_layers"], 1)

    def forward(self, h):
        _, hidden_size = h.size()
        alpha_t = torch.log(1 + torch.exp(self.sigma(h))) + 1e-5
        mu_t = torch.log(1 + torch.exp(self.mu(h)))
        return mu_t, alpha_t

def negative_binomial_sample(mu, alpha):
    var = mu + mu * mu * alpha
    ypred = mu + torch.randn(mu.size()) * torch.sqrt(var)
    return ypred
