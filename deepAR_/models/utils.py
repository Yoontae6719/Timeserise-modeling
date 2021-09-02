import torch
import torch.nn as nn
import math
import numpy as np

def G_loss_fun(
             mu : torch.Tensor,
             sigma : torch.Tensor,
             labels : torch.Tensor,
             ):
    """
    Gaussian likilihood_loss
    Args:
        z : true obs. (i, t)
        mu :    mean  (i, t)
        sigma : std   (i, t)
    """

    log_likelihood_negative =   torch.log(sigma + 1) + ((labels - mu) ** 2) / (2 * sigma ** 2) + 5
    return log_likelihood_negative.mean()

def N_loss_fun(mu, alpha, lables):

    likelihood = torch.lgamma(lables + 1. / alpha) - torch.lgamma(lables + 1) - torch.lgamma(1. / alpha) \
        - 1. / alpha * torch.log(1 + alpha * mu) \
        + lables * torch.log(alpha * mu / (1 + alpha * mu))
    return - likelihood.mean()

def init_metrics(sample=True):
    metrics = {
        'ND': np.zeros(2),  # numerator, denominator
        'RMSE': np.zeros(3),  # numerator, denominator, time step count
        'test_loss': np.zeros(2),
    }
    if sample:
        metrics['rou90'] = np.zeros(2)
        metrics['rou50'] = np.zeros(2)
    return metrics