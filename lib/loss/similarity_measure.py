""" 
Different similarity measure between feature embeddings and prototyeps.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLSLoss(nn.Module, ABC):
    """ 
    Mutual Likelihood Score loss from "probabilistic face embeddings"
    """

    def __init__(self):
        super(MLSLoss, self).__init__()

    def _negative_mls_loss(self, mu, sigma):
        xx = torch.mul(mu, mu).sum(dim=1, keepdim=True)
        yy = torch.mul(mu.T, mu.T).sum(dim=0, keepdim=True)
        xy = torch.mm(mu, mu.T)
        mu_diff = xx + yy - 2 * xy
        sigma_sum = torch.mean(sigma, dim=1, keepdim=True) + \
            torch.sum(sigma.T, dim=0, keepdim=True)
        mls_score = mu_diff / (1e-8 + sigma_sum) + \
            mu.size(1) * torch.log(sigma_sum)

        return mls_score

    def forward(self, mu, log_sigma, labels=None):
        mu = F.normalize(mu)
        non_diag_mask = (1 - torch.eye(mu.size(0))).int().cuda()
        sigma = torch.exp(log_sigma)
        mls_loss = self._negative_mls_loss(mu, sigma)
        lable_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).int()
        positive_mask = non_diag_mask * lable_mask > 0
        positive_loss = mls_loss[positive_mask].mean()

        return positive_loss


class MutualLikelihoodScore(nn.Module, ABC):
    def __init__(self, configer):
        super(MutualLikelihoodScore, self).__init__()
        
        self.configer = configer
        
        
        