"""
Probabilistic contrastive loss with each pixel being a Gaussian.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.tools.logger import Logger as Log
from lib.loss.loss_helper import FSCELoss, FSAuxCELoss


class MLSLoss(nn.Module, ABC):
    """ 
    Mutual Likelihood Loss from "probabilistic face embeddings"
    """

    def __init__(self, configer):
        super(MLSLoss, self).__init__()

        self.configer = configer

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


class ProbPPCLoss(nn.Module, ABC):
    """ 
    Pixel-wise probabilistic contrastive loss
    Probability masure: mutual likelihood loss
    """

    def __init__(self, configer):
        super(ProbPPCLoss, self).__init__()

        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']


class PixelProbContrastLoss(nn.Module, ABC):
    """
    Pixel-wise probabilistic contrastive loss
    Pixel embedding: Multivariate Gaussian
    """

    def __init__(self, configer):
        super(PixelProbContrastLoss, self).__init__()

        self.configer = configer
        self.temperature = self.configer.get('prob_contrast', 'temperature')

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')
        Log.info('ignore_index: {}'.format())

        self.seg_criterion = FSCELoss(configer=configer)  # seg loss
        self.prob_ppc_loss =

    def _prob_contrastive(self, x, y, queue=None):
        """ 
        x: feature
        y: label
        queue: stored representative samples of each class
        """
        fea_num, n_view = x.shape[0], x.shape[1]

    def forward(self, feats, labels=None, predict=None):
        if labels.shape[-1] != feats.shape[-1]:
            labels = labels.unsqueeze(1).float().clone()
            labels = torch.nn.functiontional.interpolate(labels,
                                                         (feats.shape[2],
                                                          feats.shape[3]),
                                                         mode='nearest')
            labels.squeeze(1).long()
            Log.info('Upsampling labels since labels are less than features.')
        assert labels.shape[-1] == feats.shape[-1], 'labels: {}, feats: {}'.format(
            labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(
            feats.shape[0], -1, feats.shape[-1])  # [N, H*W, C]
