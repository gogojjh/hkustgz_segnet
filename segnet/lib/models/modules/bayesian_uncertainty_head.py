import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.models.tools.module_helper import ModuleHelper


class BayesianUncertaintyHead(nn.Module):
    ''' 
    "Uncertainty-Guided Transformer Reasoning for Camouflaged Object Detection"
    '''

    def __init__(self, configer):
        super(BayesianUncertaintyHead, self).__init__()
        self.configer = configer                                            

        self.proj_dim = self.configer.get('protoseg', 'proj_dim')
        self.reparam_k = self.configer.get('protoseg', 'reparam_k')
        self.mean_conv = nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=1, bias=False)
        self.var_conv = nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=1, bias=False)

    def reparameterize(self, mu, logvar, k=1):
        sample_z = []
        for _ in range(k):
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            sample_z.append(eps.mul(std).add_(mu))
        sample_z = torch.cat(sample_z, dim=0)
        return sample_z

    def forward(self, x):
        mean = self.mean_conv(x)
        logvar = self.var_conv(x)  # [b k h w]
        logvar = logvar.unsqueeze(0)  # [1 b k h w]
        logvar_reparam = self.reparameterize(mean, logvar, 1).squeeze(0)  # [b k h w]
        logvar_reparam = torch.sigmoid(logvar_reparam)
        
        uncertainty = self.reparameterize(mean, logvar, self.reparam_k)  # [8 b k h w]
        uncertainty = torch.sigmoid(uncertainty)
        uncertainty = logvar.var(dim=0)  # [b k h w]
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())

        return mean, logvar_reparam, uncertainty
