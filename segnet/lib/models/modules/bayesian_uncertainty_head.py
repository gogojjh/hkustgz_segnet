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
        self.mean_conv = nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=1, bias=False)
        self.var_conv = nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=1, bias=False)

    def forward(self, x):
        mean = self.mean_conv(x)
        logvar = self.var_conv(x)  # [b k h w]
        #todo debug
        logvar = torch.sigmoid(logvar)

        return mean, logvar
