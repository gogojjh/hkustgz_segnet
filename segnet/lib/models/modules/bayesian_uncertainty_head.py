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
        self.mean_conv = nn.Conv2d(self.proj_dim, 1, kernel_size=1, bias=False)
        self.std_conv = nn.Conv2d(self.proj_dim, 1, kernel_size=1, bias=False)
        kernel = torch.ones((7,7))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        #kernel = np.repeat(kernel, 1, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        
    def reparameterize(self, mu, logvar, k=1):
        sample_z = []
        for _ in range(k):
            # '_': inplace operation mul():dot product [b, 1, 128, 256]
            std = logvar.mul(0.5).exp_()  
            eps = std.data.new(std.size()).normal_() # mu + epsilon * var
            sample_z.append(eps.mul(std).add_(mu))
        sample_z = torch.cat(sample_z, dim=1)
        
        return sample_z
    
    def forward(self, x):
        mean = self.mean_conv(x)
        std = self.std_conv(x)
        
        prob_x = self.reparameterize(mean, std, 1) # for loss calculation [1, 1, h/128, w/256]
        prob_out2 = self.reparameterize(mean, std, 50) # for calculating uncertainty
        prob_out2 = torch.sigmoid(prob_out2)
        
        # uncertainty
        uncertainty = prob_out2.var(dim=1, keepdim=True).detach()
        
        if self.configer.get('phase') == 'train':
            # (l-7+2*3)/1+1=l
            # to make uncertainty more even for subsequent masking and uncertainty-aware features
            #todo dilate? why 3 times?
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
        # normalize
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
        
        # mean3 = prob_out2.mean(dim=1, keepdim=True) # mean
        std3 = prob_out2.var(dim=1, keepdim=True) # uncertainty
        std3 = (std3 - std3.min()) / (std3.max() - std3.min())
        
        return std3, prob_x, uncertainty, mean, std.mul(0.5).exp_() 
            
            
        
            