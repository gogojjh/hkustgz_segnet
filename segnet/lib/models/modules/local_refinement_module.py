import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.models.tools.module_helper import ModuleHelper
from einops import rearrange, repeat


class ResBlock(nn.Module):
    def __init__(self, configer):
        super(ResBlock, self).__init__()
        
        self.configer = configer
        
        in_dim = self.configer.get('protoseg', 'proj_dim')
        out_dim = in_dim
        stride=1
        
        if in_dim == out_dim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, stride=stride)
        
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=stride)
        
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x + r


class LocalRefinementModule(nn.Module):
    ''' 
    Video Object Segmentation with Adaptive Feature Bank and Uncertain-Region Refinement
    
    Use the pixels which havehigh confidence in classification to refine the other uncertain points in its neighborhood.
    '''
    def __init__(self, configer):
        super(LocalRefinementModule, self).__init__()
        self.configer = configer
        
        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.bin_size_h = self.configer.get('protoseg', 'bin_size_h')
        self.bin_size_w = self.configer.get('protoseg', 'bin_size_w')
        self.proj_dim = self.configer.get('protoseg', 'proj_dim')        
        
    def calc_pred_uncertainty(self, sim_mat):
        ''' 
        sim_mat: [n c]
        '''
        sim_mat = F.softmax(sim_mat, dim=1)
        score_top, _ = sim_mat.topk(k=2, dim=1)
        pred_uncertainty = score_top[:, 0] / (score_top[:, 1] +  1e-8)
        pred_uncertainty = torch.exp(1 - pred_uncertainty)
        
        return pred_uncertainty
    
    def patch_split(self, x, bin_size_h, bin_size_w):
        """
        b c (bh rh) (bw rw) -> b (bh bw) rh rw c
        """
        b, c, h, w = x.size()
        patch_size_h = h // bin_size_h
        patch_size_w = w // bin_size_w
        x = x.view(b, c, bin_size_h, patch_size_h, bin_size_w, patch_size_w)
        # [b, bin_size_h, bin_size_w, patch_size_h, patch_size_w, c]
        x = x.permute(0,2,4,3,5,1).contiguous() 
        # [b, bin_size_h*bin_size_w, patch_size_h, patch_size_w, c]
        x = x.view(b, -1, patch_size_h, patch_size_w, c)
        
        return x
    
    def patch_recover(self, x, bin_size_h, bin_size_w):
        """
        b (bh bw) rh rw c -> b c (bh rh) (bw rw)
        """
        b, n, patch_size_h, patch_size_w, c = x.size()
        h = patch_size_h * bin_size_h
        w = patch_size_w * bin_size_w
        x = x.view(b, bin_size_h, bin_size_w, patch_size_h, patch_size_w, c)
        # [b, c, bin_size_h, patch_size_h, bin_size_w, patch_size_w]
        x = x.permute(0,5,1,3,2,4).contiguous()
        x = x.view(b, c, h, w)
        
        return x
        
        
    def forward(self, sim_mat, x, x_var=None):
        ''' 
        sim_mat: [b c h w]
        x: [b c h w]
        '''
        residual = x
        b, k, h, w = x.size(0), x.size(1), x.size(-2), x.size(-1)
        pred_uncertainty = self.calc_pred_uncertainty(sim_mat)
        
        return pred_uncertainty, 
        
        
        
        
        
        
        
        
        
        
        
        