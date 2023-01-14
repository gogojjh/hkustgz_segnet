''' 
Boundary head for predicting class-wise boundary.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.tools.module_helper import ModuleHelper

class BoundaryHead(nn.Module):
    def __init__(self, configer, in_channels=720, mid_channels=256):
        super(BoundaryHead, self).__init__()
        self.configer = configer
        
        num_masks = 2 # edge / non-edge?
        
        self.boundary_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            ModuleHelper.BNReLU(mid_channels,
                                bn_type=self.configer.get(
                                    'network', 'bn_type'
                                )),
            nn.Conv2d(mid_channels,
                      num_masks,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False))
        
        
        