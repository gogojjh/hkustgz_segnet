''' 
Highlight the uncertain areas in the uncertainty map using predicted boundary map.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.tools.module_helper import ModuleHelper
from einops import rearrange, repeat


class BoundaryAttentionModule(nn.Module):
    def __init__(self, configer):
        super(BoundaryAttentionModule, self).__init__()
        self.configer = configer

        self.proj_dim = self.configer.get('protoseg', 'proj_dim')
        self.mid_channels = 64
        self.out_channels = 256

        # to let dim of boundary map equal to that of uncertainty/feature map
        self.key_conv = nn.Sequential(
            nn.Conv2d(1,
                      self.mid_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            ModuleHelper.BNReLU(self.mid_channels,
                                bn_type=self.configer.get(
                                    'network', 'bn_type'
                                )),
            nn.Dropout2d(0.10),
            nn.Conv2d(self.mid_channels,
                      self.out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False))

        self.query_conv = nn.Conv2d(self.proj_dim,
                                    self.out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False)

        self.value_conv = nn.Conv2d(self.proj_dim,
                                    self.proj_dim,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, boundary_map, uncertainty_map):
        ''' 
        Similar to Position Attention Module in 
        "Dual Attention Network for Scene Segmentation".

        Use boundary prediction to let the boundary areas in uncertainty map 
        with high uncertianty, and mask the non-edge areas in semantic prediction
        to learn the prototype of boundary pixels.

        Boundary prediction is learned by supervised cross-entropy loss.

        all inputs: [b proj_dim/2 h w]
        '''
        b_size, c_size, h_size, w_size = uncertainty_map.size()  # c: proj_dim
        boundary_map = F.interpolate(
                boundary_map.float(), size=(h_size, w_size), mode='nearest')
    
        proj_key = self.key_conv(boundary_map)  # [b out_channel h w] key
        proj_query = self.query_conv(uncertainty_map)  # query [b out_channel h w]
        # [b (h w) mid_channel]
        proj_query = proj_query.view(b_size, -1, h_size*w_size).permute(0, 2, 1)
        proj_key = proj_key.view(b_size, -1, h_size*w_size)  # [b mid_channel (h w)]

        proj_value = self.value_conv(uncertainty_map)
        proj_value = proj_value.view(b_size, -1, h_size*w_size)  # value [b self.proj_dim (h w)]

        # if uncertainty high + probability in boundary map high -> more likely to be boundary
        # [b (h w) mid_channel] * [b mid_channel (h w)] = [b (h w) (h w)]
        #! to cpu to save gpu memory usage
        energy = torch.bmm(proj_query.float().cpu(), proj_key.float().cpu())
        attention = F.softmax(energy, 1)  # softmax on (h w) channel

        # [b self.proj_dim (h w)]  * [b (h w/no softmax) (h w/softmax)] = [b self.proj_dim (h w/softmax)]
        out = torch.bmm(proj_value.float().cpu(), attention.permute(0, 2, 1)).cuda()
        
        # [b c h w] here h and w results from softmax(position attention)
        out = out.view(b_size, c_size, h_size, w_size)

        out = self.gamma * out + uncertainty_map  # [b c h w]
        
        del proj_query, proj_key, proj_value, uncertainty_map

        return out
