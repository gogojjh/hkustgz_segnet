''' 
'Improving Semantic Segmentation via Decoupled Body and Edge Supervision'
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.tools.module_helper import ModuleHelper
from lib.utils.tools.logger import Logger as Log
from lib.models.modules.contrast import upsample


class BodyHead(nn.Module):
    def __init__(self, configer):
        super(BodyHead, self).__init__()
        self.configer = configer

        inplane = self.configer.get('protoseg', 'proj_dim') // 2
        bn_type = self.configer.get('network', 'bn_type')

        #! depthwise convolution
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(inplane),
            nn.ReLU(inplace=True)
        )

        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).cuda()
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).cuda()
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = upsample(seg_down, size)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp

        return seg_flow_warp, seg_edge


class EdgeHead(nn.Module):
    def __init__(self, configer):
        super(EdgeHead, self).__init__()
        self.configer = configer
        self.backbone = self.configer.get('network', 'backbone')
        bn_type = self.configer.get('network', 'bn_type')

        if self.backbone == 'hrnet48':
            in_channles = 360
            low_fea_dim = 96
        else:
            Log.error('Backbone is invalid for edge head.')

        self.bot_fine = nn.Conv2d(low_fea_dim, 48, kernel_size=1, bias=False)
        self.edge_fusion = nn.Conv2d(in_channles+48, in_channles, kernel_size=1, bias=False)
        self.edge_out = nn.Sequential(
            nn.Conv2d(in_channles, 48, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )

    def forward(self, seg_edge, low_fea):
        ''' 
        low_fea: feat2 (96) in hr net
        '''
        # add low-level feature for more details
        low_fea_size = low_fea.size()[2:]
        low_fea = self.bot_fine(low_fea)

        #! downsample will destroy the edge pred otherwise
        # low_fea_size: [64 128] # [b 360 + 48 64 128]
        seg_edge = torch.cat((F.interpolate(seg_edge, size=low_fea_size, mode='bilinear',
                                            align_corners=True), low_fea), dim=1)
        seg_edge = self.edge_fusion(seg_edge)  # [b 360 h'/64 w'/128]

        seg_edge_out = self.edge_out(seg_edge)  # [b 1 h' w']

        return seg_edge_out, seg_edge
