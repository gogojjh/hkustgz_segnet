''' 
'Boundary Guided Context Aggregation for Semantic Segmentation'
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.tools.module_helper import ModuleHelper
from lib.utils.tools.logger import Logger as Log


class BCAModule(nn.Module):
    '''
    Boundary context aggregation module using attention mechanism. 
    '''
    def __init__(self, configer):
        super(BCAModule, self).__init__()
        self.configer = configer
        
        xin_channels = 720
        yin_channels = self.configer.get('protoseg', 'edge_dim')
        mid_channels = self.configer.get('protoseg', 'edge_dim')
        self.mid_channels = mid_channels
        
        bn_type = self.configer.get('network', 'bn_type')
        
        self.f_self = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(mid_channels),
        )
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels=yin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(mid_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=xin_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(mid_channels),
        )
        
        self.value_down = nn.MaxPool2d(4)
        self.key_down = nn.MaxPool2d(4)
        
        nn.init.constant_(self.f_up[1].weight, 0)
        nn.init.constant_(self.f_up[1].bias, 0)
        
    def forward(self, x, y):
        ''' 
        key: boundary map (downsampled for efficient memory) / y
        value/query: feature map / x
        '''
        batch_size = x.size(0)
        fself = self.f_self(x).view(batch_size, self.mid_channels, -1) # [b mid_dim (h w)]
        fself = fself.permute(0, 2, 1) # [b (h w) mid_dim]
        
        fx = self.f_x(x).view(batch_size, self.mid_channels, -1) # [b mid_dim (h w)]
        fx = fx.permute(0, 2, 1) # [b (h w) mid_dim]    
        
        fy = self.f_y(y).view(batch_size, self.mid_channels, -1) # [b mid_dim (h w)]
        
        #! downsample for memory usage
        fself = self.value_down(fself)
        fy = self.key_down(fy)
                
        # [b (h w) mid_dim] @ [b mid_dim (h w)/R] = [b (h w) (h w)/R]
        sim_map = torch.matmul(fx, fy) 
        
        sim_map = F.softmax(sim_map, dim=-1)    
        
        # [b (h w) (h w)/R] @ [b (h w)/R mid_dim] = [b (h w) mid_dim]
        fout = torch.matmul(sim_map, fself) 
        fout = fout.permute(0, 2, 1).contiguous() # [b mid_dim (h w)]
        fout = fout.view(batch_size, self.mid_channels, *x.size()[2:]) # [b mid_dim h w]
        
        fout = self.f_up(fout)
        
        return x + fout

class BoundaryHead(nn.Module):
    def __init__(self, configer):
        super(BoundaryHead, self).__init__()
        self.configer = configer
        
        self.backbone = self.configer.get('network', 'backbone')
        if self.backbone == 'hrnet48':
            self.edge_conv1 = self.generate_edge_conv(48)
            self.edge_conv2 = self.generate_edge_conv(96)
            self.edge_conv3 = self.generate_edge_conv(192)
            self.edge_conv4 = self.generate_edge_conv(384)  
        else: 
            Log.error('Backbone setting is invalid for boundary head.')
        
        self.edge_dim = self.configer.get('protoseg', 'edge_dim')
        self.edge_out1 = nn.Sequential(nn.Conv2d(self.edge_dim,1, 1),
                                       nn.Sigmoid())
        self.edge_out2 = nn.Sequential(nn.Conv2d(self.edge_dim, 1, 1),
                                       nn.Sigmoid())
        self.edge_out3 = nn.Sequential(nn.Conv2d(self.edge_dim, 1, 1),
                                       nn.Sigmoid())
        self.edge_out4 = nn.Sequential(nn.Conv2d(self.edge_dim, 1, 1),
                                       nn.Sigmoid())
        bn_type = self.configer.get('network', 'bn_type')
        self.edge_down = nn.Sequential(
            nn.Conv2d(self.edge_dim*4, self.edge_dim, kernel_size=3, padding=1, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.edge_dim),
            nn.ReLU(inplace=True),
        )
        
        self.attn_module = BCAModule(configer=configer)
               
    def gen_edge_conv(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, self.edge_dim, 3),
            self.BatchNorm(self.edge_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, c):
        ''' 
        x: a list of features of different resolutions to generate edge map
        at each scale.
        c: Concated feature as input for attention module. (Not normalized)
        '''
        h, w = x[4].size()[2:]
        assert self.backbone == 'hrnet48'
        
        e1 = self.edge_conv1(x[0])
        e1 = F.interpolate(e1, size=(h, w), mode='bilinear', align_corners=True)
        e1_out = self.edge_out1(e1)
        
        e2 = self.edge_conv2(x[1])
        e2 = F.interpolate(e2, size=(h, w), mode='bilinear', align_corners=True)
        e2_out = self.edge_out2(e2)
        
        e3 = self.edge_conv3(x[2])
        e3 = F.interpolate(e3, size=(h, w), mode='bilinear', align_corners=True)
        e3_out = self.edge_out3(e3)
        
        e4 = self.edge_conv3(x[3])
        e4 = F.interpolate(e4, size=(h, w), mode='bilinear', align_corners=True)
        e4_out = self.edge_out3(e4)
        
        e = torch.cat((e1, e2, e3, e4), dim=1)
        e = self.edge_down(e) # [b 256 h w] input of attention module
        e1_out = e1_out + e2_out + e3_out + e4_out
        # For boundary supervisio loss
        e1_out = F.interpolate(e1_out, size=(h, w), mode='bilinear', align_corners=True)
        
        self.attn_module(c, e)
        
        return x
        