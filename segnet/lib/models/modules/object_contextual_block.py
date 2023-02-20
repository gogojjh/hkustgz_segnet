import torch
from torch import nn
from torch.nn import functional as F

from lib.models.tools.module_helper import ModuleHelper


class SpatialGather_0CR_Module(nn.Module):
    ''' 
    Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation
    '''

    def __init__(self, configer):
        super(SpatialGather_0CR_Module, self).__init__()
        self.configer = configer

    def forward(self, feats, probs):
        ''' 
        prob_map: similarity matrix computed from prototype learning
        '''
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)  # [b c/(c m) (h w)]
        feats = feats.view(batch_size, feats.size(1), -1)  # [b k (h w)]
        feats = feats.permute(0, 2, 1)  # [b (h w) k]
        probs = F.softmax(probs, dim=2)  # along spatial size
        # # [b c/(c m) (h w)] x [b (h w) k] = [b c k]
        ocr_context = torch.matmul(probs, feats)  # ! class/object center in an image
        return ocr_context


class ContextRelation_Module(nn.Module):
    def __init__(self, configer):
        super(ContextRelation_Module, self).__init__()
        self.configer = configer

        bn_type = self.configer.get('network', 'bn_type')
        self.in_channels = self.configer.get('protoseg', 'context_dim')
        self.key_channels = self.configer.get('protoseg', 'key_dim')
        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, context):
        ''' 
        context: class/object center in an image

        Use attention mechanism to consider contextual information to augment the feature map.
        key/value: class center/region representation
        query: coarse feature map
        '''
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        query = self.query_conv(x).view(batch_size, self.key_channels, -1)  # [b k (h w)]
        query = query.permute(0, 2, 1)  # [b (h w) k]
        # context = context.permute(0, 2, 1)
        key = self.key_conv(context).view(batch_size,
                                          self.key_channels, -1)  # [b k c]
        value = self.value_conv(context).view(batch_size,
                                              self.key_channels, -1)
        value = value.permute(0, 2, 1)  # [b c k]

        #! similarity between pixel feature and region representation (attention weight)
        # [b (h w) k] @ [b k c] = [b (h w) c]
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # [b (h w) c] @ [b c k] = [b (h w) k]
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()  # [b k (h w)]
        # [b k h w]
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.up_conv(context)

        return context
