import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.models.tools.module_helper import ModuleHelper
from lib.models.modules.multi_head_self_attention import MultiHeadSelfAttention


class UncertaintyHead(nn.Module):
    def __init__(self, configer):
        super(UncertaintyHead, self).__init__()
        self.configer = configer
        
        self.proj_dim = self.configer.get('protoseg', 'proj_dim')
        d_in = self.proj_dim
        d_out = self.proj_dim
        
        self.avgpool = nn.AvgPool2d((7, 7))
        
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.fc2 = nn.Linear(d_in, d_out)
        self.embed_dim = d_in
        
        self.attention = MultiHeadSelfAttention(configer=configer)
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x):
        x_reshape = x.view(-1, self.proj_dim, 8, 8)
        x = self.avgpool(x_reshape).view(-1, self.proj_dim) # [b, proj_dim]
        x = self.fc(x) #!mean layer [b, proj_dim]

        x_reshape = x_reshape.view(-1, self.proj_dim, 8 * 8)
        # ([b, proj_dim], [-1, proj_dim, 7*7])
        logsigma = self.uncer_module(x, x_reshape.transpose(1, 2)) 
        
        return x, logsigma
        

class AttentionUncertaintyModule(nn.Module):
    def __init__(self, configer):
        super(AttentionUncertaintyModule, self).__init__()
        self.configer = configer
        
        
        
        
    def forward(self, out, x, pad_mask=None):
        residual = self.attention(x, pad_mask)
        
        out = self.fc2(out)
        out = self.fc(residual) + out # logsigma
        
        return out
        
        
        
        
        