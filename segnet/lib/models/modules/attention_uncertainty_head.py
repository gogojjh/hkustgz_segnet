import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.models.tools.module_helper import ModuleHelper
from lib.models.modules.multi_head_self_attention import MultiHeadSelfAttention


class UncertaintyHead(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super(UncertaintyHead, self).__init__()
        
        self.avgpool = nn.AvgPool2d((7, 7))
        
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.fc2 = nn.Linear(d_in, d_out)
        
        self.attention = MultiHeadSelfAttention(1, d_in, d_h)
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, out, x, pad_mask=None):
        ''' 
        out: pooled feature
        x: reshaped features before pooling
        '''
        residual, attn = self.attention(x, pad_mask)
        
        fc_out = self.fc2(out)  
        out = self.fc(residual) + fc_out
        
        
        
        

        
        
        
        