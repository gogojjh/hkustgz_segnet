''' 
Adapted from 'Partial Class Activation Attention for Semantic Segmentation'
'''
import torch.nn as nn
from torch.nn import functional as F
import torch

from lib.models.tools.module_helper import ModuleHelper
from einops import rearrange, repeat


class GCN(nn.Module):
    def __init__(self, configer):
        super(GCN, self).__init__()
        self.configer = configer
        self.bin_size_h = self.configer.get('protoseg', 'bin_size_h')
        self.bin_size_w = self.configer.get('protoseg', 'bin_size_w')
        num_node = self.bin_size_h * self.bin_size_w
        # num_channel = self.configer.get('protoseg', 'proj_dim')
        num_channel = 720
        
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)
    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out


class CAAHead(nn.Module):
    ''' 
    Adapted from 'Partial Class Activation Attention for Semantic Segmentation'.
    '''
    def __init__(self, configer):
        super(CAAHead, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        # self.proj_dim = self.configer.get('protoseg', 'proj_dim')
        self.proj_dim = 720
        self.mid_dim = self.proj_dim // 2
        self.bin_size_h = self.configer.get('protoseg', 'bin_size_h')
        self.bin_size_w = self.configer.get('protoseg', 'bin_size_w')
        self.sim_measure = self.configer.get('protoseg', 'similarity_measure')
        self.pretrain_prototype = self.configer.get(
            'protoseg', 'pretrain_prototype')
        self.use_prototype = self.configer.get('protoseg', 'use_prototype')
        self.update_prototype = self.configer.get(
            'protoseg', 'update_prototype')
        
        self.dropout = nn.Dropout2d(0.1)
        self.conv_cam = nn.Conv2d(self.proj_dim, self.num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.pool_cam = nn.AdaptiveAvgPool2d((self.bin_size_h, self.bin_size_w))
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.proj_query = nn.Linear(self.proj_dim, self.mid_dim)
        self.proj_key = nn.Linear(self.proj_dim, self.mid_dim)
        self.proj_value = nn.Linear(self.proj_dim, self.mid_dim)
        self.fuse = nn.Conv2d(self.bin_size_h*self.bin_size_w, 1, kernel_size=1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.mid_dim, self.proj_dim, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(self.proj_dim,
                bn_type=self.configer.get('network', 'bn_type'))
                                    )
        self.gcn = GCN(configer=configer)
        
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
        
    def forward(self, x):
        residual = x # [B, C, H, W]
        #! CAM map
        cam = self.conv_cam(self.dropout(x)) # [B, K, H, W]
        #! for each patch, the overl all(pooling) score for K classes
        cls_score = self.sigmoid(self.pool_cam(cam)) # [B, K, bin_num_h, bin_num_w]
        
        cam = self.patch_split(cam, self.bin_size_h, self.bin_size_w) # [B, bin_num_h * bin_num_w, rH, rW, K]
        x = self.patch_split(x, self.bin_size_h, self.bin_size_w) # [B, bin_num_h * bin_num_w, rH, rW, C]

        B = cam.shape[0]
        rH = cam.shape[2]
        rW = cam.shape[3]
        K = cam.shape[-1]
        C = x.shape[-1]
        cam = cam.view(B, -1, rH*rW, K) # [B, bin_num_h * bin_num_w, rH * rW, K]
        x = x.view(B, -1, rH*rW, C) # [B, bin_num_h * bin_num_w, rH * rW, C]

        bin_confidence = cls_score.view(B,K,-1).transpose(1,2).unsqueeze(3) # [B, bin_num_h * bin_num_w, K, 1]
        #! pixel_confidence: inside each patch, for each pixel, its prob. belonging to K classes 
        cam = F.softmax(cam, dim=2) 
        #! local cls centers (weigted sum (weight: bin_confidence))
        #! torch.matmul: aggregate inside a patch
        cam = torch.matmul(cam.transpose(2, 3), x) * bin_confidence # [B, bin_num_h * bin_num_w, K, C]
        cam = self.gcn(cam) # [B, bin_num_h * bin_num_w, K, C]
        global_feats = self.fuse(cam) # [B, 1, K, C]
        global_feats = self.relu(global_feats).repeat(1, x.shape[1], 1, 1) # [B, bin_num_h * bin_num_w, K, C]
        
        x = self.proj_query(x) # [B, bin_num_h * bin_num_w, rH * rW, C//2]
        cam = self.proj_key(cam) # [B, bin_num_h * bin_num_w, K, C//2]
        global_feats = self.proj_value(global_feats) # [B, bin_num_h * bin_num_w, K, C//2]
        
        out = torch.matmul(x, cam.transpose(2, 3)) # [B, bin_num_h * bin_num_w, rH * rW, K]
        out = F.softmax(out, dim=-1)
        #! attention
        out = torch.matmul(out, global_feats) # [B, bin_num_h * bin_num_w, rH * rW, C]
        
        out = out.view(B, -1, rH, rW, global_feats.shape[-1]) # [B, bin_num_h * bin_num_w, rH, rW, C]
        out = self.patch_recover(out, self.bin_size_h, self.bin_size_w) # [B, C, H, W]

        out = residual + self.conv_out(out)
        
        del x, global_feats, cam, residual
        
        return out, cls_score
