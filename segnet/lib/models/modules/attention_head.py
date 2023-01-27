import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.tools.module_helper import ModuleHelper
from einops import rearrange, repeat


class GCN(nn.Module):
    def __init__(self, configer):
        super(GCN, self).__init__()
        self.configer = configer
        self.bin_size_h = self.configfer.get('protoseg', 'bin_size_h')
        self.bin_size_w = self.configfer.get('protoseg', 'bin_size_w')
        num_node = self.bin_size_h * self.bin_size_w
        num_channel = self.configer.get('protoseg', 'proj_dim')
        
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)
    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out


class AttentionModule(nn.Module):
    ''' 
    Adapted from 'Partial Class Activation Attention for Semantic Segmentation'.
    '''
    def __init__(self, configer):
        super(self, AttentionModule).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.proj_dim = self.configer.get('protoseg', 'proj_dim')
        self.mid_dim = self.proj_dim // 2
        self.num_local_prototype = self.configer.get('protoseg', 'num_local_prototype')
        self.bin_size_h = self.configfer.get('protoseg', 'bin_size_h')
        self.bin_size_w = self.configfer.get('protoseg', 'bin_size_w')
        self.sim_measure = self.configer.get('protoseg', 'similarity_measure')
        self.pretrain_prototype = self.configer.get(
            'protoseg', 'pretrain_prototype')
        self.use_prototype = self.configer.get('protoseg', 'use_prototype')
        self.update_prototype = self.configer.get(
            'protoseg', 'update_prototype')
        
        self.dropout = nn.Dropout2d(0.1)
        self.conv_cam = nn.Conv2d(self.proj_dim, self.num_classes*self.num_local_prototype, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.pool_cam = nn.AdaptiveAvgPool2d((self.bin_size_h, self.bin_size_w))
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.proj_query = nn.Linear(self.proj_dim, self.mid_dim)
        self.proj_key = nn.Linear(self.proj_dim, self.mid_dim)
        self.proj_value = nn.Linear(self.proj_dim, self.mid_dim)
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
    
    def patch_recover(x, bin_size_h, bin_size_w):
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
        
    def forward(self, x, sim_mat_ori, global_cls_centers):
        ''' 
        x is uncertainty-aware features
        Use glboal prototypes obtained from prob_proto_seg_head to calculate class scores.
        '''
        b_size = x.shape[0]
        h_size = x.shape[2]
        
        #! [b, num_cls*num_local_proto, h, w] sim_mat equal to cam here
        sim_mat_ori = rearrange(sim_mat_ori, 'n (c m) -> n c m')
        sim_mat = rearrange(sim_mat_ori, 'n c m -> b (c m) h w', b=b_size, h=h_size) 
        
        #! for each patch, the overl all(pooling) score for K classes
        patch_cls_score = self.pool_cam(sim_mat) # [b (c m) bin_size bin_size]
        
        # [b bin_size*bin_size patch_size_h patch_size_w (c m)]
        sim_mat = self.patch_split(sim_mat, self.bin_size_h, self.bin_size_w)
        # [b bin_size*bin_size patch_size_h patch_size_w k]
        x = self.patch_split(x, self.bin_size_h, self.bin_size_w) 
        
        b, _, patch_size_h, patch_size_w, num_sub_cls = sim_mat.size()
        k = x.shape[-1]
        # [b bin_size*bin_size patch_size_h*patch_size_w (c m)]
        sim_mat = sim_mat.view(b, -1, patch_size_h*patch_size_w, num_sub_cls)
        # [b bin_size*bin_size patch_size_h*patch_size_w k]
        x = x.view(b, -1, patch_size_h*patch_size_w, k)
        
        # [b (c m) bin_size*bin_size] -> [b bin_size*bin_size (c m) 1]
        patch_confidence = patch_cls_score.view(b, num_sub_cls, -1).transpose(1, 2).unsqueeze(3)
        
        # do softmax along patch_num dimension for subsequent local class center aggregation
        sim_mat = F.softmax(sim_mat, dim=2)
        
        # local class center
        # [b bin_size*bin_size (c m) patch_size_h*patch_size_w] @ [b bin_size*bin_size patch_size_h*patch_size_w k] = [b bin_size*bin_size (c m) k]
        #! local_cls_centers: each patch has (c m) local_cls_centers, but lacks consistency among different patches   
        local_cls_centers = torch.matmul(sim_mat.transpose(2, 3), x) * patch_confidence
        #! use prototypes obtained from OT clustering as global prototypes
        global_cls_centers = rearrange(global_cls_centers, 'c m k -> (c m) k') # [c*m k]

        global_cls_centers = repeat(global_cls_centers, 'l k -> b n l k', b=b_size, n=self.bin_size_h*self.bin_size_w) # [b (bin_size_h bin_size_w) (c m) k] 
        
        # # fuse local centers among all patches to global centers
        # global_cls_centers = self.fuse(local_cls_centers) # [b 1 (c m) k] 
        # # [b (bin_size_h bin_size_w) (c m) k] 
        # global_cls_centers = self.relu(global_cls_centers).repeat(1, self.bin_size_h*self.bin_size_w, 1, 1)
        
        # attention
        query = self.proj_query(x) # [b bin_size*bin_size patch_size_h*patch_size_w k]
        key = self.proj_key(local_cls_centers) # [b bin_size*bin_size (c m) k]
        value = self.proj_value(global_cls_centers) # [b (bin_size_h bin_size_w) (c m) k] 
        
        # [b bin_size*bin_size patch_size_h*patch_size_w k] @ [b bin_size*bin_size k (c m)] = [b bin_size*bin_size patch_size_h*patch_size_w (c m)]
        aff_map = torch.matmul(query, key.transpose(2, 3)) 
        aff_map = F.softmax(aff_map, dim=-1)
        # [b bin_size*bin_size patch_size_h*patch_size_w (c m)] @ [b (bin_size_h bin_size_w) (c m) k] = [b bin_size*bin_size patch_size_h*patch_size_w k]
        aff_map = torch.matmul(aff_map, value)
        # [b bin_size*bin_size patch_size_h patch_size_w k]
        aff_map = aff_map.view(b, -1, patch_size_h, patch_size_w, value.shape[-1])
        # [b k/2 h w] 
        aff_map = self.patch_recover(aff_map, self.bin_size_h, self.bin_size_w) 
        aff_map = self.conv_out(aff_map)  # [b k h w]
        
        return aff_map, patch_cls_score
        

class AttentionHead(nn.Module):
    def __init__(self, configer):
        super(AttentionHead, self).__init__()
        self.configer = configer
        
        self.attention_module = AttentionModule(configer=configer)
        
    def forward(self, x, sim_mat, prototypes, uncertainty=None):
        ''' 
        boundary: 0: non-edge, 1: edge 
        Both uncertainty and boundary map are used for get accurate predictions of small objects or bounary, which are better predicted when in a local context.
        '''
        if uncertainty is not None:
            # uncertainty-aware features
            x *= (1 - uncertainty)
        
        aff_map, patch_cls_score = self.attention_module(x, sim_mat, prototypes)
        
        return aff_map, patch_cls_score
        
        

        
            
            
            
        
        