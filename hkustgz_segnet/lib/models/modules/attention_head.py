import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

from lib.models.tools.module_helper import ModuleHelper
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from lib.utils.tools.logger import Logger as Log
from lib.models.modules.contrast import momentum_update, l2_normalize
from lib.models.modules.sinkhorn import distributed_sinkhorn


class CABAM(nn.Module):
    ''' 
    Class Activation Attention Module guided by boudnary and uncertainty maps.
    Adapted from 'Partial Class Activation Attention for Semantic Segmentation'.
    '''
    def __init__(self, configer):
        super(self, CABAM).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.proj_dim = self.configer.get('protoseg', 'proj_dim')
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
        self.fuse = nn.Conv2d(self.bin_size_h*self.bin_size_w, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.proj_query = nn.Linear(self.proj_dim, self.proj_dim)
        self.proj_key = nn.Linear(self.proj_dim, self.proj_dim)
        self.proj_value = nn.Linear(self.proj_dim, self.proj_dim)
        
        # [cls_num, proto_num, channel/k]
        self.local_prototypes = nn.Parameter(torch.zeros(
            self.num_classes, self.num_local_prototype, self.proj_dim), requires_grad=False)
        trunc_normal_(self.prototypes, std=0.02) 
        
        # dilate operation for binary boundary map
        kernel = torch.ones((3,3))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        
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
    
    def compute_similarity(self, x):
        if self.sim_measure == 'cosine':
            # batch product (toward d dimension) -> cosine simialrity between fea and prototypes
            # n: h*w, k: num_class, m: num_prototype
            sim_mat = torch.einsum('nd,kmd->nmk', x, self.local_prototypes)
            
            sim_mat = sim_mat.permute(0, 2, 1)
            
            return sim_mat
        else: 
            Log.error('Similarity measure is invalid.')
            
    def prototype_learning(self, sim_mat, out_seg, gt_seg, _c):
        """
        Prototype selection and update
        _c: (normalized) feature embedding
        each pixel corresponds to a mix to multiple protoyptes
        "Prototype selectiprototype_learninglamda * tr(T * log(T) - 1 * 1_T)_t)
        M = 1 - C, M: cost matrx, C: similarity matrix
        T: optimal transport plan
        Class-wise unsupervised clustering, which means this clustering only selects the prototype in its gt class. (Intra-clss unsupervised clustering)
        sim_mat: [b*h*w, num_cls, num_proto]
        gt_seg: [b*h*w]
        out_seg:  # [b*h*w, num_cls]
        """
        # largest score inside a class
        pred_seg = torch.max(out_seg, dim=1)[1]  # [b*h*w]

        mask = (gt_seg == pred_seg.view(-1))  # [b*h*w] bool

        #! pixel-to-prototype online clustering
        protos = self.prototypes.data.clone()
        # to store predictions of proto in each class
        proto_target = gt_seg.clone().float()
        sim_mat = sim_mat.permute(0, 2, 1)  # [n m c]
        for i in range(self.num_classes):
            if i == 255:
                continue

            init_q = sim_mat[..., i]  # [b*h*w, num_proto]
            # select the right preidctions inside i-th class
            init_q = init_q[gt_seg == i, ...]  # [n, num_proto]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q, sinkhorn_iterations=self.configer.get('protoseg', 'sinkhorn_iterations'), epsilon=self.configer.get(
                'protoseg', 'sinkhorn_epsilon'))  # q: mapping mat [n, num_proto] indexs: ind of largest proto in this cls[n]

            try:
                assert torch.isnan(q).int().sum() <= 0
            except:
                # process nan
                Log.info('-'*10 + 'NaN in mapping matrix!' + '-'*10)
                q[torch.isnan(q)] = 0
                indexs[torch.isnan(q).int().sum(dim=1)] = 255 - \
                    (self.num_prototype * i)

            m_k = mask[gt_seg == i]

            m_k_tile = repeat(m_k, 'n -> n tile',
                              tile=self.num_prototype)  # [n, num_proto], bool

            # ! select the prototypes of the correctly predicted pixels in i-th class
            m_q = q * m_k_tile  # [n, num_proto]

            c_k = _c[gt_seg == i, ...]  # [n, embed_dim]

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            # correctly predicted pixel embed  [n embed_dim]
            c_q = c_k * c_k_tile

            # f = m_q.transpose(0, 1) @ c_q  # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]

            # the prototypes that are being selected calcualted by sum
            n = torch.sum(m_q, dim=0)  # [num_proto]

            if self.update_prototype is True and torch.sum(n) > 0:
                f = m_q.transpose(0, 1) @ c_q  # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=protos[i, n != 0, :], new_value=f[n != 0, :],
                                        momentum=self.mean_gamma, debug=False)
                protos[i, n != 0, :] = new_value
                    
            # each class has a target id between [0, num_proto * c]
            #! ignore_label are still -1, and not being modified
            proto_target[gt_seg == i] = indexs.float(
            ) + (self.num_prototype * i)  # n samples -> n*m labels

        self.prototypes = nn.Parameter(
            l2_normalize(protos), requires_grad=False)  # make norm of proto equal to 1

        if dist.is_available() and dist.is_initialized():  # distributed learning
            protos = self.prototypes.data.clone()
            """
            To get average result across all gpus: 
            first average the sum
            then sum the tensors on all gpus, and copy the sum to all gpus
            """
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_target  # [n]
        
    def forward(self, x, gt_semantic_seg=None):
        ''' 
        x is uncertainty-aware features
        Use glboal prototypes obtained from prob_proto_seg_head to calculate class scores.
        '''
        b_size = x.shape[0]
        h_size = x.shape[2]
        gt_size = x.size()[2:]
        
        #! prototype classifier to replace softmax classifier
        x = rearrange(x, 'b c h w -> (b h w) c')
        sim_mat_ori = self.compute_similarity(x) # [n c m] sim_mat equals to cam here
        
        if self.pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = F.interpolate(
                gt_semantic_seg.float(), size=gt_size, mode='nearest').view(-1)
            contrast_target = self.prototype_learning(sim_mat_ori, out_seg, gt_seg, x)
            
        out_seg = torch.amax(sim_mat_ori, dim=2)  # [n, num_cls]
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, '(b h w) c -> b c h w',
                            b=b_size, h=h_size)
        x = rearrange(x, '(b h w) c -> b c h w',
                            b=b_size, h=h_size)
        #! [b, num_cls*num_local_proto, h, w] sim_mat equal to cam here
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
        global_cls_centers = self.local_prototypes.data.clone() # [num_cls, num_proto, k]
        global_cls_centers = rearrange(global_cls_centers, 'c m k -> (c m) k') # [c*m k]
        #todo ==============================
        #todo use updated protoytypes or proto in this frame???
        global_cls_centers = 
        
        # # fuse local centers among all patches to global centers
        # global_cls_centers = self.fuse(local_cls_centers) # [b 1 (c m) k] 
        # # [b (bin_size_h bin_size_w) (c m) k] 
        # global_cls_centers = self.relu(global_cls_centers).repeat(1, self.bin_size_h*self.bin_size_w, 1, 1)
        
        # attention
        query = self.proj_query(x) # [b bin_size*bin_size patch_size_h*patch_size_w k]
        key = self.proj_key(local_cls_centers) # [b bin_size*bin_size (c m) k]
        value = self.proj_value(global_cls_centers) # []
        
        
        
        
        
        
        
        
        
        
        

class AttentionHead(nn.Module):
    def __init__(self, configer):
        super(AttentionHead, self).__init__()
        self.configer = configer
        
    def forward(self, uncertainty, boundary_map, x):
        ''' 
        boundary: 0: non-edge, 1: edge 
        '''
        # uncertainty-aware features
        x *= (1 - uncertainty)

        
            
            
            
        
        