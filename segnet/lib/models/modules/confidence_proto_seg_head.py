import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from lib.utils.tools.logger import Logger as Log
from lib.models.modules.contrast import momentum_update, l2_normalize
from lib.models.modules.sinkhorn import distributed_sinkhorn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from lib.utils.distributed import get_world_size, get_rank, is_distributed


class ConfidenceProtoSegHead(nn.Module):
    """
    Similarity between pixel embeddings and prototypes

    x: batch-wise pixel embeddings [(b h w) c]
    x_var: var of x [(b h w) c]
    prototypes: [cls_num, proto_num, channel/c]
    c: similarity []
    gt_seg: [b*h*w]
    """

    def __init__(self, configer):
        super(ConfidenceProtoSegHead, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.sim_measure = self.configer.get('protoseg', 'similarity_measure')
        self.use_uncertainty = self.configer.get('protoseg', 'use_uncertainty')
        self.use_boundary = self.configer.get('protoseg', 'use_boundary')
        self.proj_dim = self.configer.get('protoseg', 'proj_dim')
        self.pretrain_prototype = self.configer.get(
            'protoseg', 'pretrain_prototype')
        self.mean_gamma = self.configer.get('protoseg', 'mean_gamma')
        self.var_gamma = self.configer.get('protoseg', 'var_gamma')
        self.use_prototype = self.configer.get('protoseg', 'use_prototype')
        self.update_prototype = self.configer.get(
            'protoseg', 'update_prototype')

        self.feat_norm = nn.LayerNorm(self.proj_dim)  # normalize each row
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.proto_norm = nn.LayerNorm(self.num_classes * self.num_prototype)

        # [cls_num, proto_num, channel/k]
        self.prototypes = nn.Parameter(torch.zeros(
            self.num_classes, self.num_prototype, self.proj_dim), requires_grad=False)
        trunc_normal_(self.prototypes, std=0.02)

        self.cosine_classifier = self.configer.get('protoseg', 'cosine_classifier')

    def compute_similarity(self, x, x_var=None, sim_measure='wasserstein'):
        ''' 
        x/x_var: [(b h w) k]
        proto_mean: [c m k]
        Use reparameterization trick to compute probabilistic distance for gradient flow.
        '''
        proto_mean = self.prototypes.data.clone()
        if sim_measure == 'cosine':
            # batch product (toward d dimension) -> cosine simialrity between fea and prototypes
            # n: h*w, k: num_class, m: num_prototype
            sim_mat = torch.einsum('nd,kmd->nmk', x, proto_mean)  # [c m n]

            sim_mat = sim_mat.permute(0, 2, 1)  # [c n m]
        else:
            Log.error('Similarity measure is invalid.')
        return sim_mat

    def confidence_guided_prototype_learning(self, sim_mat, out_seg, gt_seg, _c, confidence):
        """
        Prototype selection and update
        _c: (normalized) feature embedding
        each pixel corresponds to a mix to multiple protoyptes
        lamda * tr(T * log(T) - 1 * 1_T)_t)
        M = 1 - C, M: cost matrx, C: similarity matrix                      
        T: optimal transport plan
        Class-wise unsupervised clustering, which means this clustering only selects the prototype in its gt class. (Intra-clss unsupervised clustering)
        sim_mat: [b*h*w, num_cls, num_proto]
        gt_seg: [b*h*w]
        out_seg:  # [b*h*w, num_cls]
        confidence: [b*h*w]
        """
        # largest score inside a class
        pred_seg = torch.max(out_seg, dim=1)[1]  # [b*h*w]

        mask = (gt_seg == pred_seg.view(-1))  # [b*h*w] bool

        #! pixel-to-prototype online clustering
        protos = self.prototypes.data.clone()
        if self.use_uncertainty:
            proto_var = self.proto_var.data.clone()
        # to store predictions of proto in each class
        proto_target = gt_seg.clone().float()
        sim_mat = sim_mat.permute(0, 2, 1)  # [n m c]
        for i in range(self.num_classes):
            init_q = sim_mat[..., i]  # [b*h*w, num_proto]
            # select the right preidctions inside i-th class
            init_q = init_q[gt_seg == i, ...]  # [n, num_proto]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q, sinkhorn_iterations=self.configer.get('protoseg', 'sinkhorn_iterations'), epsilon=self.configer.get(
                'protoseg', 'sinkhorn_epsilon'))  # q: mapping mat [n, num_proto] indexs: ind of largest proto in this cls[n]

            m_k = mask[gt_seg == i]

            m_k_tile = repeat(m_k, 'n -> n tile',
                              tile=self.num_prototype)  # [n, num_proto], bool

            # ! select the prototypes of the correctly predicted pixels in i-th class
            m_q = q * m_k_tile  # [n, num_proto]

            c_k = _c[gt_seg == i, ...]  # [n, embed_dim]

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            # correctly predicted pixel embed  [n embed_dim]
            c_q = c_k * c_k_tile

            # the prototypes that are being selected calcualted by sum
            n = torch.sum(m_q, dim=0)  # [num_proto]

            if self.update_prototype is True and torch.sum(n) > 0:
                # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                f = m_q.transpose(0, 1) @ c_q
                f = F.normalize(f, p=2, dim=-1)
                protos[i, n != 0, :] = momentum_update(
                    old_value=protos[i, n != 0, :], new_value=f[n != 0, :], momentum=self.mean_gamma, debug=False)

            # each class has a target id between [0, num_proto * c]
            #! ignore_label are still -1, and not being modified
            proto_target[gt_seg == i] = indexs.float(
            ) + (self.num_prototype * i)  # n samples -> n*m labels

            del c_q, c_k_tile, c_k, m_q, m_k_tile, m_k, indexs, q

        self.prototypes = nn.Parameter(
            l2_normalize(protos), requires_grad=False)  # make norm of proto equal to 1

        if dist.is_available() and dist.is_initialized():  # distributed learning
            """
            To get average result across all gpus: 
            first average the sum
            then sum the tensors on all gpus, and copy the sum to all gpus
            """
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_target  # [n]


    def forward(self, x, x_var=None, gt_semantic_seg=None, boundary_pred=None, gt_boundary=None):
        ''' 
        boundary_pred: [(b h w) 2]
        '''
        b_size = x.shape[0]
        h_size = x.shape[2]
        gt_size = x.size()[2:]

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # sim_mat: # [n c m]
        x = rearrange(x, 'b c h w -> (b h w) c')
        if self.use_uncertainty and x_var is not None:
            x_var = rearrange(x_var, 'b c h w -> (b h w) c')

        sim_mat = self.compute_similarity(
            x, x_var=x_var, sim_measure=self.sim_measure)

        out_seg = torch.amax(sim_mat, dim=2)  # [n, num_cls]
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, '(b h w) c -> b c h w',
                            b=b_size, h=h_size)

        if self.use_boundary and gt_boundary is not None:
            gt_boundary = F.interpolate(
                gt_boundary.float(), size=gt_size, mode='nearest').view(-1)

        if self.pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            assert torch.unique(gt_semantic_seg).shape[0] != 1
            gt_seg = F.interpolate(
                gt_semantic_seg.float(), size=gt_size, mode='nearest').view(-1)
            if self.use_boundary and gt_boundary is not None:
                contrast_target = self.prototype_learning_boundary(
                    sim_mat, out_seg, gt_seg, x, x_var, gt_boundary)
            else:
                contrast_target = self.prototype_learning(
                    sim_mat, out_seg, gt_seg, x, x_var)

            sim_mat = rearrange(sim_mat, 'n c m -> n (c m)')

            return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target}
        
        if self.configer.get('proto_visualizer', 'vis_prototype'):
            sim_mat = sim_mat.reshape(b_size, h_size, -1, self.num_classes * self.num_prototype)
            return {'seg': out_seg, 'logits': sim_mat}
        else:
            return out_seg
