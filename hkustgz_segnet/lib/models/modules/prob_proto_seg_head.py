import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from lib.utils.tools.logger import Logger as Log
from lib.models.modules.contrast import momentum_update, l2_normalize
from lib.models.modules.sinkhorn import distributed_sinkhorn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat


class ProbProtoSegHead(nn.Module):
    """
    Similarity between pixel embeddings and prototypes

    x: batch-wise pixel embeddings [(b h w) c]
    x_var: var of x [(b h w) c]
    prototypes: [cls_num, proto_num, channel/c]
    c: similarity []
    gt_seg: [b*h*w]
    """

    def __init__(self, configer):
        super(ProbProtoSegHead, self).__init__()
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
        self.cov_gamma = self.configer.get('protoseg', 'cov_gamma')
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

        self.use_gt_proto_learning = self.configer.get('protoseg', 'use_gt_proto_learning')

    def compute_similarity(self, x):
        proto_mean = self.prototypes.data.clone()

        if self.sim_measure == 'fast_mls':
            x_var = torch.mean(x_var, dim=-1)  # x_var: [n, k] -> [n]
            proto_var = torch.mean(proto_var, dim=-1)  # [c m]
            proto_var = rearrange(proto_var, 'c m -> c m 1')  # [c m 1]
            proto_var = repeat(proto_var, 'c m 1 -> c m n',
                                n=x_var.shape[0])  # [c m n]
            sim_mat = torch.einsum(
                'nd,kmd->nmk', x, proto_mean)  # [n m c]
            sim_mat = sim_mat.permute(2, 1, 0)  # [c m n]
            # [c m n]
            sim_mat = - (2 - 2 * sim_mat) / (proto_var +
                                                x_var) - torch.log(proto_var + x_var)
            sim_mat = sim_mat.permute(2, 0, 1)  # [n c m]
            
        elif self.sim_measure == 'cosine':
            # batch product (toward d dimension) -> cosine simialrity between fea and prototypes
            # n: h*w, k: num_class, m: num_prototype
            sim_mat = torch.einsum('nd,kmd->nmk', x, proto_mean)
            
            sim_mat = sim_mat.permute(0, 2, 1)
            
        else: 
            Log.error('Similarity measure is invalid.')
            
        return sim_mat

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

    def prototype_learning_boundary(self, sim_mat, out_seg, gt_seg, _c, gt_boundary=None):
        """
        Prototype selection and update
        _c: (normalized) feature embedding [(b h w) c]
        _c_var: [(b h w) c]
        M = 1 - C, M: cost matrx, C: similarity matrix
        T: optimal transport plan

        Class-wise unsupervised clustering, which means this clustering only selects the prototype in its gt class. (Intra-clss unsupervised clustering)

        If use_boundary is True: (m - 1) prototypes for class-wise non-edge pixel embeddings,
        and 1 prototype for class-wise edge pixel embeddings._c_var    

        gt_boundary: 0: non-edge, 1: edge 
        """
        #! pixel-to-prototype online clustering
        protos = self.prototypes.data.clone()
        non_edge_protos = protos[:, :-1, :]
        edge_protos = protos[:, -1, :] # [c k]

        sim_mat = sim_mat.permute(0, 2, 1)  # [n m c]

        non_edge_proto_num = self.num_prototype - 1

        #! filter out the edge pixels in sim_mat for subsequent intra-class unsupervised clustering, let the last prototype in each class be the boundary prototype
        gt_seg_ori = gt_seg.clone()
        _c_ori = _c.clone()
        if self.use_prototype and _c_var is not None:
            _c_var_ori = _c_var.clone()
        # 0: non-edge, 255: edge
        non_boundary_mask = gt_boundary == 0
        sim_mat = sim_mat[non_boundary_mask, :-1, ...]  # [non_edge_num, (m-1), c]
        if self.use_prototype and _c_var is not None:
            _c_var = _c_var[non_boundary_mask, ...]
        _c = _c[non_boundary_mask, ...]
        gt_seg = gt_seg[non_boundary_mask]

        # largest score inside a class
        pred_seg = torch.max(out_seg, dim=1)[1]  # [b*h*w]
        mask = (gt_seg == pred_seg.view(-1)[non_boundary_mask])  # [b*h*w] bool

        proto_target = gt_seg_ori.clone().float()

        for i in range(self.num_classes):
            if i == 255:
                continue
            # the ones are boundary & gt is class i & correctly predicted
            boundary_cls_mask = torch.logical_and(
                gt_boundary == 1, gt_seg_ori == i)  # [b*h*w]
            boundary_cls_mask = torch.logical_and(boundary_cls_mask, pred_seg.view(-1) == i)
            # the ones are predicted correctly to class i
            if torch.count_nonzero(boundary_cls_mask) > 0:
                boundary_c_cls = _c_ori[boundary_cls_mask]  # [n k]
                boundary_c_cls = torch.mean(boundary_c_cls, dim=0) # [k]
                boundary_c_cls = F.normalize(boundary_c_cls, p=2, dim=-1)
                
                if self.update_prototype:
                    edge_protos[i, ...] = momentum_update(old_value=edge_protos[i, ...],
                                                        new_value=boundary_c_cls,
                                                        momentum=self.mean_gamma)
            #!=====non-boundary prototype learning and update ======#
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
                              tile=non_edge_proto_num)  # [n, num_proto], bool

            # ! select the prototypes of the correctly predicted pixels in i-th class
            m_q = q * m_k_tile  # [n, num_proto]

            c_k = _c[gt_seg == i, ...]  # [n, embed_dim]

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            # correctly predicted pixel embed  [n embed_dim]
            c_q = c_k * c_k_tile

            # f = m_q.transpose(0, 1) @ c_q  # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim] mean?

            var_k = _c_var[gt_seg == i, ...]

            var_q = var_k * c_k_tile

            # the prototypes that are being selected calcualted by sum
            n = torch.sum(m_q, dim=0)  # [num_proto]

            if self.update_prototype is True and torch.sum(n) > 0:
                f = m_q.transpose(0, 1) @ c_q  # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=non_edge_protos[i, n != 0, :], new_value=f[n != 0, :], momentum=self.mean_gamma, debug=False)
                non_edge_protos[i, n != 0, :] = new_value
                
            # each class has a target id between [0, num_proto * c]
            #! ignore_label are still -1, and not being modified
            # non-edge pixels
            proto_target[torch.logical_and(gt_seg_ori == i, non_boundary_mask)] = indexs.float(
            ) + (self.num_prototype * i)  # n samples -> n*m labels
            # edge pixels
            proto_target[torch.logical_and(gt_seg_ori == i, torch.logical_not(non_boundary_mask))] = float(
                self.num_prototype - 1) + (self.num_prototype * i)  # n samples -> n*m labels

        edge_protos = edge_protos.unsqueeze(1) # [c 1 k]
        protos = torch.cat((non_edge_protos, edge_protos), dim=1) # [c m k]
        
        self.prototypes = nn.Parameter(
            l2_normalize(protos), requires_grad=False)  # make norm of proto equal to 1

        del gt_seg, gt_seg_ori, _c, gt_boundary, sim_mat, non_boundary_mask, pred_seg, mask

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
    
    def forward(self, x, gt_semantic_seg=None, boundary_pred=None, gt_boundary=None):
        ''' 
        boundary_pred: [(b h w) 2]
        '''
        b_size = x.shape[0]
        h_size = x.shape[2]
        w_size = x.shape[3]
        gt_size = x.size()[2:]

        x = rearrange(x, 'b c h w -> (b h w) c')

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # sim_mat: # [n c m]
        sim_mat = self.compute_similarity(x)

        out_seg = torch.amax(sim_mat, dim=2)  # [n, num_cls]
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, '(b h w) c -> b c h w',
                            b=b_size, h=h_size)

        if self.use_boundary and gt_boundary is not None:
            gt_boundary = F.interpolate(
                gt_boundary.float(), size=gt_size, mode='nearest').view(-1)

        if self.pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = F.interpolate(
                gt_semantic_seg.float(), size=gt_size, mode='nearest').view(-1)

            if self.use_boundary and gt_boundary is not None:
                contrast_target = self.prototype_learning_boundary(
                    sim_mat, out_seg, gt_seg, x, gt_boundary)
            else:
                contrast_target = self.prototype_learning(
                    sim_mat, out_seg, gt_seg, x)

            sim_mat = rearrange(sim_mat, 'n c m -> n (c m)')

            if boundary_pred is not None and self.use_boundary:
                return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target, "boundary": boundary_pred}
            else:
                return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target}

        return out_seg
