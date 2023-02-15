import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import gc

from lib.utils.tools.logger import Logger as Log
from lib.models.modules.contrast import momentum_update, l2_normalize
from lib.models.modules.sinkhorn import distributed_sinkhorn
from timm.models.layers import trunc_normal_
from lib.utils.distributed import get_world_size, get_rank, is_distributed
from einops import rearrange, repeat
# from lib.models.modules.local_refinement_module import LocalRefinementModule


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
        self.use_attention = self.configer.get('protoseg', 'use_attention')
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
        if self.use_uncertainty:
            #! sigma (not log(sigma))
            self.proto_var = nn.Parameter(torch.ones(
                self.num_classes, self.num_prototype, self.proj_dim), requires_grad=False)
            #! weight between mean and variance when calculating similarity
            if self.sim_measure == 'wasserstein':
                self.lamda = 1 / self.proj_dim  # scaling factor for b_distance

            self.avg_update_proto = self.configer.get('protoseg', 'avg_update_proto')
            self.weighted_ppd_loss = self.configer.get('protoseg', 'weighted_ppd_loss')

        self.use_temperature = self.configer.get('protoseg', 'use_temperature')

        self.attention_proto = self.configer.get('protoseg', 'attention_proto')
        if self.attention_proto:
            self.lamda_p = self.configer.get('protoseg', 'lamda_p')
        self.local_refinement = self.configer.get('protoseg', 'local_refinement')
        # if self.local_refinement:
        #     self.local_refine_module = LocalRefinementModule(configer=configer)
        
    def reparameterize(self, mu, logvar, k=1):
        sample_z = []
        for _ in range(k):
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            sample_z.append(eps.mul(std).add_(mu))
        sample_z = torch.cat(sample_z, dim=0)
        return sample_z

    def compute_similarity(self, x, x_var=None, sim_measure='wasserstein'):
        ''' 
        x/x_var: [(b h w) k]
        proto_mean: [c m k]
        Use reparameterization trick to compute probabilistic distance for gradient flow.
        '''
        proto_mean = self.prototypes.data.clone()
        if self.use_uncertainty and x_var is not None:
            proto_var = self.proto_var.data.clone()
            if sim_measure == 'wasserstein':
                # ======== x * prototypes =========
                # -[c m n] + [n] + [c m 1] - 2 * [c m n]
                #! lamda depends on dimension of embeddings
                sim_mat = torch.einsum('nd,kmd->nmk', x, proto_mean)  # [n c m]
                sim_mat = sim_mat.permute(2, 1, 0)  # [c m n]
                # ======= whole wasserstein loss ==========
                sim_mat = 2 - 2 * sim_mat + self.lamda * (x_var.sum(-1) + proto_var.sum(-1, keepdim=True) - 2 * torch.einsum(
                    'nk,cmk->cmn', torch.sqrt(x_var), torch.sqrt(proto_var)))
                sim_mat = sim_mat.permute(2, 0, 1)  # [n c m]
                sim_mat = -sim_mat
            elif sim_measure == 'fast_mls':
                sim_mat = torch.einsum('nd,kmd->nmk', x, proto_mean)  # [n c m]
                sim_mat = sim_mat.permute(2, 1, 0)  # [c m n]

                x_var = torch.mean(x_var, dim=-1, keepdim=True)  # [n, 1]
                proto_var = torch.mean(proto_var, dim=-1, keepdim=True)  # [c m 1]
                proto_mean = proto_mean.unsqueeze(-2)  # [c m 1 k]
                proto_var = proto_var.unsqueeze(-2)  # [c m 1 k]

                sim_mat = (
                    2 - sim_mat) / (x_var + proto_var).sum(-1) + torch.log(x_var).sum(-1) + torch.log(proto_var).sum(-1)
                sim_mat = sim_mat.permute(2, 0, 1)  # [n c m]
                sim_mat = -0.5 * sim_mat
            elif sim_measure == 'match_prob':
                self.reparameterize()

            del x_var, x, proto_mean, proto_var

        else:
            if sim_measure == 'cosine':
                # batch product (toward d dimension) -> cosine simialrity between fea and prototypes
                # n: h*w, k: num_class, m: num_prototype
                sim_mat = torch.einsum('nd,kmd->nmk', x, proto_mean)  # [c m n]

                sim_mat = sim_mat.permute(0, 2, 1)  # [c n m]
            else:
                Log.error('Similarity measure is invalid.')
        return sim_mat

    def sample_gaussian_tensors(self, mu, sigma, num_samples):
        eps = torch.randn(num_samples, mu.size(0), mu.size(1), dtype=mu.dtype, device=mu.device)
        samples = eps.mul(sigma.unsqueeze(0)).add_(
            mu.unsqueeze(0))
        return samples  # [sample_time n k]

    def prototype_learning(self, sim_mat, out_seg, gt_seg, _c, _c_var):
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

            # if self.use_temperature:
            #     q_k_tile = repeat(m_k, 'n -> n tile', tile=init_q.shape[-1])
            #     init_q = init_q * q_k_tile # sim mat of correctly predicted pixels

            # f = m_q.transpose(0, 1) @ c_q  # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
            if self.use_uncertainty and _c_var is not None:
                var_k = _c_var[gt_seg == i, ...]

                var_q = var_k * c_k_tile  # [n embed_dim]

            # the prototypes that are being selected calcualted by sum
            n = torch.sum(m_q, dim=0)  # [num_proto]

            if self.update_prototype is True and torch.sum(n) > 0:
                if not self.use_uncertainty:
                    # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                    f = m_q.transpose(0, 1) @ c_q
                    f = F.normalize(f, p=2, dim=-1)
                    protos[i, n != 0, :] = momentum_update(
                        old_value=protos[i, n != 0, :], new_value=f[n != 0, :], momentum=self.mean_gamma, debug=False)

                else:
                    m_q_sum = m_q.sum(dim=0)  # [num_proto]
                    if not self.avg_update_proto:
                        # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                        f_v = 1 / ((m_q.transpose(0, 1) @ (1 / (var_q + 1e-3))) /
                                   (m_q_sum.unsqueeze(-1) + 1e-3) + 1e-3)
                        # [1 num_proto embed_dim] / [[n 1 embed_dim]] =[n num_proto embed_dim]
                        f = (f_v.unsqueeze(0) / (var_q.unsqueeze(1) + 1e-3)) * c_q.unsqueeze(1)
                        f = torch.einsum('nm,nmk->mk', m_q, f)
                        # todo debug
                        f = f / (m_q_sum.unsqueeze(-1) + 1e-3)
                        f = F.normalize(f, p=2, dim=-1)
                    else:
                        # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                        f = m_q.transpose(0, 1) @ c_q
                        f = F.normalize(f, p=2, dim=-1)

                        protos[i, n != 0, :] = momentum_update(
                            old_value=protos[i, n != 0, :], new_value=f[n != 0, :], momentum=self.mean_gamma, debug=False)

                        f_v = (m_q.transpose(0, 1) @ (var_q + (c_q ** 2))) / (m_q_sum.unsqueeze(-1) +
                                                                              1e-3) - ((m_q.transpose(0, 1) @ c_q) / (m_q_sum.unsqueeze(-1) + 1e-3)) ** 2
                        assert torch.count_nonzero(torch.isinf(f_v)) == 0
                        #! normalize for f_v
                        # f_v = torch.exp(torch.sigmoid(torch.log(f_v)))
                    protos[i, n != 0, :] = momentum_update(
                        old_value=protos[i, n != 0, :], new_value=f[n != 0, :], momentum=self.mean_gamma, debug=False)
                    proto_var[i, n != 0, :] = momentum_update(
                        old_value=proto_var[i, n != 0, :], new_value=f_v[n != 0, :], momentum=self.var_gamma, debug=False)

                    if self.attention_proto:
                        for m in range(self.num_prototype):
                            proto_score = -0.5 * ((protos[i, m, :] - protos[i, ...]) ** 2 / (
                                proto_var[i, m, :] + proto_var[i, ...]) + torch.log(proto_var[i, m, :] + proto_var[i, ...])).mean(-1)  # [m]
                            ind_mask = torch.arange(self.num_prototype).cuda() != m
                            proto_score = proto_score.masked_select(
                                ind_mask)  # [self.num_Proto - 1]
                            proto_score = proto_score / proto_score.sum()
                            protos[i, m, :] = protos[i, m, :] + self.lamda_p * (proto_score.unsqueeze(-1) * protos[i, ...].masked_select(
                                ind_mask.unsqueeze(-1)).reshape(-1, self.proj_dim)).sum(0)
                    del var_q, var_k, f, f_v

            # each class has a target id between [0, num_proto * c]
            #! ignore_label are still -1, and not being modified
            proto_target[gt_seg == i] = indexs.float(
            ) + (self.num_prototype * i)  # n samples -> n*m labels

            del c_q, c_k_tile, c_k, m_q, m_k_tile, m_k, indexs, q

        self.prototypes = nn.Parameter(
            l2_normalize(protos), requires_grad=False)  # make norm of proto equal to 1
        if self.use_uncertainty:
            self.proto_var = nn.Parameter(proto_var, requires_grad=False)

        if dist.is_available() and dist.is_initialized():  # distributed learning
            """
            To get average result across all gpus: 
            first average the sum
            then sum the tensors on all gpus, and copy the sum to all gpus
            """
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)
            if self.use_uncertainty:
                proto_var = self.proto_var.data.clone()
                dist.all_reduce(proto_var.div_(dist.get_world_size()))
                self.proto_var = nn.Parameter(proto_var, requires_grad=False)

        return proto_target  # [n]

    def prototype_learning_boundary(self, sim_mat, out_seg, gt_seg, _c, _c_var, gt_boundary=None):
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
        edge_protos = protos[:, -1, :]  # [c k]

        sim_mat = sim_mat.permute(0, 2, 1)  # [n m c]

        non_edge_proto_num = self.num_prototype - 1

        #! filter out the edge pixels in sim_mat for subsequent intra-class unsupervised clustering, let the last prototype in each class be the boundary prototype
        gt_seg_ori = gt_seg.clone()
        _c_ori = _c.clone()
        # 0: non-edge, 255: edge
        non_boundary_mask = gt_boundary == 0
        sim_mat = sim_mat[non_boundary_mask, :-1, ...]  # [non_edge_num, (m-1), c]
        _c = _c[non_boundary_mask, ...]
        gt_seg = gt_seg[non_boundary_mask]

        if self.use_prototype and _c_var is not None:
            _c_var_ori = _c_var.clone()
            _c_var = _c_var[non_boundary_mask, ...]
            proto_var = self.proto_var.data.clone()
            non_edge_proto_var = proto_var[:, :-1, :]
            edge_proto_var = proto_var[:, -1, :]  # [c k]

        # largest score inside a class
        pred_seg = torch.max(out_seg, dim=1)[1]  # [b*h*w]
        mask = (gt_seg == pred_seg.view(-1)[non_boundary_mask])  # [b*h*w] bool

        proto_target = gt_seg_ori.clone().float()

        for i in range(self.num_classes):
            # the ones are boundary & gt is class i & correctly predicted
            boundary_cls_mask = torch.logical_and(
                gt_boundary == 1, gt_seg_ori == i)  # [b*h*w]
            boundary_cls_mask = torch.logical_and(boundary_cls_mask, pred_seg.view(-1) == i)
            # the ones are predicted correctly to class i
            if torch.count_nonzero(boundary_cls_mask) > 0 and self.update_prototype:
                boundary_c_cls = _c_ori[boundary_cls_mask]  # [n k]

                if not self.use_uncertainty:
                    boundary_c_cls = torch.sum(boundary_c_cls, dim=0)  # [k]
                    boundary_c_cls = F.normalize(boundary_c_cls, p=2, dim=-1)
                    edge_protos[i, ...] = momentum_update(old_value=edge_protos[i, ...],
                                                          new_value=boundary_c_cls,
                                                          momentum=self.mean_gamma)
                else:
                    boundary_c_var_cls = _c_var_ori[boundary_cls_mask]  # [n k]
                    if not self.avg_update_proto:
                        n = boundary_c_var_cls.shape[0]
                        b_v = 1 / (((1 / (boundary_c_var_cls + 1e-3)).mean(0)) + 1e-3)
                        # [1 num_proto embed_dim] / [[n 1 embed_dim]] =[n num_proto embed_dim]
                        b = ((b_v.unsqueeze(0) / (boundary_c_var_cls.unsqueeze(1) + 1e-3))
                             * boundary_c_cls.unsqueeze(1)).mean(0)
                        b = F.normalize(b, p=2, dim=-1)
                    else:
                        # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                        b = boundary_c_cls.sum(0)
                        b = F.normalize(b, p=2, dim=-1)
                        protos[i, n != 0, :] = momentum_update(
                            old_value=protos[i, n != 0, :], new_value=f[n != 0, :], momentum=self.mean_gamma, debug=False)
                        f_v = (m_q.transpose(0, 1) @ (var_q / n)) + (m_q.transpose(0, 1)
                                                                     @ c_q ** 2) / n - (m_q.transpose(0, 1) @ c_q / n) ** 2
                    edge_protos[i, ...] = momentum_update(old_value=edge_protos[i, ...],
                                                          new_value=b,
                                                          momentum=self.mean_gamma)
                    edge_proto_var[i, ...] = momentum_update(old_value=edge_proto_var[i, ...],
                                                             new_value=b_v,
                                                             momentum=self.var_gamma)

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

            if self.use_uncertainty and _c_var is not None:

                var_k = _c_var[gt_seg == i, ...]

                var_q = var_k * c_k_tile

            # the prototypes that are being selected calcualted by sum
            n = torch.sum(m_q, dim=0)  # [num_proto]

            if self.update_prototype is True and torch.sum(n) > 0:
                if not self.use_uncertainty:
                    # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                    f = m_q.transpose(0, 1) @ c_q
                    f = F.normalize(f, p=2, dim=-1)
                    protos[i, n != 0, :] = momentum_update(
                        old_value=non_edge_protos[i, n != 0, :], new_value=f[n != 0, :], momentum=self.mean_gamma, debug=False)
                else:
                    m_q_sum = m_q.sum(dim=0)  # [num_proto]
                    if not self.avg_update_proto:
                        # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                        f_v = 1 / ((m_q.transpose(0, 1) @ (1 / (var_q + 1e-3))) /
                                   (m_q_sum.unsqueeze(-1) + 1e-3) + 1e-3)
                        # [1 num_proto embed_dim] / [[n 1 embed_dim]] =[n num_proto embed_dim]
                        f = (f_v.unsqueeze(0) / (var_q.unsqueeze(1) + 1e-3)) * c_q.unsqueeze(1)
                        f = torch.einsum('nm,nmk->mk', m_q, f)
                        f = f / (m_q_sum.unsqueeze(-1) + 1e-3)
                        f = F.normalize(f, p=2, dim=-1)
                    else:
                        # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                        f = m_q.transpose(0, 1) @ c_q
                        f = F.normalize(f, p=2, dim=-1)

                        protos[i, n != 0, :] = momentum_update(
                            old_value=protos[i, n != 0, :], new_value=f[n != 0, :], momentum=self.mean_gamma, debug=False)

                        f_v = (m_q.transpose(0, 1) @ (var_q / n)) + (m_q.transpose(0, 1)
                                                                     @ c_q ** 2) / n - (m_q.transpose(0, 1) @ c_q / n) ** 2
                    non_edge_protos[i, n != 0, :] = momentum_update(
                        old_value=non_edge_protos[i, n != 0, :],
                        new_value=f[n != 0, :],
                        momentum=self.mean_gamma, debug=False)
                    non_edge_proto_var[i, n != 0, :] = momentum_update(
                        old_value=non_edge_proto_var[i, n != 0, :],
                        new_value=f_v[n != 0, :],
                        momentum=self.var_gamma, debug=False)

            # each class has a target id between [0, num_proto * c]
            #! ignore_label are still -1, and not being modified
            # non-edge pixels
            proto_target[torch.logical_and(gt_seg_ori == i, non_boundary_mask)] = indexs.float(
            ) + (self.num_prototype * i)  # n samples -> n*m labels
            # edge pixels
            proto_target[torch.logical_and(gt_seg_ori == i, torch.logical_not(non_boundary_mask))] = float(
                self.num_prototype - 1) + (self.num_prototype * i)  # n samples -> n*m labels

            del c_q, c_k_tile, c_k, m_q, m_k_tile, m_k, indexs, q

        edge_protos = edge_protos.unsqueeze(1)  # [c 1 k]
        protos = torch.cat((non_edge_protos, edge_protos), dim=1)  # [c m k]

        if self.use_uncertainty:
            edge_proto_var = edge_proto_var.unsqueeze(1)  # [c 1 k]
            proto_var = torch.cat((non_edge_proto_var, edge_proto_var), dim=1)  # [c m k]
            self.proto_var = nn.Parameter(proto_var, requires_grad=False)

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
            if self.use_uncertainty:
                proto_var = self.proto_var.data.clone()
                dist.all_reduce(proto_var.div_(dist.get_world_size()))
                self.proto_var = nn.Parameter(proto_var, requires_grad=False)

        return proto_target  # [n]

    def forward(self, x, x_var=None, gt_semantic_seg=None, boundary_pred=None, gt_boundary=None):
        ''' 
        boundary_pred: [(b h w) 2]
        '''
        b_size = x.shape[0]
        h_size = x.shape[2]
        k_size = x.shape[1]
        gt_size = x.size()[2:]

        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        # self.proto_var.data.copy_(torch.exp(torch.sigmoid(torch.log(self.proto_var))))

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

            if self.use_uncertainty:
                proto_var = self.proto_var.data.clone()

                if self.configer.get('iters') % 100 == 0:
                    Log.info(proto_var)


                if self.weighted_ppd_loss:
                    proto_var = self.proto_var.data.clone()
                    loss_weight1 = torch.einsum('nk,cmk->cmn', x_var, proto_var)  # [c m n]
                    loss_weight1 = (loss_weight1 - loss_weight1.min()
                                    ) / (loss_weight1.max() - loss_weight1.min())
                    loss_weight1 = loss_weight1.mean()
                    # [n] + [c m 1] = [c m n]
                    loss_weight2 = torch.log(
                        x_var).sum(-1) + torch.log(proto_var).sum(-1, keepdim=True)
                    loss_weight2 = (loss_weight2 - loss_weight2.min()
                                    ) / (loss_weight2.max() - loss_weight2.min())
                    loss_weight2 = loss_weight2.mean()
                    x = x.reshape(b_size, h_size, -1, k_size)
                    x_var = x_var.reshape(b_size, h_size, -1, k_size)
                    return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target, 'w1': loss_weight1, 'w2': loss_weight2, 'x_mean': x, 'x_var': x_var}
                else:
                    return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target}
            else:
                return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target}
        return out_seg
