import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import ot
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property
import math

from lib.utils.tools.logger import Logger as Log
from lib.models.modules.contrast import momentum_update, l2_normalize
from lib.models.modules.sinkhorn import distributed_sinkhorn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat


def _batch_vector_diag(bvec):
    """
    Returns the diagonal matrices of a batch of vectors.
    """
    n = bvec.size(-1)
    bmat = bvec.new_zeros(bvec.shape + (n,))
    bmat.view(bvec.shape[:-1] + (-1,))[..., ::n + 1] = bvec
    return bmat


class MultivariateNormalDiag(Distribution):
    arg_constraints = {"loc": constraints.real,
                       "scale_diag": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale_diag, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        event_shape = loc.shape[-1:]
        if scale_diag.shape[-1:] != event_shape:
            raise ValueError(
                "scale_diag must be a batch of vectors with shape {}".format(event_shape))
        try:
            self.loc, self.scale_diag = torch.broadcast_tensors(
                loc, scale_diag)
        except RuntimeError:
            raise ValueError("Incompatible batch shapes: loc {}, scale_diag {}"
                             .format(loc.shape, scale_diag.shape))
        batch_shape = self.loc.shape[:-1]
        super(MultivariateNormalDiag, self).__init__(batch_shape, event_shape,
                                                     validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @lazy_property
    def variance(self):
        return self.scale_diag.pow(2)

    @lazy_property
    def covariance_matrix(self):
        return _batch_vector_diag(self.scale_diag.pow(2))

    @lazy_property
    def precision_matrix(self):
        return _batch_vector_diag(self.scale_diag.pow(-2))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new_empty(shape).normal_()
        return self.loc + self.scale_diag * eps

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        return (
            -0.5 * self._event_shape[0] * math.log(2 * math.pi)
            - self.scale_diag.log().sum(-1)
            - 0.5 * (diff / self.scale_diag).pow(2).sum(-1)
        )

    def entropy(self):
        return (
            0.5 * self._event_shape[0] * (math.log(2 * math.pi) + 1) +
            self.scale_diag.log().sum(-1)
        )


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
        self.use_probability = self.configer.get('protoseg', 'use_probability')
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
        trunc_normal_(self.prototypes, std=0.02)  # ori: 0.02

        if self.use_probability:
            self.proto_var = nn.Parameter(torch.ones(
                self.num_classes, self.num_prototype, self.proj_dim), requires_grad=False)

    def compute_similarity(self, x, x_var):
        cosine_dist = None
        if self.use_probability:
            proto_mean = self.prototypes.data.clone()
            proto_var = self.proto_var.data.clone()

            if self.sim_measure == 'mls':  # larger mls -> similar
                proto_mean = rearrange(
                    proto_mean, 'c m k -> c m 1 k')  # [c m 1 k]
                proto_var = rearrange(
                    proto_var, 'c m k -> c m 1 k')  # [c m 1 k]

                # [c m 1 k] - [n k]=[c m n k]
                mean_diff = torch.square((proto_mean - x))  # [c m n k]

                var_sum = x_var + proto_var  # [c m 1 k]

                sim_mat = mean_diff / var_sum + torch.log(var_sum)
                sim_mat = - 0.5 * torch.mean(sim_mat, dim=-1)

                sim_mat = sim_mat.permute(2, 0, 1)  # [n c m]

            if self.sim_measure == 'fast_mls':
                x_var = torch.mean(x_var, dim=-1)  # x_var: [n, k] -> [n]
                proto_var = torch.mean(proto_var, dim=-1)  # [c m]
                proto_var = rearrange(proto_var, 'c m -> c m 1')  # [c m 1]
                proto_var = repeat(proto_var, 'c m 1 -> c m n',
                                   n=x_var.shape[0])  # [c m n]
                cosine_dist = torch.einsum(
                    'nd,kmd->nmk', x, self.prototypes)  # [n m c]
                cosine_dist = cosine_dist.permute(2, 1, 0)  # [c m n]
                # [c m n]
                sim_mat = - (2 - 2 * cosine_dist) / (proto_var +
                                                     x_var) - torch.log(proto_var + x_var)
                sim_mat = sim_mat.permute(2, 0, 1)  # [n c m]

            elif self.sim_measure == "wasserstein":  # smaller -> similar
                proto_mean = rearrange(
                    proto_mean, 'c m k -> c m 1 k')  # [c m 1 k]
                proto_var = rearrange(
                    proto_var, 'c m k -> c m 1 k')  # [c m 1 k]

                x_var = torch.sqrt(x_var + 1e-8)
                proto_var = torch.sqrt(proto_var + 1e-8)

                sim_mat = torch.zeros((self.num_classes, self.num_prototype, x.shape[0])).cuda()

                # for memory usage
                for i in range(self.num_classes):
                    sim_mat[i] = (proto_mean[i] ** 2).mean(-1) + \
                        (x ** 2).mean(-1) - (2 * proto_mean[i] * x).mean(-1) + \
                        (proto_var[i] ** 2).mean(-1) + (x_var ** 2).mean(-1) - \
                        (2 * proto_var[i] * x_var).mean(-1)

                sim_mat = - sim_mat

                sim_mat = sim_mat.permute(2, 0, 1)  # [n c m]

            elif self.sim_measure == 'bhattacharyya':
                proto_mean = rearrange(
                    proto_mean, 'c m k -> c m 1 k')  # [c m 1 k]
                proto_var = rearrange(
                    proto_var, 'c m k -> c m 1 k')  # [c m 1 k]

                sim_mat = 0.25 * ((proto_mean ** 2).mean(-1) + (x ** 2).mean(-1) -
                                  (2 * proto_mean * x).mean(-1))

                x_var = torch.sqrt(x_var)
                proto_var = torch.sqrt(proto_var)

                sim_mat += 0.5 * torch.log(x_var / proto_var + proto_var / x_var).mean(-1)

                sim_mat = - sim_mat  # smaller bhattacharyya -> similar

                sim_mat = sim_mat.permute(2, 0, 1)  # [n c m]

                # [c m 1 k] + [n k] - [c m n k] + [c m n k] / [c m n k] -> [c m n]

            elif self.sim_measure == 'gmm_log_prob':
                """
                log probability of GMM classifier as similarixty measure
                """
                factor_n = self.configer.get('GMM', 'factor_n')
                factor_c = self.configer.get('GMM', 'factor_c')
                factor_p = self.configer.get('GMM', 'factor_p')

                _prob_n = []
                # divide calculating log prob of pixel embeddings for several times for memory usage
                _n_group = x.shape[0] // factor_n
                # divide calculate gaussian for several times or memory usage
                _c_group = self.num_classes // factor_c

                for _c in range(0, self.num_classes, _c_group):
                    _prob_c = []
                    _c_means = proto_mean[_c:_c+_c_group]
                    _c_cov = proto_var[_c:_c+_c_group]

                    _c_gauss = MultivariateNormalDiag(
                        _c_means.view(-1, self.proj_dim),
                        scale_diag=_c_cov.view(-1, self.proj_dim))  # c * m multivariate gaussian

                    for _n in range(0, x.shape[0], _n_group):
                        _prob_c.append(_c_gauss.log_prob(
                            x[_n:_n+_n_group, None, ...]))

                    _c_probs = torch.cat(_prob_c, dim=0)  # [n, c*m?]
                    _c_probs = _c_probs.contiguous().view(
                        _c_probs.shape[0], -1, self.num_prototype)  # [n, c, m]
                    _prob_n.append(_c_probs)
                sim_mat = torch.cat(_prob_n, dim=1)  # todo [n, c, m]
                # sim_mat = sim_mat.contiguous().view(sim_mat.shape[0], -1)  # [n, (c m)]
                # sim_mat = rearrange(sim_mat, 'n (c m) -> n c m')

        else:
            if self.sim_measure == 'cosine':
                # batch product (toward d dimension) -> cosine simialrity between fea and prototypes
                # n: h*w, k: num_class, m: num_prototype
                sim_mat = torch.einsum('nd,kmd->nmk', x, self.prototypes)

                sim_mat = sim_mat.permute(2, 0, 1)  # [n c m]

        del proto_var, proto_mean, x, x_var

        return sim_mat, cosine_dist

    def shifted_var(x, sim_mat, row_var=True, bias=True):
        """ 
        row_var: each row is a variance of a prototype
        """
        x = x if row_var else x.transpose(-1, -2)
        factor = 1 / (x.shape[-1] - int(not bool(bias)))

        # row_var is False -> x: [proj_dim, n]
        x = (sim_mat[None, ...] * (x ** 2)).sum(-1)

        return factor * x

    def prototype_learning(self, sim_mat, out_seg, gt_seg, _c, _c_var):
        """
        Prototype selection and update
        _c: (normalized) feature embin_channels
        each pixel corresponds to a mix to multiple protoyptes
        "Prototype selectiprototype_learninglamda * tr(T * log(T) - 1 * 1_T)_t)
        M = 1 - C, M: cost matrx, C: similarity matrix
        T: optimal transport plan

        sim_mat: [b*h*w, num_cls, num_proto]
        gt_seg: [b*h*w]
        out_seg: [b*h*w, m]
        """
        # largest score inside a class
        pred_seg = torch.max(out_seg, dim=1)[1]  # [b*h*w]

        mask = (gt_seg == pred_seg.view(-1))  # [b*h*w] bool

        #! pixel-to-prototype online clustering
        protos = self.prototypes.data.clone()
        proto_var = self.proto_var.data.clone()
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

            m_k = mask[gt_seg == i]  # the correctly predicted ones

            m_k_tile = repeat(m_k, 'n -> n tile',
                              tile=self.num_prototype)  # [n, num_proto], bool

            # ! select the prototypes of the correctly predicted pixels in i-th class
            m_q = q * m_k_tile  # [n, num_proto]

            c_k = _c[gt_seg == i, ...]  # [n, embed_dim]

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            # correctly predicted pixel embed  [n embed_dim]
            c_q = c_k * c_k_tile

            # f = m_q.transpose(0, 1) @ c_q  # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]

            var_k = _c_var[gt_seg == i, ...]

            var_q = var_k * c_k_tile

            # the prototypes that are being selected calcualted by sum
            n = torch.sum(m_q, dim=0)  # [num_proto]

            if self.update_prototype is True and torch.sum(n) > 0:
                if self.use_probability:
                    for k in range(self.num_prototype):
                        if torch.sum(m_q[..., k] > 0):
                            """
                            1. correctly predicted: c_q / var_q
                            2. choose the ones belonging to this prototype in c_q / var_q 
                            """
                            proto_ind = torch.nonzero(
                                (m_q[..., k] != 0), as_tuple=True)

                            if self.sim_measure == 'gmm_log_prob':
                                # todo: why do not multiply with probs of GMM????????
                                mean_gamma = torch.sum(
                                    (init_q[proto_ind, k] * c_q[proto_ind]),
                                    dinm=0) / (c_q[proto_ind].shape[0])  # [proj_dim]

                                mean_gamma = l2_normalize(mean_gamma)

                                # [n_proto, proj_dim]
                                _shfit_fea = c_q[proto_ind] - \
                                    mean_gamma[None, ...]

                                _cov = self.shifted_var(
                                    _shfit_fea, row_var=False)  # [proj_dim]
                                # TODO: How to utilize variance of pixel embeddings?
                            else:
                                # [embed_dim]
                                var_hat = 1 / \
                                    (torch.sum(
                                        (1/var_q[proto_ind]), dim=0)) * (var_q[proto_ind].shape[0])  # [embed_dim]

                                # var_hat = torch.log(var_hat)
                                # var_hat = torch.sigmoid(var_hat)
                                # var_hat = torch.exp(var_hat)

                                mean_gamma = torch.mean(
                                    ((var_hat * c_q[proto_ind]
                                      ) / var_q[proto_ind]),
                                    dim=0)  # [fea_dim]
                                mean_gamma = l2_normalize(mean_gamma)

                            #! momentum update
                            protos[i, k, :] = momentum_update(old_value=protos[i, k, :],
                                                              new_value=mean_gamma,
                                                              momentum=self.mean_gamma)
                            proto_var[i, k, :] = momentum_update(old_value=proto_var[i, k, :],
                                                                 new_value=var_hat,
                                                                 momentum=self.cov_gamma)

                elif self.use_probability is False:
                    return
            # each class has a target id between [0, num_proto * c]
            proto_target[gt_seg == i] = indexs.float(
            ) + (self.num_prototype * i)  # n samples -> n*m labels

        self.prototypes = nn.Parameter(
            l2_normalize(protos), requires_grad=False)  # make norm of proto equal to 1
        #! =============== rescale variance into proper value range ===============
        self.proto_var = nn.Parameter(proto_var, requires_grad=False)

        if dist.is_available() and dist.is_initialized():  # distributed learning
            protos = self.prototypes.data.clone()
            """
            To get average result across all gpus: 
            first average the sum
            then sum the tensors on all gpus, and copy the sum to all gpus
            """
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)
            dist.all_reduce(proto_var.div_(dist.get_world_size()))
            self.proto_var = nn.Parameter(proto_var, requires_grad=False)

        return proto_target  # [n]

    def forward(self, x, x_var, gt_semantic_seg=None):
        b_size = x.shape[0]
        h_size = x.shape[2]
        w_size = x.shape[3]
        gt_size = x.size()[2:]

        x = rearrange(x, 'b c h w -> (b h w) c')
        x_var = rearrange(x_var, 'b c h w -> (b h w) c')

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # sim_mat: # [n c m]
        sim_mat, cosine_dist = self.compute_similarity(x, x_var)

        out_seg = torch.amax(sim_mat, dim=2)  # [n, num_cls]
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, '(b h w) c -> b c h w',
                            b=b_size, h=h_size)

        sim_mat = rearrange(sim_mat, 'n c m -> n (c m)')
        sim_mat = self.proto_norm(sim_mat)
        sim_mat = rearrange(sim_mat, 'n (c m) -> n c m', c=self.num_classes)

        if self.pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = F.interpolate(
                gt_semantic_seg.float(), size=gt_size, mode='nearest').view(-1)

            contrast_target = self.prototype_learning(
                sim_mat, out_seg, gt_seg, x, x_var)

            sim_mat = rearrange(sim_mat, 'n c m -> n (c m)')

        # TODO: ====================== use var as temperature ======================

            if self.configer.get('uncertainty_visualizer', 'vis_uncertainty') and self.configer.get(
                    'phase') == 'val':
                x_var = x_var.mean(-1)  # [(b h w)]
                x_var = rearrange(x_var, '(b h w) -> b h w',
                                  b=b_size, h=h_size)
                return {'seg': out_seg, 'uncertainty': x_var}

            elif self.configer.get('protoseg', 'var_temp'):
                proto_var = self.proto_var.data.clone()
                proto_var = proto_var.mean(-1)
                return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target, 'proto_var': proto_var}

            else:
                return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target}

        return out_seg
