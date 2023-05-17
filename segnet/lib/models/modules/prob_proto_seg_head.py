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
        self.reparam_k = self.configer.get('protoseg', 'reparam_k')

        # [cls_num, proto_num, channel/k]
        self.prototypes = nn.Parameter(torch.zeros(
            self.num_classes, self.num_prototype, self.proj_dim), requires_grad=False)
        trunc_normal_(self.prototypes, std=0.02)
        if self.use_uncertainty:
            self.proto_var = nn.Parameter(torch.ones(
                self.num_classes, self.num_prototype, self.proj_dim), requires_grad=False)
            #! weight between mean and variance when calculating similarity
            if self.sim_measure == 'wasserstein':
                self.lamda = 1 / self.proj_dim  # scaling factor for b_distance

        self.cosine_classifier = self.configer.get('protoseg', 'cosine_classifier')

    def reparameterize(self, mu, var, k=1, protos=False):
        sample_z = []
        for _ in range(k):
            # std = logvar.mul(0.5).exp_()
            std = torch.sqrt(var)
            eps = std.data.new(std.size()).normal_()
            z = eps.mul(std).add_(mu)

            if not protos:
                z = rearrange(z, 'l n k -> (l n) k')
            else:
                z = rearrange(z, 'l c m k -> (l c m) k')
            z = self.feat_norm(z)
            z = l2_normalize(z)
            if not protos:
                z = z.reshape(1, -1, self.proj_dim)
            else:
                z = z.reshape(1, self.num_classes, self.num_prototype, -1)

            sample_z.append(z)
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
                x = self.reparameterize(x.unsqueeze(0), x_var.unsqueeze(
                    0), self.reparam_k, protos=False)  # [sample_num n k]
                proto_mean = self.reparameterize(
                    proto_mean.unsqueeze(0),
                    proto_var.unsqueeze(0),
                    self.reparam_k, protos=True)  # [sample_num c m k]
                x = repeat(x, 's n k -> s r n k', r=self.reparam_k)
                proto_mean = repeat(proto_mean, 's c m k -> r s c m k', r=self.reparam_k)

                # [reparam_k reparam_k n m c]
                sim_mat = torch.einsum('srnd,zjkmd->sjnmk', x, proto_mean)
                sim_mat = rearrange(sim_mat, 'r l n m c -> (r l) n m c')
                # sim_mat = sim_mat.permute(0, 3, 1, 2)  # [(sample_k sample_k) c n m]

                sim_mat = torch.mean(sim_mat, dim=0)  # [n m c]cc
                sim_mat = sim_mat.permute(0, 2, 1)  # [n c m]
            else:
                Log.error('Similarity measure is invalid.')
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
        mean_gamma = self.mean_gamma
        
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

            if self.use_uncertainty and _c_var is not None and not self.cosine_classifier:
                var_k = _c_var[gt_seg == i, ...]

                var_q = var_k * c_k_tile  # [n embed_dim]

            # the prototypes that are being selected calcualted by sum
            n = torch.sum(m_q, dim=0)  # [num_proto]

            if self.update_prototype is True and torch.sum(n) > 0:
                # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                f = m_q.transpose(0, 1) @ c_q
                f = F.normalize(f, p=2, dim=-1)
                protos[i, n != 0, :] = momentum_update(
                    old_value=protos[i, n != 0, :], new_value=f[n != 0, :], momentum=mean_gamma, debug=False)

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

        sim_mat = self.compute_similarity(
            x, x_var=x_var, sim_measure=self.sim_measure)

        out_seg = torch.amax(sim_mat, dim=2)  # [n, num_cls]
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, '(b h w) c -> b c h w',
                            b=b_size, h=h_size)

        if self.pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            # assert torch.unique(gt_semantic_seg).shape[0] != 1
            gt_seg = F.interpolate(
                gt_semantic_seg.float(), size=gt_size, mode='nearest').view(-1)

            contrast_target = self.prototype_learning(
                sim_mat, out_seg, gt_seg, x, x_var)

            sim_mat = rearrange(sim_mat, 'n c m -> n (c m)')

            return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target}
        
        elif self.configer.get('proto_visualizer', 'vis_prototype'):
            sim_mat = sim_mat.reshape(b_size, h_size, -1, self.num_classes * self.num_prototype)
            return {'seg': out_seg, 'logits': sim_mat}
        elif self.configer.get('phase') == 'test_ros':
            return {'seg': out_seg}
        else:
            return out_seg
