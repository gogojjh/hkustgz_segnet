import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

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
        self.use_attention = self.configer.get('protoseg', 'use_attention')
        if self.use_attention:
            self.proj_dim = 2 * self.proj_dim
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

            self.reparam_k = self.configer.get('protoseg', 'reparam_k')
            #! weight between mean and variance when calculating similarity
            if self.sim_measure == 'wasserstein':
                self.lamda = 1 / self.proj_dim  # scaling factor for b_distance
                
            self.avg_update_proto = self.configer.get('protoseg', 'avg_update_proto')
            
        self.use_temperature = self.configer.get('protoseg', 'use_temperature')
        if self.use_temperature:
            self.alfa = self.configer.get('protoseg', 'alfa')
            proto_confidence = torch.ones((self.num_classes, self.num_prototype)).cuda()
            self.proto_confidence = nn.Parameter(proto_confidence, requires_grad=False)
        
        if self.sim_measure == 'match_prob':
            a = self.configer.get('protoseg', 'init_a') * torch.ones(1).cuda()
            b = self.configer.get('protoseg', 'init_b') * torch.ones(1).cuda()
            self.a = nn.Parameter(a)
            self.b = nn.Parameter(b)
        
        kernel = torch.ones((7,7))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        #kernel = np.repeat(kernel, 1, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.sigmoid = nn.Sigmoid()
        
        self.uncertainy_aware_fea = self.configer.get('protoseg', 'uncertainy_aware_fea')
        self.weighted_ppd_loss = self.configer.get('protoseg', 'weighted_ppd_loss')

    def get_uncertainty(self, c):
        ''' 
        c: [sample_num, (b h w ) k)]
        '''
        c = torch.sigmoid(c)
        c = c.var(dim=0, keepdim=True).detach() # [1 b h w]
        if self.configer.get('phase') == 'train':
            # (l-7+2*3)/1+1=l
            c = F.conv2d(c, self.weight, padding=3, groups=1)
            c = F.conv2d(c, self.weight, padding=3, groups=1)
            c = F.conv2d(c, self.weight, padding=3, groups=1)
        # normalize
        c = (c - c.min()) / (c.max() - c.min())

        return c
        
    def compute_similarity(self, x, x_var=None):
        ''' 
        x/x_var: [(b h w) k]
        proto_mean: [c m k]
        Use reparameterization trick to compute probabilistic distance for gradient flow.
        '''
        proto_mean = self.prototypes.data.clone()
        if self.use_uncertainty and x_var is not None:
            proto_var = self.proto_var.data.clone()
            #? reparam
            # [reparam_k (b h w ) k]
            if self.uncertainy_aware_fea:
                x_sample = self.sample_gaussian_tensors(x, x_var, num_samples=self.reparam_k) 
                x_sample = self.get_uncertainty(x_sample)
                x *= (1.0 - x_sample.squeeze(0))
                x = self.feat_norm(x)
                x = l2_normalize(x)
            
            # proto_mean = self.sample_gaussian_tensors(proto_mean.view(-1, self.proj_dim), proto_var.view(-1, self.proj_dim), num_samples=self.reparam_k) # [reparam_k (c m) k]
            # proto_mean = self.feat_norm(proto_mean)
            # proto_mean = l2_normalize(proto_mean)
            # proto_mean = proto_mean.reshape(-1, self.num_classes, self.num_prototype, self.proj_dim)
            # x = x.unsqueeze(1) # [reparam_k 1 (b h w ) k]
            # proto_mean = proto_mean.unsqueeze(0) # [1 reparam_k c m k]
            
            #! use var not log(var) for similarity
            x_var = x_var
            proto_var = proto_var
            
            if self.sim_measure == 'wasserstein':
                # monte-carlo eucl dist
                
            #     sim_mat = torch.einsum('rlnd,abkmd->rbnmk', x, proto_mean) 
            # # [(reparam_k reparam_k) (b h w) m c]
            #     sim_mat = rearrange(sim_mat, 'r l n m c -> (r l) n m c') 
            #     sim_mat = torch.mean(sim_mat, dim=0).permute(2, 1, 0) # [c m n]
                
                # -[c m n] + [n] + [c m 1] - 2 * [c m n]
                #! lamda depends on dimension of embeddings
                #? non-reparam
                sim_mat = torch.einsum('nd,kmd->nmk', x, proto_mean) # [n c m]
                sim_mat = sim_mat.permute(2, 1, 0) # [c m n]
                
                sim_mat = 2 - 2 * sim_mat + self.lamda * (x_var.sum(-1) + \
                    proto_var.sum(-1, keepdim=True) \
                 - 2 * torch.einsum('nk,cmk->cmn', torch.sqrt(x_var), torch.sqrt(proto_var)))    
                sim_mat = sim_mat.permute(2, 0, 1) # [n c m]       
                sim_mat = -sim_mat 
                
            elif self.sim_measure == 'match_prob':
                ''' 
                'Probabilistic Representations for Video Contrastive Learning' /
                'Probabilistic Embeddings for Cross-Modal Retrieval'
                Use average value of multiple samples
                '''
                # [reparam_k reparam_k (b h w) m c]
                sim_mat = torch.einsum('rlnd,abkmd->rbnmk', x, proto_mean) # [num_sample, ]
            # [(reparam_k reparam_k) (b h w) m c]
                sim_mat = rearrange(sim_mat, 'r l n m c -> (r l) n m c') 
                sim_mat = torch.mean(sim_mat, dim=0).permute(2, 1, 0)  # [n m c] -> [c m n]
                
                sim_mat = - self.a * (2 - 2 * sim_mat) + self.b
                
                sim_mat = self.sigmoid(sim_mat) # [c m n]
                sim_mat = sim_mat.permute(2, 0, 1) # [n c m]  
                
            del x_var, x, proto_mean, proto_var
                    
        else:
            if self.sim_measure == 'cosine':
                # batch product (toward d dimension) -> cosine simialrity between fea and prototypes
                # n: h*w, k: num_class, m: num_prototype
                sim_mat = torch.einsum('nd,kmd->nmk', x, proto_mean) # [c m n]
                
                sim_mat = sim_mat.permute(0, 2, 1) # [c n m]
            else: 
                Log.error('Similarity measure is invalid.')
        return sim_mat
    
    def sample_gaussian_tensors(self, mu, logsigma, num_samples):
        eps = torch.randn(num_samples, mu.size(0), mu.size(1), dtype=mu.dtype, device=mu.device)

        # samples = eps.mul(torch.exp(logsigma.unsqueeze(0))).add_(
        #     mu.unsqueeze(0))
        samples = eps.mul(logsigma.unsqueeze(0)).add_(
            mu.unsqueeze(0))
        return samples # [sample_time n k]

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
            
            if self.use_temperature:
                q_k_tile = repeat(m_k, 'n -> n tile', tile=init_q.shape[-1])
                init_q = init_q * q_k_tile # sim mat of correctly predicted pixels 

            # f = m_q.transpose(0, 1) @ c_q  # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
            if self.use_uncertainty and _c_var is not None:
                _c_var = _c_var
                var_k = _c_var[gt_seg == i, ...]

                var_q = var_k * c_k_tile # [n embed_dim]

            # the prototypes that are being selected calcualted by sum
            n = torch.sum(m_q, dim=0)  # [num_proto]

            if self.update_prototype is True and torch.sum(n) > 0:
                if not self.use_uncertainty:
                    f = m_q.transpose(0, 1) @ c_q  # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                    f = F.normalize(f, p=2, dim=-1)
                    protos[i, n != 0, :]  = momentum_update(old_value=protos[i, n != 0, :], new_value=f[n != 0, :], momentum=self.mean_gamma, debug=False)
                
                else:
                    n = m_q.shape[0]
                    if not self.avg_update_proto:
                    # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                        f_v = 1 / (m_q.transpose(0, 1) @ (1 / ((var_q + 1e-3)) + 1e-3) / n)
                        # [1 num_proto embed_dim] / [[n 1 embed_dim]] =[n num_proto embed_dim]
                        f_v = torch.exp(torch.sigmoid(torch.log(f_v)))
                        f = (f_v.unsqueeze(0) / (var_q.unsqueeze(1) + 1e-3)) * c_q.unsqueeze(1)
                        f = torch.einsum('nm,nmk->mk', m_q, f)
                        f = F.normalize(f, p=2, dim=-1)
                        # if self.use_temperature:
                        #     for k in range(self.num_prototype):
                        #             if self.use_temperature:
                        #                 proto_ind = torch.nonzero((m_q[..., k] != 0), as_tuple=True)
                        #                 num = proto_ind[0].shape[0]
                        #                 if num > 0:
                        #                     temp = - init_q[proto_ind[0], k].sum(0) / (num * np.log(num + self.alfa))
                        #                     self.proto_confidence[i, k] = temp
                    else: 
                        f = m_q.transpose(0, 1) @ c_q # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                        f = F.normalize(f, p=2, dim=-1)
                        protos[i, n != 0, :]  = momentum_update(old_value=protos[i, n != 0, :], new_value=f[n != 0, :], momentum=self.mean_gamma, debug=False)
                        f_v = (m_q.transpose(0, 1) @ (var_q / n)) + (m_q.transpose(0, 1) @ c_q ** 2) / n - \
                        (m_q.transpose(0, 1) @ c_q / n) ** 2
                        #! normalize for f_v
                        f_v = torch.exp(torch.sigmoid(torch.log(f_v)))
                    protos[i, n != 0, :]  = momentum_update(old_value=protos[i, n != 0, :], new_value=f[n != 0, :], momentum=self.mean_gamma, debug=False)
                    proto_var[i, n != 0, :] = momentum_update(old_value=proto_var[i, n != 0, :], new_value=f_v[n != 0, :], momentum=self.var_gamma, debug=False)
                    
                    del var_q, var_k, f, f_v
                                    
            # each class has a target id between [0, num_proto * c]
            #! ignore_label are still -1, and not being modified
            proto_target[gt_seg == i] = indexs.float(
            ) + (self.num_prototype * i)  # n samples -> n*m labels
            
            del c_q, c_k_tile, c_k, m_q, m_k_tile, m_k, indexs, q

        self.prototypes = nn.Parameter(
            l2_normalize(protos), requires_grad=False)  # make norm of proto equal to 1
        if self.use_uncertainty:
            # proto_var = torch.exp(torch.sigmoid(torch.log(proto_var)))
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
    
    def forward(self, x, x_var=None, gt_semantic_seg=None, boundary_pred=None, gt_boundary=None):
        ''' 
        boundary_pred: [(b h w) 2]
        '''
        b_size = x.shape[0]
        h_size = x.shape[2]
        k_size = x.shape[-1]
        gt_size = x.size()[2:]

        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        # self.proto_var.data.copy_(torch.exp(torch.sigmoid(torch.log(self.proto_var))))

        # sim_mat: # [n c m]
        x = rearrange(x, 'b c h w -> (b h w) c')
        if self.use_uncertainty and x_var is not None:
            x_var = rearrange(x_var, 'b c h w -> (b h w) c')
        sim_mat = self.compute_similarity(x, x_var)

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
                    sim_mat, out_seg, gt_seg, x, x_var)

            sim_mat = rearrange(sim_mat, 'n c m -> n (c m)')
            
            prototypes = self.prototypes.data.clone()

            if boundary_pred is not None and self.use_boundary:
                return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target, "boundary": boundary_pred, 'prototypes': prototypes}
            elif self.use_uncertainty:
                x = x.reshape(b_size, h_size, -1, k_size)
                x_var = x_var.reshape(b_size, h_size, -1, k_size)
                if self.weighted_ppd_loss:
                    proto_var = self.proto_var.data.clone()
                    loss_weight1 = torch.einsum('nk,cmk->cmn', x_var, proto_var) # [c m n]
                    loss_weight1 = (loss_weight1 - loss_weight1.min()) / (loss_weight1.max() - loss_weight1.min())
                    loss_weight1 = loss_weight1.mean()
                    # [n] + [c m 1] = [c m n]
                    loss_weight2 = x_var.sum(-1) + proto_var.sum(-1, keepdim=True)
                    loss_weight2 = (loss_weight2 - loss_weight2.min()) / (loss_weight2.max() - loss_weight2.min())
                    loss_weight2 = loss_weight2.mean()
                    return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target, 'w1': loss_weight1, 'w2': loss_weight2}
                
                elif self.use_temperature:
                    proto_confidence = self.proto_var.data.clone() # [c m k]
                    proto_confidence = proto_confidence.mean(-1) # [c m]
                    return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target, 'proto_confidence': proto_confidence}
                else:
                    return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target,
                            'x_mean': x, 'x_var': x_var}
            else:
                return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target}

        return out_seg