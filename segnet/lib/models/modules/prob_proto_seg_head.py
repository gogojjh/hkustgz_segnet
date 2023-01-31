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
            self.proto_var = nn.Parameter(torch.ones(
                self.num_classes, self.num_prototype, self.proj_dim), requires_grad=False)

            self.reparam_k = self.configer.get('protoseg', 'reparam_k')
            #! weight between mean and variance when calculating similarity
            if self.sim_measure == 'wasserstein':
                self.lamda = 1 / (self.proj_dim // 4) # scaling factor for b_distance

        self.use_gt_proto_learning = self.configer.get('protoseg', 'use_gt_proto_learning')
        
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

    def get_uncertainty(self, c, c_var):
        # uncertainty = self.reparameterize(c, c_var, k=50)
        uncertainty = self.sample_gaussian_tensors(c, c_var, k=10)
        uncertainty = torch.sigmoid(uncertainty)
        uncertainty = uncertainty.var(dim=1, keepdim=True).detach() # [b 1 h w]
        if self.configer.get('phase') == 'train':
            # (l-7+2*3)/1+1=l
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
        # normalize
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())

        return uncertainty
        
    def compute_similarity(self, x, x_var=None):
        ''' 
        x/x_var: [(b h w) k]
        proto_mean: [c m k]
        Use reparameterization trick to compute probabilistic distance for gradient flow.
        '''
        proto_mean = self.prototypes.data.clone()
        if self.use_uncertainty and x_var is not None:
            proto_var = self.proto_var.data.clone()
            proto_mean = self.reparameterize(proto_mean.unsqueeze(0), proto_var.unsqueeze(0), k=self.reparam_k) # [reparam_k c m k]
            proto_mean = self.feat_norm(proto_mean)
            proto_mean = l2_normalize(proto_mean)
            
            x = self.reparameterize(x.unsqueeze(0), x_var.unsqueeze(0), k=self.reparam_k) # [reparam_k (b h w ) k]
            x = self.feat_norm(x)
            x = l2_normalize(x)
            
            #! use var not log(var) for similarity
            x_var = torch.exp(x_var)
            proto_var = torch.exp(proto_var)
            
            if self.sim_measure == 'wasserstein':
                # monte-carlo eucl dist
                x = x.unsqueeze(1) # [reparam_k 1 (b h w ) k]
                proto_mean = proto_mean.unsqueeze(0) # [1 reparam_k c m k]
                
                sim_mat = torch.einsum('rlnd,abkmd->rbnmk', x, proto_mean) 
            # [(reparam_k reparam_k) (b h w) m c]
                sim_mat = rearrange(sim_mat, 'r l n m c -> (r l) n m c') 
                sim_mat = torch.mean(sim_mat, dim=0).permute(2, 1, 0)  # [n m c] -> [c m n]
                
                # -[c m n] + [n] + [c m 1] - 2 * [c m n]
                #! lamda depends on dimension of embeddings
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
                
                sim_mat = torch.einsum('rlnd,abkmd->rbnmk', x, proto_mean) 
            # [(reparam_k reparam_k) (b h w) m c]
                sim_mat = rearrange(sim_mat, 'r l n m c -> (r l) n m c') 
                sim_mat = torch.mean(sim_mat, dim=0).permute(2, 1, 0)  # [n m c] -> [c m n]
                
                sim_mat = - self.a * (2 - 2 * sim_mat) + self.b
                
                sim_mat = self.sigmoid(sim_mat)
                
                sim_mat = - sim_mat
                
            del x_var, x, proto_mean, proto_var
                    
        else:
            if self.sim_measure == 'cosine':
                # batch product (toward d dimension) -> cosine simialrity between fea and prototypes
                # n: h*w, k: num_class, m: num_prototype
                sim_mat = torch.einsum('nd,kmd->nmk', x, proto_mean)
                
                sim_mat = sim_mat.permute(0, 2, 1) # [c n m]
            
            else: 
                Log.error('Similarity measure is invalid.')
            
        return sim_mat
    
    def reparameterize(self, mu, logvar, k=1):
        ''' 
        mu/var: [1 n k]
        '''
        sample_z = []
        for _ in range(k):
            # '_': inplace operation mul():dot product
            std = logvar.mul(0.5).exp_()  
            eps = std.data.new(std.size()).normal_() # mu + epsilon * var
            sample_z.append(eps.mul(std).add_(mu))
        sample_z = torch.cat(sample_z, dim=0) # [sample_time n k]
        
        return sample_z
    
    def sample_gaussian_tensors(self, mu, logsigma, num_samples):
        eps = torch.randn(num_samples, mu.size(0), mu.size(1), dtype=mu.dtype, device=mu.device)

        samples = eps.mul(torch.exp(logsigma.unsqueeze(0))).add_(
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

            # f = m_q.transpose(0, 1) @ c_q  # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
            if self.use_uncertainty and _c_var is not None:
                var_k = _c_var[gt_seg == i, ...]

                var_q = var_k * c_k_tile # [n embed_dim]

            # the prototypes that are being selected calcualted by sum
            n = torch.sum(m_q, dim=0)  # [num_proto]

            if self.update_prototype is True and torch.sum(n) > 0:
                f = m_q.transpose(0, 1) @ c_q  # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim]
                f = F.normalize(f, p=2, dim=-1)
                protos[i, n != 0, :]  = momentum_update(old_value=protos[i, n != 0, :], new_value=f[n != 0, :], momentum=self.mean_gamma, debug=False)
                
                if self.use_uncertainty:
                    #? self.lamda is used to scale between c and c_var
                    f_v = m_q.transpose(0, 1) @ (l2_normalize(torch.exp(var_q)) + c_q ** 2) - f ** 2
                    f_v = torch.sigmoid(torch.log(f_v))
                    proto_var[i, n != 0, :] = momentum_update(old_value=proto_var[i, n != 0, :], new_value=f_v[n != 0, :], momentum=self.var_gamma, debug=False)
                
                
                # else:
                #     for k in range(self.num_prototype):
                #          if torch.sum(m_q[..., k] > 0):
                #             """
                #             1. correctly predicted: c_q / var_q
                #             2. choose the ones belonging to this prototype in c_q / var_q 
                #             """
                #             proto_ind = torch.nonzero(
                #                 (m_q[..., k] != 0), as_tuple=True)
                #             # [num_proto, n] @ [n embed_dim] = [num_proto embed_dim], correct pixel var for each proto
                #             f_v = 1 / ((1 / var_q[proto_ind]).mean(dim=0))  
                #             f = f_v * ((c_q[proto_ind] / var_q[proto_ind]).mean(dim=0))
                #             f = F.normalize(f, p=2, dim=-1)
                #             protos[i, k, :] = momentum_update(old_value=protos[i, k, :], new_value=f, momentum=self.mean_gamma, debug=False)
                #             proto_var[i, k, :] = momentum_update(old_value=proto_var[i, k, :], new_value=f_v, momentum=self.var_gamma, debug=False)
                    
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
                proto_var = self.proto_var.data.clone()
                loss_weight1 = (0.0005 * torch.einsum('nk,cmk->cmn', torch.exp(x_var), torch.exp(proto_var))).mean() # [c m n]
                # [n] + [c m 1] = [c m n]
                loss_weight2 = (0.001 * (x_var.sum(-1) + proto_var.sum(-1, keepdim=True))).mean() 
                return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target, 'prototypes': prototypes, 'loss_weight1': loss_weight1, 'loss_weight2': loss_weight2}
            else:
                return {'seg': out_seg, 'logits': sim_mat, 'target': contrast_target, 'prototypes': prototypes}

        return out_seg
