"""
Probabilistic contrastive loss with each pixel being a Gaussian.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.tools.logger import Logger as Log
from lib.loss.loss_helper import FSCELoss
from lib.utils.tools.rampscheduler import RampdownScheduler
from einops import rearrange, repeat


class InstanceContrativeLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(InstanceContrativeLoss, self).__init__()
        self.configer = configer


class FocalLoss(nn.Module, ABC):
    ''' focal loss '''
    def __init__(self, configer):
        super(FocalLoss, self).__init__()
        self.configer = configer

        self.gamma = self.configer.get('loss', 'focal_gamma')
        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def binary_focal_loss(self, input, target, valid_mask):
        input = input[valid_mask]
        target = target[valid_mask]
        pt = torch.where(target == 1, input, 1 - input)
        ce_loss = self.crit(input, target)
        loss = torch.pow(1 - pt, self.gamma) * ce_loss
        loss = loss.mean()
        return loss
        
    def	forward(self, input, target):
        valid_mask = (target != self.ignore_label)
        K = target.shape[0]
        total_loss = 0
        for i in range(K):
            # total_loss += self.binary_focal_loss(input[:,i], target[:,i], valid_mask[:,i])
            total_loss += self.binary_focal_loss(input[i], target[i], valid_mask[i])
        return total_loss / K


class PatchClsLoss(nn.Module, ABC):
    ''' 
    'Partial Class Activation Attention for Semantic Segmentation'
    '''
    def __init__(self, configer):
        super(PatchClsLoss, self).__init__()
        self.configer = configer
        
        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        self.num_classes = self.configer.get('data', 'num_classes')
        self.bin_size_h = self.configer.get('protoseg', 'bin_size_h')
        self.bin_size_w = self.configer.get('protoseg', 'bin_size_w')
        
        self.seg_criterion = FocalLoss(configer=configer)
            
    def get_onehot_label(self, label):
        # label: [b h w] 
        # deal with the void class
        assert self.ignore_label == -1
        label = label + 1
        label = F.one_hot(label, num_classes=self.num_classes+1).to(torch.float32) # [b h w num_cls+1]
        label = label.permute(0, 3, 1, 2) #! [b num_cls+1 h w] cls 0 is ignore class
        
        return label
            
    def get_patch_label(self, label_onehot, th=0.01):
        ''' 
        label_onehot: [b num_cls h w]
        For each patch, there is a unique class label which dominates this patch pixels.
        '''
        # [b num_cls+1 bin_size bin_size]
        #! since ignore label is negative, it is possible that pooled resulta are below 0.
        cls_percentage = F.adaptive_avg_pool2d(label_onehot, (self.bin_size_h, self.bin_size_w))
        cls_label = torch.where(cls_percentage>0, torch.ones_like(cls_percentage), torch.zeros_like(cls_percentage)) # float cls_label to integer cls_label
        cls_label[(cls_percentage<th)&(cls_percentage>0)] = self.ignore_label # [0, 1, -1]?
        
        return cls_label
        
            
    def forward(self, patch_cls_score, target):
        ''' 
        patch_cls_score: [b, num_cls, bin_num_h, bin_num_w]
        '''        
        label_onehot = self.get_onehot_label(target) # [b h w num_cls+1]
        patch_cls_gt = self.get_patch_label(label_onehot) # [b num_cls+1 bin_size bin_size]
        # [num_cls+1 b*bin_size*bin_size]

        patch_cls_gt = patch_cls_gt[:, 1:, ...]
        patch_cls_gt = rearrange(patch_cls_gt, 'b n h w -> n (b h w)')
        patch_cls_score = rearrange(patch_cls_score, 'b n h w -> n (b h w)')  
        focal_loss = self.seg_criterion(patch_cls_score, patch_cls_gt[1:, ...])
        
        return focal_loss
        
    
class BoundaryLoss(nn.Module, ABC):
    ''' 
    Cross entropy loss between boundary prediction and boundary gt.
    '''

    def __init__(self, configer):
        super(BoundaryLoss, self).__init__()
        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def forward(self, boundary_pred, boundary_gt, sem_gt):
        mask = sem_gt == self.ignore_label # [b h w]
        boundary_gt[mask] = self.ignore_label
        
        boundary_loss = F.cross_entropy(boundary_pred, boundary_gt, ignore_index=self.ignore_label)

        return boundary_loss


class ConfidenceLoss(nn.Module, ABC):
    ''' 
    Video Object Segmentation with Adaptive Feature Bank and Uncertain-Region Refinement
    '''

    def __init__(self, configer):
        super(ConfidenceLoss, self).__init__()

        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')

    def forward(self, sim_mat):
        # sim_mat: [n (c m)]
        if self.configer.get(
                'protoseg', 'similarity_measure') == 'fast_mls' or self.configer.get(
                'protoseg', 'similarity_measure') == 'mls':
            sim_mat = torch.exp(sim_mat)
        score_top, _ = sim_mat.topk(k=2, dim=1)
        confidence = score_top[:, 0] / score_top[:, 1]
        confidence = torch.exp(1 - confidence).mean(-1)

        return confidence


class ProbPPCLoss(nn.Module, ABC):
    """ 
    Pixel-wise probabilistic contrastive loss.
    """

    def __init__(self, configer):
        super(ProbPPCLoss, self).__init__()

        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.proto_norm = nn.LayerNorm(self.num_classes * self.num_prototype)

    def forward(self, contrast_logits, contrast_target, x_var=None, proto_confidence=None):  
        ''' 
        x_var: [c m k]
        '''
        if proto_confidence is not None:
            # proto_confidence: [(num_cls num_proto)]
            # contrast_logits: [n c m]
            proto_confidence = rearrange(proto_confidence, 'c m -> (c m)')
            contrast_logits = contrast_logits / proto_confidence.unsqueeze(0)
        
        contrast_logits = self.proto_norm(contrast_logits)
        prob_ppc_loss = F.cross_entropy(
            contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)
        
        if x_var is not None:
            w = x_var.mean()
            # w2 = torch.log(x_var).mean()
            prob_ppc_loss = 2 / (w + 1e-3) * prob_ppc_loss + w

        return prob_ppc_loss
    

class KLLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(KLLoss, self).__init__()

        self.configer = configer
        
        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def forward(self, x_mean, x_var, sem_gt=None, proto=False):
        ''' 
        x / x_var: [b h w c]
        proto_mean / proto_var: [c m k]
        '''
        if not proto:
            h, w = x_mean.size(1), x_mean.size(2)
            sem_gt = F.interpolate(input=sem_gt.unsqueeze(1).float(), size=(
                    h, w), mode='nearest')
            
            mask = sem_gt.squeeze(1) == self.ignore_label # [b h w]
            x_mean[mask, ...] = 0
            x_var[mask, ...] = 1
        
        kl_loss = 0.5 * (x_mean ** 2 + x_var - torch.log(x_var) - 1).sum(-1)
        kl_loss = kl_loss.mean()
        
        return kl_loss


class AleatoricUncertaintyLoss(nn.Module, ABC):
    """ 
    Geometry and Uncertainty in Deep Learning for Computer Vision
    """

    def __init__(self, configer):
        super(AleatoricUncertaintyLoss, self).__init__()
        self.configer = configer
        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def forward(self, x_var, target, pred):  # x_var: [b 1 h w]
        x_var = torch.mean(x_var, dim=1, keepdim=True)
        x_var = F.interpolate(
            input=x_var, size=(target.shape[1],
                               target.shape[2]),
            mode='nearest')  # [b 1 h_ori w_ori]
        x_var = x_var.squeeze(1)  # [b h w]
        x_var = rearrange(x_var, 'b h w -> (b h w)')

        pred = torch.argmax(pred, 1)  # [b h w]
        pred = rearrange(pred, 'b h w -> (b h w)')
        target = rearrange(target, 'b h w -> (b h w)')

        # ignore the -1 label pixel
        ignore_mask = (target != self.ignore_label)
        target = target[ignore_mask]
        pred = pred[ignore_mask]
        x_var = x_var[ignore_mask]

        #! change l2-norm into l1-norm to avoid large outlier
        aleatoric_uncer_loss = torch.mean(
            (0.5 * torch.abs(target.float() - pred.float()) / x_var + 0.5 * torch.log(x_var)))

        return aleatoric_uncer_loss


class ProbPPDLoss(nn.Module, ABC):
    """ 
    Minimize intra-class compactness using distance between probabilistic distributions (MLS Distance).
    """

    def __init__(self, configer):
        super(ProbPPDLoss, self).__init__()

        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']
            
        self.sim_measure = self.configer.get('protoseg', 'similarity_measure')

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target !=
                                          self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1,
                              contrast_target[:, None].long()).squeeze(-1)
        
        if self.sim_measure == 'cosine' or 'match_prob': 
            prob_ppd_loss = (1 - logits).pow(2).mean()
        elif self.sim_measure == 'wasserstein' or 'mls' or 'fast_mls':
            prob_ppd_loss = - logits.mean()

        return prob_ppd_loss
    
    
class BoundaryContrastiveLoss(nn.Module, ABC):
    ''' 
    - pixels belonging to boundary prototype(last proto) have high uncertainty because they are likely to predicted as nearby class
    - This loss is conducted between the pixels around the boundary.
    '''
    def __init__(self, configer):
        super(BoundaryContrastiveLoss, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        
        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False)
        #! for summation inside the kernel
        self.conv1.weight = torch.nn.Parameter(torch.ones_like((self.conv1.weight)))
        self.pad = nn.ReplicationPad2d(1)
        
        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.proto_norm = nn.LayerNorm(self.num_classes * self.num_prototype)
    
    def get_nearby_pixel(self, contrast_logits, sem_gt, boundary_gt):
        l = contrast_logits.size(-1)
        boundary_gt = boundary_gt.unsqueeze(1)
        sem_gt = sem_gt.unsqueeze(1)
        bound_mask = boundary_gt == 1 # [b 1 h' w']
        
        bound_mask = self.pad(bound_mask.float()) # [b 1 h+2 w+2]
        # (l - 3 + 2 * 1/padding) / 1 + 1 = l 
         #! sum the boundary pixel inside the 7x7 sliding window
        bound_mask = self.conv1(bound_mask.float()) # [b 1 h w]
        #! if bound_mask == 0: no boundary pixels inside the nearby window -> mask out
        bound_mask = torch.where((bound_mask == 0), 0, 1)
        bound_mask = bound_mask.squeeze(1).unsqueeze(-1) # [b h w 1]
        
        contrast_logits = contrast_logits.masked_select(bound_mask.bool())
        contrast_logits = contrast_logits.reshape(-1, l) # [n, (c m)]
        
        sem_gt = sem_gt.squeeze(1).unsqueeze(-1)
        sem_gt = sem_gt.masked_select(bound_mask.bool())
        
        return contrast_logits, sem_gt
        
    def forward(self, contrast_logits, boundary_gt, sem_gt):
        ''' 
        sem_gt: [b h w]
        boundary_gt: [b h w]
        contrast_logits: [b h w (c m)]
        '''
        h, w = sem_gt.size(1), sem_gt.size(2) # 128, 256

        contrast_logits = contrast_logits.permute(0, 3, 1, 2) # [b (c m) h w]
        contrast_logits = F.interpolate(input=contrast_logits, size=(
                h, w), mode='bilinear', align_corners=True)
        contrast_logits = contrast_logits.permute(0, 2, 3, 1) # [b h w (c m)]
        
        contrast_logits, sem_gt = self.get_nearby_pixel(contrast_logits, sem_gt, boundary_gt)
        
        contrast_logits = self.proto_norm(contrast_logits)
        bound_contrast_loss = F.cross_entropy(contrast_logits, sem_gt.long(), ignore_index=self.ignore_label)
        
        return bound_contrast_loss
        

class PixelProbContrastLoss(nn.Module, ABC):
    """
    Pixel-wise probabilistic contrastive loss
    Pixel embedding: Multivariate Gaussian
    """

    def __init__(self, configer):
        super(PixelProbContrastLoss, self).__init__()

        self.configer = configer
        # self.temperature = self.configer.get('prob_contrast', 'temperature')

        ignore_index = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.prob_ppd_weight = self.configer.get('protoseg', 'prob_ppd_weight')
        self.prob_ppc_weight = self.configer.get('protoseg', 'prob_ppc_weight')
        
        self.prob_ppc_criterion = ProbPPCLoss(configer=configer)

        self.prob_ppd_criterion = ProbPPDLoss(configer=configer)
        self.seg_criterion = FSCELoss(configer=configer)
        
        self.use_uncertainty = self.configer.get('protoseg', 'use_uncertainty')
        if self.use_uncertainty:
            self.kl_loss = KLLoss(configer=configer)
            self.aleatoric_loss = AleatoricUncertaintyLoss(configer=configer)
            self.kl_loss_weight = self.configer.get('protoseg', 'kl_loss_weight')
        self.use_attention = self.configer.get('protoseg', 'use_attention')
        if self.use_attention:
            self.patch_cls_loss = PatchClsLoss(configer=configer)
            self.patch_cls_weight = self.configer.get('protoseg', 'patch_cls_weight')

        # initialize scheudler for uncer_loss_weight
        self.rampdown_scheduler = RampdownScheduler(
            begin_epoch=self.configer.get('rampdownscheduler', 'begin_epoch'),
            max_epoch=self.configer.get('rampdownscheduler', 'max_epoch'),
            current_epoch=self.configer.get('epoch'),
            max_value=self.configer.get('rampdownscheduler', 'max_value'),
            min_value=self.configer.get('rampdownscheduler', 'min_value'),
            ramp_mult=self.configer.get('rampdownscheduler', 'ramp_mult'),
            configer=configer)

        self.use_boundary = self.configer.get('protoseg', 'use_boundary')
        if self.use_boundary:
            # self.boundary_loss = BoundaryLoss(configer=configer)
            self.bound_contrast_loss = BoundaryContrastiveLoss(configer=configer)
            self.boundary_loss_weight = self.configer.get('protoseg', 'boundary_loss_weight')
        self.use_temperature = self.configer.get('protoseg', 'use_temperature')
        self.weighted_ppd_loss = self.configer.get('protoseg', 'weighted_ppd_loss')

    def get_uncer_loss_weight(self):
        uncer_loss_weight = self.rampdown_scheduler.value

        return uncer_loss_weight

    def forward(self, preds, target, gt_boundary=None):
        b, h, w = target.size(0), target.size(1), target.size(2)
        
        if isinstance(preds, dict):
            assert 'seg' in preds
            assert 'logits' in preds
            assert 'target' in preds

            seg = preds['seg']  # [b c h w]
            contrast_logits = preds['logits']
            contrast_target = preds['target']  # prototype selection [n]
            
            if self.use_uncertainty:
                proto_confidence = None
                x_var = None
                if self.use_temperature:
                    proto_confidence = preds['proto_confidence']
                if self.weighted_ppd_loss:
                    x_var = preds['x_var']

                prob_ppc_loss = self.prob_ppc_criterion(contrast_logits, contrast_target, proto_confidence=proto_confidence, x_var=x_var)
                
            else: 
                prob_ppc_loss = self.prob_ppc_criterion(contrast_logits, contrast_target)

            prob_ppd_loss = self.prob_ppd_criterion(
                contrast_logits, contrast_target)
            
            if self.use_boundary and gt_boundary is not None:
                h_d = seg.shape[-2]
                w_d = seg.shape[-1]
                contrast_logits = contrast_logits.reshape(b, h_d, w_d, -1)
                # contrast_logits = F.interpolate(input=contrast_logits, size=(
                # h, w), mode='bilinear', align_corners=True)
                
                # if torch.count_nonzero(gt_boundary) == 0:
                #     bound_contrast_loss = prob_ppd_loss * 0
                # else:
                #     bound_contrast_loss = self.bound_contrast_loss(contrast_logits, gt_boundary.squeeze(1), target)
                    
                # boundary prototype contrastive learning
                
                # assert 'boundary' in preds
                
                # h_bound, w_bound = gt_boundary.size(1), gt_boundary.size(2) 

                # boundary_pred = preds['boundary']  # [b 2 h w]
                # boundary_pred = F.interpolate(input=boundary_pred,
                #                               size=(h_bound, w_bound),
                #                               mode='bilinear',
                #                               align_corners=True)

                # boundary_loss = self.boundary_loss(boundary_pred, gt_boundary, target)

            pred = F.interpolate(input=seg, size=(
                h, w), mode='bilinear', align_corners=True)

            seg_loss = self.seg_criterion(pred, target)

            # prob_ppc_weight = self.get_uncer_loss_weight()
            
            # x_mean = preds['x_mean']
            # x_var = preds['x_var']
            # kl_loss = self.kl_loss(x_mean, x_var, sem_gt=target)
                
            if self.use_attention:
                patch_cls_score = preds['patch_cls_score']
                patch_cls_loss = self.patch_cls_loss(patch_cls_score, target)
                
                loss = seg_loss + self.prob_ppc_weight * prob_ppc_loss + self.prob_ppd_weight * prob_ppd_loss + self.patch_cls_weight * patch_cls_loss
                
                assert not torch.isnan(loss)
                
                return {'loss': loss, 'seg_loss': seg_loss, 'prob_ppc_loss': prob_ppc_loss, 'prob_ppd_loss': prob_ppd_loss, 'patch_cls_loss': patch_cls_loss}

            else:
                loss = seg_loss + self.prob_ppc_weight * prob_ppc_loss + self.prob_ppd_weight * prob_ppd_loss
                
                assert not torch.isnan(loss)

                return {'loss': loss, 'seg_loss': seg_loss, 'prob_ppc_loss': prob_ppc_loss, 'prob_ppd_loss': prob_ppd_loss}

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)

        seg_loss = self.seg_criterion(pred, target)

        return seg_loss
