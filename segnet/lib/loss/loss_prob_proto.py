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


class PredUncertaintyLoss(nn.Module, ABC):
    ''' 
    Minimzie difference between data variance and predictive uncertainty.
    Only backpropated to uncertainty head.
    '''
    def __init__(self, configer):
        super(PredUncertaintyLoss, self).__init__()
        self.configer = configer
        self.seg_criterion = FSCELoss(configer=configer)
    
    def get_uncertainty_label(self, pred, target):
        uncer_label = torch.mul(target, (1 - pred)) + torch.mul((1- target), pred)
        return uncer_label 
            
    def forward(self, x_var, pred, target):
        uncer_label = self.get_uncertainty_label(pred, target)
        uncer_seg_loss = self.seg_criterion(torch.sigmoid(x_var), uncer_label)
        
        return uncer_seg_loss
        

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
        score_top, _ = sim_mat.topk(k=2, dim=1)
        confidence = score_top[:, 0] / (score_top[:, 1] + 1e-8)
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

    def forward(self, contrast_logits, contrast_target, w1=None, w2=None, proto_confidence=None):
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

        if w1 is not None and w2 is not None:
            # w2 = torch.log(x_var).mean()
            prob_ppc_loss = 1 / (4 * w1 + 1e-3) * prob_ppc_loss + w2 * 0.5

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

            mask = sem_gt.squeeze(1) == self.ignore_label  # [b h w]
            x_mean[mask, ...] = 0
            x_var[mask, ...] = 1

        kl_loss = 0.5 * (x_mean ** 2 + x_var - torch.log(x_var) - 1).sum(-1)
        kl_loss = kl_loss.mean()

        return kl_loss


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


class PixelProbContrastLoss(nn.Module, ABC):
    """
    Pixel-wise probabilistic contrastive loss
    Pixel embedding: Multivariate Gaussian
    """

    def __init__(self, configer):
        super(PixelProbContrastLoss, self).__init__()
        self.configer = configer

        ignore_index = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.prob_ppd_weight = self.configer.get('protoseg', 'prob_ppd_weight')
        self.prob_ppc_weight = self.configer.get('protoseg', 'prob_ppc_weight')
        self.coarse_seg_weight = self.configer.get('protoseg', 'coarse_seg_weight')

        self.prob_ppc_criterion = ProbPPCLoss(configer=configer)

        self.prob_ppd_criterion = ProbPPDLoss(configer=configer)
        self.seg_criterion = FSCELoss(configer=configer)

        self.use_uncertainty = self.configer.get('protoseg', 'use_uncertainty')

        # initialize scheudler for uncer_loss_weight
        # self.rampdown_scheduler = RampdownScheduler(
        #     begin_epoch=self.configer.get('rampdownscheduler', 'begin_epoch'),
        #     max_epoch=self.configer.get('rampdownscheduler', 'max_epoch'),
        #     current_epoch=self.configer.get('epoch'),
        #     max_value=self.configer.get('rampdownscheduler', 'max_value'),
        #     min_value=self.configer.get('rampdownscheduler', 'min_value'),
        #     ramp_mult=self.configer.get('rampdownscheduler', 'ramp_mult'),
        #     configer=configer)
        self.use_temperature = self.configer.get('protoseg', 'use_temperature')
        self.weighted_ppd_loss = self.configer.get('protoseg', 'weighted_ppd_loss')
        self.kl_loss = KLLoss(configer=configer)
        self.kl_loss_weight = self.configer.get('protoseg', 'kl_loss_weight')

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
            contrast_target = preds['target']  

            if self.use_uncertainty:
                w1 = None
                w2 = None
                proto_confidence = None
                if self.use_temperature:
                    proto_confidence = preds['proto_confidence']
                if self.weighted_ppd_loss:
                    w1 = preds['w1']
                    w2 = preds['w2']

                prob_ppc_loss = self.prob_ppc_criterion(
                    contrast_logits, contrast_target, w1=w1, w2=w2,
                    proto_confidence=proto_confidence)

            else:
                prob_ppc_loss = self.prob_ppc_criterion(contrast_logits, contrast_target)

            prob_ppd_loss = self.prob_ppd_criterion(
                contrast_logits, contrast_target)

            pred = F.interpolate(input=seg, size=(
                h, w), mode='bilinear', align_corners=True)

            seg_loss = self.seg_criterion(pred, target)

            coarse_pred = preds['coarse_seg']
            coarse_seg_loss = self.seg_criterion(coarse_pred, target)

            x_mean = preds['x_mean']
            x_var = preds['x_var']
            kl_loss = self.kl_loss(x_mean, x_var, target)

            loss = seg_loss + self.prob_ppc_weight * prob_ppc_loss + self.prob_ppd_weight * \
                prob_ppd_loss + self.kl_loss_weight * kl_loss + self.coarse_seg_weight * coarse_seg_loss
            assert not torch.isnan(loss)

            return {'loss': loss, 'seg_loss': seg_loss, 'prob_ppc_loss': prob_ppc_loss, 'prob_ppd_loss': prob_ppd_loss, 'kl_loss': kl_loss}

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)

        seg_loss = self.seg_criterion(pred, target)

        return seg_loss
