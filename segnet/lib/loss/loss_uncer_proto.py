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
    Construct the multi-class classification problem into binary classification problem using the 
    top 2 classification probability/distance.
    '''

    def __init__(self, configer):
        super(PredUncertaintyLoss, self).__init__()
        self.configer = configer
        self.seg_criterion = torch.nn.BCEWithLogitsLoss()

        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def get_uncertainty_label(self, binary_pred, binary_sem_gt):
        ''' 
        1. correct pred: min(1-pred, pred)
        2. wrong pred: max(1-pred, pred)

        pred/sem_gt [b h w]
        '''
        mask = torch.zeros_like(binary_sem_gt)  # mask=0:wrong pred, mask=1:correct pred
        mask[binary_sem_gt == 1] = 1

        correct_uncer_label = torch.cat(
            (binary_pred.unsqueeze(1),
             (1 - binary_pred).unsqueeze(1)),
            dim=1)  # [b 2 h w]
        correct_uncer_label = binary_sem_gt * torch.min(correct_uncer_label, dim=1)[0]  # [b h w]

        wrong_uncer_label = torch.cat(
            (binary_pred.unsqueeze(1),
             (1 - binary_pred).unsqueeze(1)),
            dim=1)  # [b 2 h w]
        wrong_uncer_label = (1 - binary_sem_gt) * torch.max(wrong_uncer_label, dim=1)[0]  # [b h w]

        uncer_label = torch.zeros_like(mask)
        uncer_label.masked_scatter_(mask.bool(), correct_uncer_label)
        uncer_label.masked_scatter_(~(mask.bool()), wrong_uncer_label)

        return uncer_label

    def get_binary_sem_label(self, pred, sem_gt):
        ''' 
        Construct the lable for binary classificatio using the top2 segmentation probability.
        Correct prediction: 1
        Wrong prediction: 0
        '''
        binary_label = torch.zeros_like(sem_gt).cuda()
        pred = torch.argmax(pred, dim=1)
        binary_label[pred == sem_gt] = 1  # [b h w]
        return binary_label

    def get_binary_prediction(self, pred):
        ''' 
        Use the top2 segmentation probability to contruct a binary classifier.
        '''
        score_top, _ = pred.topk(k=2, dim=1)  # [b 2 h w]
        score_top = F.softmax(score_top, dim=1)  # prob of binary classifiers
        return score_top

    def forward(self, confidence, pred, sem_gt):
        ''' 
        confidence: [b h w]
        pred: [b num_cls h w]
        '''
        h, w = confidence.size(1), confidence.size(2)
        sem_gt = F.interpolate(input=sem_gt.unsqueeze(1).float(), size=(
            h, w), mode='nearest')
        sem_gt = sem_gt.squeeze(1)

        binary_label = self.get_binary_sem_label(pred, sem_gt)  # [b h w]
        binary_pred = self.get_binary_prediction(pred)  # [b 2 h w]

        binary_pred = torch.max(binary_pred, dim=1)[0]  # [b h w]
        uncer_label = self.get_uncertainty_label(binary_pred, binary_label)  # [b h w]

        # mask out the ignored label
        mask = sem_gt != self.ignore_label

        uncer_seg_loss = self.seg_criterion(confidence[mask].float(), uncer_label[mask])

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


class PixelUncerContrastLoss(nn.Module, ABC):
    """
    Use both images and predictions to obtain uncertainty/confidence.
    Uncertainty is utilized in uncertainty-aware learning framework.
    """

    def __init__(self, configer):
        super(PixelUncerContrastLoss, self).__init__()
        self.configer = configer

        ignore_index = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')

        self.prob_ppd_weight = self.configer.get('protoseg', 'prob_ppd_weight')
        self.prob_ppc_weight = self.configer.get('protoseg', 'prob_ppc_weight')
        self.coarse_seg_weight = self.configer.get('protoseg', 'coarse_seg_weight')
        self.uncer_seg_loss_weight = self.configer.get('protoseg', 'uncer_seg_loss_weight')
        self.use_weighted_seg_loss = self.configer.get('protoseg', 'use_weighted_seg_loss')
        if self.use_weighted_seg_loss:
            self.confidence_seg_loss_weight = self.configer.get(
                'protoseg', 'confidence_seg_loss_weight')

        self.seg_criterion = FSCELoss(configer=configer)

        self.prob_ppc_criterion = ProbPPCLoss(configer=configer)

        self.prob_ppd_criterion = ProbPPDLoss(configer=configer)

        self.use_uncertainty = self.configer.get('protoseg', 'use_uncertainty')

        self.use_temperature = self.configer.get('protoseg', 'use_temperature')
        self.use_context = self.configer.get('protoseg', 'use_context')
        self.weighted_ppd_loss = self.configer.get('protoseg', 'weighted_ppd_loss')
        self.uncer_seg_loss = PredUncertaintyLoss(configer=configer)

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

            if self.use_weighted_seg_loss:
                confidence = preds['confidence']
                contrast_logits = contrast_logits.reshape(-1, self.num_classes, self.num_prototype)
                contrast_logits = torch.max(contrast_logits, dim=-1)[0]
                b_train, h_train, w_train = seg.size(0), seg.size(2), seg.size(3)
                contrast_logits = contrast_logits.reshape(
                    b_train, h_train, w_train, self.num_classes)
                #! pred is detached when caluclating supervision for uncertainty
                uncer_seg_loss = self.uncer_seg_loss(
                    confidence, contrast_logits.permute(0, -1, 1, 2).detach(), target)

                confidence = F.interpolate(input=confidence.unsqueeze(1), size=(
                    h, w), mode='bilinear', align_corners=True)  # [b 1 h w]
                #! confidence is detached when calculating seg loss
                seg_loss = self.seg_criterion(
                    pred, target, confidence_wieght=confidence.squeeze(1).detach())
            else:
                seg_loss = self.seg_criterion(pred, target)

            if self.use_context:
                coarse_pred = preds['coarse_seg']
                coarse_seg_loss = self.seg_criterion(coarse_pred, target)

            loss = seg_loss + self.prob_ppc_weight * prob_ppc_loss + self.prob_ppd_weight * \
                prob_ppd_loss + self.uncer_seg_loss_weight * uncer_seg_loss
            assert not torch.isnan(loss)

            return {'loss': loss, 'seg_loss': seg_loss, 'prob_ppc_loss': prob_ppc_loss, 'prob_ppd_loss': prob_ppd_loss, 'uncer_seg_loss': uncer_seg_loss}

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)

        seg_loss = self.seg_criterion(pred, target)

        return seg_loss
