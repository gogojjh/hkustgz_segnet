from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.tools.logger import Logger as Log
from lib.loss.loss_helper import FSCELoss
from einops import rearrange


class ProtoDiverseLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(ProtoDiverseLoss, self).__init__()
        self.configer = configer
        
        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']
            
        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
    
    def forward(self, proto):
        ''' 
        Minimize the similarity between different prototypes.
        prototypes:[c m k]
        '''
        proto = rearrange(proto, 'c m k -> (c m) k') # [(c m) k]
        proto_inv = proto.permute(1, 0) # [k (c m)]
        proto_sim_mat = torch.einsum('nk,km->nm', proto, proto_inv) # [(c m) (c m)]
        proto_diverse_loss = (1 + proto_sim_mat).pow(2).mean()
        
        return proto_diverse_loss


class PixelContrastiveLoss(nn.Module, ABC):
    ''' 
    Confidence-guided hard example sampling.
    '''

    def __init__(self, configer):
        super(PixelContrastiveLoss, self).__init__()
        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    # def _construct_region_center(self, feats, sem_gt, pred, confidence):

    def forward(self, feats, confidence, sem_gt):
        sem_gt = sem_gt.unsqueeze(1).float().clone()
        sem_gt = F.interpolate(sem_gt,
                               (feats.shape[2], feats.shape[3]), mode='nearest')
        sem_gt = sem_gt.squeeze(1).long()

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])


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
        self.top_k_num = self.configer.get('protoseg', 'top_k_num')

    def get_uncer_label(self, pred_var, binary_sem_gt):
        ''' 
        1. correct pred: min(1-pred, pred)
        2. wrong pred: max(1-pred, pred)

        pred/sem_gt [b h w]

        binary prediction:
        use variance of prediction probability as guidance to generate uncertainty label.

        Compard with naive predicitve variance, we also considered the case where a pixel
        has small uncertainty/large pred variance, but with wrong prediction.
        '''
        mask = torch.zeros_like(binary_sem_gt).cuda()  # mask=0:wrong pred, mask=1:correct pred
        mask[binary_sem_gt == 1] = 1

        uncer_label = torch.zeros_like(binary_sem_gt).cuda()

        # correct pred -> 0
        # wrong pred -> max[pred_var, 1-pred_var]
        pred_var = torch.cat(((pred_var).unsqueeze(1), (1 - pred_var).unsqueeze(1)), dim=1)

        uncer_label.masked_scatter_(~mask.bool(), torch.max(pred_var, dim=1)[0])
        uncer_label.masked_scatter_(mask.bool(), torch.min(pred_var, dim=1)[0])

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

    def get_pred_var(self, pred):
        ''' 
        'ARM: A Confidence-Based Adversarial Reweighting Module for Coarse Semantic Segmentation'

        Use variance of prediction probability as uncertainty prediction. 
        But we only take the top 4 classes for variance calculation.
        '''
        score_top, _ = pred.topk(k=self.top_k_num, dim=1)  # [b 2/4 h w]
        score_top = F.softmax(score_top, dim=1)  # prob of binary classifiers

        mean = 1 / self.top_k_num
        var_map = torch.sigmoid(score_top - mean)  # [b top_k h w]
        pred_var = var_map.var(dim=1)

        # normalization
        pred_var = (pred_var - pred_var.min()) / (pred_var.max() - pred_var.min())
        return pred_var

    def forward(self, confidence, pred, sem_gt):
        ''' 
        confidence: [b h w]
        pred: [b num_cls h w]

        Use l1 norm between prediction variance and confidence as supervision of uncertainty.
        L1 norm is used for robustness to outleirs.
        '''
        h, w = confidence.size(1), confidence.size(2)
        sem_gt = F.interpolate(input=sem_gt.unsqueeze(1).float(), size=(
            h, w), mode='nearest')
        sem_gt = sem_gt.squeeze(1)

        binary_label = self.get_binary_sem_label(pred, sem_gt)  # [b h w]
        pred_var = self.get_pred_var(pred)  # [b h w]

        uncer_label = self.get_uncer_label(pred_var, binary_label)  # [b h w]

        # mask out the ignored label
        mask = sem_gt != self.ignore_label

        uncer_seg_loss = torch.abs((confidence[mask].float() - uncer_label[mask])).mean()

        return uncer_seg_loss


class ConfidenceLoss(nn.Module, ABC):
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

    def forward(self, confidence, contrast_target):
        ''' 
        confidence: [b h w]
        contrast_target: [(b h w)]
        Minimize the uncertainty of the easy samples to refine the uncertain areas.
        '''
        confidence = rearrange(confidence, 'b h w -> (b h w)')  # [n]
        mask = contrast_target != self.ignore_label
        confidence = torch.masked_select(confidence, mask)

        confidence_loss = torch.exp(confidence).mean()

        return confidence_loss


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
        self.confidence_seg_loss_weight = self.configer.get(
            'protoseg', 'confidence_seg_loss_weight')

    def forward(self, contrast_logits, contrast_target, confidence=None):
        ''' 
        confidence: [b h w]
        contrast_logits: [n c m]
        '''
        contrast_logits = self.proto_norm(contrast_logits)
        if confidence is not None:
            confidence = rearrange(confidence, 'b h w -> (b h w)')
            confidence = 1 + self.confidence_seg_loss_weight * torch.sigmoid(confidence)
            prob_ppc_loss = F.cross_entropy(
                contrast_logits, contrast_target.long(),
                ignore_index=self.ignore_label, reduction='none') * confidence
            prob_ppc_loss = torch.mean(prob_ppc_loss)
        else:
            prob_ppc_loss = F.cross_entropy(
                contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)

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
        self.uncer_seg_loss_weight = self.configer.get('protoseg', 'uncer_seg_loss_weight')

        self.seg_criterion = FSCELoss(configer=configer)

        self.prob_ppc_criterion = ProbPPCLoss(configer=configer)

        self.prob_ppd_criterion = ProbPPDLoss(configer=configer)

        self.use_uncertainty = self.configer.get('protoseg', 'use_uncertainty')
        self.uncer_seg_loss = PredUncertaintyLoss(configer=configer)
        self.use_prototype = self.configer.get('protoseg', 'use_prototype')

    def get_uncer_loss_weight(self):
        uncer_loss_weight = self.rampdown_scheduler.value

        return uncer_loss_weight

    def forward(self, preds, target, gt_boundary=None):
        b, h, w = target.size(0), target.size(1), target.size(2)

        if isinstance(preds, dict) and self.use_prototype:
            assert 'seg' in preds
            assert 'logits' in preds
            assert 'target' in preds

            seg = preds['seg']  # [b c h w]
            contrast_logits = preds['logits']
            contrast_target = preds['target']  # prototype selection [n]

            prob_ppd_loss = self.prob_ppd_criterion(
                contrast_logits, contrast_target)

            prob_ppc_loss = self.prob_ppc_criterion(contrast_logits, contrast_target)

            pred = F.interpolate(input=seg, size=(
                h, w), mode='bilinear', align_corners=True)
            
            confidence = preds['confidence']
            # uncer_seg_loss
            contrast_logits = contrast_logits.reshape(-1, self.num_classes, self.num_prototype)
            contrast_logits = torch.max(contrast_logits, dim=-1)[0]
            b_train, h_train, w_train = seg.size(0), seg.size(2), seg.size(3)
            contrast_logits = contrast_logits.reshape(
                b_train, h_train, w_train, self.num_classes)
            #! pred is detached when caluclating supervision for uncertainty
            uncer_seg_loss = self.uncer_seg_loss(
                confidence, contrast_logits.permute(0, -1, 1, 2).detach(), target)

            seg_loss = self.seg_criterion(pred, target)

            loss = seg_loss + self.prob_ppc_weight * prob_ppc_loss + self.prob_ppd_weight * prob_ppd_loss + \
                self.uncer_seg_loss_weight * uncer_seg_loss
            assert not torch.isnan(loss)

            return {'loss': loss, 'seg_loss': seg_loss, 'prob_ppc_loss': prob_ppc_loss, 'prob_ppd_loss': prob_ppd_loss, 'uncer_seg_loss': uncer_seg_loss}

        if isinstance(preds, dict):
            seg = preds['seg']  # [b c h w]
        else:
            seg = preds
            
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)

        seg_loss = self.seg_criterion(pred, target)

        return seg_loss
