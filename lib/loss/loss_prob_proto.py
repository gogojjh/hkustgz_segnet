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


# todo: class-wise contrastive loss
# todo "Probabilistic Representations for Video Contrastive Learning"
# todo: video loss
class ProbPPCLoss(nn.Module, ABC):
    """ 
    Pixel-wise probabilistic contrastive loss (instanse-wise contrastive loss)
    Probability masure: mutual likelihood loss
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

    def forward(self, contrast_logits, contrast_target):
        prob_ppc_loss = F.cross_entropy(
            contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)

        return prob_ppc_loss


class stoch_contrastive_loss(nn.Module, ABC):
    """ 
    Probabilistic Representations for Video Contrastive Learning
    """

    def __init__(self, configer):
        super(stoch_contrastive_loss, self).__init__()

        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def forward(self, cosine_dist, contrast_target, x_var, proto_var):
        stoch_conta_loss = F.cross_entropy(
            cosine_dist, contrast_target.long(), ignore_index=self.ignore_label, reduction='none')
        stoch_conta_loss = stoch_conta_loss[contrast_target !=
                                            self.ignore_label]
        stoch_conta_loss = (0.25 * stoch_conta_loss / (x_var * proto_var + 1e-8) + 0.5 *
                            (torch.log(x_var + 1e-8) + torch.log(proto_var + 1e-8))).mean()

        return stoch_conta_loss


class ProbPPDLoss(nn.Module, ABC):
    """ 
    Minimize intra-class compactness using distance between probabilistic distributions (MLS Distance).
    """

    def __init__(self, configer):
        super(ProbPPDLoss, self).__init__()

        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_incccdex' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target !=
                                          self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1,
                              contrast_target[:, None].long())

        if self.configer.get('protoseg', 'similarity_measure') == "mls":
            # exp(-log_likelihood)
            if logits.shape[0] > 0:
                prob_ppd_loss = torch.mean(torch.exp(-logits))  # torch.exp(negative MLS)
            else:
                print('0 in logits')
                prob_ppd_loss = 0
        elif self.configer.get('protoseg', 'similarity_measure') == "wasserstein":  # mls larger -> similar
            prob_ppd_loss = torch.mean(-logits)

        return prob_ppd_loss


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

        # initialize scheudler for uncer_loss_weight
        self.rampdown_scheduler = RampdownScheduler(
            begin_epoch=self.configer.get('rampdownscheduler', 'begin_epoch'),
            max_epoch=self.configer.get('rampdownscheduler', 'max_epoch'),
            current_epoch=self.configer.get('epoch'),
            max_value=self.configer.get('rampdownscheduler', 'max_value'),
            min_value=self.configer.get('rampdownscheduler', 'min_value'),
            ramp_mult=self.configer.get('rampdownscheduler', 'ramp_mult'),
            configer=configer)

        self.stoch_contrastive_loss = stoch_contrastive_loss(configer=configer)

    def get_uncer_loss_weight(self):
        uncer_loss_weight = self.rampdown_scheduler.value

        return uncer_loss_weight

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert 'seg' in preds
            assert 'logits' in preds
            assert 'target' in preds

            seg = preds['seg']  # [b c h w]
            contrast_logits = preds['logits']
            contrast_target = preds['target']  # prototype selection [n]
            contrast_logits_ori = preds['sim_mat_ori']

            temp_sim_mat = None

            if self.configer.get('loss', 'stochastic_contrastive_loss'):
                x_var = preds['x_var']
                proto_var = preds['proto_var']
                cosine_dist = preds['cosine_dist']
                prob_ppc_loss = self.stoch_contrastive_loss(
                    cosine_dist, contrast_target, x_var, proto_var)
            else:
                prob_ppc_loss = self.prob_ppc_criterion(
                    contrast_logits, contrast_target)

            prob_ppd_loss = self.prob_ppd_criterion(
                contrast_logits_ori, contrast_target)

            pred = F.interpolate(input=seg, size=(
                h, w), mode='bilinear', align_corners=True)

            seg_loss = self.seg_criterion(pred, target)

            # prob_ppc_weight = self.get_uncer_loss_weight()

            loss = seg_loss + self.prob_ppc_weight * prob_ppc_loss + self.prob_ppd_weight * prob_ppd_loss
            if prob_ppd_loss == 0:
                prob_ppd_loss = prob_ppc_loss * 0
            return {'loss': loss,
                    'seg_loss': seg_loss, 'prob_ppc_loss': prob_ppc_loss, 'prob_ppd_loss': prob_ppd_loss}

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)

        seg_loss = self.seg_criterion(pred, target)

        return seg_loss
