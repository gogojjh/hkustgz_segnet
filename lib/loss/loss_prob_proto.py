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

        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.proto_norm = nn.LayerNorm(self.num_classes * self.num_prototype)

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = self.proto_norm(contrast_logits)

        prob_ppc_loss = F.cross_entropy(
            contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)

        return prob_ppc_loss


class KLLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(KLLoss, self).__init__()

        self.configer = configer

    def forward(self, proto_mean, proto_var):
        kl_loss = 0.5 * torch.sum(
            (torch.square(proto_mean) + proto_var - torch.log(proto_var) - 1),
            dim=-1).mean()

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

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target !=
                                          self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1,
                              contrast_target[:, None].long())

        if logits.shape[0] == 0:
            prob_ppd_loss = 0
            return prob_ppd_loss

        if self.configer.get(
                'protoseg', 'similarity_measure') == "mls" or self.configer.get(
                'protoseg', 'similarity_measure') == "fast_mls":
            prob_ppd_loss = torch.mean(-logits)  # negative MLS
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

        if self.configer.get('loss', 'aleatoric_uncer_loss'):
            self.aleatoric_uncer_loss = AleatoricUncertaintyLoss(configer=configer)
            self.aleatoric_uncer_weight = self.configer.get('protoseg', 'aleatoric_uncer_weight')

        if self.configer.get('loss', 'kl_loss'):
            self.kl_loss = KLLoss(configer=configer)

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

            prob_ppc_loss = self.prob_ppc_criterion(
                contrast_logits, contrast_target)

            prob_ppd_loss = self.prob_ppd_criterion(
                contrast_logits, contrast_target)

            pred = F.interpolate(input=seg, size=(
                h, w), mode='bilinear', align_corners=True)

            seg_loss = self.seg_criterion(pred, target)

            if self.configer.get('loss', 'kl_loss'):
                proto_var = preds['proto_var']
                proto_mean = preds['proto_mean']
                kl_loss = self.kl_loss(proto_mean, proto_var)

            # prob_ppc_weight = self.get_uncer_loss_weight()

            if prob_ppd_loss == 0:
                prob_ppd_loss = seg_loss * 0

            # if self.configer.get('loss', 'aleatoric_uncer_loss'):
            #     x_var = preds['uncertainty']
            #     aleatoric_uncer_loss = self.aleatoric_uncer_loss(x_var, target, pred)

            #     loss = seg_loss + self.prob_ppc_weight * prob_ppc_loss + self.prob_ppd_weight * \
            #         prob_ppd_loss + self.aleatoric_uncer_weight * aleatoric_uncer_loss

            #     return {'loss': loss,
            #             'seg_loss': seg_loss, 'prob_ppc_loss': prob_ppc_loss, 'prob_ppd_loss': prob_ppd_loss, 'aleatoric_uncer_loss': aleatoric_uncer_loss}

            loss = seg_loss + self.prob_ppc_weight * prob_ppc_loss + self.prob_ppd_weight * prob_ppd_loss

            return {'loss': loss,
                    'seg_loss': seg_loss, 'prob_ppc_loss': prob_ppc_loss, 'prob_ppd_loss': prob_ppd_loss}

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)

        seg_loss = self.seg_criterion(pred, target)

        return seg_loss
