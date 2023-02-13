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

    def forward(self, input, target):
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
        # [b h w num_cls+1]
        label = F.one_hot(label, num_classes=self.num_classes+1).to(torch.float32)
        label = label.permute(0, 3, 1, 2)  # ! [b num_cls+1 h w] cls 0 is ignore class

        return label

    def get_patch_label(self, label_onehot, th=0.01):
        ''' 
        label_onehot: [b num_cls h w]
        For each patch, there is a unique class label which dominates this patch pixels.
        '''
        # [b num_cls+1 bin_size bin_size]
        #! since ignore label is negative, it is possible that pooled resulta are below 0.
        cls_percentage = F.adaptive_avg_pool2d(label_onehot, (self.bin_size_h, self.bin_size_w))
        cls_label = torch.where(
            cls_percentage > 0, torch.ones_like(cls_percentage),
            torch.zeros_like(cls_percentage))  # float cls_label to integer cls_label
        cls_label[(cls_percentage < th) & (cls_percentage > 0)] = self.ignore_label  # [0, 1, -1]?

        return cls_label

    def forward(self, patch_cls_score, target):
        ''' 
        patch_cls_score: [b, num_cls, bin_num_h, bin_num_w]
        '''
        label_onehot = self.get_onehot_label(target)  # [b h w num_cls+1]
        patch_cls_gt = self.get_patch_label(label_onehot)  # [b num_cls+1 bin_size bin_size]
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
        mask = sem_gt == self.ignore_label  # [b h w]
        boundary_gt[mask] = self.ignore_label

        boundary_loss = F.cross_entropy(boundary_pred, boundary_gt, ignore_index=self.ignore_label)

        return boundary_loss