"""
Code adapted from 'Improving Semantic Segmentation via Decoupled Body and Edge Supervision'.

Use predictio from boundary prototype classifier for boundary/body loss calculation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.tools.logger import Logger as Log


class EdgeBodyLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(EdgeBodyLoss, self).__init__()
        self.configer = configer

        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def body_loss(self, body_pred, body_gt, confidence):
        ''' 
        Use confidence map to extract the uncertain body pixels for loss calculation.

        body_pred: # [b 360 h w]
        body_pred: 
        '''

    def forward(self, preds, target, gt_boundary, sem_gt):
        ''' 
        gt_boundary: 1->boundary 0->non-boundary
        Use contrast target instead of sem_gt for more detailed supervision.
        '''
        seg_edge = preds['seg_edge']
        seg_body = preds['seg_body']
        contrast_logits = preds['logits']  # [(b h w) (c m)]
        contrast_target = preds['target']  # [(b h w)]
        confidence = preds['confidence']
        b, _, h, w = seg_edge.size()
        # contrast_target = contrast_target.reshape(-1, h, w) # [b h w]

        # extract the pixels predicted as boundary (last prototype in each class)
        edge_contrast_logits = torch.zeros_like(contrast_target).cuda()  # [(b h w)]
        body_contrast_logits = torch.zeros_like(contrast_target).cuda()  # [(b h w)]
        # [(b h w)]
        pred_logits = torch.scatter(src=contrast_logits, dim=1, index=contrast_target.int())
        for i in range(self.num_classes):
            edge_cls_logits = torch.amax(contrast_logits, dim=1)  # [(b h w) m]
            edge_mask = contrast_target == float(
                self.num_prototype - 1) + (self.num_prototype * i)  # [(b h w)]

            edge_contrast_logits.masked_scatter_(edge_mask.bool(), pred_logits)
            body_contrast_logits.masked_scatter_(~edge_mask.bool(), pred_logits)

        #! set the boundary gt pixel to ignore label to generate body gt
        contrast_target.masked_fill_(gt_boundary, self.ignore_label)

        body_loss = self.body_loss(seg_body, contrast_target, confidence)
