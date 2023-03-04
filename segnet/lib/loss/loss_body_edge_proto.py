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
        
    def forward(self, preds, target, gt_boundary):
        seg_edge = preds['seg_edge']
        seg_body = preds['seg_body']
        contrast_logits = preds['logits'] # [(b h w) (c m)]
        contrast_target = preds['target']  # [(b h w)]
        b, _, h, w = seg_edge.size()
        # contrast_target = contrast_target.reshape(-1, h, w) # [b h w]
        # [(b h w) c m]
        contrast_logits = contrast_logits.reshape(-1, self.num_classes, self.num_prototype) 
        
        # extract the pixels predicted as boundary (last prototype in each class)
        edge_contrast_logits = torch.zeros_like(contrast_target).cuda() # [(b h w)]
        for i in range(self.num_classes):
            edge_cls_logits = contrast_logits
            edge_mask = contrast_target == float(
                self.num_prototype - 1) + (self.num_prototype * i)  # [(b h w)]
            edge_contrast_logits.masked_scatter_(edge_mask.bool(), 1)
        
        
        