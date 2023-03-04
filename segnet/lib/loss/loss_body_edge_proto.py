"""
Code adapted from 'Improving Semantic Segmentation via Decoupled Body and Edge Supervision'.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.tools.logger import Logger as Log
from einops import rearrange, repeat


class EdgeBodyLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(EdgeBodyLoss, self).__init__()
        self.configer = configer
        
    def forward(self, preds, target, gt_boundary):
        seg_edge = preds['seg_edge']
        seg_body = preds['seg_body']
        contrast_logits = preds['logits']
        contrast_target = preds['target']  
        
        c
        
        