"""
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
from lib.loss.loss_helper import FSCELoss


# todo class uniform sampling in cropping the images using confidence map
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

        self.seg_criterion = FSCELoss(configer=configer)
    
    def edge_attention(self, input, target, edge):
        ''' 
        intput: [b c h w] (sem_seg)
        target: [b h w] (sem_gt)
        '''
        filler = torch.ones_like(target) * self.ignore_label
        #! only evaluate the segmentation performance where it is predicted as edge
        edge_attn_loss = self.seg_criterion(input, torch.where(edge.max(1)[0] > 0.8, target, filler))
        return edge_attn_loss
    
    def bce2d(self, input, target):
        # [b 1 h w] -> [b h w 1] -> [1 (b h w)]
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()
        pos_index = (target_t == 1) # edge
        neg_index = (target_t == 0) # non-edge
        ignore_index = (target_t > 1)
        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num
        weight[ignore_index] = 0
        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

        return loss

    def forward(self, preds, gt_boundary, sem_gt):
        ''' 
        gt_boundary: 1->boundary 0->non-boundary
        Use contrast target instead of sem_gt for more detailed supervision.
        '''
        b, h, w = sem_gt.size(0), sem_gt.size(1), sem_gt.size(2)
        
        seg_edge = preds['seg_edge'] # [b 1 h w]
        # seg_body = preds['seg_body']
        # contrast_logits = preds['logits']  # [(b h w) (c m)]
        # contrast_target = preds['target']  # [(b h w)]
        # confidence = preds['confidence']
        # b, _, h, w = seg_edge.size()
        # h_gt, w_gt = target.size()[1:]

        # # extract the pixels predicted as boundary (last prototype in each class)
        # edge_contrast_logits = torch.zeros_like(contrast_target).cuda()  # [(b h w)]
        # body_contrast_logits = torch.zeros_like(contrast_target).cuda()  # [(b h w)]
        # # [(b h w)]
        # pred_logits = torch.scatter(src=contrast_logits, dim=1, index=contrast_target.int())
        # for i in range(self.num_classes):
        #     edge_cls_logits = torch.amax(contrast_logits, dim=1)  # [(b h w) m]
        #     edge_mask = contrast_target == float(
        #         self.num_prototype - 1) + (self.num_prototype * i)  # [(b h w)]

        #     edge_contrast_logits.masked_scatter_(edge_mask.bool(), pred_logits)
        #     body_contrast_logits.masked_scatter_(~edge_mask.bool(), pred_logits)

        # #! set the boundary gt pixel to ignore label to generate body gt
        # contrast_target.masked_fill_(gt_boundary, self.ignore_label)
        # contrast_target = contrast_target.reshape(-1, h, w)  # [b h w]

        # seg_body = F.interpolate(input=seg_body, size=(
        #     h_gt, w_gt), mode='bilinear', align_corners=True)
        # gt_body = torch.ones_like(sem_gt) * self.ignore_label
        # gt_body.masked_scatter_(~gt_boundary.bool(), sem_gt)
        # confidence = F.interpolate(input=confidence.unsqueeze(1), size=(
        #     h_gt, w_gt), mode='bilinear', align_corners=True)  # [b 1 h w]
        # body_loss = self.body_seg_criterion(
        #     seg_body, gt_body, confidence_wieght=confidence.squeeze(1).detach())
        
        seg_edge = F.interpolate(input=seg_edge, size=(
                h, w), mode='bilinear', align_corners=True)
        edge_bce_loss = self.bce2d(seg_edge, gt_boundary) 
        return edge_bce_loss