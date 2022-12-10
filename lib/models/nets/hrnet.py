# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: RainbowSecret
# Microsoft Research
# yuyua@microsoft.com
# Copyright (c) 2018
##
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper
from lib.utils.tools.logger import Logger as Log
from lib.models.modules.hanet_attention import HANet_Conv
from lib.models.modules.contrast import momentum_update, l2_normalize, ProjectionHead
from lib.models.modules.prob_embedding import UncertaintyHead, PredictionHead
from lib.models.modules.sinkhorn import distributed_sinkhorn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat


class HRNet_W48_Prob_Contrast_Proto(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48_Prob_Contrast_Proto, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        # prototype config
        self.use_prototype = self.configer.get('protoseg', 'use_prototype')
        self.update_prototype = self.configer.get(
            'protoseg', 'update_prototype')
        self.pretrain_prototype = self.configer.get(
            'protoseg', 'pretrain_prototype')
        # similarity measure between features and prototypes
        self.sim_measure = self.configer.get('protoseg', 'similarity_measure')
        self.use_probability = self.configer.get('protoseg', 'use_probability')
        # multi_proxy = True: sum of M prototypes
        # multi_proxy = False: each proto corresponds to a pixel
        self.multi_proxy = self.configer.get('protoseg', 'multi_proxy')

        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 720
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Drop2(0.10)
        )

        self.uncertainty_head = UncertaintyHead(
            in_channels=in_channels)  # predict variance of each gaussian

        # [cls_num, proto_num, channel/k]
        self.prototypes = nn.Parameter(torch.zeros(
            self.num_classes, self.num_prototype, in_channels), requires_grad=False)
        if self.use_probability:
            self.proto_var = nn.Parameter(torch.ones(
                self.num_classes, self.num_prototype, in_channels), requires_grad=True)
        else:
            trunc_normal_(self.prototypes, std=0.02)

        self.proj_head = ProjectionHead(in_channels, in_channels)
        self.proj_dim = self.configer.get('prob_contrast', 'proj_dim')
        self.feat_norm = nn.LayerNorm(in_channels)  # normalize each row
        self.mask_norm = nn.LayerNorm(self.num_classes)

    def similarity_compute(self, x, x_var=None):
        """ 
        Similarity between pixel embeddings and prototypes

        x: batch-wise pixel embeddings [(b h w) c]
        sigma: var of x [(b h w) c]
        prototypes: [cls_num, proto_num, channel/c]
        c: similarity [] 
        gt_seg: [b*h*w]?
        """
        #! similarity measure between features and prototypes
        if self.use_probability and self.sim_measure == "mls":
            pt_num = x.shape[0]
            # [num_class, num_prototype, b*h*w, c]
            x_tile = x.unsqueeze(0).expand(
                self.num_classes, self.num_prototype, pt_num, x.shape[-1])
            x_var_tile = x_var.unsqueeze(0).expand(
                self.num_classes, self.num_prototype, pt_num, x.shape[-1])

            # [num_class, num_prototype, b*h*w, c]
            proto_mean = self.prototypes.expand(
                self.num_classes, self.num_prototype, pt_num, self.prototypes.shape[-1])
            proto_var = self.proto_var.expand(
                self.num_classes, self.num_prototype, pt_num, self.prototypes.shape[-1])

            square_term = torch.square(
                x_tile - proto_mean) / (x_var_tile + proto_var) + torch.log(x_var_tile + proto_var)
            # [num_class, num_prototype, b*h*w]
            sim_mat = - torch.sum(square_term, dim=2) / \
                2 - x_tile.shape[-1] * np.log(2 * np.pi) / 2
            sim_mat = sim_mat.permute(2, 0, 1)

        elif self.use_probability is False and self.sim_measure == "cosine":
            # l2-normalization for cosine similarity
            x = self.feat_norm(x)
            x = l2_normalize(x)
            self.prototypes.data.copy_(l2_normalize(self.prototypes))

            # batch product (toward d dimension) -> cosine simialrity between fea and prototypes
            # n: h*w, k: num_class, m: num_prototype
            sim_mat = torch.einsum('nd,kmd->nmk', x, self.prototypes)

        return sim_mat

    def prototype_learning(self, sim_mat, gt_seg):
        """ 
        Prototype selection and update
        Non-uniform matching flow using optimal transport.
        min(tr(M * T_t) + lamda * tr(T * log(T) - 1 * 1_T)_t)
        M = 1 - C, M: cost matrx, C: similarity matrix
        T: optimal transport plan

        sim_mat: [b*h*w, num_cls, num_proto]
        gt_seg: [b*h*w]
        """
        # largest score inside a class, # [num_class, b*h*w]
        sim_mat = torch.amax(sim_mat, dim=1)
        pred_seg = torch.max(sim_mat, dim=0)[1]  # [b*h*w]
        proto_activate = torch.ones_like(sim_mat)
        mask = (gt_seg == pred_seg.view(-1))  # [b*h*w]

        #! pixel-to-prototype online clustering
        if not self.multi_proxy:
            for i in range(self.num_classes):
                init_q = sim_mat[..., i]  # [b*h*w, num_proto]
                # select the right preidctions inside i-th class
                init_q = init_q[gt_seg == i, ...]  # [n, num_proto]
                if init_q.shape[0] == 0:
                    continue

                q, indexs = distributed_sinkhorn(init_q)
        return

    def forward(self, x_, gt_semantic_seg=None, pretrain_prototype=False):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(
            h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(
            h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(
            h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        c = self.cls_head(feats)

        c = self.proj_head(c)
        _c = rearrange(c, 'b c h w -> (b h w) c')

        x_var = None
        if self.use_probability:
            x_var = self.uncertainty_head(_c)  # (b h w) c

        sim_mat = self.similarity_compute(
            x, x_var)  # [b*h*w, num_cls, num_proto]

        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[
                                   2:], mode="nearest").view(-1)  # [b*h*w]?
            self.prototype_learning(sim_mat, gt_seg)


class HRNet_W48(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(
            h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(
            h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(
            h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out = self.cls_head(feats)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)),
                            mode="bilinear", align_corners=True)
        return out


class HRNet_W48_CONTRAST(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48_CONTRAST, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.proj_dim = self.configer.get('contrast', 'proj_dim')

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )

        self.proj_head = ProjectionHead(
            dim_in=in_channels, proj_dim=self.proj_dim)

    def forward(self, x_, with_embed=False, is_eval=False):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(
            h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(
            h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(
            h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out = self.cls_head(feats)

        emb = self.proj_head(feats)
        return {'seg': out, 'embed': emb}


class HRNet_W48_OCR_CONTRAST(nn.Module):
    def __init__(self, configer):
        super(HRNet_W48_OCR_CONTRAST, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.proj_dim = self.configer.get('contrast', 'proj_dim')

        in_channels = 720
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.spatial_ocr_block import SpatialGather_Module
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Module
        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 bn_type=self.configer.get('network', 'bn_type'))
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1,
                      stride=1, padding=0, bias=True)
        )

        self.proj_head = ProjectionHead(
            dim_in=in_channels, proj_dim=self.proj_dim)

    def forward(self, x_, with_embed=False, is_eval=False):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(
            h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(
            h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(
            h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        emb = self.proj_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        return {'seg': out, 'seg_aux': out_aux, 'embed': emb}


class HRNet_W48_MEM(nn.Module):
    def __init__(self, configer, dim=256, m=0.999, with_masked_ppm=False):
        super(HRNet_W48_MEM, self).__init__()
        self.configer = configer
        self.m = m
        self.r = self.configer.get('contrast', 'memory_size')
        self.with_masked_ppm = with_masked_ppm

        num_classes = self.configer.get('data', 'num_classes')

        self.encoder_q = HRNet_W48_CONTRAST(configer)

        self.register_buffer(
            "segment_queue", torch.randn(num_classes, self.r, dim))
        self.segment_queue = nn.functional.normalize(
            self.segment_queue, p=2, dim=2)
        self.register_buffer("segment_queue_ptr", torch.zeros(
            num_classes, dtype=torch.long))

        self.register_buffer(
            "pixel_queue", torch.randn(num_classes, self.r, dim))
        self.pixel_queue = nn.functional.normalize(
            self.pixel_queue, p=2, dim=2)
        self.register_buffer("pixel_queue_ptr", torch.zeros(
            num_classes, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, im_q, lb_q=None, with_embed=True, is_eval=False):
        if is_eval is True or lb_q is None:
            ret = self.encoder_q(im_q, with_embed=with_embed)
            return ret

        ret = self.encoder_q(im_q)

        q = ret['embed']
        out = ret['seg']

        return {'seg': out, 'embed': q, 'key': q.detach(), 'lb_key': lb_q.detach()}


class HRNet_W48_OCR(nn.Module):
    def __init__(self, configer):
        super(HRNet_W48_OCR, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 720
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.spatial_ocr_block import SpatialGather_Module
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Module
        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 bn_type=self.configer.get('network', 'bn_type'))
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1,
                      stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(
            h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(
            h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(
            h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(out_aux, size=(
            x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)),
                            mode="bilinear", align_corners=True)
        return out_aux, out


class HRNet_W48_OCR_B(nn.Module):
    """
    Considering that the 3x3 convolution on the 4x resolution feature map is expensive,
    we can decrease the intermediate channels from 512 to 256 w/o performance loss.
    """

    def __init__(self, configer):
        super(HRNet_W48_OCR_B, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 720  # 48 + 96 + 192 + 384
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                256, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.spatial_ocr_block import SpatialGather_Module
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Module
        self.ocr_distri_head = SpatialOCR_Module(in_channels=256,
                                                 key_channels=128,
                                                 out_channels=256,
                                                 scale=1,
                                                 dropout=0.05,
                                                 bn_type=self.configer.get('network', 'bn_type'))

        self.cls_head = nn.Conv2d(
            256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                256, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(256, self.num_classes, kernel_size=1,
                      stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(
            h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(
            h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(
            h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(out_aux, size=(
            x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)),
                            mode="bilinear", align_corners=True)
        return out_aux, out


class HRNet_W48_OCR_B_HA(nn.Module):
    """
    Considering that the 3x3 convolution on the 4x resolution feature map is expensive,
    we can decrease the intermediate channels from 512 to 256 w/o performance loss.
    """

    def __init__(self, configer):
        super(HRNet_W48_OCR_B_HA, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 720  # 48 + 96 + 192 + 384
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                256, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.spatial_ocr_block import SpatialGather_Module
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Module
        self.ocr_distri_head = SpatialOCR_Module(in_channels=256,
                                                 key_channels=128,
                                                 out_channels=256,
                                                 scale=1,
                                                 dropout=0.05,
                                                 bn_type=self.configer.get('network', 'bn_type'))
        self.cls_head = nn.Conv2d(
            256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                256, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(256, self.num_classes, kernel_size=1,
                      stride=1, padding=0, bias=True)
        )

        self.ha1 = HANet_Conv(
            384, 384, bn_type=self.configer.get('network', 'bn_type'))
        self.ha2 = HANet_Conv(
            192, 192, bn_type=self.configer.get('network', 'bn_type'))
        self.ha3 = HANet_Conv(
            96, 96, bn_type=self.configer.get('network', 'bn_type'))
        self.ha4 = HANet_Conv(
            48, 48, bn_type=self.configer.get('network', 'bn_type'))

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        x[0] = x[0] + self.ha1(x[0])
        x[1] = x[1] + self.ha1(x[1])
        x[2] = x[2] + self.ha1(x[2])
        x[3] = x[3] + self.ha1(x[3])

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(
            h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(
            h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(
            h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(out_aux, size=(
            x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)),
                            mode="bilinear", align_corners=True)
        return out_aux, out
