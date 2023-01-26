# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: RainbowSecret
# Microsoft Research
# yuyua@microsoft.com
# Copyright (c) 2018
##
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper
from lib.utils.tools.logger import Logger as Log
from lib.models.modules.hanet_attention import HANet_Conv
from lib.models.modules.contrast import momentum_update, l2_normalize, ProjectionHead
from lib.models.modules.uncertainty_head import UncertaintyHead
from lib.models.modules.sinkhorn import distributed_sinkhorn
from lib.models.modules.prob_proto_seg_head import ProbProtoSegHead
from lib.models.modules.boundary_head import BoundaryHead
from hkustgz_segnet.lib.models.modules.bayesian_uncertainty_head import BayesianUncertaintyHead

from timm.models.layers import trunc_normal_
from einops import rearrange, repeat


class HRNet_W48_Prob_Bound_Proto(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48_Prob_Bound_Proto, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        # prototype config
        self.use_prototype = self.configer.get('protoseg', 'use_prototype')
        self.update_prototype = self.configer.get(
            'protoseg', 'update_prototype')
        self.pretrain_prototype = self.configer.get(
            'protoseg', 'pretrain_prototype')
        self.use_probability = self.configer.get('protoseg', 'use_probability')
        self.use_uncertainty = self.configer.get('protoseg', 'use_uncertainty')
        self.use_boundary = self.configer.get('protoseg', 'use_boundary')
        self.use_ros = self.configer.get('ros', 'use_ros')

        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 720
        self.proj_dim = self.configer.get('protoseg', 'proj_dim')

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10)
        )
        
        if self.use_uncertainty:
            self.uncertainty_head = BayesianUncertaintyHead(configer=configer)

        self.proj_head = ProjectionHead(in_channels, self.proj_dim)
        
        self.prob_seg_head = ProbProtoSegHead(configer=configer)

        if self.use_boundary:
            self.boundary_head = BoundaryHead(
                configer=configer, in_channels=self.proj_dim, mid_channels=self.configer.get(
                    'protoseg', 'boundary_head_mid_channel'))
            # self.boundary_attention_module = BoundaryAttentionModule(configer=configer)

        self.feat_norm = nn.LayerNorm(self.proj_dim)  # normalize each row
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.bound_norm = nn.LayerNorm(2)

    def forward(self, x_, gt_semantic_seg=None, gt_boundary=None, pretrain_prototype=False):
        x = self.backbone(x_)
        b, _, h, w = x[0].size()  # 128, 256

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(
            h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(
            h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(
            h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)  # sent to boundary head

        c = self.cls_head(feats)  # 720
        c = self.proj_head(c)  

        del feats, feat1, feat2, feat3, feat4

        gt_size = c.size()[2:]
        c = rearrange(c, 'b c h w -> (b h w) c')
        c = self.feat_norm(c)  # ! along channel dimension
        c = l2_normalize(c)  # ! l2_norm along num_class dimension

        c = rearrange(c, '(b h w) c -> b c h w',
                      h=gt_size[0], w=gt_size[1])

        boundary_pred = None
        if self.use_boundary:
            boundary_map = self.boundary_head(c)  # [b 2 h w]
            boundary_map = rearrange(boundary_map, 'b c h w -> (b h w) c') # [(b h w) 2]
            boundary_map = self.bound_norm(boundary_map)
            boundary_map = rearrange(boundary_map, '(b h w) c -> b c h w',
                      h=gt_size[0], w=gt_size[1])

        if self.use_uncertainty:
            # masked_c: the most uncertain pixels are masked out
            # prob_c: only sample once for loss calculation to promote diversity
            uncertainty, prob_c, conv_uncertainty, mean, var = self.uncertainty_head(c)
            
        preds = self.prob_seg_head(
            c, gt_semantic_seg=gt_semantic_seg, boundary_pred=boundary_pred,
            gt_boundary=gt_boundary)
        if self.use_uncertainty and self.configer.get('phase') == 'train':
            preds['prob_x'] = prob_c
            preds['x_mean'] = mean
            preds['x_var'] = var

        return preds


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
        self.use_probability = self.configer.get('protoseg', 'use_probability')
        self.use_boundary = self.configer.get('protoseg', 'use_boundary')
        self.use_ros = self.configer.get('ros', 'use_ros')

        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 720
        self.proj_dim = self.configer.get('protoseg', 'proj_dim')

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10)
        )
        if self.use_probability:
            self.uncertainty_head = UncertaintyHead(
                in_feat=self.configer.get('protoseg', 'proj_dim'),
                out_feat=self.configer.get('protoseg', 'proj_dim'))  # predict variance of each gaussian

        self.proj_head = ProjectionHead(in_channels, self.proj_dim)

        self.prob_seg_head = ProbProtoSegHead(configer=configer)

        if self.use_boundary:
            self.boundary_head = BoundaryHead(
                configer=configer, in_channels=self.proj_dim, mid_channels=self.configer.get(
                    'protoseg', 'boundary_head_mid_channel'))
            # self.boundary_attention_module = BoundaryAttentionModule(configer=configer)

        self.feat_norm = nn.LayerNorm(self.proj_dim)  # normalize each row
        self.mask_norm = nn.LayerNorm(self.num_classes)

    def forward(self, x_, gt_semantic_seg=None, gt_boundary=None, pretrain_prototype=False):
        x = self.backbone(x_)
        b, _, h, w = x[0].size()  # 128, 256

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(
            h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(
            h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(
            h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)  # sent to boundary head

        c = self.cls_head(feats)  # 720
        c = self.proj_head(c)  # 256

        del feats, feat1, feat2, feat3, feat4

        #! seghead
        gt_size = c.size()[2:]
        c = rearrange(c, 'b c h w -> (b h w) c')
        c = self.feat_norm(c)  # ! along channel dimension
        c = l2_normalize(c)  # ! l2_norm along num_class dimension

        c = rearrange(c, '(b h w) c -> b c h w',
                      h=gt_size[0], w=gt_size[1])

        boundary_pred = None
        if self.use_boundary:
            boundary_pred = self.boundary_head(c)  # [b 2 h w]
            # boundary_pred = rearrange(boundary_pred, 'b c h w -> (b h w) c') # [(b h w) 2]

        c_var = None
        if self.use_probability:
            c_var = self.uncertainty_head(c)  # ! b c h w, log(sigma^2)
            c_var = torch.exp(c_var)  # ! variance x_var should > 0!

            # c_var = rearrange(c_var, 'b c h w -> (b h w) c')
            # c_var = self.feat_norm(c_var)  # ! along channel dimension
            # c_var = l2_normalize(c_var)  # ! l2_norm along num_class dimension

            # c_var = rearrange(c_var, '(b h w) c -> b c h w',
            #                   h=gt_size[0], w=gt_size[1])

        preds = self.prob_seg_head(
            c, c_var, gt_semantic_seg=gt_semantic_seg, boundary_pred=boundary_pred,
            gt_boundary=gt_boundary)

        return preds


class HRNet_W48_Proto(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48_Proto, self).__init__()
        self.configer = configer
        self.gamma = self.configer.get('protoseg', 'gamma')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.use_prototype = self.configer.get('protoseg', 'use_prototype')
        self.update_prototype = self.configer.get(
            'protoseg', 'update_prototype')
        self.pretrain_prototype = self.configer.get(
            'protoseg', 'pretrain_prototype')
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 720
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10)
        )

        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.num_prototype, in_channels),
            requires_grad=True)

        self.proj_head = ProjectionHead(in_channels, in_channels)
        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        cosine_similarity = torch.mm(
            _c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())
        # [c m k] -> [(c m) k] -> [k (c m)]
        # n k * k (c m)] = [n (c m)]

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        #! clustering for each class and momentum update of prototypes
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            # self.num_prototype x embedding_dim, @: mat product
            f = m_q.transpose(0, 1) @ c_q

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + \
                (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)

        #! update proto in distributed learning
        if dist.is_available() and dist.is_initialized():  # distributed learning
            protos = self.prototypes.data.clone()
            """
            To get average result across all gpus: 
            first average the sum
            then sum the tensors on all gpus, and copy the sum to all gpus
            """
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

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

        # we use prototypes for classification, so input of proj head is c instead of feats
        c = self.proj_head(c)
        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)  # ! along channel dimension
        _c = l2_normalize(_c)  # ! along num_class dimension

        #! channel-wise normalize the prototypes because of cosine simialrity
        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # n: h*w, k: num_class, m: num_prototype
        # unnormalized cosine similarity
        masks = torch.einsum(
            'nd,kmd->nmk', _c, self.prototypes)  # similarity_mat [n num_proto num_cls]

        out_seg = torch.amax(masks, dim=1)  # [n num_cls]
        # normalized cosine similarity
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w",
                            b=feats.shape[0], h=feats.shape[2])

        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[
                                   2:], mode='nearest').view(-1)
            contrast_logits, contrast_target = self.prototype_learning(
                _c, out_seg, gt_seg, masks)
            return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}

        return out_seg


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
        # out: for segmentation evaluation?, emb: for loss calculation
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
