import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.models.tools.module_helper import ModuleHelper
from einops import rearrange, repeat


class ResBlock(nn.Module):
    def __init__(self, configer):
        super(ResBlock, self).__init__()

        self.configer = configer

        in_dim = self.configer.get('protoseg', 'proj_dim')
        out_dim = in_dim
        stride = 1

        if in_dim == out_dim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=stride)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class GCN(nn.Module):
    def __init__(self, configer):
        super(GCN, self).__init__()
        self.configer = configer
        self.bin_size_h = self.configer.get('protoseg', 'bin_size_h')
        self.bin_size_w = self.configer.get('protoseg', 'bin_size_w')
        num_node = self.bin_size_h * self.bin_size_w
        # num_channel = self.configer.get('protoseg', 'proj_dim')
        num_channel = 720

        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)

    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out


class LocalRefinementModule(nn.Module):
    ''' 
    Video Object Segmentation with Adaptive Feature Bank and Uncertain-Region Refinement

    Use the pixels which havehigh confidence in classification to refine the other uncertain points in its neighborhood.
    '''

    def __init__(self, configer):
        super(LocalRefinementModule, self).__init__()
        self.configer = configer

        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.bin_size_h = self.configer.get('protoseg', 'bin_size_h')
        self.bin_size_w = self.configer.get('protoseg', 'bin_size_w')
        self.proj_dim = self.configer.get('protoseg', 'proj_dim')

        self.dropout = nn.Dropout2d(0.1)
        self.conv_cam = nn.Conv2d(self.proj_dim, self.num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.pool_cam = nn.AdaptiveAvgPool2d((self.bin_size_h, self.bin_size_w))
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.relu = nn.ReLU(inplace=True)
        # self.avg_pool = nn.AdaptiveAvgPool2d()
        self.local_res_module = ResBlock(configer=configer)
        self.gcn = GCN(configer=configer)

    def calc_pred_uncertainty(self, sim_mat):
        ''' 
        sim_mat: [n c]
        '''
        sim_mat = F.softmax(sim_mat, dim=1)
        score_top, _ = sim_mat.topk(k=2, dim=1)
        pred_uncertainty = score_top[:, 0] / (score_top[:, 1] + 1e-8)
        pred_uncertainty = torch.exp(1 - pred_uncertainty)

        return pred_uncertainty

    def patch_split(self, x, bin_size_h, bin_size_w):
        """
        b c (bh rh) (bw rw) -> b (bh bw) rh rw c
        """
        b, c, h, w = x.size()
        patch_size_h = h // bin_size_h
        patch_size_w = w // bin_size_w
        x = x.view(b, c, bin_size_h, patch_size_h, bin_size_w, patch_size_w)
        # [b, bin_size_h, bin_size_w, patch_size_h, patch_size_w, c]
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        # [b, bin_size_h*bin_size_w, patch_size_h, patch_size_w, c]
        x = x.view(b, -1, patch_size_h, patch_size_w, c)

        return x

    def patch_recover(self, x, bin_size_h, bin_size_w):
        """
        b (bh bw) rh rw c -> b c (bh rh) (bw rw)
        """
        b, n, patch_size_h, patch_size_w, c = x.size()
        h = patch_size_h * bin_size_h
        w = patch_size_w * bin_size_w
        x = x.view(b, bin_size_h, bin_size_w, patch_size_h, patch_size_w, c)
        # [b, c, bin_size_h, patch_size_h, bin_size_w, patch_size_w]
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, h, w)

        return x

    def forward(self, sim_mat, x, x_var=None):
        ''' 
        sim_mat: [b c h w]
        x: [b c h w]
        '''
        residual = x
        b, k, h, w = x.size(0), x.size(1), x.size(-2), x.size(-1)
        pred_uncertainty = self.calc_pred_uncertainty(sim_mat)

        cam = self.conv_cam(self.dropout(x))  # [B, C, H, W]
        #! for each patch, the overl all(pooling) score for K classes
        cls_score = self.sigmoid(self.pool_cam(cam))  # [B, C, bin_num_h, bin_num_w]

        # [B, bin_num_h * bin_num_w, rH, rW, c]
        cam = self.patch_split(cam, self.bin_size_h, self.bin_size_w)
        # [B, bin_num_h * bin_num_w, rH, rW, k] [1, 32, 32, 32, 720]
        x = self.patch_split(x, self.bin_size_h, self.bin_size_w)

        B = cam.shape[0]
        rH = cam.shape[2]
        rW = cam.shape[3]
        K = cam.shape[-1]
        C = x.shape[-1]
        cam = cam.view(B, -1, rH*rW, K)  # [B, bin_num_h * bin_num_w, rH * rW, c] [1, 32, 1024, 19]
        x = x.view(B, -1, rH*rW, C)  # [B, bin_num_h * bin_num_w, rH * rW, k] [1, 32, 1024, 720]

        # [B, bin_num_h * bin_num_w, K, 1] [1, 32, 19, 1]
        bin_confidence = cls_score.view(B, K, -1).transpose(1, 2).unsqueeze(3)
        #! pixel_confidence: inside each patch, for each pixel, its prob. belonging to K classes
        cam = F.softmax(cam, dim=2)
        #! local cls centers (weigted sum (weight: bin_confidence))
        #! torch.matmul: aggregate inside a patch
        # [B, bin_num_h * bin_num_w, c, Ck [1, 32, 19, 720]
        cam = torch.matmul(cam.transpose(2, 3), x) * bin_confidence
        cam = self.gcn(cam)  # [B, bin_num_h * bin_num_w, K, C] [1, 32, 19, 720]

        #! local refinement mask
        local_max = torch.max(cam, dim=1)[0]  # [b, 19, 720] cls-wise local max
        x = x.permute(0, 3, 1, 2)  # [1, 720, 32, 1024]
        q = self.local_res_module(x)  # [1, 720, 32, 1024]

        return
